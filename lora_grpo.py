# finetune_demo/peft_grpo_duallora.py
import argparse, os, random, logging, csv, time, torch
import numpy as np
from typing import List
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import HfDeepSpeedConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils.omnimedqkv import OmniMedVQA_Dataset
from datetime import datetime
# RL 组件
from rl.embedder import MedClipEmbedder
from rl.rewards import build_rewards, standardize_each_then_mix, group_norm_and_clip
from rl.roi import set_local_crops_mode
from rl.utils import (
    split_prompt_from_sft_batch, per_token_logps, first_eos_mask, ensure_cogvlm_images
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grpo-duallora")

def parse_args():
    p = argparse.ArgumentParser("GRPO for Med-CogVLM (dual-LoRA, ZeRO-2)")

    # 路径
    p.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model")
    p.add_argument("--dataset_path", type=str, default=None, help="Path to the conversation dataset")
    p.add_argument("--save_path", type=str, default=None, help="Path to save the finetuned model")
    p.add_argument("--actor_lora_path", type=str, default=None)
    p.add_argument("--ds_config", type=str, default="finetune_demo/ds_config.yaml")
    p.add_argument("--torch_type", type=str, default="torch.bfloat16")

    # 数据 & batch
    p.add_argument("--max_input_len", type=int, default=128)
    p.add_argument("--max_output_len", type=int, default=128)
    p.add_argument("--train_dataset_rate", type=float, default=1)
    p.add_argument("--batch_size", type=int, default=1)

    # 生成（K）
    p.add_argument("--k", type=int, default=4)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target", type=eval,default=["vision_expert_query_key_value","language_expert_query_key_value"])

    # 优化 & 调度
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_step", type=int, default=100)

    # KL
    p.add_argument("--kl_coef", type=float, default=0.02)

    # 奖励权重
    p.add_argument("--w_acc", type=float, default=0.30)      #早期（acc < 35%）—偏探索/看图 w_acc=0.30, w_vec_g=0.35, w_vec_l=0.25, w_dep=0.10
    p.add_argument("--w_vec_g", type=float, default=0.35)   #中期（35% ≤ acc < 65%）—稳步收敛 w_acc=0.45, w_vec_g=0.25, w_vec_l=0.20, w_dep=0.10
    p.add_argument("--w_vec_l", type=float, default=0.25)   #后期（acc ≥ 65%）—强化真图依赖 w_acc=0.40, w_vec_g=0.25, w_vec_l=0.20, w_dep=0.15
    p.add_argument("--w_dep", type=float, default=0.10)
    p.add_argument("--dep_blur_sigma", type=float, default=3.0)
    p.add_argument("--dep_shuffle_grid", type=int, default=4)
    p.add_argument("--n_rois", type=int, default=9)  # 默认九裁剪

    # 采样策略
    p.add_argument("--visdep_sidecar", type=str, default="./dataset/iodep_scores.jsonl", help="JSON/JSONL: {image_path, visdep_score}")
    p.add_argument("--visdep_min_score", type=float, default=None, help="Filter samples with score < threshold")
    p.add_argument("--visdep_weighting", action="store_true", help="Return visdep_weight = 0.5+0.5*score")
    p.add_argument("--dep_sidecar", type=str, default="./dataset/visdep_scores.jsonl", help="JSONL/JSON: mapping question_id → dep")
    p.add_argument("--dep_min", type=float, default=0.0, help="Filter threshold (e.g., 0.0 means drop samples with dep < 0)")
    p.add_argument("--keep_if_dep_missing", action="store_true", help="Keep samples if dep is missing in sidecar")

    # 稳定化
    p.add_argument("--adv_clip", type=float, default=5.0)
    # acc 日志路径文件
    p.add_argument("--log_path", type=str, default="./log/rewards.log")
    p.add_argument("--acc_log_path", type=str, default="./log/acc_rewards.log")
    p.add_argument("--vec_g_log_path", type=str, default="./log/vec_g_rewards.log")
    p.add_argument("--vec_l_log_path", type=str, default="./log/vec_l_rewards.log")
    p.add_argument("--dep_log_path", type=str, default="./log/dep_rewards.log")

    p.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return p.parse_args()

def main():
    args = parse_args()
    dtype = eval(args.torch_type)
    
    # DeepSpeed / Accelerate
    import yaml
    with open(args.ds_config, "r") as f:
        ds_cfg = yaml.safe_load(f)
    hf_ds_config = HfDeepSpeedConfig(ds_cfg)
    ds_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)

    # Tokenizer & 左填充
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left" #设置左填充 → 所有句子尾部对齐，生成段（completion）在右边对齐，方便对比和计算 loss/reward。

    # 基座
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )

    # 数据集
    dataset = OmniMedVQA_Dataset(
        root_dir=args.dataset_path,
        tokenizer=tokenizer,
        model=base,
        torch_type=dtype,
        input_length=args.max_input_len,
        output_length=args.max_output_len,
        visdep_sidecar=args.visdep_sidecar,
        visdep_min_score=args.visdep_min_score,
        visdep_weighting=args.visdep_weighting,
        dep_sidecar=args.dep_sidecar,
        dep_min=args.dep_min,
        keep_if_dep_missing=args.keep_if_dep_missing,
    )

    total = len(dataset)
    train_size = int(args.train_dataset_rate * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
      
    # ===== 采样加权：让高视觉依赖样本更常被抽到（仅训练集）=====
    subset_indices = train_ds.indices  # random_split 之后的原始索引
    scores = torch.tensor(
        [dataset.samples[i]['visdep_score'] for i in subset_indices],
        dtype=torch.float32,
    )

    gamma = 2.0  # 温度/幂指数，越大越偏向高分
    weights = (0.5 + 0.5 * scores).pow(gamma)  # ∈ [0.5^γ, 1]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(subset_indices),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=dataset.custom_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.custom_collate_fn
    )

    # 单 LoRA（actor）
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target,
        inference_mode=False,
    )
    if args.actor_lora_path:
        model = PeftModel.from_pretrained(base, args.actor_lora_path, is_trainable=True)
        ckpt_name = os.path.basename(args.actor_lora_path.rstrip("/"))
        step_str = ckpt_name.split("_")[2]
        step_global=int(step_str)
        logger.info(f"Resumed from {args.actor_lora_path}, step={step_global}")
    else:
        model = get_peft_model(base, lcfg)
        step_global = 0

    logger.info(">> Loading MedCLIP embedder ...")
    embedder = MedClipEmbedder()
    embedder.eval()
    logger.info(">> MedCLIP ready.")
    set_local_crops_mode(n_rois=args.n_rois)

    # Optim
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_loader) * args.num_epochs),
    )

    model, optim, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optim, train_loader, val_loader, lr_scheduler
    )

    os.makedirs(args.save_path, exist_ok=True)

    def set_seed_time(step: int = 0, i: int = 0) -> int:
        """
        时间种子：
        - 默认仅用当前时间 + pid；可选再混入 step / i；
        - 自动混入分布式 rank（若已初始化）。
        返回最终 seed 便于日志记录。
        """
        seed = time.time_ns() ^ os.getpid() ^ (step * 0x9E3779B1) ^ (i * 0x85EBCA77)
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                seed ^= (dist.get_rank() * 0xC2B2AE35)
        except Exception:
            pass
        seed &= 0xFFFFFFFF

        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return seed

    # ---- 统计器：累计 + EMA（平滑） ----
    metrics_sum = defaultdict(float)   # 累计和
    metrics_n = 0                      # 累计步数（自上次写盘起）
    ema = {}                           # 指标的 EMA
    EMA_BETA = 0.98                    # 平滑系数，可在 0.95~0.995 间调

    def _ema_update(name, value):
        if value is None: 
            return
        if name not in ema:
            ema[name] = value
        else:
            ema[name] = EMA_BETA * ema[name] + (1.0 - EMA_BETA) * value

    # ---- CSV 路径与表头 ----
    csv_path = os.path.join(args.save_path, "train_metrics.csv")
    if accelerator.is_main_process and (not os.path.exists(csv_path)):
        os.makedirs(args.save_path, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "wall_time","epoch","step_global",
                "loss_mean","kl_mean","r_mean_mean",
                "adv_mean_mean","adv_std_mean",
                "parts_acc_mean","parts_vec_g_mean","parts_vec_l_mean","parts_dep_mean",
                "loss_ema","kl_ema","r_mean_ema","adv_mean_ema","adv_std_ema",
                "parts_acc_ema","parts_vec_g_ema","parts_vec_l_ema","parts_dep_ema"
            ])

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, initial=step_global, desc=f"Epoch {epoch}")):
            step_global += 1

            # =========================
            # 1) 从 SFT 批里切出 prompt 段（不含 completion）
            #    依据 labels == -100 的掩码来确定 prompt 长度
            #    同时，如果 batch 里有 token_type_ids（tti），也切相同的 prompt 段
            # =========================
            prompts, prompt_attn, prompt_lens, images_raw, extra = split_prompt_from_sft_batch(batch, tokenizer) 
            prompt_tti = extra.get("prompt_token_type_ids", None)

            if (images_raw is not None) and (prompt_tti is None):
                raise RuntimeError("batch 缺少 token_type_ids；请在 collate_fn 中提供（与 SFT 保持一致）。")

            B, P = prompts.size(0), prompts.size(1)
            device = accelerator.device

            # =========================
            # 2) 整理 images / tti 以供 generate()
            #    CogVLM 期望 images: 长度为 B 的 list，每个元素是 [tensor]（单图）
            # =========================
            if images_raw is None:
                images_for_gen = None
                tti_for_gen = torch.zeros_like(prompts, device=device)  # 纯文本时，tti=0
            else:
                images_for_gen = ensure_cogvlm_images(images_raw, device, dtype)
                tti_for_gen = prompt_tti.to(device)

            # =========================
            # 3) 基于同一 prompt 生成 K 个候选（actor=LoRA）
            #    注意：为了多样性，可以为每个候选设置不同 seed
            # =========================
            gen_inputs = {
                "input_ids": prompts.to(device),
                "token_type_ids": tti_for_gen,
                "attention_mask": prompt_attn.to(device),
                "images": images_for_gen,
            }
            gen_kwargs = {
                "max_new_tokens": args.max_output_len,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                "top_p": 0.95,       # nucleus sampling
                "temperature": 0.9,  # 温度
                "do_sample": True,
            }

            completions_ids = []  # List[Tensor[B, L_i]]
            with torch.no_grad():
                for i in range(args.k):
                    set_seed_time()
                    outputs = accelerator.unwrap_model(model).generate(**gen_inputs, **gen_kwargs)  # outputs: [B, P + L_i]
                    comp_i = outputs[:, P:]  # 仅取 completion 段
                    completions_ids.append(comp_i)

            # ---------- 对 K 个候选按最大长度右侧 pad 到同一长度 ----------
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
            if eos_id is None:
                eos_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            Lmax = max(c.size(1) for c in completions_ids) if len(completions_ids) > 0 else 1
            comp_padded = []
            for c in completions_ids:
                if c.size(1) < Lmax:
                    pad = c.new_full((c.size(0), Lmax - c.size(1)), eos_id)
                    comp_padded.append(torch.cat([c, pad], dim=1))
                else:
                    comp_padded.append(c)
            # comp_cat: [B*K, Lmax]
            comp_cat = torch.cat(comp_padded, dim=0)

            # completion 有效位置的 mask（按“第一个 EOS 含在内”截断）
            comp_mask = first_eos_mask(comp_cat, eos_id=eos_id).to(prompts.dtype)  # [B*K, Lmax]

            # 文本解码（用于奖励）
            comp_texts: List[str] = [
                tokenizer.decode(row, skip_special_tokens=True).strip()
                for row in comp_cat
            ]
            assert len(comp_texts) == B * args.k, f"comp_texts 长度异常：{len(comp_texts)} vs {B*args.k}"

            # =========================
            # 4) 拼回完整序列（prompt + completion），以计算逐 token logp
            #    注意：需要同步构造 attention_mask 与 token_type_ids
            # =========================
            # 将 prompt 重复 K 次，与 completion 对齐
            prompts_k   = prompts.repeat(args.k, 1)          # [B*K, P]
            prompt_attn_k = prompt_attn.repeat(args.k, 1)    # [B*K, P]
            if images_raw is None:
                prompt_tti_k = torch.zeros_like(prompts_k, device=device)  # 纯文本时 tti=0
            else:
                prompt_tti_k = prompt_tti.repeat(args.k, 1).to(device)     # [B*K, P]

            # completion 段的 tti 全 0（文本）
            comp_tti = torch.zeros_like(comp_cat, device=device)           # [B*K, Lmax]

            # 拼接
            pc_ids  = torch.cat([prompts_k, comp_cat.to(device)], dim=1)   # [B*K, P+Lmax]
            pc_attn = torch.cat([prompt_attn_k.to(device), comp_mask.to(device)], dim=1)  # [B*K, P+Lmax]
            pc_tti  = torch.cat([prompt_tti_k, comp_tti], dim=1)           # [B*K, P+Lmax]

            # =========================
            # 5) 广播图像到 B*K 以匹配文本 batch
            #    CogVLM 接口规范：images 是长度为 N 的 list，每个元素是 [tensor]（单图）
            # =========================
            if images_for_gen is None:
                images_k = None
            elif isinstance(images_for_gen, list):
                # images_for_gen: 长度 B，每个元素形如 [tensor]
                # 广播 K 次
                images_k = []
                for _ in range(args.k):
                    images_k.extend(images_for_gen)
                assert len(images_k) == B * args.k, f"images_k 长度异常：{len(images_k)} vs {B*args.k}"
            else:
                images_k = [images_for_gen for _ in range(args.k)]

            # =========================
            # 6) 计算 actor/ref 的逐 token logp 与 KL（ref=基座）
            #    - actor: 激活 LoRA（已 set_adapter(actor)）
            #    - ref: 通过 no_adapter 临时禁用 LoRA，回到“纯基座”
            #    - KL 使用凸上界：exp(Δ) - Δ - 1，Δ=logp_ref - logp_actor
            # =========================
            start = max(P-1, 0)
            logp_actor = per_token_logps(
                model,
                input_ids=pc_ids, attention_mask=pc_attn, images=images_k,
                use_cache=False, token_type_ids=pc_tti
            )  # [B*K, P+Lmax-1]

            logp_actor = logp_actor[:, start:]  # [B*K, Lmax]

            # 计算 ref logp（禁用 LoRA）
            with torch.no_grad():
                kl_coef_this_step = float(args.kl_coef)
                with accelerator.unwrap_model(model).disable_adapter():
                    logp_ref = per_token_logps(
                        model,
                        input_ids=pc_ids, attention_mask=pc_attn, images=images_k,
                        use_cache=False, token_type_ids=pc_tti
                    )  # [B*K, P+Lmax-1]
                    logp_ref   = logp_ref[:, start:]  # [B*K, Lmax]

            # KL 上界（逐 token）
            if kl_coef_this_step > 0.0:
                delta = (logp_ref - logp_actor)                 # [B*K, Lmax]
                per_tok_kl = torch.exp(delta) - delta - 1.0     # [B*K, Lmax]
            else:
                per_tok_kl = torch.zeros_like(logp_actor)

            # 形状对齐检查
            assert logp_actor.size() == per_tok_kl.size() == (B * args.k, Lmax), \
                f"logp/kl 形状异常：{logp_actor.size()} vs {per_tok_kl.size()}, 期望 {(B*args.k, Lmax)}"

            # =========================
            # 7) 计算奖励（acc/format + VEC/DEP），并组内标准化得到优势（adv）
            # =========================
            gold_texts = batch.get("answer_text", None)
            if gold_texts is None:
                gold_texts = [""] * B

            r_acc, r_vec_g, r_vec_l, r_dep = build_rewards(
                pred_texts=comp_texts,
                gold_texts=gold_texts,
                embedder=embedder,
                images=images_raw,
                B=B, K=args.k,device=device,
                w_acc=args.w_acc,w_g=args.w_vec_g, w_l=args.w_vec_l, n_rois=args.n_rois,
                w_dep=args.w_dep, blur_sigma=args.dep_blur_sigma, shuffle_grid=args.dep_shuffle_grid,
                acc_log_path=args.acc_log_path,vec_g_log_path=args.vec_g_log_path,
                vec_l_log_path=args.vec_l_log_path,dep_log_path=args.dep_log_path,
            )

            # 统一在 GPU 上混合
            def _to_dev(x): return None if x is None else x.to(device)
            r_acc, r_vec_g, r_vec_l, r_dep = map(_to_dev, (r_acc, r_vec_g, r_vec_l, r_dep))

            rewards_mix, parts = standardize_each_then_mix({
                "acc":    (r_acc,    args.w_acc),
                "vec_g":  (r_vec_g,  args.w_vec_g),
                "vec_l":  (r_vec_l,  args.w_vec_l),
                "dep":    (r_dep,    args.w_dep),
            })  # rewards_mix: [B, K]

            # 组内标准化（按每个样本的 K 个候选），并裁剪
            advantages = group_norm_and_clip(rewards_mix, clip=args.adv_clip)   # [B, K]
            adv_flat = advantages.reshape(B * args.k, 1)                        # [B*K, 1]

            # =========================
            # 8) 构造损失：REINFORCE 风格 + KL 正则
            #    注意：重要性权重写成 exp(logπ - logπ_detach) == 1（占位形式）
            # =========================
            imp_weight = torch.exp(logp_actor - logp_actor.detach())  # 恒为 1，占位写法，便于未来替换 old_logp
            per_token_loss = imp_weight * adv_flat                    # [B*K, Lmax] 广播到第二维
            per_token_loss = -(per_token_loss - kl_coef_this_step * per_tok_kl)  # 最小化负的优势 + KL

            # 只在有效 completion 位置聚合
            comp_mask_f = comp_mask.to(per_token_loss.dtype)          # [B*K, Lmax]
            loss_per_seq = (per_token_loss * comp_mask_f).sum(dim=1) / (comp_mask_f.sum(dim=1) + 1e-8)  # [B*K]
            loss = loss_per_seq.mean()

            # =========================
            # 9) 反向传播与优化
            # =========================
            optim.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            # accelerator.clip_grad_norm_(model.parameters(), 1.0)  # RL 波动大，加裁剪
            optim.step()
            lr_scheduler.step()

            # =========================
            # 10) 累计、日志与 checkpoint
            # =========================
            with torch.no_grad():
                if kl_coef_this_step > 0:
                    mean_kl = ((per_tok_kl * comp_mask_f).sum(dim=1) / (comp_mask_f.sum(dim=1) + 1e-8)).mean().item()
                else:
                    mean_kl = 0.0
                r_mean = rewards_mix.mean().item()

            adv_mean = advantages.mean().item()
            adv_std  = advantages.std().item()

            parts_acc   = float(parts.get('acc',    0.0))
            parts_vec_g = float(parts.get('vec_g',  0.0))
            parts_vec_l = float(parts.get('vec_l',  0.0))
            parts_dep   = float(parts.get('dep',    0.0))

            metrics_sum["loss"]        += float(loss.item())
            metrics_sum["kl"]          += float(mean_kl)
            metrics_sum["r_mean"]      += float(r_mean)
            metrics_sum["adv_mean"]    += float(adv_mean)
            metrics_sum["adv_std"]     += float(adv_std)
            metrics_sum["parts_acc"]   += parts_acc
            metrics_sum["parts_vec_g"] += parts_vec_g
            metrics_sum["parts_vec_l"] += parts_vec_l
            metrics_sum["parts_dep"]   += parts_dep
            metrics_n += 1

            if (step_global + 1) % args.save_step == 0:
                eps = 1e-12
                loss_mean        = metrics_sum["loss"]        / max(metrics_n, 1)
                kl_mean          = metrics_sum["kl"]          / max(metrics_n, 1)
                r_mean_mean      = metrics_sum["r_mean"]      / max(metrics_n, 1)
                adv_mean_mean    = metrics_sum["adv_mean"]    / max(metrics_n, 1)
                adv_std_mean     = metrics_sum["adv_std"]     / max(metrics_n, 1)
                parts_acc_mean   = metrics_sum["parts_acc"]   / max(metrics_n, 1)
                parts_vec_g_mean = metrics_sum["parts_vec_g"] / max(metrics_n, 1)
                parts_vec_l_mean = metrics_sum["parts_vec_l"] / max(metrics_n, 1)
                parts_dep_mean   = metrics_sum["parts_dep"]   / max(metrics_n, 1)

                _ema_update("loss",        loss_mean)
                _ema_update("kl",          kl_mean)
                _ema_update("r_mean",      r_mean_mean)
                _ema_update("adv_mean",    adv_mean_mean)
                _ema_update("adv_std",     adv_std_mean)
                _ema_update("parts_acc",   parts_acc_mean)
                _ema_update("parts_vec_g", parts_vec_g_mean)
                _ema_update("parts_vec_l", parts_vec_l_mean)
                _ema_update("parts_dep",   parts_dep_mean)

                logger.info(
                    f"[RL] epoch {epoch} step {step_global} "
                    f"| mean(loss)={loss_mean:.6f} "
                    f"| mean(KL)={kl_mean:.6f} (coef={kl_coef_this_step:.6f}) "
                    f"| mean(r_mean)={r_mean_mean:.6f} "
                    f"| mean(adv_mean)={adv_mean_mean:.3e} | mean(adv_std)={adv_std_mean:.3e} "
                    f"| mean(parts: acc={parts_acc_mean:.6f}, g={parts_vec_g_mean:.6f}, "
                    f"l={parts_vec_l_mean:.6f}, dep={parts_dep_mean:.6f}) "
                    f"|| EMA(loss)={ema.get('loss',0):.6f}, EMA(KL)={ema.get('kl',0):.6f}, "
                    f"EMA(r_mean)={ema.get('r_mean',0):.6f}, EMA(acc)={ema.get('parts_acc',0):.6f}"
                )

                if accelerator.is_main_process:
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([
                            time.time(), epoch, int(step_global),
                            loss_mean, kl_mean, r_mean_mean,
                            adv_mean_mean, adv_std_mean,
                            parts_acc_mean, parts_vec_g_mean, parts_vec_l_mean, parts_dep_mean,
                            ema.get("loss",0.0), ema.get("kl",0.0), ema.get("r_mean",0.0),
                            ema.get("adv_mean",0.0), ema.get("adv_std",0.0),
                            ema.get("parts_acc",0.0), ema.get("parts_vec_g",0.0),
                            ema.get("parts_vec_l",0.0), ema.get("parts_dep",0.0),
                        ])

                if args.log_path and accelerator.is_main_process:
                    try:
                        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                        with open(args.log_path, "a", encoding="utf-8") as f:
                            f.write(f"------------- {current_time} Step {step_global} -------------\n")
                            f.write(
                                f"mean(loss)={loss_mean:.6f} | mean(KL)={kl_mean:.6f} (coef={kl_coef_this_step:.6f}) "
                                f"| mean(r_mean)={r_mean_mean:.6f} | mean(adv_mean)={adv_mean_mean:.3e} | mean(adv_std)={adv_std_mean:.3e}\n"
                            )
                            f.write(
                                f"mean(parts): acc={parts_acc_mean:.6f}, g={parts_vec_g_mean:.6f}, "
                                f"l={parts_vec_l_mean:.6f}, dep={parts_dep_mean:.6f}\n"
                            )
                            f.write(
                                f"EMA: loss={ema.get('loss',0):.6f}, KL={ema.get('kl',0):.6f}, "
                                f"r_mean={ema.get('r_mean',0):.6f}, acc={ema.get('parts_acc',0):.6f}\n\n"
                            )
                    except Exception as e:
                        logger.warning(f"写日志失败: {e}")

                ckpt = os.path.join(args.save_path, f"checkpoint_step_{step_global}")
                os.makedirs(ckpt, exist_ok=True)
                model.save_pretrained(ckpt, safe_serialization=True)

                metrics_sum.clear()
                metrics_n = 0

    logger.info("Training finished.")

if __name__ == "__main__":
    main()
