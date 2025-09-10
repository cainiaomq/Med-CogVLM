# finetune_demo/rl/utils.py
"""
通用工具：
- pad_left：左填充（或裁剪）到统一长度
- split_prompt_from_sft_batch：根据 labels==-100 切出 prompt 段（可同时切出 tti）
- per_token_logps：逐 token log p（自回归对齐）
- first_eos_mask：按首个 EOS 截断的有效 mask
- ensure_cogvlm_images：把多种图像批格式整理成 CogVLM2 期望的 [[tensor], ...]
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

def pad_left(seqs: List[torch.Tensor], pad_id: int, to_len: int = None) -> torch.Tensor:
    """
    对若干 1D 序列做左填充或裁剪，使其等长，并 stack 成 [N, L]
    - 左填充：在序列左侧补 pad_id
    - 裁剪：保留右侧 L 个 token（右对齐）
    """
    if not isinstance(seqs, list) or len(seqs) == 0:
        raise ValueError("pad_left expects a non-empty list of 1D tensors.")
    L = to_len or max(int(x.size(0)) for x in seqs)
    out = []
    for x in seqs:
        x = x.view(-1)  # 保证是 1D
        if x.size(0) < L:
            pad = x.new_full((L - x.size(0),), pad_id)
            out.append(torch.cat([pad, x], dim=0))
        else:
            out.append(x[-L:])
    return torch.stack(out, dim=0)  # [N, L]


def split_prompt_from_sft_batch(batch: Dict, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, List[int], object, Dict]:
    """
    利用 labels 的 -100 掩码切出 prompt 段：
    - labels == -100 → prompt；labels != -100 → completion
    如果 batch 里有 token_type_ids，则把 prompt 段的 tti 一起切出来，放入 extra。
    返回：
    - prompts:      [B, P*]
    - prompt_attn:  [B, P*]
    - prompt_lens:  List[int]
    - images:       原样返回（供 generate 使用）
    - extra:        {"prompt_token_type_ids": [B, P*]}（若 batch 提供了 token_type_ids）
    """
    input_ids: torch.Tensor = batch["input_ids"]
    labels: torch.Tensor = batch["labels"]
    attn: torch.Tensor = batch["attention_mask"]
    images = batch.get("images", None)
    tti_full: torch.Tensor = batch.get("token_type_ids", None)

    B, L = int(input_ids.size(0)), int(input_ids.size(1))
    prompt_lens: List[int] = []

    # 根据 labels!=-100 的第一个位置判定 prompt 长度
    for b in range(B):
        lb = labels[b]
        idx = (lb != -100).nonzero(as_tuple=False)
        p_len = int(idx[0].item()) if idx.numel() > 0 else L
        prompt_lens.append(p_len)

    # 切出 prompt 段，并左填充到相同长度
    seqs, masks, tti_prompts = [], [], []
    for b in range(B):
        p_len = prompt_lens[b]
        seqs.append(input_ids[b, :p_len])
        masks.append(attn[b, :p_len])
        if tti_full is not None:
            tti_prompts.append(tti_full[b, :p_len])

    prompts = pad_left(seqs, tokenizer.pad_token_id)
    prompt_attn = pad_left(masks, 0, to_len=prompts.size(1))

    extra: Dict = {}
    if tti_full is not None:
        prompt_tti = pad_left(tti_prompts, 0, to_len=prompts.size(1))
        extra["prompt_token_type_ids"] = prompt_tti

    return prompts, prompt_attn, prompt_lens, images, extra


def per_token_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    images=None,
    use_cache: bool = False,
    token_type_ids: torch.Tensor = None,
) -> torch.Tensor:
    """
    返回逐 token 的 log p(input_ids) → [B, L-1]
    对齐方式：用位置 t 的 logits 预测位置 t+1 的 token
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        use_cache=use_cache,
        token_type_ids=token_type_ids,
    )
    logits = out.logits[:, :-1, :]      # [B, L-1, V]
    targets = input_ids[:, 1:]          # [B, L-1]
    lp = F.log_softmax(logits, dim=-1)  # [B, L-1, V]
    tok = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    return tok


def first_eos_mask(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """
    对每行，首个 EOS（含）之前为 1，其后为 0；若无 EOS 则整行 1。
    返回 long 型 mask（0/1）。
    """
    B, L = int(ids.size(0)), int(ids.size(1))
    device = ids.device
    is_eos = (ids == eos_id)

    # 默认把最后一位当作 EOS（处理“无 EOS”的行）
    eos_idx = torch.full((B,), L - 1, dtype=torch.long, device=device)
    has = is_eos.any(dim=1)
    if has.any():
        # argmax 在多处 True 时返回第一处的下标；配合 has 做行选择
        eos_idx[has] = is_eos.int().argmax(dim=1)[has]

    seq_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    mask = (seq_idx <= eos_idx.unsqueeze(1)).to(torch.long)
    return mask


def ensure_cogvlm_images(images, device: torch.device, torch_type: torch.dtype):
    """
    CogVLM2 期望 images 形如：长度为 B 的 list，每个元素是 [tensor]（单图对话）
    - 若 None：直接返回 None（纯文本）
    - 若已经是 [[...], [...]]：原样返回
    - 若是 list[tensor]：包装成 [[t], [t], ...]
    - 若是 batched tensor (B, ...): 拆成 [[x0], [x1], ...]
    - 若是单张 tensor：包装成 [[tensor]]
    """
    if images is None:
        return None

    # 二维 list（每样本一小列表）
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
        out = []
        for sub in images:
            if len(sub) == 0:
                out.append([])
            else:
                out.append([sub[0].to(device).to(torch_type)])
        return out

    # list[tensor] -> [[tensor], ...]
    if isinstance(images, list) and len(images) > 0 and not isinstance(images[0], list):
        return [[img.to(device).to(torch_type)] for img in images]

    # batched tensor -> 拆分
    if hasattr(images, "dim") and images.dim() >= 1:
        B = images.size(0)
        return [[images[i].to(device).to(torch_type)] for i in range(B)]

    # 兜底：单张 -> [[tensor]]
    return [[images.to(device).to(torch_type)]]
