# finetune_demo/rl/rewards.py
# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Dict
import os
import re
from datetime import datetime
import string
import torch
import torch.nn.functional as F
from rl.roi import get_local_crops, make_counterfactuals


# =========================
# 工具：余弦相似度与编码
# =========================
def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    规范化后点积：支持形状广播。
    a: [..., D], b: [..., D] -> return: [... ]
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(-1)


@torch.no_grad()
def _encode_img(embedder, images) -> torch.Tensor:
    """
    统一入口：encode B 张图像 -> [B, D]
    - embedder 需实现 encode_image(list[PIL] or tensor batch)
    """
    return embedder.encode_image(images)  # [B, D]


@torch.no_grad()
def _encode_txt(embedder, texts: List[str]) -> torch.Tensor:
    """
    统一入口：encode 文本列表 -> [N, D]
    """
    return embedder.encode_text(texts)  # [N, D]


# =========================
# 文本解析与格式工具
# =========================
_ANS_TAG_RE = re.compile(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.I | re.S)
_LETTER_HEAD_RE = re.compile(r'^\s*([A-D1-4])[\)\].、．\s-]*', re.I)

# 一些常见同义归一（可按需补充）
_SYNONYM_MAP = {
    "x ray": "xray", "xray": "xray", "radiograph": "xray", "plain film": "xray",
    "ct": "ct", "computed tomography": "ct",
    "mri": "mri", "magnetic resonance imaging": "mri",
    "yes": "yes", "true": "yes", "positive": "yes",
    "no": "no", "false": "no", "negative": "no",
}

def _extract_answer_tag(s: str) -> str | None:
    if not isinstance(s, str): return None
    m = _ANS_TAG_RE.search(s)
    if m:
        return m.group(1).strip()
    return None

def _extract_head_letter_and_tail(s: str) -> tuple[str | None, str]:
    """返回 (选项字母 or None, 去掉字母头后的尾部文本)"""
    if not isinstance(s, str): return None, ""
    m = _LETTER_HEAD_RE.match(s)
    if m:
        letter = m.group(1).upper()
        tail = s[m.end():].strip()
        return letter, tail
    return None, s.strip()

def _normalize_string(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = _ANS_TAG_RE.sub(lambda m: m.group(1).strip().lower(), s)
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    toks = [_SYNONYM_MAP.get(t, t) for t in toks]
    return " ".join(toks)

def _normalize_loose(s: str) -> str:
    """更宽松的归一：去标点小写空格压缩，适合做包含判断"""
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# 准确率奖励（示例逻辑：符号验证优先，其次字符串/标签匹配）
# =========================
@torch.no_grad()
def accuracy_reward_bk(
    pred_texts: List[str],
    gold_texts: List[str],
    B: int,
    K: int,
    device: torch.device,
    log_path: Optional[str] = None,
) -> torch.Tensor:
    """
    返回 [B, K]，元素∈{0.0, 1.0}
    判定优先级：
      1) 解析 <answer> 标签：若 gold / pred 任一含标签，优先用标签内文本进行后续匹配；
      2) 选项字母匹配：若 gold/pred 均能提取到 A-D 字母，直接比字母；
      3) 文本匹配：规范化后严格相等 or 宽松包含；
      4) 兜底：原始完整串的规范化严格相等。
    索引顺序保持你现有实现：pred_texts[i*B + b]
    """
    vals: List[float] = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for i in range(K):
        for b in range(B):
            content_full = pred_texts[i * B + b]
            gold_full = gold_texts[b]

            # 先抽取 <answer> 标签（若存在）
            content_core = _extract_answer_tag(content_full)
            if content_core is None:
                content_core = content_full

            gold_core = _extract_answer_tag(gold_full)
            if gold_core is None:
                gold_core = gold_full

            reward = 0.0
            try:
                # 1) 选项字母匹配（优先）
                pl, p_tail = _extract_head_letter_and_tail(content_core)
                gl, g_tail = _extract_head_letter_and_tail(gold_core)
                if pl and gl and (pl == gl):
                    reward = 1.0

                # 2) 文本匹配（规范化/宽松包含）
                if reward == 0.0:
                    p_norm = _normalize_string(p_tail if pl else content_core)
                    g_norm = _normalize_string(g_tail if gl else gold_core)
                    # 完全相等（规范化后）
                    if p_norm and g_norm and (p_norm == g_norm):
                        reward = 1.0
                    else:
                        # 宽松包含（缓解冗余描述）
                        p_loose = _normalize_loose(p_tail if pl else content_core)
                        g_loose = _normalize_loose(g_tail if gl else gold_core)
                        if len(p_loose) >= 3 and len(g_loose) >= 3 and (g_loose in p_loose or p_loose in g_loose):
                            reward = 1.0

                # 3) 兜底：原始完整串规范化相等
                if reward == 0.0:
                    if _normalize_string(content_full) == _normalize_string(gold_full):
                        reward = 1.0

            except Exception:
                # 静默失败，按 0 计
                pass

            vals.append(reward)

            # 调试日志
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"Content: {content_full}\n")
                        f.write(f"Solution: {gold_full}\n")
                except Exception:
                    pass

    return torch.tensor(vals, device=device, dtype=torch.float32).view(K, B).T.contiguous()

# =========================
# 视觉一致性奖励：global / local / DEP （带日志）
# =========================
@torch.no_grad()
def vec_global(
    embedder, images, texts: List[str], B: int, K: int,
    cached_txt: torch.Tensor = None, cached_img: torch.Tensor = None,
    log_path: Optional[str] = None
) -> torch.Tensor:
    """
    整图-文本的余弦相似度，返回 [B, K]
    可选：传入 cached_txt / cached_img 以避免重复编码。
    """
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    img = cached_img if cached_img is not None else _encode_img(embedder, images)  # [B, D]
    txt_all = cached_txt if cached_txt is not None else _encode_txt(embedder, texts)  # [B*K, D]
    txt = txt_all.view(K, B, -1).transpose(0, 1).contiguous()  # [B, K, D]
    sim = _cos(img.unsqueeze(1), txt)  # [B, K]

    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for b in range(B):
                    for i in range(K):
                        f.write(f"------------- {current_time} VecGlobal sim[{b},{i}]: {sim[b,i].item():.7f} -------------\n")
                        f.write(f"Text: {texts[i*B+b]}\n")
        except Exception:
            pass

    return sim


@torch.no_grad()
def vec_local(
    embedder,
    images,
    texts: List[str],
    B: int,
    K: int,
    n_rois: int = 9,
    reduce: str = "max",
    cached_txt: torch.Tensor = None,
    log_path: Optional[str] = None,
) -> torch.Tensor:
    """
    ROI-文本相似度，返回 [B, K]
    """
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    roi_lists = get_local_crops(images, n_rois=n_rois)  # List[B][n]
    flat_rois = [r for rs in roi_lists for r in rs]     # B*n

    img_bn = embedder.encode_image(flat_rois).view(B, n_rois, -1)  # [B, n, D]

    txt_all = cached_txt if cached_txt is not None else _encode_txt(embedder, texts)  # [B*K, D]
    txt = txt_all.view(K, B, -1).transpose(0, 1).contiguous()  # [B, K, D]

    img_n = F.normalize(img_bn, dim=-1).unsqueeze(2)  # [B, n, 1, D]
    txt_n = F.normalize(txt, dim=-1).unsqueeze(1)     # [B, 1, K, D]
    sim = (img_n * txt_n).sum(-1)                     # [B, n, K]

    if reduce == "max":
        r = sim.max(dim=1).values   # [B, K]
    else:
        r = sim.mean(dim=1)         # [B, K]

    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for b in range(B):
                    for i in range(K):
                        f.write(f"------------- {current_time} VecLocal sim[{b},{i}]: {r[b,i].item():.7f} -------------\n")
                        f.write(f"Text: {texts[i*B+b]}\n")
        except Exception:
            pass

    return r


@torch.no_grad()
def dep(
    embedder,
    images,
    texts: List[str],
    B: int,
    K: int,
    blur_sigma: float = 3.0,
    shuffle_grid: int = 4,
    cached_txt: torch.Tensor = None,
    cached_img: torch.Tensor = None,
    log_path: Optional[str] = None,
) -> torch.Tensor:
    """
    real - counterfactual：差值越大说明依赖真实视觉细节
    """
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    real = vec_global(embedder, images, texts, B, K, cached_txt=cached_txt, cached_img=cached_img)
    cf = make_counterfactuals(images, blur_sigma=blur_sigma, shuffle_grid=shuffle_grid)
    cf_sim = vec_global(embedder, cf, texts, B, K, cached_txt=cached_txt)
    diff = real - cf_sim

    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for b in range(B):
                    for i in range(K):
                        f.write(f"------------- {current_time} DEP diff[{b},{i}]: {diff[b,i].item():.7f} -------------\n")
                        f.write(f"Text: {texts[i*B+b]}\n")
                f.write("\n")
        except Exception:
            pass

    return diff

# =========================
# 统一打包入口（包含 acc / fmt / global / local / dep）
# =========================
@torch.no_grad()
def build_rewards(
    *,
    pred_texts: List[str],    # 长度 B*K，顺序：先 K 后 B
    gold_texts: List[str],    # 长度 B
    embedder=None,
    images=None,
    B: int,
    K: int,
    device: torch.device,
    w_acc: float = 1.0,
    w_g: float = 0.0,
    w_l: float = 0.0,
    n_rois: int = 9,
    w_dep: float = 0.0,
    blur_sigma: float = 3.0,
    shuffle_grid: int = 4,
    acc_log_path: Optional[str] = None,
    vec_g_log_path: Optional[str] = None,
    vec_l_log_path: Optional[str] = None,
    dep_log_path: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    统一构建奖励，返回 (r_acc, r_g, r_l, r_dep)，各项形状均为 [B, K] 或 None。
    说明：
      - acc/fmt 不依赖 embedder/images；
      - global/local/dep 依赖 embedder/images，且内部做了必要的缓存避免重复编码；
      - 这里不做加权与标准化，只提供原始项（便于主循环做标准化、组内归一、裁剪等策略）。
    """
    # 1) 文本类奖励
    r_acc = accuracy_reward_bk(pred_texts, gold_texts, B, K, device, acc_log_path) if w_acc > 0 else None

    # 2) 视觉类奖励（可选缓存）
    cached_txt = None
    cached_img = None
    if (w_g > 0 or w_l > 0 or w_dep > 0):
        # 仅当至少一个视觉奖励启用时才做编码缓存
        if (embedder is not None) and (images is not None):
            if (w_g > 0 or w_l > 0 or w_dep > 0):
                cached_txt = _encode_txt(embedder, pred_texts) if (w_g > 0 or w_l > 0 or w_dep > 0) else None
            if (w_g > 0 or w_dep > 0):
                cached_img = _encode_img(embedder, images) if (w_g > 0 or w_dep > 0) else None
        else:
            pass

    r_g = vec_global(embedder, images, pred_texts, B, K, cached_txt=cached_txt, cached_img=cached_img, log_path=vec_g_log_path) if (w_g > 0 and embedder is not None and images is not None) else None
    r_l = vec_local(embedder, images, pred_texts, B, K, n_rois=n_rois, cached_txt=cached_txt, log_path=vec_l_log_path)          if (w_l > 0 and embedder is not None and images is not None) else None
    r_dep = dep(
        embedder, images, pred_texts, B, K,
        blur_sigma=blur_sigma, shuffle_grid=shuffle_grid,
        cached_txt=cached_txt, cached_img=cached_img, log_path=dep_log_path
    ) if (w_dep > 0 and embedder is not None and images is not None) else None

    return r_acc, r_g, r_l, r_dep


# ========== 标准化/混合/优势 ==========
def _std_norm(x: torch.Tensor, eps=1e-6):
    mu = x.mean(); sd = x.std() + eps
    return (x-mu)/sd

def standardize_each_then_mix(parts: Dict[str, Tuple[torch.Tensor, float]]):
    """
    对每个奖励 r_i 先做全局（批维度）标准化 → w_i * r_i → 求和
    返回：合成奖励 [B,K] 与各项贡献均值（便于日志）
    """
    out = None; logs={}
    for name, (ri, wi) in parts.items():
        if wi == 0.0 or ri is None: logs[name]=0.0; continue
        logs[name] = float(ri.mean().item())
        r_norm = _std_norm(ri)
        contrib = wi * r_norm
        out = contrib if out is None else (out + contrib)
    if out is None:
        first = next(iter(parts.values()))[0]
        out = torch.zeros_like(first)
    return out, logs

def group_norm_and_clip(r_bk: torch.Tensor, clip: float = 5.0):
    """
    先按组（每行 B）做 z-norm → 再裁剪稳定数值 → 返回优势矩阵 [B,K]
    """
    B, K = r_bk.size()
    mu = r_bk.mean(dim=1, keepdim=True)
    sd = r_bk.std(dim=1, keepdim=True) + 1e-6
    adv = (r_bk - mu) / sd
    return adv.clamp(-clip, clip)