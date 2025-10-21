from typing import List, Optional, Tuple, Dict
import re
from datetime import datetime
import string
import torch
import torch.nn.functional as F
from rl.roi import get_local_crops, make_counterfactuals


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(-1)


@torch.no_grad()
def _encode_img(embedder, images) -> torch.Tensor:
    return embedder.encode_image(images)

@torch.no_grad()
def _encode_txt(embedder, texts: List[str]) -> torch.Tensor:
    return embedder.encode_text(texts)


_ANS_TAG_RE = re.compile(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.I | re.S)
_LETTER_ANY_RE = re.compile(
    r'(?<![A-Za-z0-9Ａ-Ｚａ-ｚ０-９])'
    r'([A-DＡ-Ｄa-dａ-ｄ1-4１-４])'
    r'(?:\s*[\)\].、．：:]\s*|\s+|$)',
    re.I
)

_SYNONYM_MAP = {
    "x ray": "xray", "xray": "xray", "radiograph": "xray", "plain film": "xray",
    "ct": "ct", "computed tomography": "ct",
    "mri": "mri", "magnetic resonance imaging": "mri",
    "yes": "yes", "true": "yes", "positive": "yes",
    "no": "no", "false": "no", "negative": "no",
}

def _normalize_letter(x: str) -> str | None:
    x = x.translate(str.maketrans("ＡＢＣＤａｂｃｄ１２３４", "ABCDabcd1234")).upper()
    if x in {"A","B","C","D"}: return x
    if x in {"1","2","3","4"}: return "ABCD"[int(x)-1]
    return None

def _extract_answer_tag(s: str) -> str | None:
    if not isinstance(s, str): return None
    m = _ANS_TAG_RE.search(s)
    if m:
        return m.group(1).strip()
    return None

def _extract_head_letter_and_tail(s: str):
    if not isinstance(s, str): return None, ""
    m = _LETTER_ANY_RE.search(s)
    if m:
        letter = _normalize_letter(m.group(1))
        if letter:
            return letter, s[m.end():].strip()
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
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    table = str.maketrans({c: " " for c in string.punctuation})
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@torch.no_grad()
def accuracy_reward_bk(
    pred_texts: List[str],
    gold_texts: List[str],
    question_id: List[str],
    B: int,
    K: int,
    device: torch.device,
    log_path: Optional[str] = None,
) -> torch.Tensor:
    vals: List[float] = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for i in range(K):
        for b in range(B):
            content_full = pred_texts[i * B + b]
            gold_full = gold_texts[b]
            question_full = question_id[b]

            content_core = _extract_answer_tag(content_full)
            if content_core is None:
                content_core = content_full

            gold_core = _extract_answer_tag(gold_full)
            if gold_core is None:
                gold_core = gold_full

            reward = 0.0
            try:
                pl, p_tail = _extract_head_letter_and_tail(content_core)
                gl, g_tail = _extract_head_letter_and_tail(gold_core)
                if pl and gl and (pl == gl):
                    reward = 1.0
                
                if reward == 0.0:
                    p_norm = _normalize_string(p_tail if pl else content_core)
                    g_norm = _normalize_string(g_tail if gl else gold_core)
                    if p_norm and g_norm and (p_norm == g_norm):
                        reward = 1.0
                    else:
                        p_loose = _normalize_loose(p_tail if pl else content_core)
                        g_loose = _normalize_loose(g_tail if gl else gold_core)
                        if p_loose and g_loose:
                            if len(g_loose) <= 3:
                                if re.search(rf'\b{re.escape(g_loose)}\b', p_loose):
                                    reward = 1.0
                            elif len(p_loose) <= 3:
                                if re.search(rf'\b{re.escape(p_loose)}\b', g_loose):
                                    reward = 1.0
                            else:
                                if (g_loose in p_loose) or (p_loose in g_loose):
                                    reward = 1.0
                if reward == 0.0:
                    if _normalize_string(content_full) == _normalize_string(gold_full):
                        reward = 1.0
                
            except Exception:
                pass

            vals.append(reward)

            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} ----- question_id: {question_full} ----- Accuracy reward: {reward} -------------\n")
                        f.write(f"Content: {content_full}\n")
                        f.write(f"Solution: {gold_full}\n")
                except Exception:
                    pass

    return torch.tensor(vals, device=device, dtype=torch.float32).view(K, B).T.contiguous()

@torch.no_grad()
def format_reward_simple(
    pred_texts: List[str],
    B: int,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    vals = []
    ts = datetime.now().strftime("%d-%H-%M-%S-%f")

    for i in range(K):
        for b in range(B):
            s = pred_texts[i * B + b]
            r = 0.0
            if isinstance(s, str):
                if "evidence:" in s.lower() or "Explanation:" in s.lower():
                    r = 0.5
            vals.append(r)

    return torch.tensor(vals, device=device, dtype=torch.float32).view(K, B).T.contiguous()

@torch.no_grad()
def vec_global(
    embedder, images, texts: List[str], B: int, K: int,
    cached_txt: torch.Tensor = None, cached_img: torch.Tensor = None,
    log_path: Optional[str] = None
) -> torch.Tensor:
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    img = cached_img if cached_img is not None else _encode_img(embedder, images)
    txt_all = cached_txt if cached_txt is not None else _encode_txt(embedder, texts)
    txt = txt_all.view(K, B, -1).transpose(0, 1).contiguous()
    sim = _cos(img.unsqueeze(1), txt)

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
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    roi_lists = get_local_crops(images, n_rois=n_rois)
    flat_rois = [r for rs in roi_lists for r in rs]

    img_bn = embedder.encode_image(flat_rois).view(B, n_rois, -1)

    txt_all = cached_txt if cached_txt is not None else _encode_txt(embedder, texts)
    txt = txt_all.view(K, B, -1).transpose(0, 1).contiguous()

    img_n = F.normalize(img_bn, dim=-1).unsqueeze(2)
    txt_n = F.normalize(txt, dim=-1).unsqueeze(1)
    sim = (img_n * txt_n).sum(-1)

    if reduce == "max":
        r = sim.max(dim=1).values
    else:
        r = sim.mean(dim=1)

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

@torch.no_grad()
def build_rewards(
    *,
    pred_texts: List[str],
    gold_texts: List[str],
    question_id: List[str],
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
    r_acc = accuracy_reward_bk(pred_texts, gold_texts, question_id,B, K, device, acc_log_path) if w_acc > 0 else None
    r_fmt = format_reward_simple(pred_texts, B, K, device)
    r_acc = r_acc + r_fmt

    cached_txt = None
    cached_img = None
    if (w_g > 0 or w_l > 0 or w_dep > 0):
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

def _std_norm(x: torch.Tensor, eps=1e-6):
    mu = x.mean(); sd = x.std() + eps
    return (x-mu)/sd

def standardize_each_then_mix(parts: Dict[str, Tuple[torch.Tensor, float]]):
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
    B, K = r_bk.size()
    mu = r_bk.mean(dim=1, keepdim=True)
    sd = r_bk.std(dim=1, keepdim=True) + 1e-6
    adv = (r_bk - mu) / sd
    return adv.clamp(-clip, clip)