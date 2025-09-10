# finetune_demo/rl/roi.py
"""
ROI 与反事实（counterfactual）工具
- 接受多种输入格式：PIL / torch.Tensor / numpy.ndarray / dict / list
- 统一转成 PIL 列表，提供：
  * get_local_crops: five-crops / nine-grid
  * make_counterfactuals: blur and shuffle-blocks
"""

from typing import Any, List
import numpy as np
from PIL import Image, ImageFilter
import random
import torch


_local_mode = {"n_rois": 9}


def set_local_crops_mode(n_rois: int = 9):
    """设置默认 ROI 数（1~9）"""
    _local_mode["n_rois"] = max(1, min(9, int(n_rois)))


# ============== 各类输入 → PIL ==============
def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    支持 [C,H,W] / [H,W,C] / [H,W] / [B,C,H,W]
    浮点认为在 [0,1]，会缩放到 0-255 并转 uint8
    """
    t = t.detach().cpu()
    if t.dtype.is_floating_point:
        t = (t.clamp(0, 1) * 255).to(torch.uint8)

    if t.ndim == 4:
        # 批量：取第一张兜底；真正批处理走 to_pil_list
        return _tensor_to_pil(t[0])

    if t.ndim == 3:
        # 可能是 [C,H,W] 或 [H,W,C]
        if t.size(0) in (1, 3):  # [C,H,W]
            c, h, w = t.size(0), t.size(1), t.size(2)
            if c == 1:
                t = t.repeat(3, 1, 1)
            t = t.permute(1, 2, 0)  # → [H,W,C]
        # 否则默认已是 [H,W,C]
    elif t.ndim == 2:
        # 灰度 → 三通道
        t = t.unsqueeze(-1).repeat(1, 1, 3)

    return Image.fromarray(t.numpy())


def _numpy_to_pil(a: np.ndarray) -> Image.Image:
    """
    支持 [C,H,W] / [H,W,C] / [H,W] / [B,C,H,W]
    """
    if a.ndim == 4:
        return _numpy_to_pil(a[0])

    if a.ndim == 3:
        if a.shape[0] in (1, 3):          # [C,H,W]
            a = np.transpose(a, (1, 2, 0))
            if a.shape[2] == 1:
                a = np.repeat(a, 3, axis=2)
        elif a.shape[2] == 1:             # [H,W,1]
            a = np.repeat(a, 3, axis=2)
    elif a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)

    a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _maybe_extract_from_dict(x: dict) -> Any:
    """尝试从 dict 中抽取常见字段（兼容 HF processor）"""
    for k in ("image", "images", "pixel_values", "pixels", "img"):
        if k in x:
            return x[k]
    return x


def _to_pil(x: Any) -> Image.Image:
    """把单个元素转成 PIL；list/tuple 递归取第一个非空元素"""
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict):
        x = _maybe_extract_from_dict(x)

    if torch.is_tensor(x):
        return _tensor_to_pil(x)
    if isinstance(x, np.ndarray):
        return _numpy_to_pil(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_pil(x[0])

    raise TypeError(f"Unsupported image type in to_pil: {type(x)}")


def to_pil_list(images) -> List[Image.Image]:
    """
    批量入口：统一返回 List[PIL]
    支持：
    - list of PIL/Tensor/ndarray/dict
    - torch.Tensor: [B,C,H,W] or [B,H,W,C]
    - numpy.ndarray: [B,C,H,W] or [B,H,W,C]
    - 单张任意类型（最终也会包装成长度为 1 的 List）
    """
    if isinstance(images, list):
        return [_to_pil(i) for i in images]

    if torch.is_tensor(images) and images.ndim == 4:
        # 兼容 [B,H,W,C] 与 [B,C,H,W]
        b = images.size(0)
        return [_tensor_to_pil(images[i]) for i in range(b)]

    if isinstance(images, np.ndarray) and images.ndim == 4:
        b = images.shape[0]
        return [_numpy_to_pil(images[i]) for i in range(b)]

    return [_to_pil(images)]


# ============== ROI 裁剪 ==============
def _five_crops(im: Image.Image, ratio: float = 0.6):
    """
    常用 five-crops：四角 + 中心
    """
    W, H = im.size
    w, h = int(W * ratio), int(H * ratio)
    boxes = [
        (0, 0, w, h),                     # 左上
        (W - w, 0, W, h),                 # 右上
        (0, H - h, w, H),                 # 左下
        (W - w, H - h, W, H),             # 右下（补全）
        ((W - w) // 2, (H - h) // 2, (W + w) // 2, (H + h) // 2),  # 中心
    ]
    return [im.crop(b) for b in boxes]


def _nine_grid(im: Image.Image):
    """九宫格裁剪"""
    W, H = im.size
    thirds_x = [0, W // 3, 2 * W // 3, W]
    thirds_y = [0, H // 3, 2 * H // 3, H]
    crops = []
    for yi in range(3):
        for xi in range(3):
            crops.append(im.crop((thirds_x[xi], thirds_y[yi], thirds_x[xi + 1], thirds_y[yi + 1])))
    return crops


def get_local_crops(images, n_rois: int = None) -> List[List[Image.Image]]:
    """
    返回 List[List[PIL]]，外层 B、内层 n（每张图 n 个裁块）
    - n<=5 走 five-crops（随机打乱取前 n）
    - n>5  走九宫格（最多 9 个）
    """
    n = n_rois if n_rois is not None else _local_mode["n_rois"]
    ims = to_pil_list(images)
    outs: List[List[Image.Image]] = []
    for im in ims:
        if n <= 5:
            cs = _five_crops(im)
            random.shuffle(cs)
            outs.append(cs[:n])
        else:
            cs = _nine_grid(im)
            outs.append(cs[: min(n, 9)])
    return outs


# ============== 反事实（counterfactual） ==============
def _shuffle_blocks(im: Image.Image, grid: int = 4) -> Image.Image:
    """将图像切成 grid×grid 小块后打乱重排，破坏空间结构"""
    W, H = im.size
    bw, bh = W // grid, H // grid
    blocks = []
    for gy in range(grid):
        for gx in range(grid):
            blocks.append(im.crop((gx * bw, gy * bh, (gx + 1) * bw, (gy + 1) * bh)))
    random.shuffle(blocks)

    out = Image.new("RGB", (W, H))
    idx = 0
    for gy in range(grid):
        for gx in range(grid):
            out.paste(blocks[idx], (gx * bw, gy * bh))
            idx += 1
    return out


def make_counterfactuals(images, blur_sigma: float = 3.0, shuffle_grid: int = 4):
    """
    生成反事实图像列表（与输入数量一致）：
    - blur：高斯模糊，破坏细节
    - shuffle：网格打乱，破坏全局结构
    """
    ims = to_pil_list(images)
    def first(x):  return x.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    def second(x): return _shuffle_blocks(x, grid=shuffle_grid)
    return [second(first(im)) for im in ims]