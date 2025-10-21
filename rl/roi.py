from typing import Any, List
import numpy as np
from PIL import Image, ImageFilter
import random
import torch


_local_mode = {"n_rois": 9}


def set_local_crops_mode(n_rois: int = 9):
    _local_mode["n_rois"] = max(1, min(9, int(n_rois)))


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu()
    if t.dtype.is_floating_point:
        t = (t.clamp(0, 1) * 255).to(torch.uint8)

    if t.ndim == 4:
        return _tensor_to_pil(t[0])

    if t.ndim == 3:
        if t.size(0) in (1, 3):
            c, h, w = t.size(0), t.size(1), t.size(2)
            if c == 1:
                t = t.repeat(3, 1, 1)
            t = t.permute(1, 2, 0)
    elif t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)

    return Image.fromarray(t.numpy())


def _numpy_to_pil(a: np.ndarray) -> Image.Image:
    if a.ndim == 4:
        return _numpy_to_pil(a[0])

    if a.ndim == 3:
        if a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
            if a.shape[2] == 1:
                a = np.repeat(a, 3, axis=2)
        elif a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
    elif a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)

    a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _maybe_extract_from_dict(x: dict) -> Any:
    for k in ("image", "images", "pixel_values", "pixels", "img"):
        if k in x:
            return x[k]
    return x


def _to_pil(x: Any) -> Image.Image:
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
    if isinstance(images, list):
        return [_to_pil(i) for i in images]

    if torch.is_tensor(images) and images.ndim == 4:
        b = images.size(0)
        return [_tensor_to_pil(images[i]) for i in range(b)]

    if isinstance(images, np.ndarray) and images.ndim == 4:
        b = images.shape[0]
        return [_numpy_to_pil(images[i]) for i in range(b)]

    return [_to_pil(images)]


def _five_crops(im: Image.Image, ratio: float = 0.6):
    W, H = im.size
    w, h = int(W * ratio), int(H * ratio)
    boxes = [
        (0, 0, w, h),
        (W - w, 0, W, h),
        (0, H - h, w, H),
        (W - w, H - h, W, H),
        ((W - w) // 2, (H - h) // 2, (W + w) // 2, (H + h) // 2),
    ]
    return [im.crop(b) for b in boxes]


def _nine_grid(im: Image.Image):
    W, H = im.size
    thirds_x = [0, W // 3, 2 * W // 3, W]
    thirds_y = [0, H // 3, 2 * H // 3, H]
    crops = []
    for yi in range(3):
        for xi in range(3):
            crops.append(im.crop((thirds_x[xi], thirds_y[yi], thirds_x[xi + 1], thirds_y[yi + 1])))
    return crops


def get_local_crops(images, n_rois: int = None) -> List[List[Image.Image]]:
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


def _shuffle_blocks(im: Image.Image, grid: int = 4) -> Image.Image:
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
    ims = to_pil_list(images)
    def first(x):  return x.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    def second(x): return _shuffle_blocks(x, grid=shuffle_grid)
    return [second(first(im)) for im in ims]