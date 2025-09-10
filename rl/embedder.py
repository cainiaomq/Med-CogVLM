# finetune_demo/rl/embedder.py
import os
import re
import json
import torch
import torch.nn as nn
from typing import List, Optional
from rl.roi import to_pil_list

import open_clip
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from transformers import AutoTokenizer

_MED_CLIP_DIR = "/home/lvzeyu/.cache/huggingface/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_CFG_JSON = os.path.join(_MED_CLIP_DIR, "open_clip_config.json")
_WEIGHT_BIN = os.path.join(_MED_CLIP_DIR, "open_clip_pytorch_model.bin")
_LOCAL_NAME = "biomedclip_local"

_ANSWER_RE = re.compile(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.I | re.S)

def _extract_answer_only(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip()
        return ans if ans else text.strip()
    return text.strip()

class MedClipEmbedder(nn.Module):
    def __init__(self, device: str = "cuda", dtype: Optional[torch.dtype] = torch.float16):
        super().__init__()
        self.device = torch.device(device)
        with open(_CFG_JSON, "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        if (not _LOCAL_NAME.startswith(HF_HUB_PREFIX)
            and _LOCAL_NAME not in _MODEL_CONFIGS
            and config is not None):
            _MODEL_CONFIGS[_LOCAL_NAME] = model_cfg 

        self.tokenizer = open_clip.get_tokenizer(_LOCAL_NAME)
        image_kwargs = {f"image_{k}": v for k, v in preprocess_cfg.items()}
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=_LOCAL_NAME,
            pretrained=_WEIGHT_BIN,
            **image_kwargs,
        )
        self.model = self.model.to(self.device, dtype=dtype).eval()

    @torch.no_grad()
    def encode_image(self, images):
        pil_list = to_pil_list(images)
        imgs = torch.stack([self.preprocess(im) for im in pil_list]).to(self.device)

        param_dtype = next(self.model.parameters()).dtype
        if imgs.dtype != param_dtype:
            imgs = imgs.to(dtype=param_dtype)

        feats = self.model.encode_image(imgs)
        return feats  # [B, D]

    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        ans_texts = [_extract_answer_only(t) for t in texts]
        token_ids = self.tokenizer(ans_texts, context_length=256).to(self.device)
        feats = self.model.encode_text(token_ids)
        return feats  # [B, D]
