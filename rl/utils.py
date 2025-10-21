from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

def pad_left(seqs: List[torch.Tensor], pad_id: int, to_len: int = None) -> torch.Tensor:
    if not isinstance(seqs, list) or len(seqs) == 0:
        raise ValueError("pad_left expects a non-empty list of 1D tensors.")
    L = to_len or max(int(x.size(0)) for x in seqs)
    out = []
    for x in seqs:
        x = x.view(-1)
        if x.size(0) < L:
            pad = x.new_full((L - x.size(0),), pad_id)
            out.append(torch.cat([pad, x], dim=0))
        else:
            out.append(x[-L:])
    return torch.stack(out, dim=0)


def split_prompt_from_sft_batch(batch: Dict, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, List[int], object, Dict]:
    input_ids: torch.Tensor = batch["input_ids"]
    labels: torch.Tensor = batch["labels"]
    attn: torch.Tensor = batch["attention_mask"]
    images = batch.get("images", None)
    tti_full: torch.Tensor = batch.get("token_type_ids", None)

    B, L = int(input_ids.size(0)), int(input_ids.size(1))
    prompt_lens: List[int] = []

    for b in range(B):
        lb = labels[b]
        idx = (lb != -100).nonzero(as_tuple=False)
        p_len = int(idx[0].item()) if idx.numel() > 0 else L
        prompt_lens.append(p_len)

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
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        use_cache=use_cache,
        token_type_ids=token_type_ids,
    )
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    lp = F.log_softmax(logits, dim=-1)
    tok = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok


def first_eos_mask(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    B, L = int(ids.size(0)), int(ids.size(1))
    device = ids.device
    is_eos = (ids == eos_id)

    eos_idx = torch.full((B,), L - 1, dtype=torch.long, device=device)
    has = is_eos.any(dim=1)
    if has.any():
        eos_idx[has] = is_eos.int().argmax(dim=1)[has]

    seq_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    mask = (seq_idx <= eos_idx.unsqueeze(1)).to(torch.long)
    return mask


def ensure_cogvlm_images(images, device: torch.device, torch_type: torch.dtype):
    if images is None:
        return None

    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
        out = []
        for sub in images:
            if len(sub) == 0:
                out.append([])
            else:
                out.append([sub[0].to(device).to(torch_type)])
        return out
    
    if isinstance(images, list) and len(images) > 0 and not isinstance(images[0], list):
        return [[img.to(device).to(torch_type)] for img in images]

    if hasattr(images, "dim") and images.dim() >= 1:
        B = images.size(0)
        return [[images[i].to(device).to(torch_type)] for i in range(B)]

    return [[images.to(device).to(torch_type)]]
