# -*- coding: utf-8 -*-
import re
import os
import json
import argparse
from typing import List
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.omnimedqkv import OmniMedVQA_Dataset
from rl.utils import split_prompt_from_sft_batch, ensure_cogvlm_images
from rl.embedder import MedClipEmbedder
from rl.roi import make_counterfactuals
import torch.nn.functional as F

_LETTER_ANY_RE = re.compile(
    r'(?<![A-Za-z0-9Ａ-Ｚａ-ｚ０-９])'
    r'([A-DＡ-Ｄa-dａ-ｄ1-4１-４])'
)
_FULL_TO_HALF = str.maketrans('ａｂｃｄＡＢＣＤ１２３４', 'abcdABCD1234')

def normalize_letter(txt: str) -> str:
    if not txt:
        return ""
    t = txt.strip().translate(_FULL_TO_HALF)
    m = _LETTER_ANY_RE.search(t)
    if not m:
        return ""
    ch = m.group(1).upper()
    if ch in ['1', '2', '3', '4']:
        return {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}[ch]
    if ch in ['A', 'B', 'C', 'D']:
        return ch
    return ""

def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(-1)


@torch.no_grad()
def compute_vds(
    embedder,
    images,
    texts: List[str],
) -> List[float]:
    img_emb = embedder.encode_image(images)
    txt_emb = embedder.encode_text(texts)
    sim_real = _cos(img_emb, txt_emb)
    vds = sim_real
    return vds.cpu().tolist()


@torch.no_grad()
def run_vds_evaluation(
    model_path: str,
    dataset_path: str,
    out_pred_jsonl: str,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
    temperature: float = 0.7,
    dtype_str: str = "torch.bfloat16",
    visdep_sidecar: str = None,
    visdep_min_score: float = None,
    access: str = "open",
    resume: bool = False,
):
    os.makedirs(os.path.dirname(out_pred_jsonl), exist_ok=True)
    dtype = eval(dtype_str)

    print(f"[VDS Eval] Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).eval().to("cuda")
    print(f"[VDS Eval] Model loaded")

    dataset = OmniMedVQA_Dataset(
        root_dir=dataset_path,
        tokenizer=tokenizer,
        model=model,
        torch_type=dtype,
        input_length=128,
        output_length=max_new_tokens,
        access=access,
        visdep_sidecar=visdep_sidecar,
        visdep_min_score=visdep_min_score,
    )

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=OmniMedVQA_Dataset.custom_collate_fn
    )

    print(">> Loading MedCLIP embedder...")
    embedder = MedClipEmbedder()
    embedder.eval()
    print(">> MedCLIP ready.")

    done_qids = set()
    if resume and os.path.exists(out_pred_jsonl):
        with open(out_pred_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    qid = str(obj.get("question_id") or "")
                    if qid:
                        done_qids.add(qid)
                except Exception:
                    continue
        print(f"[VDS Eval/Resume] already finished: {len(done_qids)} samples")

    file_mode = "a" if (resume and os.path.exists(out_pred_jsonl)) else "w"
    wrote = 0
    
    with open(out_pred_jsonl, file_mode, encoding="utf-8") as fout:
        for batch in tqdm(loader, total=len(loader), desc="VDS Evaluation"):
            qids = [str(x) for x in batch.get("question_id", [])]
            keep_idx = [i for i, q in enumerate(qids) if q and q not in done_qids]
            if len(keep_idx) == 0:
                continue

            def _take(v):
                if isinstance(v, list):
                    return [v[i] for i in keep_idx]
                if isinstance(v, torch.Tensor):
                    return v.index_select(0, torch.tensor(keep_idx, device=v.device if v.is_cuda else 'cpu'))
                return v

            batch = {k: _take(v) for k, v in batch.items()}

            prompts, prompt_attn, _, images_raw, extra = split_prompt_from_sft_batch(batch, tokenizer)
            
            if images_raw is None:
                images_for_gen = None
                tti_for_gen = torch.zeros_like(prompts, device=prompts.device)
            else:
                images_for_gen = ensure_cogvlm_images(images_raw, prompts.device, dtype)
                tti_for_gen = extra["prompt_token_type_ids"].to(prompts.device)

            gen_inputs = {
                "input_ids": prompts.to(model.device),
                "token_type_ids": tti_for_gen.to(model.device),
                "attention_mask": prompt_attn.to(model.device),
                "images": images_for_gen,
            }
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                top_p=top_p,
                temperature=temperature,
                do_sample=(temperature > 0),
            )

            out_ids = model.generate(**gen_inputs, **gen_kwargs)
            P = prompts.shape[1]
            comp_ids = out_ids[:, P:]
            pred_texts = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in comp_ids]

            if images_raw is not None and len(pred_texts) > 0:
                vds_scores = compute_vds(
                    embedder=embedder,
                    images=images_raw,
                    texts=pred_texts,
                )
            else:
                vds_scores = [0.0] * len(pred_texts)

            B = len(pred_texts)
            for i in range(B):
                qid = str(batch.get("question_id", [""] * B)[i])
                if qid in done_qids:
                    continue

                gt_text = batch.get("answer_text", [""] * B)[i]
                opt_text = batch.get("options_text", [""] * B)[i]
                pred_letter = normalize_letter(pred_texts[i])
                gt_letter = normalize_letter(gt_text)

                rec = {
                    "question_id": qid,
                    "question_type": (batch.get("question_type", [""] * B)[i] or "").strip(),
                    "modality_type": (batch.get("modality_type", [""] * B)[i] or "").strip(),
                    "options_text": opt_text,
                    "pred_text": pred_texts[i],
                    "gt_answer_text": gt_text,
                    "pred_option": pred_letter,
                    "gt_option": gt_letter,
                    "vds": vds_scores[i],
                    "text_length": len(pred_texts[i]),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
                done_qids.add(qid)

            if wrote % 10 == 0:
                fout.flush()
                os.fsync(fout.fileno())

    print(f"\n[VDS Eval] inference finished. wrote {wrote} records to {out_pred_jsonl}")


def analyze_vds_statistics(pred_jsonl: str, out_summary_json: str):
    from collections import Counter
    
    records = []
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue

    if len(records) == 0:
        print("[VDS Eval] No records found")
        return

    total, correct = 0, 0
    by_qtype_total, by_qtype_correct = Counter(), Counter()
    by_modal_total, by_modal_correct = Counter(), Counter()

    vds_values = []
    text_lengths = []
    vds_by_qtype = defaultdict(list)
    vds_by_modal = defaultdict(list)

    for r in records:
        qtype = r.get("question_type", "Unknown")
        modal = r.get("modality_type", "Unknown")
        
        pred_opt = (r.get("pred_option") or "").upper()
        gt_opt = (r.get("gt_option") or "").upper()
        
        if gt_opt:
            total += 1
            by_qtype_total[qtype] += 1
            by_modal_total[modal] += 1
            
            if pred_opt == gt_opt:
                correct += 1
                by_qtype_correct[qtype] += 1
                by_modal_correct[modal] += 1
        
        if "vds" in r:
            vds_values.append(r["vds"])
            vds_by_qtype[qtype].append(r["vds"])
            vds_by_modal[modal].append(r["vds"])
        
        if "text_length" in r:
            text_lengths.append(r["text_length"])

    import numpy as np

    def stats(vals):
        if not vals:
            return {}
        return {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }

    def _pack_acc(total_ctr: Counter, correct_ctr: Counter):
        keys = sorted(total_ctr.keys())
        out = {}
        for k in keys:
            t = total_ctr[k]
            c = correct_ctr.get(k, 0)
            out[k] = {
                "total": t, 
                "correct": c, 
                "accuracy": (c / t) if t > 0 else 0.0
            }
        return out

    acc = (correct / total) if total > 0 else 0.0
    
    summary = {
        "accuracy": {
            "overall": {"total": total, "correct": correct, "accuracy": acc},
            "by_question_type": _pack_acc(by_qtype_total, by_qtype_correct),
            "by_modality_type": _pack_acc(by_modal_total, by_modal_correct),
        },
        "vds": {
            "overall": stats(vds_values),
            "text_length": stats(text_lengths),
            "by_question_type": {k: stats(v) for k, v in vds_by_qtype.items()},
            "by_modality_type": {k: stats(v) for k, v in vds_by_modal.items()},
        }
    }

    os.makedirs(os.path.dirname(out_summary_json), exist_ok=True)
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[VDS Eval] Statistics saved to {out_summary_json}")
    print(f"{'='*60}")
    
    print(f"ACCURACY:")
    print(f"  Overall: {acc:.4f} ({correct}/{total})")
    
    print(f"\nVDS:")
    print(f"  Overall: mean={summary['vds']['overall']['mean']:.4f}, "
          f"std={summary['vds']['overall']['std']:.4f}, "
          f"median={summary['vds']['overall']['median']:.4f}")
    print(f"  Text Length: mean={summary['vds']['text_length']['mean']:.1f}, "
          f"median={summary['vds']['text_length']['median']:.1f}")
    
    print(f"\nBy Question Type:")
    all_qtypes = sorted(set(list(by_qtype_total.keys()) + list(vds_by_qtype.keys())))
    for k in all_qtypes:
        acc_info = summary["accuracy"]["by_question_type"].get(k, {})
        vds_info = summary["vds"]["by_question_type"].get(k, {})
        
        acc_str = f"Acc={acc_info.get('accuracy', 0):.4f} ({acc_info.get('correct', 0)}/{acc_info.get('total', 0)})" if acc_info else "Acc=N/A"
        vds_str = f"VDS={vds_info.get('mean', 0):.4f}±{vds_info.get('std', 0):.4f}" if vds_info else "VDS=N/A"
        
        print(f"  {k:35s}: {acc_str:25s} | {vds_str}")
    
    print(f"\nBy Modality Type:")
    all_modals = sorted(set(list(by_modal_total.keys()) + list(vds_by_modal.keys())))
    for k in all_modals:
        acc_info = summary["accuracy"]["by_modality_type"].get(k, {})
        vds_info = summary["vds"]["by_modality_type"].get(k, {})
        
        acc_str = f"Acc={acc_info.get('accuracy', 0):.4f} ({acc_info.get('correct', 0)}/{acc_info.get('total', 0)})" if acc_info else "Acc=N/A"
        vds_str = f"VDS={vds_info.get('mean', 0):.4f}±{vds_info.get('std', 0):.4f}" if vds_info else "VDS=N/A"
        
        print(f"  {k:35s}: {acc_str:25s} | {vds_str}")
    
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)

    ap.add_argument("--access", type=str, default="open", choices=["open", "both"])
    ap.add_argument("--visdep_sidecar", type=str, default=None)
    ap.add_argument("--visdep_min_score", type=float, default=0.65)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--torch_dtype", type=str, default="torch.bfloat16")

    ap.add_argument("--resume", default=None)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_jsonl = os.path.join(args.out_dir, "vds_predictions.jsonl")
    summary_json = os.path.join(args.out_dir, "vds_summary.json")

    print(f"\n{'='*60}")
    print(f"OmniMedVQA VDS Evaluation - CogVLM")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.out_dir}")
    print(f"VDS Filter: visdep_score >= {args.visdep_min_score}")
    print(f"Generation: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}")
    print(f"{'='*60}\n")

    run_vds_evaluation(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        out_pred_jsonl=pred_jsonl,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        dtype_str=args.torch_dtype,
        visdep_sidecar=args.visdep_sidecar,
        visdep_min_score=args.visdep_min_score,
        access=args.access,
        resume=args.resume,
    )

    analyze_vds_statistics(pred_jsonl=pred_jsonl, out_summary_json=summary_json)


if __name__ == "__main__":
    main()