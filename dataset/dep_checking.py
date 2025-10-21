from rl.embedder import MedClipEmbedder
import argparse, logging
import os, json
from typing import List, Optional
from tqdm import tqdm
from rl.roi import make_counterfactuals
import torch.nn.functional as F
import torch
from PIL import Image
import collections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dep_checking")

def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(-1)

def iter_items(dataset_path: str, access: str, split_files: Optional[List[str]]):
    qa_base = os.path.join(dataset_path, "QA_information")
    dirs = []
    if access == "open":
        dirs.append(os.path.join(qa_base, "Open-access"))
    if access == "both":
        dirs.append(os.path.join(qa_base, "Restricted-access"))

    for qa_dir in dirs:
        if not os.path.isdir(qa_dir):
            continue
        for fname in os.listdir(qa_dir):
            if not fname.endswith(".json"):
                continue
            dataset_name = fname[:-5]
            if split_files and dataset_name not in split_files:
                continue
            path = os.path.join(qa_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            for it in data:
                yield it

def load_done_set(out_jsonl: str) -> set:
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("question_id")
            if qid:
                done.add(qid)
    return done

def main():
    ap = argparse.ArgumentParser("Annotate OmniMedVQA visual dependency by GPT (text-only)")
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument("--access", type=str, default="both", choices=["open", "both"])
    ap.add_argument("--out_jsonl", type=str, default="visdep_scores.jsonl")
    ap.add_argument("--split_files", type=str, default=None, help="comma-separated dataset names (no .json)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)

    split_list = None
    if args.split_files:
        split_list = [x.strip() for x in args.split_files.split(",") if x.strip()]

    items_all = list(iter_items(args.dataset_path, args.access, split_list))
    done_set = load_done_set(args.out_jsonl)
    items = [it for it in items_all if it.get("question_id") not in done_set]

    print(f"[Info] Collected {len(items_all)} items; resume skips {len(done_set)} done; pending {len(items)}.")

    logger.info(">> Loading MedCLIP embedder ...")
    embedder = MedClipEmbedder()
    embedder.eval()
    logger.info(">> MedCLIP ready.")

    with open(args.out_jsonl, "a", encoding="utf-8") as fout:
        pbar = tqdm(total=len(items), desc="Annotating", dynamic_ncols=True)

        for item in items:
            try:
                img_rel = item.get("image_path")
                if not img_rel:
                    pbar.update(1)
                    continue
                if args.access is open:
                    img_abs = os.path.join(args.dataset_path, img_rel)
                else:
                    img_abs = img_rel.replace("${dataset_root_path}", args.dataset_path)
                if not os.path.exists(img_abs):
                    pbar.update(1)
                    continue
                img_abs = os.path.abspath(img_abs)

                gt_answer = item.get("gt_answer")
                if gt_answer is None or (isinstance(gt_answer, str) and gt_answer.strip() == ""):
                    pbar.update(1)
                    continue

                pil_img = Image.open(img_abs).convert("RGB")
                cf_raw = make_counterfactuals(pil_img, blur_sigma=3.0, shuffle_grid=4)

                def ensure_pil_list(x):
                    if isinstance(x, Image.Image):
                        return [x]
                    if isinstance(x, str):
                        return [Image.open(x).convert("RGB")]
                    if isinstance(x, collections.abc.Sequence):
                        out = []
                        for xi in x:
                            if isinstance(xi, Image.Image):
                                out.append(xi)
                            elif isinstance(xi, str):
                                out.append(Image.open(xi).convert("RGB"))
                            else:
                                out.append(xi)
                        return out
                    return [x]

                cf_imgs = ensure_pil_list(cf_raw)
                n_runs = 3  
                sims, sims_cf, diffs = [], [], []

                with torch.no_grad():
                    for _ in range(n_runs):
                        img = embedder.encode_image(pil_img)
                        img_cf = embedder.encode_image(cf_imgs)
                        txt = embedder.encode_text([gt_answer])

                        sim = _cos(img, txt)
                        sim_cf = _cos(img_cf, txt)
                        diff = sim - sim_cf

                        sims.append(sim)
                        sims_cf.append(sim_cf)
                        diffs.append(diff)

                    sim = torch.stack(sims).mean(0)
                    sim_cf = torch.stack(sims_cf).mean(0)
                    diff = torch.stack(diffs).mean(0)

                rec = {
                    "question_id": item.get("question_id"),
                    "vec_g": sim.squeeze().item() if torch.is_tensor(sim) else sim,
                    "vec-cf": sim_cf.squeeze().tolist() if torch.is_tensor(sim_cf) else sim_cf,
                    "dep": diff.squeeze().tolist() if torch.is_tensor(diff) else diff,
                }

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

            except Exception as e:
                print(f"[Error] {item.get('question_id')}error:{e}")

            pbar.update(1)

        pbar.close()

if __name__ == "__main__":
    main()
