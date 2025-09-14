import os, json, re, argparse, time
import string
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are a medical data annotator. Your task is to judge how visually-dependent a multiple-choice medical VQA question is, given only the text (no images).
"Visually-dependent" means the question cannot be reliably answered without looking at the image (e.g., requires spatial localization, radiologic signs/patterns, intensity/signal characteristics, comparing views/timepoints/sides, or interpreting markers/arrows).
Use ONLY textual cues in the prompt. Do NOT hallucinate image content. Be consistent and deterministic.
Return strict JSON with the required keys only.
"""

USER_PROMPT_TEMPLATE = """Question:
{question}

Options (choose-one):
{options_block}

(You do NOT have access to the image. You only see text.)

Label each boolean using the following deterministic rules:

1) needs_spatial = true if the text implies localization or region selection is required.
2) needs_visual_sign = true if the text requires recognizing a radiological sign/pattern or signal/intensity/attenuation/appearance.
3) needs_compare_views = true if the text requires comparing different views/slices/timepoints/sides/modalities.
4) needs_annot_ref = true if the text asks about a region indicated by an arrow/marker/ROI/box/label.
5) is_knowledge_question = true if the question can be answered purely by general medical knowledge without needing to see the image.
6) instruction_mentions_image = true if the wording explicitly references an image or modality.

Answer selection (text_only_answer / text_only_confidence):
- Step 1: Decide the most plausible option you could answer using only the text 
  (general medical knowledge, lexical cues, or common-sense priors). 
  If no textual basis at all, set text_only_answer="unknown".
- Step 2: Assign confidence (text_only_confidence ∈ [0,1]) according to the following rules:
  • 0.8–1.0: Strong textual cue or widely-known fact clearly selects one option 
             (e.g., modality names, textbook facts, unambiguous keywords).  
  • 0.5–0.7: Some textual priors suggest one option, but uncertainty remains 
             (e.g., partial cues, plausible but not certain).  
  • 0.2–0.4: Only weak lexical cues or common-sense priors are available; 
             guessing is possible but unreliable.  
  • 0.0: No textual basis at all → set text_only_answer="unknown".  
- Important: Never assign both text_only_answer="unknown" and confidence > 0.0. 
  If you guess (confidence ≥ 0.2), you must pick an actual option A–D.

Reasoning constraint:
- Use exactly one concise sentence in "reason_short" explaining the textual cues that determined the flags and answerability. Do not reference unseen image content.

Now produce JSON with exactly these keys:
{{
  "needs_spatial": true/false,
  "needs_visual_sign": true/false,
  "needs_compare_views": true/false,
  "needs_annot_ref": true/false,
  "is_knowledge_question": true/false,
  "instruction_mentions_image": true/false,
  "text_only_answer": "A|B|C|D",
  "text_only_confidence": number in [0,1],
  "reason_short": "one-sentence justification using only textual cues"
}}
"""


# ------------------------------------------------------------
# 正则表达式：提取选项字母（A-D）
# ------------------------------------------------------------
# 用于匹配答案开头的字母（A-D），后面可跟分隔符，如 ":)．-" 等；忽略大小写
LETTER_RE = re.compile(r'^\s*([A-D])(\b|:|\)|\.|、|．|-)?', re.I)

# ------------------------------------------------------------
# 参数数据类（命令行参数收集到这里，方便传递）
# ------------------------------------------------------------
@dataclass
class Args:
    root_dir: str                          # 数据集根目录
    access: str = "both"                   # 读取范围："open" 或 "both"（包含受限）
    out_jsonl: str = "visdep_scores.jsonl" # 输出 JSONL 路径
    out_merge_json: Optional[str] = None   # 可选：合并输出 JSON（image_path -> 结果映射）
    model: str = "gpt-4o-mini"             # 调用的模型名
    temperature: float = 0.2               # 生成温度（越低越稳定）
    max_workers: int = 8                   # 线程池最大并发数
    qps: float = 2.0                       # 每秒请求上限（粗略节流）
    thresholds: tuple = (0.35, 0.70)       # 分段阈值（low/medium/high）
    split_files: Optional[List[str]] = None# 只处理指定的数据集 JSON（文件名不含 .json）

# ------------------------------------------------------------
# 工具函数：解析、拼装与健壮性处理
# ------------------------------------------------------------
def _norm_text(x: str) -> str:
    """小写 + 去首尾空白 + 去掉常见标点 + 合并多空格"""
    if x is None:
        return ""
    s = x.strip().lower()
    table = str.maketrans("", "", string.punctuation + "，。；：？！【】（）「」『』、．")
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s)
    return s
def parse_gt_letter(item: Dict[str, Any]) -> Optional[str]:
    gt_raw = item.get("gt_answer", "")
    gt_norm = _norm_text(str(gt_raw))
    options = {
        "A": item.get("option_A", ""),
        "B": item.get("option_B", ""),
        "C": item.get("option_C", ""),
        "D": item.get("option_D", ""),
    }
    for L, opt in options.items():
        if _norm_text(str(opt)) == gt_norm and gt_norm != "":
            return L
        
def build_options_block(item: Dict[str, Any]) -> str:
    opts = []
    for k in ["option_A", "option_B", "option_C", "option_D"]:
        if item.get(k):
            letter = k[-1].upper()
            opts.append(f"{letter}. {item[k]}")
    if not opts:
        opts = ["A. Option A", "B. Option B", "C. Option C", "D. Option D"]
    return "\n".join(opts)

# 尝试安全解析 JSON 字符串；若失败，尝试截取最外层大括号再解析
def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start >= 0 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            return None

# ------------------------------------------------------------
# 评分函数（根据 GPT 返回的布尔特征与文本答题能力计算视觉依赖分）
# ------------------------------------------------------------
# 输入：feat（GPT 的 JSON 输出 + 后续补充字段） -> 输出：分数 [0,1]
def visdep_score_from_features(feat: Dict[str, Any]) -> float:
    def b(x): return 1.0 if bool(x) else 0.0
    def clip01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    needs_spatial         = b(feat.get("needs_spatial", False))
    needs_visual_sign     = b(feat.get("needs_visual_sign", False))
    needs_compare_views   = b(feat.get("needs_compare_views", False))
    needs_annot_ref       = b(feat.get("needs_annot_ref", False))
    is_knowledge_question = b(feat.get("is_knowledge_question", False))
    mentions_image        = b(feat.get("instruction_mentions_image", False))

    conf = clip01(float(feat.get("text_only_confidence", 0.0)))
    ans  = str(feat.get("text_only_answer", "unknown")).strip().lower()
    is_unknown = 1.0 if ans in ("", "unknown", "n/a", "na", "none") else 0.0
    correct = clip01(float(feat.get("text_only_correct", 0.0)))

    # ---------- 视觉需求 ----------
    # 权重总和≈1，突出“空间/征象”，比较/标注次之，“仅提到图像”为轻信号
    v_base = (
        0.34 * needs_spatial +
        0.25 * needs_visual_sign +
        0.16 * needs_compare_views +
        0.15 * needs_annot_ref +
        0.10 * mentions_image
    )
    v_need = v_base * (0.7 if is_knowledge_question else 1.0)

    # ---------- 文本可答性（连续化 reducibility ∈ [0,1]）----------
    reduc_if_correct = correct * (0.5 + 0.5 * conf)
    reduc_if_wrong_small = (1.0 - correct) * (1.0 - is_unknown) * (0.2 * conf)  # 0~0.2
    reducibility = clip01(reduc_if_correct + reduc_if_wrong_small)

    text_term = 1.0 - reducibility
    if is_knowledge_question:
        text_term *= 0.5

    # ---------- 综合得分 ----------
    score = 0.60 * v_need + 0.40 * text_term
    return clip01(score)

def score_to_tag(score: float, t_low: float, t_high: float) -> str:
    if score < t_low: return "low"
    if score < t_high: return "medium"
    return "high"

# ------------------------------------------------------------
# GPT 调用：带重试、固定 response_format 要求 JSON
# ------------------------------------------------------------
@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(6))
def call_gpt(model: str, system: str, user: str, temperature: float) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    obj = safe_json_loads(content) or {}
    return obj

# ------------------------------------------------------------
# 处理单条样本：构造 prompt -> 调用 GPT -> 计算是否答对 -> 打分与打标签
# ------------------------------------------------------------
def annotate_one(item: Dict[str, Any], args: Args) -> Optional[Dict[str, Any]]:
    q = item.get("question") or ""
    options_block = build_options_block(item)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=q, options_block=options_block)

    out = call_gpt(args.model, SYSTEM_PROMPT, user_prompt, args.temperature)

    gt_letter = parse_gt_letter(item)
    ans = str(out.get("text_only_answer", "unknown")).upper()
    text_only_correct = 1 if (gt_letter is not None and ans == gt_letter) else 0
    out["text_only_correct"] = text_only_correct

    score = visdep_score_from_features(out)
    tag = score_to_tag(score, *args.thresholds)

    rec = {
        "question_id": item.get("question_id"),
        "vis_features": out,
        "visdep_score": score,
        "visdep_tag": tag,
    }
    return rec

# ------------------------------------------------------------
# 遍历数据集：按 access 选择 Open-access / Restricted-access 下的 JSON 文件
# ------------------------------------------------------------
def iter_items(root_dir: str, access: str, split_files: Optional[List[str]]):
    qa_base = os.path.join(root_dir, "QA_information")
    dirs = []
    if access in ("open", "both"):
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
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception:
                continue
            for it in data:
                yield it

def _abs_img_path(root_dir: str, item: dict) -> str:
    img_rel = item.get("question_id") or ""
    return os.path.abspath(os.path.join(root_dir, img_rel))

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
            p = obj.get("question_id")
            if p:
                done.add(p)
    return done

# ------------------------------------------------------------
# 主入口：解析参数 -> 收集样本 -> 并发标注 -> 增量写结果
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Annotate OmniMedVQA visual dependency by GPT (text-only)")
    ap.add_argument("--root_dir", type=str, default="./")
    ap.add_argument("--access", type=str, default="open", choices=["open", "both"])
    ap.add_argument("--out_jsonl", type=str, default="visdep_scores.jsonl")
    ap.add_argument("--out_merge_json", type=str, default=None)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--max_inflight", type=int, default=32)
    ap.add_argument("--threshold_low", type=float, default=0.35)
    ap.add_argument("--threshold_high", type=float, default=0.70)
    ap.add_argument("--split_files", type=str, default=None, help="comma-separated dataset names (no .json)")
    ap.add_argument("--preview_every", type=int, default=500, help="每处理多少条打印一次预览")
    ap.add_argument("--show_k", type=int, default=5, help="预览时展示的样本条数")
    args_ns = ap.parse_args()

    args = Args(
        root_dir=args_ns.root_dir,
        access=args_ns.access,
        out_jsonl=args_ns.out_jsonl,
        out_merge_json=args_ns.out_merge_json,
        model=args_ns.model,
        temperature=args_ns.temperature,
        max_workers=args_ns.max_workers,
        qps=args_ns.qps,
        thresholds=(args_ns.threshold_low, args_ns.threshold_high),
        split_files=args_ns.split_files.split(",") if args_ns.split_files else None,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)

    items_all = list(iter_items(args.root_dir, args.access, args.split_files))
    done_set = load_done_set(args.out_jsonl)
    items = [it for it in items_all if it.get("question_id") not in done_set]

    print(f"[Info] Collected {len(items_all)} items; resume skips {len(done_set)} done; pending {len(items)}.")

    qps = max(1, int(args_ns.qps))
    max_inflight = max(1, int(args_ns.max_inflight))
    preview_every = max(0, int(args_ns.preview_every))
    show_k = max(1, int(args_ns.show_k))

    results_for_merge = [] if args_ns.out_merge_json else None
    running_sum, done_cnt = 0.0, 0
    preview_buf = deque(maxlen=show_k)

    with ThreadPoolExecutor(max_workers=args_ns.max_workers) as ex, \
         open(args.out_jsonl, "a", encoding="utf-8") as fout:

        pbar = tqdm(total=len(items), desc="Annotating", dynamic_ncols=True)
        inflight = {}
        tick = time.time()
        submitted = 0
        it = iter(items)

        while True:
            while len(inflight) < max_inflight:
                try:
                    item = next(it)
                except StopIteration:
                    break
                fut = ex.submit(annotate_one, item, args)
                inflight[fut] = item

                submitted += 1
                if submitted % qps == 0:
                    dt = time.time() - tick
                    if dt < 1.0:
                        time.sleep(1.0 - dt)
                    tick = time.time()

            if not inflight:
                break

            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                item = inflight.pop(f)
                try:
                    rec = f.result()
                except Exception:
                    rec = None

                if rec:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()

                    if results_for_merge is not None:
                        results_for_merge.append(rec)

                    score = float(rec.get("visdep_score", 0.5))
                    running_sum += score
                    done_cnt += 1
                    pbar.set_postfix({"done": done_cnt, "mean": f"{running_sum / max(1, done_cnt):.3f}"})
                    preview_buf.append({"image_path": rec.get("image_path", ""), "visdep_score": score})

                    if preview_every and done_cnt % preview_every == 0:
                        print(f"\n[Preview] 最近 {len(preview_buf)} 条：")
                        for x in list(preview_buf):
                            print(f"  score={x['visdep_score']:.3f}  |  {x['image_path']}")

                pbar.update(1)

        pbar.close()

    if args_ns.out_merge_json and results_for_merge:
        mapping = {
            r["image_path"]: {
                "visdep_score": r["visdep_score"],
                "visdep_tag":   r.get("visdep_tag"),
                "vis_features": r.get("vis_features"),
            } for r in results_for_merge
        }
        with open(args_ns.out_merge_json, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote merged mapping JSON: {args_ns.out_merge_json}")

if __name__ == "__main__":
    main()
