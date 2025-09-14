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

client = OpenAI()  # 建议改为从环境变量读取，避免泄漏

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
    model: str = "gpt-4o-mini"            # 调用的模型名
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
    # 去标点
    table = str.maketrans("", "", string.punctuation + "，。；：？！【】（）「」『』、．")
    s = s.translate(table)
    # 合并多空格
    s = re.sub(r"\s+", " ", s)
    return s
# 从 gt 字符串中提取选项字母（支持 "B" 或 "B: fibroid" 之类格式）
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
        
# 将 item 中的 A-D 四个选项拼成一段文本，供 prompt 模板使用
def build_options_block(item: Dict[str, Any]) -> str:
    opts = []
    for k in ["option_A", "option_B", "option_C", "option_D"]:  # 依次检查四个选项键
        if item.get(k):                                           # 若存在该选项
            letter = k[-1].upper()                                # 从键名末尾取字母 A-D
            opts.append(f"{letter}. {item[k]}")                   # 拼为 "A. 选项内容"
    if not opts:
        # 极端兜底（数据缺失时，保证模板完整）
        opts = ["A. Option A", "B. Option B", "C. Option C", "D. Option D"]
    return "\n".join(opts)                                      # 以换行连接

# 尝试安全解析 JSON 字符串；若失败，尝试截取最外层大括号再解析
def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)                                      # 首选直接解析
    except Exception:
        # 兜底：找到最外层的 "{" 与 "}"，截取其中内容再解析
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start >= 0 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            return None                                           # 最终失败返回 None

# ------------------------------------------------------------
# 评分函数（根据 GPT 返回的布尔特征与文本答题能力计算视觉依赖分）
# ------------------------------------------------------------
# 输入：feat（GPT 的 JSON 输出 + 后续补充字段） -> 输出：分数 [0,1]
def visdep_score_from_features(feat: Dict[str, Any]) -> float:
    # 布尔 -> float
    def b(x): return 1.0 if bool(x) else 0.0
    # 裁剪到 [0,1]
    def clip01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # ---------- 读取特征 ----------
    needs_spatial         = b(feat.get("needs_spatial", False))
    needs_visual_sign     = b(feat.get("needs_visual_sign", False))
    needs_compare_views   = b(feat.get("needs_compare_views", False))
    needs_annot_ref       = b(feat.get("needs_annot_ref", False))
    is_knowledge_question = b(feat.get("is_knowledge_question", False))
    mentions_image        = b(feat.get("instruction_mentions_image", False))

    conf = clip01(float(feat.get("text_only_confidence", 0.0)))
    ans  = str(feat.get("text_only_answer", "unknown")).strip().lower()
    is_unknown = 1.0 if ans in ("", "unknown", "n/a", "na", "none") else 0.0
    correct = clip01(float(feat.get("text_only_correct", 0.0)))  # 允许 0/1 或概率

    # ---------- 视觉需求 ----------
    # 权重总和≈1，突出“空间/征象”，比较/标注次之，“仅提到图像”为轻信号
    v_base = (
        0.34 * needs_spatial +
        0.25 * needs_visual_sign +
        0.16 * needs_compare_views +
        0.15 * needs_annot_ref +
        0.10 * mentions_image
    )
    # 知识题：整体弱化视觉必要性（避免相减变负）
    v_need = v_base * (0.7 if is_knowledge_question else 1.0)

    # ---------- 文本可答性（连续化 reducibility ∈ [0,1]）----------
    # 答对：0.5~1.0 强加成（置信度越高越可由文本回答）
    reduc_if_correct = correct * (0.5 + 0.5 * conf)            # 0, 或 [0.5,1.0]
    # 答错但非 unknown：给很小的置信度加成（体现“有点依据但不充分”）
    reduc_if_wrong_small = (1.0 - correct) * (1.0 - is_unknown) * (0.2 * conf)  # 0~0.2
    # unknown：不吃到任何置信度加成
    reducibility = clip01(reduc_if_correct + reduc_if_wrong_small)

    # 知识题：文本主导，剩余“需要度”折半
    text_term = 1.0 - reducibility
    if is_knowledge_question:
        text_term *= 0.5

    # ---------- 综合得分 ----------
    score = 0.60 * v_need + 0.40 * text_term
    return clip01(score)

# 分数映射为标签：low / medium / high
def score_to_tag(score: float, t_low: float, t_high: float) -> str:
    if score < t_low: return "low"
    if score < t_high: return "medium"
    return "high"

# ------------------------------------------------------------
# GPT 调用：带重试、固定 response_format 要求 JSON
# ------------------------------------------------------------
# 装饰器：指数退避重试，最多 6 次
@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(6))
def call_gpt(model: str, system: str, user: str, temperature: float) -> Dict[str, Any]:
    # 发送 Chat Completions 请求；要求以 JSON 对象形式返回
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    # 取第一条候选的内容
    content = resp.choices[0].message.content
    # 尝试解析为 JSON（失败有兜底）
    obj = safe_json_loads(content) or {}
    return obj

# ------------------------------------------------------------
# 处理单条样本：构造 prompt -> 调用 GPT -> 计算是否答对 -> 打分与打标签
# ------------------------------------------------------------
def annotate_one(item: Dict[str, Any], args: Args) -> Optional[Dict[str, Any]]:
    q = item.get("question") or ""                      # 题干文本
    options_block = build_options_block(item)             # 选项文本块
    # 将题干与选项填入用户模板，得到最终 user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(question=q, options_block=options_block)

    # 调用 GPT：根据文本判断视觉依赖性，返回 JSON 特征
    out = call_gpt(args.model, SYSTEM_PROMPT, user_prompt, args.temperature)

    # 计算 text_only_correct：比较 GPT 给出的 text_only_answer 与 GT 答案首字母
    gt_letter = parse_gt_letter(item)
    ans = str(out.get("text_only_answer", "unknown")).upper()
    text_only_correct = 1 if (gt_letter is not None and ans == gt_letter) else 0
    out["text_only_correct"] = text_only_correct  # 回写到特征中，便于后续打分

    # 计算视觉依赖分与标签
    score = visdep_score_from_features(out)
    tag = score_to_tag(score, *args.thresholds)

    # 组织输出记录，包含路径、元信息、特征、分数与标签
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
    qa_base = os.path.join(root_dir, "QA_information")   # 基路径：QA 信息目录
    dirs = []
    if access in ("open", "both"):                      # open 或 both：包含 Open-access
        dirs.append(os.path.join(qa_base, "Open-access"))
    if access == "both":                                  # both：再包含 Restricted-access
        dirs.append(os.path.join(qa_base, "Restricted-access"))
    for qa_dir in dirs:
        if not os.path.isdir(qa_dir):                     # 如果目录不存在，跳过
            continue
        for fname in os.listdir(qa_dir):                  # 遍历目录中文件
            if not fname.endswith(".json"):              # 只处理 .json 文件
                continue
            dataset_name = fname[:-5]                     # 去掉 ".json"，得到数据集名
            if split_files and dataset_name not in split_files:  # 若有限定名单，过滤
                continue
            path = os.path.join(qa_dir, fname)
            try:
                data = json.load(open(path, "r", encoding="utf-8"))  # 读取 JSON 列表
            except Exception:
                continue                                              # 读取失败则跳过
            for it in data:
                yield it                                              # 逐条产出样本 item

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
                # 末尾行可能被中断写坏，直接跳过
                continue
            p = obj.get("question_id")
            if p:
                # 规范化为绝对路径，避免重复
                done.add(p)
    return done

# ------------------------------------------------------------
# 主入口：解析参数 -> 收集样本 -> 并发标注 -> 增量写结果
# ------------------------------------------------------------
def main():
    # 1) 命令行参数
    ap = argparse.ArgumentParser("Annotate OmniMedVQA visual dependency by GPT (text-only)")
    ap.add_argument("--root_dir", type=str, default="./")
    ap.add_argument("--access", type=str, default="open", choices=["open", "both"])
    ap.add_argument("--out_jsonl", type=str, default="visdep_scores.jsonl")
    ap.add_argument("--out_merge_json", type=str, default=None)  # 可选：最终汇总映射
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--qps", type=float, default=2.0)            # 每秒最多提交多少请求
    ap.add_argument("--max_inflight", type=int, default=32)      # 在飞任务上限（避免堆太多 future）
    ap.add_argument("--threshold_low", type=float, default=0.35)
    ap.add_argument("--threshold_high", type=float, default=0.70)
    ap.add_argument("--split_files", type=str, default=None, help="comma-separated dataset names (no .json)")
    ap.add_argument("--preview_every", type=int, default=500, help="每处理多少条打印一次预览")
    ap.add_argument("--show_k", type=int, default=5, help="预览时展示的样本条数")
    args_ns = ap.parse_args()

    # 2) 打包成内部 Args（你的 annotate_one 里用到）
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

    # 3) 准备输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)

    # 4) 收集样本 + 断点续跑（跳过已完成）
    items_all = list(iter_items(args.root_dir, args.access, args.split_files))
    done_set = load_done_set(args.out_jsonl)     # ← 读取已完成集合
    items = [it for it in items_all if it.get("question_id") not in done_set]

    print(f"[Info] Collected {len(items_all)} items; resume skips {len(done_set)} done; pending {len(items)}.")

    # 5) 并发：边提交边接收，QPS 限速，增量写 JSONL
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
        inflight = {}  # {future: item}
        tick = time.time()
        submitted = 0
        it = iter(items)

        while True:
            # 补齐在飞任务（受在飞上限与 QPS 限制）
            while len(inflight) < max_inflight:
                try:
                    item = next(it)
                except StopIteration:
                    break
                fut = ex.submit(annotate_one, item, args)  # 立刻后台执行
                inflight[fut] = item

                submitted += 1
                if submitted % qps == 0:
                    dt = time.time() - tick
                    if dt < 1.0:
                        time.sleep(1.0 - dt)  # 粗粒度限速：每提交 qps 个至少用 1 秒
                    tick = time.time()

            # 没有在飞任务且也没有可提交的 => 结束
            if not inflight:
                break

            # 等至少一个完成，立刻处理（边提交边接收）
            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                item = inflight.pop(f)
                try:
                    rec = f.result()  # annotate_one 的返回
                except Exception:
                    rec = None

                if rec:
                    # 增量写出（断点续跑稳妥）
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()

                    # 训练外汇总（仅当需要 out_merge_json 时）
                    if results_for_merge is not None:
                        results_for_merge.append(rec)

                    # 进度/统计/预览
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

    # 6) 可选：汇总写 mapping（仅当传了 --out_merge_json）
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

# Python 脚本常见入口：仅当作为主程序运行时才执行 main()
if __name__ == "__main__":
    main()
