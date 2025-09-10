import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class OmniMedVQA_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024,
                 use_doctor_prompt: bool = True,
                 access='open',                    # 'open' or 'both'
                 split_files=None,
                 visdep_sidecar: str = None,       # 可选：JSONL(推荐)/JSON，按 image_path → visdep_score
                 visdep_min_score: float = None,   # 可选：过滤阈值（如 0.7 只留高依赖）
                 visdep_weighting: bool = False    # 可选：输出 visdep_weight = 0.5+0.5*score
                 ):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.torch_type = torch_type
        self.input_length = input_length
        self.output_length = output_length
        self.padding_len = 2303
        self.max_length = self.input_length + self.output_length + self.padding_len
        self.access = access
        self.split_files = split_files

        self.use_doctor_prompt = use_doctor_prompt
        self.doctor_prompt = (
            "You are a medical AI assistant. Read the image and the question carefully. "
            "Then answer strictly by selecting one of the given options. "
            "Do not generate anything outside the provided options."
        )

        self._visdep_map = self._load_visdep_sidecar(visdep_sidecar) if visdep_sidecar else {}

        qa_base = os.path.join(root_dir, 'QA_information')
        self.qa_dirs = []
        if access in ['open', 'both']:
            self.qa_dirs.append(os.path.join(qa_base, 'Open-access'))
        if access == 'both':
            self.qa_dirs.append(os.path.join(qa_base, 'Restricted-access'))
        if not self.qa_dirs:
            raise ValueError(f"参数 access='{access}' 无效或不包含任何 QA 信息路径。")

        self.visdep_min_score = visdep_min_score
        self.visdep_weighting = visdep_weighting

        self.samples = self._load_samples()
        # === 可选：按分数过滤 ===
        if self.visdep_min_score is not None:
            before = len(self.samples)
            self.samples = [s for s in self.samples if s.get('visdep_score', 0.5) >= float(self.visdep_min_score)]
            print(f"[OmniMedVQA_Dataset] 视觉依赖过滤: {before} → {len(self.samples)} (阈值≥{self.visdep_min_score})")

        print(f"[OmniMedVQA_Dataset] 成功加载样本数: {len(self.samples)}")

    def _load_visdep_sidecar(self, path):
        if not os.path.exists(path):
            print(f"[OmniMedVQA_Dataset] visdep_sidecar 文件不存在: {path}")
            return {}
        m = {}
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        ip = obj.get("image_path")
                        sc = obj.get("visdep_score")
                        if ip is not None and isinstance(sc, (int, float)):
                            m[os.path.abspath(ip)] = float(max(0.0, min(1.0, sc)))
                    except Exception:
                        continue
        else:  # json
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
                # 支持两种格式：list[{"image_path":..., "visdep_score":...}] 或 dict[image_path]=score
                if isinstance(data, list):
                    for obj in data:
                        ip = obj.get("image_path")
                        sc = obj.get("visdep_score")
                        if ip is not None and isinstance(sc, (int, float)):
                            m[os.path.abspath(ip)] = float(max(0.0, min(1.0, sc)))
                elif isinstance(data, dict):
                    for ip, sc in data.items():
                        if isinstance(sc, (int, float)):
                            m[os.path.abspath(ip)] = float(max(0.0, min(1.0, sc)))
            except Exception as e:
                print(f"[OmniMedVQA_Dataset] 解析 sidecar 失败: {path} | {e}")
        print(f"[OmniMedVQA_Dataset] 载入 visdep_sidecar 条目数: {len(m)}")
        return m

    def _load_samples(self):
        samples = []
        for qa_dir in self.qa_dirs:
            if not os.path.isdir(qa_dir):
                print(f"[OmniMedVQA_Dataset] QA 文件夹不存在: {qa_dir}")
                continue
            for fname in os.listdir(qa_dir):
                if not fname.endswith('.json'):
                    continue
                dataset_name = fname[:-5]
                if self.split_files and dataset_name not in self.split_files:
                    continue
                path = os.path.join(qa_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[OmniMedVQA_Dataset] 解析 JSON 失败: {path} | {e}")
                    continue

                for item in data:
                    img_rel = item.get('image_path')
                    if img_rel is None:
                        continue
                    img_abs = os.path.join(self.root_dir, img_rel)
                    if not os.path.exists(img_abs):
                        img_abs = os.path.join(self.root_dir, 'Images', img_rel)
                    if not os.path.exists(img_abs):
                        print(f"[Missing Image] {img_abs} 不存在，跳过该项。")
                        continue
                    img_abs = os.path.abspath(img_abs)

                    # === 读取 visdep_score：优先题目 JSON，其次 sidecar，默认为 0.5 中性 ===
                    vscore = item.get('visdep_score', None)
                    if not isinstance(vscore, (int, float)):
                        vscore = self._visdep_map.get(img_abs, 0.5)
                    vscore = float(max(0.0, min(1.0, vscore)))

                    samples.append({
                        'image_path': img_abs,
                        'question': item.get('question'),
                        'options': [item.get(k) for k in ['option_A', 'option_B', 'option_C', 'option_D'] if item.get(k)],
                        'gt_answer': item.get('gt_answer'),
                        'question_type': item.get('question_type'),
                        'modality_type': item.get('modality_type'),
                        'dataset_name': item.get('dataset'),
                        'visdep_score': vscore,
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def custom_collate_fn(batch):
        batched = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched[key] = [x[key] for x in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched[key] = torch.stack([x[key] for x in batch])
            else:
                batched[key] = [x[key] for x in batch]
        return batched

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert("RGB")
        question = s['question']
        options = s.get('options', [])
        gt = s.get('gt_answer', '')

        opt_text = '\n'.join(f"- {opt}" for _, opt in enumerate(options))
        prefix = (self.doctor_prompt + "\n") if self.use_doctor_prompt else ""
        query = prefix + f"{question}\nOptions:\n{opt_text}"

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            history=None,
            images=[image],
            answer=gt
        )

        def pad_to_len(t, length, pad_value=0):
            l = len(t)
            return t[:length] if l >= length else torch.cat(
                (t, torch.full([length - l], fill_value=pad_value, dtype=t.dtype, device=t.device)), dim=0
            )

        input_data['input_ids'] = pad_to_len(input_data['input_ids'], self.max_length, pad_value=128002)
        input_data['attention_mask'] = pad_to_len(input_data['attention_mask'], self.max_length, pad_value=0)
        input_data['token_type_ids'] = pad_to_len(input_data['token_type_ids'], self.max_length, pad_value=0)
        input_data['labels'] = pad_to_len(input_data['labels'], self.max_length, pad_value=-100)

        norm = lambda x: " ".join((x or "").strip().lower().rstrip(".。").split())
        g = norm(gt)
        input_data['answer_text'] = g
        input_data['options_text'] = opt_text  # 便于日志
        vs = float(s['visdep_score'])
        input_data['visdep_score'] = torch.tensor(vs, device=self.device, dtype=torch.float32)
        if self.visdep_weighting:
            input_data['visdep_weight'] = torch.tensor(0.5 + 0.5 * vs, device=self.device, dtype=torch.float32)

        for key in input_data:
            if key == 'images':
                input_data[key] = [img.to(self.device).to(self.torch_type) for img in input_data[key]]
            else:
                if isinstance(input_data[key], torch.Tensor):
                    input_data[key] = input_data[key].to(self.device)

        return input_data
