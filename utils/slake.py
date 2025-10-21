import os
import json
import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
except Exception:
    load_dataset = None
    hf_hub_download = None


class SLAKE_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024,
                 split='train',
                 lang=None,
                 answer_required=True,
                 cache_dir=None,
                 imgs_dir=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.torch_type = torch_type
        self.input_length = input_length
        self.output_length = output_length
        self.padding_len = 2303
        self.max_length = self.input_length + self.output_length + self.padding_len
        self.split = split
        self.lang = lang
        self.answer_required = answer_required
        self.cache_dir = cache_dir

        if os.path.isdir(self.root_dir):
            json_path = os.path.join(self.root_dir, f"{'validation' if split=='validation' else split}.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Cannot find {json_path}, please confirm that the local contains three JSON files: train/validation/test.")
            with open(json_path, 'r', encoding='utf-8') as f:
                self.records = json.load(f)
        else:
            if load_dataset is None:
                raise ImportError("Missing dataset library, please first 'pip install datasets huggingface'.")
            ds = load_dataset(self.root_dir, split=split, cache_dir=cache_dir)
            self.records = [ds[i] for i in range(len(ds))]

        self.img_base_dirs = []
        if imgs_dir is not None:
            self.img_base_dirs.append(imgs_dir)
        if os.path.isdir(self.root_dir):
            for cand in ['imgs', 'images', 'img', 'Images']:
                p = os.path.join(self.root_dir, cand)
                if os.path.isdir(p):
                    self.img_base_dirs.append(p)

        if not self.img_base_dirs and not os.path.isdir(self.root_dir):
            if hf_hub_download is None:
                raise ImportError("Lack of huggingface_cub library, unable to automatically download imgs.rip.")
            try:
                zip_path = hf_hub_download(repo_id=self.root_dir, filename='imgs.zip', repo_type='dataset', cache_dir=cache_dir)
                extract_dir = os.path.join(os.path.dirname(zip_path), 'imgs_extracted')
                if not os.path.isdir(extract_dir):
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(extract_dir)
                self.img_base_dirs.append(extract_dir)
                print(f"[SLAKE_Sataset] The image has been decompressed from HF to:{extract_dir}")
            except Exception as e:
                print(f"[SLAKE_Sataset] Automatic download/decompression of imgs. zip failed: {e}")

        if not self.img_base_dirs:
            print("Warning: Image directory not explicitly found, will attempt to open by relative path (may fail).")

        self.samples = self._build_samples(self.records)
        print(f"[SLAKE_Sataset] Number of successfully loaded samples: {len(self.samples)}")

    def _resolve_image_path(self, img_name: str):
        for base in self.img_base_dirs:
            cand = os.path.join(base, img_name)
            if os.path.exists(cand):
                return cand
            cand2 = os.path.join(base, 'imgs', img_name)
            if os.path.exists(cand2):
                return cand2
        cand = os.path.join(self.root_dir, img_name)
        return cand

    def _build_samples(self, records):
        samples = []
        drop_no_img = 0
        drop_empty_ans = 0
        drop_lang = 0

        for idx, r in enumerate(records):
            try:
                q = r.get('question', None)
                a = r.get('answer', None)
                img_name = r.get('img_name', None)
                q_lang = r.get('q_lang', None)

                if self.lang is not None and q_lang is not None and q_lang != self.lang:
                    drop_lang += 1
                    continue
                if self.answer_required and (a is None or str(a).strip() == ''):
                    drop_empty_ans += 1
                    continue
                if img_name is None:
                    print(f"[Idx {idx}] is missing the imd_name field, skip.")
                    continue

                img_path = self._resolve_image_path(img_name)
                if not os.path.exists(img_path):
                    drop_no_img += 1
                    if drop_no_img <= 10:
                        print(f"[Idx {idx}] image does not exist: {img_path}")
                    continue

                samples.append({
                    'image_path': img_path,
                    'question': q,
                    'answer': a
                })
            except Exception as e:
                print(f"[Idx {idx}] parsing failed: {e} | Content:{r}")

        if drop_lang:
            print(f"[SLAKE_Sataset] Language filtering discard: {drop_lang}")
        if drop_empty_ans:
            print(f"[SLAKE_Sataset] Empty answer discarded: {drop_empty_ans}")
        if drop_no_img:
            print(f"[SLAKE_Sataset] Image not found, discard:{drop_no_img}")

        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError("Unsupported datatype in custom collate_fn")
        return batched_data

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        question = sample['question']
        answer = sample['answer']

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=question if question else "Please answer the medical question.",
            history=None,
            images=[image],
            answer=answer if answer is not None else ""
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            current_length = len(unpadded_tensor)
            if current_length >= pad_to_length:
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor,
                 torch.full([pad_to_length - current_length],
                            fill_value=pad_value,
                            dtype=unpadded_tensor.dtype,
                            device=unpadded_tensor.device)), dim=0)

        input_data['input_ids'] = pad_to_len(input_data['input_ids'], self.max_length, pad_value=128002)
        input_data['attention_mask'] = pad_to_len(input_data['attention_mask'], self.max_length, pad_value=0)
        input_data['token_type_ids'] = pad_to_len(input_data['token_type_ids'], self.max_length, pad_value=0)
        input_data['labels'] = pad_to_len(input_data['labels'], self.max_length, pad_value=-100)

        for key in input_data:
            if key == 'images':
                input_data[key] = [img.to(self.device).to(self.torch_type) for img in input_data[key]]
            else:
                input_data[key] = input_data[key].to(self.device)

        return input_data