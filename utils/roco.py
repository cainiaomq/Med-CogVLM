import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class ROCO_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024,
                 split='train',
                 category='non-radiology'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.image_dir = os.path.join(root_dir, split, category, 'images')
        self.caption_file = os.path.join(root_dir, split, category, 'captions.txt')
        self.input_length = input_length
        self.output_length = output_length
        self.device = device
        self.torch_type = torch_type
        self.padding_len = 2303
        self.max_length = self.input_length + self.output_length + self.padding_len
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        with open(self.caption_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line or '\t' not in line:
                    print(f"[Line {line_idx}] 格式错误或空行: {line}")
                    continue
                try:
                    image_id, caption = line.split('\t', 1)
                    img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
                    if os.path.exists(img_path):
                        samples.append({'image_path': img_path, 'caption': caption})
                    else:
                        print(f"[Line {line_idx}] 图像不存在: {img_path}")
                except Exception as e:
                    print(f"[Line {line_idx}] 解析失败: {e} | 内容: {line}")
        print(f"[ROCO_Dataset] 成功加载样本数: {len(samples)}")
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
        caption = sample['caption']

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query="Please describe the medical image.",
            history=None,
            images=[image],
            answer=caption
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
