import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset


class ROCOv2_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024,
                 split='train'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.torch_type = torch_type
        self.input_length = input_length
        self.output_length = output_length
        self.padding_len = 2303
        self.max_length = self.input_length + self.output_length + self.padding_len

        data_path = os.path.join(self.root_dir, "data")
        self.dataset = load_dataset("parquet", data_dir=data_path, split=split)
        print(f"[ROCOv2_Dataset] 成功加载样本数: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [item[key] for item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError(f"Unsupported datatype for key {key} in custom collate_fn")
        return batched_data

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Hugging Face datasets 会自动 decode 为 PIL Image，但本地 parquet 可能为路径/bytes，需要手动加载
        image = sample["image"]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(image["path"] if "path" in image else image["bytes"]).convert("RGB")
        elif isinstance(image, str) and os.path.exists(image):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"无法识别的 image 格式: {type(image)}")

        caption = sample["caption"]

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
            return torch.cat((
                unpadded_tensor,
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
