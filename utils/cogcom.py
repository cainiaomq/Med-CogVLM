import os
import json
import torch
import random
from torch.utils.data import Dataset
from PIL import Image

class CogCoM_Dataset(Dataset):
    START_PROMPTS = [
        "Given a question, please solve the question step-by-step with a chain of manipulations. {QUESTION}",
        "{QUESTION} Please proceed with the question by optionally applying a series of manipulations.",
        "Please solve the problem step by step. {QUESTION}",
    ]

    END_PROMPTS = [
        "So the answer is {ANSWER}.",
        "Therefore, the answer to the question is {ANSWER}.",
        "Hence the final answer is {ANSWER}.",
    ]

    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024,
                 split='comp',
                 use_com_reasoning=True,
                 use_prompt_template=True,
                 pure_reasoning_only=True):
        
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
        self.use_com_reasoning = use_com_reasoning
        self.use_prompt_template = use_prompt_template
        self.pure_reasoning_only = pure_reasoning_only

        self.samples = self._load_dataset()
        print(f"[CogCoM_Dataset] split={split}, use_com_reasoning={use_com_reasoning}, pure_reasoning_only={pure_reasoning_only}")
        print(f"[CogCoM_Dataset] Number of samples successfully loaded: {len(self.samples)}")

    def _load_dataset(self):
        jsonl_path = os.path.join(self.root_dir, f"{self.split}.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Not found {jsonl_path}")

        samples = []
        drop_no_img = 0
        drop_empty = 0
        drop_no_reasoning = 0
        drop_has_tool = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    
                    image_path = record.get('image_path', None)
                    if not image_path:
                        continue
                    
                    full_image_path = os.path.join(self.root_dir, image_path)
                    if not os.path.exists(full_image_path):
                        drop_no_img += 1
                        if drop_no_img <= 5:
                            print(f"[Line {line_num}] Image does not exist: {full_image_path}")
                        continue
                    
                    metadata_list = record.get('metadata', [])
                    for item in metadata_list:
                        if self.pure_reasoning_only:
                            if not self._is_pure_reasoning(item.get('final_com', {})):
                                drop_has_tool += 1
                                continue
                        
                        sample = self._parse_metadata_item(item, full_image_path)
                        if sample is None:
                            drop_empty += 1
                            continue
                        if not sample['answer']:
                            drop_no_reasoning += 1
                            continue
                        samples.append(sample)
                        
                except json.JSONDecodeError as e:
                    print(f"[Line {line_num}] JSON Parsing error: {e}")

        if drop_no_img:
            print(f"[CogCoM_Dataset] Image cannot be found and discarded: {drop_no_img}")
        if drop_empty:
            print(f"[CogCoM_Dataset] Empty problem discarded: {drop_empty}")
        if drop_no_reasoning:
            print(f"[CogCoM_Dataset] Empty reasoning chain dropout: {drop_no_reasoning}")
        if drop_has_tool:
            print(f"[CogCoM_Dataset] Discard with tool call: {drop_has_tool}")

        return samples

    def _parse_metadata_item(self, item, image_path):
        pid = item.get('pid', -1)
        question = item.get('question', None)
        answer = item.get('answer', None)
        final_com = item.get('final_com', None)

        if isinstance(question, list):
            question_text = ' '.join(question) if question else ""
        else:
            question_text = str(question) if question else ""

        if not question_text:
            return None

        if self.use_prompt_template:
            prompt = random.choice(self.START_PROMPTS).format(QUESTION=question_text)
        else:
            prompt = question_text

        if self.use_com_reasoning and final_com:
            reasoning_text = self._format_com_reasoning(final_com)
            if answer:
                final_answer = random.choice(self.END_PROMPTS).format(ANSWER=str(answer))
                answer_text = reasoning_text + " " + final_answer if reasoning_text else final_answer
            else:
                answer_text = reasoning_text
        else:
            answer_text = str(answer) if answer else ""

        return {
            'pid': pid,
            'image_path': image_path,
            'question': prompt,
            'answer': answer_text
        }

    def _is_pure_reasoning(self, final_com):
        if not final_com or not isinstance(final_com, dict):
            return True
        
        for key, node in final_com.items():
            if '--' not in key:
                continue
            if not isinstance(node, dict):
                continue
            
            func = node.get('func')
            if func is not None and func not in ['none', 'null', '']:
                return False
        
        return True

    def _format_com_reasoning(self, final_com):
        """格式化CoM推理链为文本"""
        if isinstance(final_com, str):
            try:
                final_com = json.loads(final_com)
            except json.JSONDecodeError:
                return final_com

        if not isinstance(final_com, dict):
            return str(final_com)

        chain = self._find_valid_chain(final_com)
        
        reasoning_steps = []
        for node in chain:
            if isinstance(node, dict):
                desc = node.get('desc', '')
                
                variables = node.get('variables', {})
                if variables and desc:
                    for var_key, var_val in variables.items():
                        if var_val is not None:
                            placeholder = f'`{var_key}`'
                            if placeholder in desc:
                                desc = desc.replace(placeholder, str(var_val))
                
                if desc and desc.strip():
                    reasoning_steps.append(desc.strip())

        return " ".join(reasoning_steps) if reasoning_steps else ""

    def _find_valid_chain(self, final_com):
        tree = {}
        
        for key, node in final_com.items():
            if '--' not in key:
                continue
            parent_id, current_id = key.split('--')
            parent_id_norm = parent_id.replace('*', '0')
            
            if parent_id_norm not in tree:
                tree[parent_id_norm] = []
            tree[parent_id_norm].append((current_id, node))
        
        def dfs(node_id, path):
            node_id_norm = node_id.replace('*', '0')
            if node_id_norm not in tree:
                return None
            for child_id, node in tree[node_id_norm]:
                new_path = path + [node]
                if node.get('found', False):
                    return new_path
                result = dfs(child_id, new_path)
                if result:
                    return result
            return None
        
        chain = dfs('-1,0', [])
        if chain:
            return chain
        
        try:
            def sort_key(k):
                k = k.replace('*', '0')
                parts = k.replace('--', ',').split(',')
                return tuple(int(p) for p in parts)
            
            sorted_keys = sorted(
                [k for k in final_com.keys() if '--' in k],
                key=sort_key
            )
            return [final_com[key] for key in sorted_keys]
        except:
            return list(final_com.values())

    def __len__(self):
        return len(self.samples)

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
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert("RGB")
        question = sample['question']
        answer = sample['answer']

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=question,
            history=None,
            images=[image],
            answer=answer
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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("python cogcom_dataset.py /path/to/cogcom_data")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    jsonl_path = os.path.join(root_dir, "com.jsonl")
    
    print(f"Test reading: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            record = json.loads(line)
            print(f"\n=== image {i+1}: {record['image_path']} ===")
            
            for j, item in enumerate(record.get('metadata', [])):
                print(f"\n  Q&A pair {j+1}:")
                print(f"    question: {item.get('question')}")
                print(f"    answer: {item.get('answer')}")
                
                final_com = item.get('final_com', {})
                if final_com:
                    class TempDataset(CogCoM_Dataset):
                        def __init__(self):
                            self.use_com_reasoning = True
                            self.use_prompt_template = False
                    
                    temp = TempDataset.__new__(TempDataset)
                    temp.use_com_reasoning = True
                    reasoning = temp._format_com_reasoning(final_com)
                    print(f"    chain of reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"    chain of reasoning: {reasoning}")