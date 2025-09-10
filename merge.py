from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 原始模型路径
base_model_path = None

# LoRA adapter 的保存路径（你的保存路径里应该有 adapter_config.json）
lora_model_path = None

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

# 加载 LoRA adapter 到模型
model = PeftModel.from_pretrained(model, lora_model_path)

model = model.merge_and_unload()  # 合并 LoRA 权重

lora_model_path_output = None

model.save_pretrained(lora_model_path_output, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(lora_model_path_output)
