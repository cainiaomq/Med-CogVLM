from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = None
lora_model_path = None

model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.merge_and_unload()
lora_model_path_output = None

model.save_pretrained(lora_model_path_output, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(lora_model_path_output)
