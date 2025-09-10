# Fine-tune Med-CogVLM Model

[中文版 README](./README_zh.md)

Run this demo to fine-tune **Med-CogVLM** with LoRA — a medical multimodal expert model built upon CogVLM2.

## Project Overview

+ **Med-CogVLM** is a medical multimodal large model based on CogVLM2 (llama3-based VL), focusing on medical image QA tasks.
+ We adopt a **progressive LoRA fine-tuning** strategy:  
  1. Shallow fine-tuning on **ROCO**;  
  2. Deep fine-tuning on **ROCOv2**;  
  3. Instruction fine-tuning on **SLAKE**;  
  4. Final reinforcement learning (**GRPO**) training on **OmniMedVQA**.  
+ The goal is to enhance the model’s **visual dependency** in medical tasks, reducing reliance on text-only priors.

## Minimum Configuration

- Verified on **A100 80GB** GPUs.  
- SFT stage: batch≈4, memory usage ≈75GB.  
- RL (GRPO) stage: **ZeRO-2**, batch=1, K=4 candidates, fits single GPU.  
- `zero3` is not supported.

## Start Fine-tuning

1. Download datasets and install dependencies

Recommended datasets:  
- [ROCO](https://github.com/razorx89/roco-dataset/tree/master)  
- [ROCOv2](https://huggingface.co/datasets/eltorio/ROCOv2-radiology)  
- [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)
- [OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA)  

Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run fine-tuning

We provide both SFT and GRPO training scripts:  

```bash
# SFT
deepspeed ./peft_lora.py --ds_config ./ds_config.yaml

# GRPO (dual-LoRA, ZeRO-2)
deepspeed f./peft_grpo_duallora.py
```

During training, Loss values will be recorded in tensorboard for monitoring convergence:
During the training process of GRPO, logs will be accumulated and written to the logs for easy viewing of reward status.

```bash
tensorboard --logdir=output
```

**We recommend BF16 precision** to avoid NaN losses.