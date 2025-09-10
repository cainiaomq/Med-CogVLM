# 微调 Med-CogVLM 模型

[Read this in English.](./README.md)

运行本 demo 来使用 LoRA 微调 **Med-CogVLM** —— 基于 CogVLM2 的医学多模态专家模型。

## 项目简介

+ **Med-CogVLM** 是在 CogVLM2（llama3-based VL）基础上构建的医学多模态大模型，专注于医学影像问答场景。
+ 我们采用 **渐进式 LoRA 微调** 策略：  
  1. 先在 **ROCO** 上做浅层微调；  
  2. 再在 **ROCOv2** 上做深层微调；  
  3. 使用 **SLAKE** 进行指令微调；  
  4. 最终在 **OmniMedVQA*** 数据上进行 **强化学习（GRPO）训练**。
+ 项目的目标是提升模型在医学任务中的**视觉依赖性**，避免仅凭语言先验作答。

## 最低配置

- 训练在 **A100 80GB** GPU 上验证通过。  
- SFT 阶段（batch≈4）显存占用约 75GB。  
- 强化学习阶段（GRPO）采用 **ZeRO-2**，batch=1、K=4 候选，单卡可控。  
- 暂不支持 `zero3`。

## 开始微调

1. 下载数据集和安装依赖

推荐的数据集包括：  
- [ROCO](https://github.com/razorx89/roco-dataset/tree/master)  
- [ROCOv2](https://huggingface.co/datasets/eltorio/ROCOv2-radiology)  
- [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)
- [OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA)  

安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行微调程序

我们提供了 SFT 与 GRPO 两类脚本：  

```bash
# SFT
deepspeed ./peft_lora.py --ds_config ./ds_config.yaml

# GRPO (dual-LoRA, ZeRO-2)
deepspeed f./peft_grpo_duallora.py
```

SFT在训练过程中，Loss 会自动写入 tensorboard，便于监控收敛情况：
GRPO在训练过程中，log会累积写入日志，便于查看奖励情况。

```bash
tensorboard --logdir=output
```

**推荐使用 BF16 精度**，避免出现 Loss 为 NaN 的情况。