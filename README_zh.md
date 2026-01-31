<div align="center">

# Med-CogVLM: 一个基于视觉依赖强化学习的医学多模态模型

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/greedno/Med-CogVLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[English](./README.md) | [中文](./README_zh.md)

</div>

---

## 📖 项目概述

**Med-CogVLM** 是基于CogVLM2深度融合架构构建的先进医学视觉语言模型。我们通过创新的**视觉依赖奖励框架(VDRF)**系统性地解决医学VLM领域的两大核心挑战:

- 🔍 **深度融合架构**: 利用CogVLM2的视觉专家机制,在所有Transformer层实现视觉与语言的密集交互
- 🎯 **视觉依赖强化学习**: 确保模型推理真正基于医学图像,而非语言先验知识

---

## 🌟 核心贡献

1. **视觉依赖奖励框架(VDRF)**: 一种结合视觉-文本一致性(VEC)和反事实依赖性(DEP)奖励的互补奖励机制,确保推理基于医学图像。

2. **架构必要性证明**: 全面的实验表明VDRF仅在深度融合架构中有效(CogVLM2: ΔAcc +13.00%, ΔVDS +0.082),而浅层融合改进微乎其微(Qwen2.5-VL: ΔAcc +0.06%, ΔVDS -0.001)。

3. **最先进性能**: 在OmniMedVQA基准上,跨8种医学成像模态和5个临床任务达到85.98%的准确率。

---

## 🏗️ 模型架构

Med-CogVLM采用三阶段训练范式:

### 阶段1: 渐进式监督微调(SFT)

- **ROCO**: 建立医学视觉-语言对齐基础(80K样本)
- **ROCOv2**: 增强医学术语理解能力(60K样本)
- **SLAKE**: 学习结构化问答模式(14K样本)
- **OmniMedVQA**: 整合多模态临床任务(89K样本)

### 阶段2: 高视觉依赖(HVD)数据采样

1. **GPT-4o-mini评分**: 自动化视觉依赖度评估
2. **反事实过滤**: 基于DEP分数的硬负样本挖掘
3. **加权采样**: 训练时优先选择高依赖样本

### 阶段3: GRPO + VDRF强化学习

**奖励组成:**
- ✓ 基础奖励: 准确率 + 格式合规性
- ✓ 视觉一致性奖励(VEC): 全局 + 局部相似度
- ✓ 反事实依赖奖励(DEP): 真实图像依赖

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/cainiaomq/Med-CogVLM.git
cd Med-CogVLM

# 创建虚拟环境
conda create -n medcogvlm python=3.10
conda activate medcogvlm

# 安装依赖
pip install -r requirements.txt
```

### 硬件需求

| 阶段 | GPU | 显存 | DeepSpeed |
|------|-----|------|-----------|
| SFT | A100 80GB | ~75GB | ZeRO-2 |
| GRPO | A100 80GB | ~75GB | ZeRO-2 |

> ⚠️ **注意**: 目前不支持ZeRO-3

---

## 📁 项目结构

```
Med-CogVLM/
├── dataset/              # HVD数据处理脚本
│   ├── annotate_visdep_omnimedvqa.py
│   └── dep_checking.py
├── rl/                   # 强化学习模块
│   ├── rewards.py        # VDRF实现
│   ├── embedder.py       # BiomedCLIP集成
│   └── utils.py
├── utils/                # 数据集加载器
│   ├── omnimedqkv.py
│   ├── roco.py
│   └── slake.py
├── lora_finetune.py      # SFT训练脚本
├── lora_grpo.py          # GRPO+VDRF训练
├── eval.py               # 评估流程
├── web_demo.py           # 交互式演示
└── requirements.txt      # 依赖项
```

---

## 💻 使用方法

### Web端在线调用模型

运行本代码以开始在WebUI中对话。

```shell
chainlit run web_demo.py
```

---

## 📊 模型训练

### 1. 数据准备

下载所需数据集:

| 数据集 | 规模 | 用途 | 链接 |
|--------|------|------|------|
| ROCO | 80K | 放射学图文对 | [GitHub](https://github.com/razorx89/roco-dataset) |
| ROCOv2 | 60K | 高质量放射学数据 | [HuggingFace](https://huggingface.co/datasets/eltorio/ROCOv2-radiology) |
| SLAKE | 14K | 结构化医学VQA | [HuggingFace](https://huggingface.co/datasets/BoKelvin/SLAKE) |
| OmniMedVQA | 89K | 多模态基准测试 | [HuggingFace](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) |

**视觉编码器:**
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224): 用于计算VEC和DEP奖励

### 2. 渐进式SFT微调

```bash
# 阶段1: ROCO
deepspeed lora_finetune.py \
    --model_path ./your_model_path \
    --dataset_path ./your_dataset_path \
    --save_path ./your_save_path \
    --resume_from ./your_save_path/checkpoint_epoch_{batch}_step_{step}
```

### 3. GRPO与VDRF训练

```bash
deepspeed grpo_vdrf.py \
    --model_path ./your_model_path \
    --dataset_path ./your_dataset_path \
    --save_path ./your_save_path \
    --actor_lora_path ./your_save_path/checkpoint_step_{step} 
```

---

## 📈 模型评估

### 运行评估

```bash
python eval.py \
    --model_path ./your_model_path \
    --dataset_path ./your_dataset_path \
    --out_dir ./your_out_dir \
    --resume ./your_out_dir/vds_predictions.jsonl
```

### 按模态分类的性能

| 指标 | CT | MRI | X光 | 超声 | 皮肤镜 | 眼底 | OCT | 显微镜 | **整体** |
|------|----|----|-----|------|--------|------|-----|--------|----------|
| **准确率 (%)** | 83.29 | 86.17 | 87.19 | 92.61 | 76.77 | 83.74 | 85.87 | 73.96 | **84.85** |

### 按临床任务分类的性能

| 指标 | 解剖识别 | 疾病诊断 | 病灶分级 | 模态识别 | 属性分析 |
|------|---------|---------|---------|---------|---------|
| **准确率 (%)** | 85.46 | 82.33 | 78.32 | 98.00 | 81.18 |

### 架构对VDRF有效性的影响

| 架构 | 模型 | 基础准确率 | +VDRF准确率 | ΔVDS |
|------|------|-----------|------------|------|
| 浅层 | Qwen2.5-VL | 45.36% | 45.42% | -0.001 |
| 浅层 | Lingshu | 77.29% | 77.35% | +0.006 |
| **深层** | **CogVLM2** | **79.04%** | **82.81%** | **+0.084** |

---

## 📝 引用

如果Med-CogVLM对您的研究有帮助,请引用:

```bibtex
@article{medcogvlm2025,
  title={Med-CogVLM: Vision-Dependent Reinforcement Learning for Medical Multimodal Understanding},
  author={},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

---

## 🤝 参与贡献

我们欢迎各种形式的贡献!请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解贡献指南。

### 贡献方式
- 🐛 报告Bug和问题
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交Pull Request

---

## 📄 开源许可

本项目采用[Apache License 2.0](LICENSE)许可证。

---

## 📧 联系方式

- **项目主页**: https://github.com/cainiaomq/Med-CogVLM
- **模型仓库**: https://huggingface.co/greedno/Med-CogVLM
- **问题反馈**: [GitHub Issues](https://github.com/cainiaomq/Med-CogVLM/issues)

---

## 🙏 致谢

特别感谢:
- CogVLM团队提供的基础架构
- OmniMedVQA数据集的贡献者
- 医学影像社区的数据支持

---

## ⚠️ 免责声明

**Med-CogVLM仅供研究使用。**本模型不应作为临床诊断的唯一依据。任何医疗决策都应在合格医疗专业人员的指导下做出。
