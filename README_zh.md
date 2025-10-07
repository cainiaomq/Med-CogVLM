# Med-CogVLM: 基于视觉依赖强化学习的医学多模态模型

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/greedno/Med-CogVLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[English](./README.md) | [中文](./README_zh.md)

</div>

---

## 📖 项目概述

**Med-CogVLM** 是基于CogVLM2深度融合架构构建的先进医学视觉语言模型。我们通过创新的**视觉依赖奖励框架（VDRF）**系统性地解决医学VLM领域的两大核心挑战：

- 🔍 **深度融合架构**：利用CogVLM2的视觉专家机制，在所有Transformer层实现视觉与语言的密集交互
- 🎯 **视觉依赖强化学习**：确保模型推理真正基于医学图像，而非语言先验知识

---

## 🌟 项目亮点

✅ **最先进性能**：在OmniMedVQA基准测试中达到84.85%的整体准确率  
✅ **多模态支持**：CT、MRI、X光、超声、皮肤镜、眼底、OCT、显微镜  
✅ **临床任务覆盖**：解剖识别、疾病诊断、病灶分级、模态识别、属性分析  
✅ **增强视觉基础**：通过VDRF使视觉依赖分数（VDS）提升30%  
✅ **生产就绪**：完整的训练流程，配备DeepSpeed优化  

---

## 🏗️ 模型架构

Med-CogVLM采用三阶段训练范式：

### 阶段1：渐进式监督微调（SFT）

- **ROCO**：建立医学视觉-语言对齐基础（80K样本）
- **ROCOv2**：增强医学术语理解能力（60K样本）
- **SLAKE**：学习结构化问答模式（14K样本）
- **OmniMedVQA**：整合多模态临床任务（89K样本）

### 阶段2：高视觉依赖（HVD）数据采样

1. **GPT-4o-mini评分**：自动化视觉依赖度评估
2. **反事实过滤**：基于DEP分数的硬负样本挖掘
3. **加权采样**：训练时优先选择高依赖样本

### 阶段3：GRPO + VDRF强化学习

**奖励组成：**
- ✓ 基础奖励：准确率 + 格式合规性
- ✓ 视觉一致性奖励（VEC）：全局 + 局部相似度
- ✓ 反事实依赖奖励（DEP）：真实图像依赖

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/cainiaomq/Med-CogVLM.git
cd Med-CogVLM

# 安装依赖
pip install -r requirements.txt
```

### 硬件需求

| 阶段 | GPU | 显存 | DeepSpeed |
|------|-----|------|-----------|
| SFT | A100 80GB | ~75GB | ZeRO-2 |
| GRPO | A100 80GB | ~75GB | ZeRO-2 |

> ⚠️ **注意**：目前不支持ZeRO-3

---

## 💻 使用方法

### Web端在线调用模型

运行本代码以开始在 WebUI 中对话。

```shell
chainlit run web_demo.py
```

---

## 📊 模型训练

### 1. 数据准备

下载所需数据集：

| 数据集 | 规模 | 用途 | 链接 |
|--------|------|------|------|
| ROCO | 80K | 放射学图文对 | [GitHub](https://github.com/razorx89/roco-dataset) |
| ROCOv2 | 60K | 高质量放射学数据 | [HuggingFace](https://huggingface.co/datasets/eltorio/ROCOv2-radiology) |
| SLAKE | 14K | 结构化医学VQA | [HuggingFace](https://huggingface.co/datasets/BoKelvin/SLAKE) |
| OmniMedVQA | 89K | 多模态基准测试 | [HuggingFace](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) |

**视觉编码器：**
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)：用于计算VEC和DEP奖励

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

| 模态 | 准确率 (%) |
|------|-----------|
| CT | 83.29 |
| MRI | 86.17 |
| X光 | 87.19 |
| 超声 | 92.61 |
| 皮肤镜 | 76.77 |
| 眼底 | 83.74 |
| OCT | 85.87 |
| 显微镜 | 73.96 |
| **整体** | **84.85** |

### 按临床任务分类的性能

| 任务类型 | 准确率 (%) |
|----------|-----------|
| 解剖识别 | 88.32 |
| 疾病诊断 | 82.47 |
| 病灶分级 | 79.15 |
| 模态识别 | 93.68 |
| 属性分析 | 85.91 |

---

## 📝 引用

如果Med-CogVLM对您的研究有帮助，请引用：

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

我们欢迎各种形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

### 贡献方式
- 🐛 报告Bug和问题
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交Pull Request

---

## 📄 开源许可

本项目采用 [Apache License 2.0](LICENSE) 许可证。

---

## 📧 联系方式

- **项目主页**：https://github.com/cainiaomq/Med-CogVLM
- **模型仓库**：https://huggingface.co/greedno/Med-CogVLM
- **问题反馈**：[GitHub Issues](https://github.com/cainiaomq/Med-CogVLM/issues)

---

## 🙏 致谢

特别感谢：
- CogVLM团队提供的基础架构
- OmniMedVQA数据集的贡献者
- 医学影像社区的数据支持

---

## ⚠️ 免责声明

**Med-CogVLM仅供研究使用。** 本模型不应作为临床诊断的唯一依据。任何医疗决策都应在合格医疗专业人员的指导下做出。