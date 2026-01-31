<div align="center">

# Med-CogVLM: A Medical Vision-Language Model with Visual Dependency Reinforcement Learning

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/greedno/Med-CogVLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[English](./README.md) | [中文](./README_zh.md)

</div>

---

## 📖 Overview

**Med-CogVLM** is an advanced medical vision-language model built upon CogVLM2's deep fusion architecture. We systematically address two fundamental challenges in medical VLMs through our innovative **Visual Dependency Reward Framework (VDRF)**:

- 🔍 **Deep Fusion Architecture**: Leverages CogVLM2's visual expert mechanism for dense vision-language interactions across all Transformer layers
- 🎯 **Visual Dependency Reinforcement Learning**: Ensures the model's reasoning is genuinely grounded in medical images rather than language priors

---

## 🌟 Key Contributions

1. **Visual Dependency Reward Framework (VDRF)**: A complementary reward mechanism combining visual-text consistency (VEC) and counterfactual dependency (DEP) rewards to ensure reasoning grounded in medical images.

2. **Architectural Necessity Proof**: Comprehensive experiments showing that VDRF is effective only with deep fusion architectures (CogVLM2: ΔAcc +13.00%, ΔVDS +0.082) while shallow fusion shows negligible improvement (Qwen2.5-VL: ΔAcc +0.06%, ΔVDS -0.001).

3. **SOTA Performance**: 85.98% accuracy on OmniMedVQA across 8 medical imaging modalities and 5 clinical tasks.

---

## �️ Architecture

Med-CogVLM employs a three-stage training paradigm:

### Stage 1: Progressive Supervised Fine-Tuning (SFT)

- **ROCO**: Establish medical vision-language alignment foundation (80K samples)
- **ROCOv2**: Enhance medical terminology understanding (60K samples)
- **SLAKE**: Learn structured QA patterns (14K samples)
- **OmniMedVQA**: Integrate multi-modal clinical tasks (89K samples)
- **CogCoM-TDIUC**: General reasoning data for enhancing reasoning strategy exploration during VDRF

### Stage 2: High Visual Dependency (HVD) Data Sampling

1. **GPT-4o-mini Scoring**: Automated visual dependency assessment
2. **Counterfactual Filtering**: Hard negative mining based on DEP scores
3. **Weighted Sampling**: Prioritize high-dependency samples during training

### Stage 3: GRPO + VDRF Reinforcement Learning

**Reward Components:**
- ✓ Base Reward: Accuracy + Format compliance
- ✓ Visual Consistency Reward (VEC): Global + Local similarity
- ✓ Counterfactual Dependency Reward (DEP): Real image grounding

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/cainiaomq/Med-CogVLM.git
cd Med-CogVLM

# Create virtual environment
conda create -n medcogvlm python=3.10
conda activate medcogvlm

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements

| Stage | GPU | Memory | DeepSpeed |
|-------|-----|--------|-----------|
| SFT | A100 80GB | ~75GB | ZeRO-2 |
| GRPO | A100 80GB | ~75GB | ZeRO-2 |

> ⚠️ **Note**: ZeRO-3 is not currently supported

---

## 📁 Project Structure

```
Med-CogVLM/
├── dataset/              # HVD Data processing scripts
│   ├── annotate_visdep_omnimedvqa.py
│   └── dep_checking.py
├── rl/                   # Reinforcement learning modules
│   ├── rewards.py        # VDRF implementation
│   ├── embedder.py       # BiomedCLIP integration
│   └── utils.py
├── utils/                # Dataset loaders
│   ├── omnimedqkv.py
│   ├── roco.py
│   └── slake.py
├── lora_finetune.py      # SFT training script
├── lora_grpo.py          # GRPO+VDRF training
├── eval.py               # Evaluation pipeline
├── web_demo.py           # Interactive demo
└── requirements.txt      # Dependencies
```

---

## 💻 Usage

### Web-based Model Inference

Run this code to start chatting in WebUI.

```shell
chainlit run web_demo.py
```

---

## 📊 Training

### 1. Data Preparation

Download required datasets:

| Dataset | Size | Purpose | Link |
|---------|------|---------|------|
| ROCO | 80K | Radiology image-text pairs | [GitHub](https://github.com/razorx89/roco-dataset) |
| ROCOv2 | 60K | High-quality radiology data | [HuggingFace](https://huggingface.co/datasets/eltorio/ROCOv2-radiology) |
| SLAKE | 14K | Structured medical VQA | [HuggingFace](https://huggingface.co/datasets/BoKelvin/SLAKE) |
| OmniMedVQA | 89K | Multi-modal benchmark | [HuggingFace](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) |
| CogCoM-TDIUC | 37K | reasoning | [HuggingFace](https://huggingface.co/qijimrc/CogCoM) |

**Visual Encoder:**
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224): For computing VEC and DEP rewards

### 2. Progressive SFT Fine-tuning

```bash
# Stage 1: ROCO
deepspeed lora_finetune.py \
    --model_path ./your_model_path \
    --dataset_path ./your_dataset_path \
    --save_path ./your_save_path \
    --resume_from ./your_save_path/checkpoint_epoch_{batch}_step_{step}
```

### 3. GRPO with VDRF Training

```bash
deepspeed grpo_vdrf.py \
    --model_path ./your_model_path \
    --dataset_path ./your_dataset_path \
    --save_path ./your_save_path \
    --actor_lora_path ./your_save_path/checkpoint_step_{step}
```

---

## 📈 Evaluation

### Run Evaluation

```bash
python eval.py \
    --model_path ./your_model_path \
    --dataset_path ./your_dataset_path \
    --out_dir ./your_out_dir \
    --resume ./your_out_dir/vds_predictions.jsonl
```

### Performance by Modality (OmniMedVQA)

| Metric | CT | MRI | X-ray | Ultrasound | Dermoscopy | Fundus | OCT | Microscopy | **Overall** |
|--------|----|----|-------|------------|------------|--------|-----|------------|-------------|
| **Accuracy (%)** | 82.3 | 86.9 | 88.9 | 98.7 | 77.9 | 85.3 | 86.0 | 73.7 | **85.98** |

### Performance by Clinical Task (OmniMedVQA)

| Metric | Anatomy Recognition | Disease Diagnosis | Lesion Grading | Modality Identification | Attribute Analysis |
|--------|-------------------|------------------|----------------|------------------------|-------------------|
| **Accuracy (%)** | 86.5 | 83.4 | 79.1 | 98.1 | 81.9 |

### Architecture Impact on VDRF Effectiveness

| Architecture | Model | Base Acc | +VDRF Acc | ΔVDS |
|-------------|-------|----------|-----------|------|
| Shallow | Qwen2.5-VL-SFT | 45.36% | 45.42% | -0.001 |
| **Deep** | **CogVLM2-SFT** | **74.66%** | **87.66%** | **+0.082** |

---

## 🤝 Contributing

We welcome contributions of all kinds! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Propose new features
- 📝 Improve documentation
- 🔧 Submit pull requests

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 📧 Contact

- **Project Homepage**: https://github.com/cainiaomq/Med-CogVLM
- **Model Hub**: https://huggingface.co/greedno/Med-CogVLM
- **Issue Tracker**: [GitHub Issues](https://github.com/cainiaomq/Med-CogVLM/issues)

---

## 🙏 Acknowledgments

Special thanks to:
- CogVLM team for the foundational architecture
- OmniMedVQA dataset contributors
- Medical imaging communities for data support

---

## ⚠️ Disclaimer

**Med-CogVLM is intended for research purposes only.** This model should not be used as the sole basis for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.
