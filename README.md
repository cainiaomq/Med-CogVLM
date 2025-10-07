<div align="center">

# Med-CogVLM: Vision-Dependent Reinforcement Learning for Medical Multimodal Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx)
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

## 🌟 Highlights

✅ **State-of-the-art Performance**: Achieves 84.85% overall accuracy on OmniMedVQA benchmark  
✅ **Multi-modal Support**: CT, MRI, X-ray, Ultrasound, Dermoscopy, Fundus, OCT, Microscopy  
✅ **Clinical Task Coverage**: Anatomy Recognition, Disease Diagnosis, Lesion Grading, Modality Identification, Attribute Analysis  
✅ **Enhanced Visual Grounding**: 30% improvement in Visual Dependency Score (VDS) through VDRF  
✅ **Production-Ready**: Complete training pipeline with DeepSpeed optimization  

---

## 🏗️ Architecture

Med-CogVLM employs a three-stage training paradigm:

### Stage 1: Progressive Supervised Fine-Tuning (SFT)

- **ROCO**: Establish medical vision-language alignment foundation (80K samples)
- **ROCOv2**: Enhance medical terminology understanding (60K samples)
- **SLAKE**: Learn structured QA patterns (14K samples)
- **OmniMedVQA**: Integrate multi-modal clinical tasks (89K samples)

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

### Performance by Modality

| Modality | Accuracy (%) |
|----------|--------------|
| CT | 83.29 |
| MRI | 86.17 |
| X-ray | 87.19 |
| Ultrasound | 92.61 |
| Dermoscopy | 76.77 |
| Fundus | 83.74 |
| OCT | 85.87 |
| Microscopy | 73.96 |
| **Overall** | **84.85** |

### Performance by Clinical Task

| Task Type | Accuracy (%) |
|-----------|--------------|
| Anatomy Recognition | 88.32 |
| Disease Diagnosis | 82.47 |
| Lesion Grading | 79.15 |
| Modality Identification | 93.68 |
| Attribute Analysis | 85.91 |

---

## 📝 Citation

If you find Med-CogVLM helpful in your research, please cite:

```bibtex
@article{medcogvlm2025,
  title={Med-CogVLM: Vision-Dependent Reinforcement Learning for Medical Multimodal Understanding},
  author={},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

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