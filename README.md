<div align="center">

# Med-CogVLM: A Medical Vision-Language Model with Visual Dependency Reinforcement Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## Overview

**Med-CogVLM** is a 19B-parameter medical vision-language model built upon CogVLM2's deep fusion architecture. We address two fundamental challenges in medical VLMs: insufficient visual-language interaction from shallow fusion designs, and over-reliance on language priors rather than image evidence. Our key contribution is the **Visual Dependency Reward Framework (VDRF)**, which uses visual-text consistency (VEC) and counterfactual dependency (DEP) rewards under GRPO to ensure model reasoning is genuinely grounded in medical images.

---

## Key Contributions

1. **Visual Dependency Reward Framework (VDRF)**: A complementary reward mechanism combining VEC and DEP rewards to optimize cross-modality information flow, transforming visual experts from passive feature extractors to active reasoning guides.

2. **Architecture as Enabling Factor**: Convergent evidence from two independent shallow fusion models shows VDRF is architecture-dependent — shallow fusion gains negligibly (Qwen2.5-VL: ΔAcc +0.46%; Lingshu-7B: ΔAcc +0.16%), while deep fusion CogVLM2 achieves ΔAcc **+13.94 pp** on the high visual dependency evaluation subset.

3. **SOTA on OmniMedVQA**: **84.8%** accuracy across 8 medical imaging modalities and 5 clinical tasks, surpassing Lingshu-7B by 3.9 pp despite using far fewer training samples.

---

## Architecture

Med-CogVLM (19B) comprises:
- **Visual Encoder**: EVA-CLIP-E (3B)
- **Language Backbone**: Meta-Llama-3-8B-Instruct
- **Visual Experts**: CogVLM2-style 8B expert modules at every Transformer layer
- **Adapters**: Lightweight Conv + SwiGLU adapters

Training follows a three-stage paradigm:

### Stage 1: Progressive Supervised Fine-Tuning (SFT)

| Dataset | Samples | Purpose |
|---------|---------|---------|
| ROCO | 16K | Radiology image-text alignment |
| ROCOv2 | 30K | Medical terminology |
| SLAKE | 14K | Structured medical VQA |
| OmniMedVQA-train | 60K | Multi-modal clinical tasks |
| CogCoM-TDIUC | 37K | Chain-of-thought reasoning |

### Stage 2: High Visual Dependency (HVD) Data Sampling

1. **GPT-4o-mini Scoring**: Automated visual dependency assessment
2. **Counterfactual Filtering**: Hard negative mining based on DEP scores
3. **Weighted Sampling**: Prioritize high-dependency samples during GRPO

### Stage 3: GRPO + VDRF Reinforcement Learning

**Reward Components (weights):**
- Base Reward (0.5): Accuracy + Format compliance
- Visual Consistency Reward VEC (0.4): Global + Local image-text similarity via BiomedCLIP
- Counterfactual Dependency Reward DEP (0.2): Anti-gaming constraint ensuring real image grounding

---

## Quick Start

### Installation

```bash
git clone <anonymous>
cd Med-CogVLM

conda create -n medcogvlm python=3.10
conda activate medcogvlm

pip install -r requirements.txt
```

### Hardware Requirements

| Stage | GPU | Memory | DeepSpeed |
|-------|-----|--------|-----------|
| SFT | A800 80GB | ~75GB | ZeRO-2 |
| GRPO | A800 80GB | ~75GB | ZeRO-2 |

> **Note**: ZeRO-3 is not currently supported. GRPO training: 2000 steps, group size K=4, AdamW lr=1e-6.

---

## Project Structure

```
Med-CogVLM/
├── dataset/              # HVD data processing scripts
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

## Training

### 1. Data Preparation

Download required datasets:

| Dataset | Link |
|---------|------|
| ROCO | [GitHub](https://github.com/razorx89/roco-dataset) |
| ROCOv2 | [PhysioNet](https://physionet.org/content/roco/1.0/) |
| SLAKE | [Official](https://www.med-vqa.com/slake/) |
| OmniMedVQA | [GitHub](https://github.com/OpenGVLab/OmniMedVQA) |
| CogCoM-TDIUC | [GitHub](https://github.com/THUDM/CogCoM) |

**Visual Encoder for VDRF rewards**: BiomedCLIP

### 2. Progressive SFT Fine-tuning

```bash
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

## Evaluation

### Run Evaluation

```bash
python eval.py \
    --model_path ./your_model_path \
    --dataset_path ./your_dataset_path \
    --out_dir ./your_out_dir \
    --resume ./your_out_dir/vds_predictions.jsonl
```

> OmniMedVQA results are reported on a clean test set (n=23,776, 89.05% of original) with image-level leakage removed.

### Cross-Modality Performance on OmniMedVQA (Accuracy %)

| Model | CT | MRI | X-ray | Ultra | Derm | Fundus | OCT | Micro | **Avg** |
|-------|----|----|-------|-------|------|--------|-----|-------|---------|
| CogVLM2-19B | 46.6 | 48.6 | 64.7 | 40.7 | 54.8 | 51.5 | 57.9 | 62.2 | 50.1 |
| Qwen2.5-VL-7B | 57.9 | 67.0 | 73.8 | 39.7 | 67.8 | 69.9 | 63.3 | 68.6 | 62.4 |
| Qwen2.5-VL-32B | 70.5 | 73.9 | 78.0 | 40.4 | 68.7 | 80.9 | 74.0 | 66.3 | 68.8 |
| LLaVA-Med-7B | 18.9 | 20.2 | 24.1 | 25.9 | 30.7 | 27.2 | 18.9 | 26.1 | 24.0 |
| MedVLM-R1-2B | 62.4 | 85.9 | 65.7 | 51.0 | 57.9 | 54.9 | 56.1 | 58.7 | 69.0 |
| MedGemma-4B-IT | 76.8 | 61.8 | 77.5 | 54.6 | 73.5 | 80.8 | 73.1 | 66.6 | 67.4 |
| InternVL2.5-14B | 71.3 | 76.1 | 84.6 | 81.6 | 79.6 | 85.8 | 79.1 | 82.9 | 78.0 |
| Lingshu-7B | 73.7 | 80.6 | 82.8 | 81.2 | **83.9** | **87.7** | **87.9** | **84.1** | 80.9 |
| **Med-CogVLM-19B** | **82.8** | **86.4** | **88.9** | **95.2** | 75.7 | 83.7 | 86.3 | 68.0 | **84.8** |

### Cross-Task and Multi-Benchmark Performance (Accuracy %)

| Model | Anat. | Dis. | Grad. | Mod. | Attr. | OmniMed Avg | PMC-VQA | MedXpertQA | **Overall** |
|-------|-------|------|-------|------|-------|-------------|---------|------------|-------------|
| CogVLM2-19B | 42.9 | 43.8 | 31.1 | 96.9 | 63.6 | 50.1 | 42.7 | 19.5 | 37.4 |
| Qwen2.5-VL-7B | 43.6 | 61.1 | 59.8 | 98.1 | 69.7 | 62.4 | 51.9 | 22.3 | 45.5 |
| Lingshu-7B | 79.8 | 77.7 | **87.6** | **99.4** | 78.4 | 80.9 | **56.3** | 26.7 | 54.6 |
| **Med-CogVLM-19B** | **86.6** | **82.0** | 84.1 | 97.5 | **85.4** | **84.8** | 53.7 | **29.5** | **56.0** |

### Architecture Impact on VDRF Effectiveness (HVD subset, n=4,339)

| Fusion | Model | Acc (%) | ΔAcc | BMCA-VDS | PLIP-VDS | UniMed-VDS |
|--------|-------|---------|------|----------|----------|------------|
| Shallow | Qwen2.5-VL-7B-SFT | 42.52 | -- | 0.2146 | 0.2526 | 0.2371 |
| Shallow | +VDRF | 42.98 | +0.46 | 0.2148 | 0.2525 | 0.2371 |
| Shallow | Lingshu-7B | 77.78 | -- | 0.2230 | 0.2536 | 0.2750 |
| Shallow | +VDRF | 77.94 | +0.16 | 0.2253 | 0.2549 | 0.2809 |
| **Deep** | CogVLM2-SFT | 75.71 | -- | 0.2013 | 0.2374 | 0.2160 |
| **Deep** | **+VDRF (Ours)** | **89.65** | **+13.94** | **0.2261** | **0.2637** | **0.3489** |

### Ablation: Reward Components (HVD subset, n=4,339)

| Configuration | Acc (%) | BMCA-VDS | PLIP-VDS | UniMed-VDS |
|---------------|---------|----------|----------|------------|
| CogVLM2-SFT | 75.71 | 0.2013 | 0.2374 | 0.2160 |
| Only R_acc | 86.89 | 0.2206 | 0.2596 | 0.3126 |
| w/o R_DEP | 86.93 | 0.2247 | 0.2630 | 0.3492 |
| w/o R_VEC | 88.02 | 0.2210 | 0.2451 | 0.2445 |
| **Full VDRF** | **89.65** | **0.2261** | **0.2637** | **0.3489** |

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

