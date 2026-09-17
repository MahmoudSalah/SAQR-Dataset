# SAQR: A Synthetic-to-Real Arabic Handwriting Recognition, Retrieval, and Demographic Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://zenodo.org/)

Official custom code repository for the paper:  
**"SAQR: Synthetic-to-Real Arabic Handwriting Recognition, Retrieval, and Demographic Benchmark"**.

---

## 📌 Repository Description

This repository contains all **custom core code** generated for the SAQR benchmark:
1. **Dataset Curation & Splits**: Deterministic splitting and manifest creation for 2,263 aligned printed-handwritten text pairs across Train (1,575), Validation (335), and Test (353) sets.
2. **GT-to-Handwriting Matching/Retrieval**: Training routines for fine-tuned Siamese networks with hard negative mining.
3. **Demographic (Gender) Classification**: Fine-tuning pipeline with class-weighted loss for gender classification from line-level handwriting crops.

---

## 📁 Repository Structure

```text
SAQR-Dataset/
├── README.md                   # Repository documentation and reproduction guide
├── requirements.txt            # Environment dependencies
├── LICENSE                     # MIT License
│
├── curation/                   # Dataset preparation scripts
│   ├── curate_dataset.py       # Manifest assembly and metadata validation
│   └── generate_splits.py      # Split generation (train: 1575, val: 335, test: 353)
│
└── training/                   # Model fine-tuning routines
    ├── train_gender_classifier.py # Gender classifier training (ViT-Base + class weighting)
    └── train_siamese_hardneg.py   # Hard-negative Siamese network training for retrieval
```

---

## ⚙️ Environment Setup & Configuration

### 1. Prerequisites
- **Operating System**: Linux / Windows / macOS
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support recommended.

### 2. Installation

```bash
git clone git@github.com:MahmoudSalah/SAQR-Dataset.git
cd SAQR-Dataset

# Create conda environment
conda create -n saqr_env python=3.10 -y
conda activate saqr_env

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

1. Download `SAQR_Dataset.zip` from our Zenodo repository.
2. Extract the dataset into `./data/clean_crops_curated`:

```bash
mkdir -p data
unzip SAQR_Dataset.zip -d data/
```

The dataset path structure:
```text
data/clean_crops_curated/
├── manifest.csv
├── gt/
└── hw/
```

---

## 🚀 Execution & Quick Start

### 1. Train Retrieval Models
Train the hard-negative Siamese retrieval model:

```bash
python training/train_siamese_hardneg.py
```

### 2. Train Gender Classifier
Fine-tune ViT-Base with class weighting:

```bash
python training/train_gender_classifier.py
```

---

## 📜 Code Availability & Third-Party Notice

- **Custom Code Statement**: All scripts in this repository (`curation/`, `training/`) were custom-authored by the paper authors for dataset processing and model training.
- **Third-Party Software**: Pre-trained vision backbones (*ViT, DINOv2*) and deep learning frameworks (*PyTorch, HuggingFace Transformers*) are pre-existing third-party software tools cited in the Methods section.

---

## 📝 Citation

If you use the SAQR dataset or code in your research, please cite our paper:

```bibtex
@article{saqr2026arabic,
  title={SAQR: Synthetic-to-Real Arabic Handwriting Recognition, Retrieval, and Demographic Benchmark},
  author={MS Kasem et al.},
  journal={Scientific Data},
  year={2026}
}
```
