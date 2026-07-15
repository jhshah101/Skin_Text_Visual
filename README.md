# Multimodal Skin Cancer Diagnosis — ViT + BERT with Cross-Attention

Multimodal classification of dermoscopic skin lesions on **HAM10000**, fusing image features from a Vision Transformer with symptom-text features from BERT via **bidirectional cross-attention**.

## Overview

This repository contains an end-to-end research notebook for comparing three models under the same training and evaluation protocol:

| Model | Input | Encoder |
|---|---|---|
| Image-only | Dermoscopic image | ViT-B/16 (`google/vit-base-patch16-224-in21k`) |
| Text-only | Symptom description | BERT-base-uncased |
| Multimodal | Image + text | ViT + BERT with bidirectional cross-attention fusion |

**Task:** 7-class lesion classification: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

## Status

The notebook has been successfully executed and tested on GPU. All code cells run without errors, and the complete pipeline has been verified from data loading through training, validation, testing, and result visualization.

The notebook generates the expected outputs, including:

- dataset statistics and class distributions
- training and validation curves
- balanced accuracy, precision, recall, and F1-score
- melanoma sensitivity
- confusion matrices
- top-k prediction analysis
- cross-attention visualizations
- exported prediction files

All three models were trained using the same preprocessing pipeline, optimization strategy, learning schedule, and evaluation protocol to ensure a fair and reproducible comparison.

## Why this notebook exists

An earlier version of this project reported about **97% accuracy** on HAM10000. That result could have been inflated by two possible failure modes. This notebook investigates both issues explicitly before interpreting or comparing model performance.

### 1. Text-label leakage

HAM10000 does not provide symptom descriptions. If symptom text is generated directly from the diagnosis label, the text can become a disguised version of the label. In that case, a text encoder may learn to recover the class directly and artificially inflate multimodal performance.

Section 3 performs a dedicated leakage audit by training a **TF-IDF + Logistic Regression** probe on symptom text alone.

| Text-only probe accuracy | Interpretation |
|---|---|
| About majority baseline (~67%) | Clean — little or no label leakage |
| 70–80% | Potential leakage — needs further review |
| >85% | Strong leakage — multimodal results should not be treated as diagnostic performance |

### 2. Lesion-level leakage in train/test splitting

HAM10000 contains multiple images for the same lesion (`lesion_id`). Random image splitting can place different images of the same lesion into both training and testing sets, allowing the model to memorize lesions instead of learning pathology.

To prevent this, Section 4 uses **lesion-level grouped splitting** with `lesion_id` and verifies that no lesion appears in more than one split.

## Evaluation

HAM10000 is highly imbalanced, with the nevus (`nv`) class accounting for approximately 67% of all images. Because of this, overall accuracy alone is not enough.

This notebook reports:

- Balanced Accuracy
- Per-class Precision, Recall, and F1-score
- Weighted and Macro Precision, Recall, and F1-score
- Macro One-vs-Rest ROC-AUC
- Melanoma Sensitivity (Recall)
- Confusion Matrices for all three models
- Top-1 and Top-3 Accuracy
- Cross-Attention Heatmaps

Attention maps are shown only as qualitative visualizations and should not be treated as proof of causality or clinical explanation.

## Repository contents

- `skin_cancer_multimodal_vit_bert.ipynb`
- `README.md`

## Notebook structure

| Section | Description |
|---|---|
| 1 | Environment setup, configuration, reproducibility |
| 2 | HAM10000 loading, preprocessing, class distribution |
| 3 | Symptom text generation and leakage audit |
| 4 | Lesion-grouped train/validation/test split |
| 5 | Dataset pipeline, augmentation, balanced sampling |
| 6 | Image-only, Text-only, and Multimodal architectures |
| 7 | Model training with AdamW, cosine scheduling, early stopping |
| 8 | Model evaluation and quantitative analysis |
| 9 | Top-k predictions and attention visualization |
| 10 | Export of predictions, metrics, and experiment artifacts |

## Setup

### Dataset

Download the HAM10000 dataset from either:

- Harvard Dataverse
- Kaggle

Expected directory structure:

```text
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

Configure the notebook:

```python
cfg.METADATA_CSV = "data/HAM10000_metadata.csv"
cfg.IMAGE_DIRS = (
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2",
)
cfg.SYMPTOM_CSV = None
```

If `SYMPTOM_CSV` is `None`, synthetic symptom descriptions are generated automatically.

### Dependencies

```bash
pip install transformers==4.44.2 torch torchvision scikit-learn pandas matplotlib seaborn tqdm pillow
```

## Running

For quick verification:

```python
cfg.SUBSET_N = 800
```

For full training:

```python
cfg.SUBSET_N = None
```

Training time depends on GPU hardware and typically ranges from about **1–2 GPU hours** for the full experiment.

## Project highlights

- Vision Transformer (ViT-B/16) image encoder
- BERT-base text encoder
- Bidirectional cross-attention fusion
- Class-balanced sampling
- Class-weighted CrossEntropy loss
- AdamW optimizer
- Cosine Annealing learning-rate scheduler
- Early stopping based on Balanced Accuracy
- Lesion-level grouped train/test splitting

## Notes

This project is designed for research and method validation. Any clinical use requires further prospective validation on independent datasets.
