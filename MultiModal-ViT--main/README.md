# Multimodal Skin Cancer Diagnosis — ViT + BERT with Cross-Attention

Multimodal classification of dermoscopic skin lesions on **HAM10000**, fusing image features
from a **Vision Transformer** with symptom-text features from **BERT** via bidirectional
**cross-attention**.

Three models are trained under an identical protocol so the comparison is meaningful:

| Model | Input | Encoder |
|---|---|---|
| Image-only | dermoscopic image | ViT-B/16 (`google/vit-base-patch16-224-in21k`) |
| Text-only | symptom description | BERT-base-uncased |
| **Multimodal** | image + text | ViT + BERT, bidirectional cross-attention fusion |

Task: 7-class lesion classification (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

---

## Status

> **The notebook has been successfully executed and thoroughly tested on GPU.** All code
> cells have been run without errors, and the complete pipeline has been verified from data
> loading through model training, validation, testing, and result visualization.
>
> The notebook generates all expected outputs, including dataset statistics, class
> distributions, training and validation curves, balanced accuracy, precision, recall,
> F1-score, melanoma sensitivity, confusion matrices, top-k prediction analysis,
> cross-attention visualizations, and exported prediction files.
>
> All three models (Image-only, Text-only, and Multimodal ViT+BERT) were trained using the
> same preprocessing pipeline, optimization strategy, learning schedule, and evaluation
> protocol to ensure a fair and reproducible comparison.
>
> The implementation has been validated for reproducibility, and the repository now
> represents a complete end-to-end multimodal skin lesion classification framework.

---

## Why this notebook exists

An earlier version of this project reported ~97% accuracy on HAM10000. That number has two
plausible failure modes that the original evaluation could not rule out. This notebook is
designed to investigate both issues explicitly before interpreting or comparing model
performance.

### 1. Text-label leakage

HAM10000 does not provide symptom descriptions. If symptom text is generated directly from the
diagnosis label (for example, `dx == "mel"` → *"irregular borders, multiple colors"*), the
text effectively becomes the diagnosis label in another form. A text encoder can therefore
learn to recover the label directly, artificially inflating multimodal performance.

**Section 3** performs a dedicated leakage audit by training a TF-IDF + Logistic Regression
probe on the symptom text alone.

| Text-only probe accuracy | Interpretation |
|---|---|
| ≈ majority baseline (~67%) | **Clean** — little or no label leakage |
| 70–80% | **Potential leakage** — requires further investigation |
| >85% | **Strong leakage** — multimodal results should not be interpreted as diagnostic performance |

### 2. Lesion-level leakage in train/test splitting

HAM10000 contains multiple images belonging to the same lesion (`lesion_id`). Random image
splitting may place different images of the same lesion into both training and testing sets,
allowing the model to memorize lesions rather than learn pathology.

To avoid this, **Section 4** performs lesion-level grouped splitting using `lesion_id` and
verifies that no lesion appears in more than one dataset split.

---

## Evaluation

HAM10000 is highly imbalanced, with the **nevus (nv)** class accounting for approximately
67% of all images. Because of this imbalance, overall accuracy alone is insufficient for
evaluating model performance.

The notebook therefore reports:

- Balanced Accuracy
- Per-class Precision, Recall, and F1-score
- Weighted and Macro Precision, Recall, and F1-score
- Macro One-vs-Rest ROC-AUC
- Melanoma Sensitivity (Recall)
- Confusion Matrices for all three models
- Top-1 and Top-3 Accuracy
- Cross-Attention Heatmaps (visual interpretation only)

Attention maps are presented as qualitative visualizations rather than explanations of model
decisions.

---

## Repository Contents

```
skin_cancer_multimodal_vit_bert.ipynb
README.md
```

Notebook organization:

| Section | Description |
|---------|-------------|
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

---

## Setup

### Dataset

Download the HAM10000 dataset from either:

- Harvard Dataverse
- Kaggle

Expected directory structure:

```
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

### Running

For quick verification:

```python
cfg.SUBSET_N = 800
```

For full training:

```python
cfg.SUBSET_N = None
```

Training time depends on GPU hardware and typically ranges from approximately 1–2 GPU hours
for the complete experiment.

---

## Project Highlights

- Vision Transformer (ViT-B/16) image encoder
- BERT-base text encoder
- Bidirectional Cross-Attention fusion
- Class-balanced sampling
- Class-weighted CrossEntropy loss
- AdamW optimizer
- Cosine Annealing learning-rate scheduler
- Early stopping based on Balanced Accuracy
- Lesion-level grouped train/test splitting
- Leakage detection pipeline
- Comprehensive evaluation metrics
- Top-k prediction analysis
- Attention visualization
- Exportable predictions and experiment logs
- Fully reproducible training pipeline

---

## Citation

Tschandl P., Rosendahl C., Kittler H.

**The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.**

*Scientific Data*, 5, 180161 (2018).
