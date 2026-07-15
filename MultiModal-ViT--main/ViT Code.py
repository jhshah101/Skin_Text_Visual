!pip install -q transformers==4.44.2 torch torchvision scikit-learn pandas matplotlib seaborn tqdm pillow

import os, re, json, math, random, warnings
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import ViTModel, ViTImageProcessor, BertModel, BertTokenizerFast

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

@dataclass
class CFG:
    # --- paths: point these at your HAM10000 copy ---
    METADATA_CSV: str = "data/HAM10000_metadata.csv"
    IMAGE_DIRS: tuple = ("data/HAM10000_images_part_1", "data/HAM10000_images_part_2")
    # Optional: a CSV with columns [image_id, symptom_text].
    # Leave as None to auto-generate text -- but then READ SECTION 3 CAREFULLY.
    SYMPTOM_CSV: str = None

    # --- models ---
    VIT_NAME: str = "google/vit-base-patch16-224-in21k"
    BERT_NAME: str = "bert-base-uncased"

    # --- data ---
    IMG_SIZE: int = 224
    MAX_TEXT_LEN: int = 64
    NUM_CLASSES: int = 7
    SUBSET_N: int = None      # e.g. 800 for a smoke test; None = full dataset

    # --- training ---
    BATCH_SIZE: int = 32
    EPOCHS: int = 12
    LR_HEAD: float = 1e-3
    LR_BACKBONE: float = 2e-5
    WEIGHT_DECAY: float = 0.01
    WARMUP_FRAC: float = 0.1
    LABEL_SMOOTH: float = 0.05
    PATIENCE: int = 3
    FUSION_DIM: int = 512
    N_HEADS: int = 8
    DROPOUT: float = 0.2
    USE_AMP: bool = True

    OUT_DIR: str = "outputs"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis-like lesion",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevus",
    "vasc":  "Vascular lesion",
}
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
MEL_IDX = CLS2IDX["mel"]
print(json.dumps({k: v for k, v in vars(cfg).items() if not k.startswith("_")}, indent=2, default=str))

# ## 2. Load HAM10000
# 
# Download from Harvard Dataverse (`DBW86T`) or Kaggle (`kmader/skin-cancer-mnist-ham10000`),
# then point `CFG.METADATA_CSV` and `CFG.IMAGE_DIRS` at it.

meta = pd.read_csv(cfg.METADATA_CSV)

def resolve_path(image_id):
    for d in cfg.IMAGE_DIRS:
        p = os.path.join(d, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None

meta["path"] = meta["image_id"].map(resolve_path)
missing = meta["path"].isna().sum()
print(f"Rows: {len(meta)} | images not found on disk: {missing}")
meta = meta.dropna(subset=["path"]).reset_index(drop=True)

meta["label"] = meta["dx"].map(CLS2IDX)
assert meta["label"].notna().all(), "Unexpected dx value in metadata"
meta["label"] = meta["label"].astype(int)

print(meta.head())

dist = meta["dx"].value_counts().reindex(CLASSES)
dist_df = pd.DataFrame({
    "count": dist,
    "share_%": (100 * dist / len(meta)).round(2),
    "diagnosis": [CLASS_FULL[c] for c in CLASSES],
})
print("\nClass distribution:")
print(dist_df)
print(f"\nMajority-class baseline (predict 'nv' always): {100*dist['nv']/len(meta):.2f}% accuracy")
print("Any reported accuracy must be read against that number.")

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=dist.index, y=dist.values, ax=ax, color="#4C72B0")
ax.axhline(dist["nv"], ls="--", c="crimson", lw=1,
           label=f"nv = {100*dist['nv']/len(meta):.1f}% of data")
ax.set_title("HAM10000 class imbalance"); ax.set_ylabel("images"); ax.legend()
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/class_distribution.png", dpi=150); plt.show()


SYMPTOM_BANK = {
    "akiec": [
        "rough scaly patch that feels like sandpaper",
        "dry crusty spot on sun exposed skin",
        "persistent red patch with flaky surface",
        "brown scaly lesion that will not heal",
        "ulcerated area with rough surface",
    ],
    "bcc": [
        "shiny pearly bump with visible small vessels",
        "pink growth with a central indentation",
        "sore that bleeds and scabs and returns",
        "waxy translucent nodule on the face",
        "flat pale scar like area that is slowly growing",
    ],
    "bkl": [
        "waxy stuck on looking growth",
        "brown warty plaque with a greasy surface",
        "rough tan patch present for many years",
        "well defined brown lesion that is slightly raised",
        "dry crusty spot that has not changed recently",
    ],
    "df": [
        "firm small nodule that dimples when pinched",
        "hard lesion with a central dimple",
        "brownish firm bump on the leg",
        "small hard nodule that is slightly itchy",
        "firm papule that has been stable for years",
    ],
    "mel": [
        "asymmetric mole with irregular borders and multiple colors",
        "dark lesion that has changed in size and shape",
        "mole with uneven pigmentation that sometimes bleeds",
        "growing pigmented spot with a ragged edge",
        "new dark patch with variable brown and black tones",
    ],
    "nv": [
        "dark brown mole with a smooth and even border",
        "small round evenly pigmented mole",
        "stable brown spot present since childhood",
        "symmetric tan lesion with a regular outline",
        "uniform brown mole that has not changed",
    ],
    "vasc": [
        "bright red raised spot that blanches with pressure",
        "small purple vascular papule",
        "red blood blister like lesion",
        "cluster of tiny red vessels under the skin",
        "reddish nodule that bleeds easily when scratched",
    ],
}

CONTEXT = [
    "patient reports mild itching",
    "no associated pain",
    "lesion noticed several months ago",
    "occasional bleeding after minor trauma",
    "no change reported by the patient",
    "history of significant sun exposure",
    "family history of skin cancer",
    "no systemic symptoms",
]

CROSS_TALK_P = 0.25   # chance of drawing the primary phrase from a DIFFERENT class
rng = random.Random(SEED)

def make_symptom(dx: str) -> str:
    if rng.random() < CROSS_TALK_P:
        other = rng.choice([c for c in CLASSES if c != dx])
        primary = rng.choice(SYMPTOM_BANK[other])
    else:
        primary = rng.choice(SYMPTOM_BANK[dx])
    ctx = rng.sample(CONTEXT, k=rng.choice([1, 2]))
    return f"{primary}, " + ", ".join(ctx)

if cfg.SYMPTOM_CSV and os.path.exists(cfg.SYMPTOM_CSV):
    sym = pd.read_csv(cfg.SYMPTOM_CSV)
    meta = meta.merge(sym[["image_id", "symptom_text"]], on="image_id", how="left")
    meta["symptom_text"] = meta["symptom_text"].fillna("no symptom description provided")
    TEXT_SOURCE = "real"
    print(f"Loaded real symptom text from {cfg.SYMPTOM_CSV}")

print(meta[["image_id", "dx", "symptom_text"]].sample(6, random_state=SEED))


Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(
    meta["symptom_text"].values, meta["label"].values,
    test_size=0.2, stratify=meta["label"].values, random_state=SEED,
)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
probe = LogisticRegression(max_iter=2000, class_weight="balanced")
probe.fit(vec.fit_transform(Xtr_t), ytr_t)
probe_pred = probe.predict(vec.transform(Xte_t))

probe_acc = accuracy_score(yte_t, probe_pred)
probe_bal = balanced_accuracy_score(yte_t, probe_pred)
majority  = (yte_t == CLS2IDX["nv"]).mean()

print("=" * 66)
print("LEAKAGE AUDIT -- text-only TF-IDF + logistic regression")
print("=" * 66)
print(f"  majority-class baseline : {majority:.4f}")
print(f"  text-only accuracy      : {probe_acc:.4f}")
print(f"  text-only balanced acc  : {probe_bal:.4f}")
print(f"  lift over baseline      : {probe_acc - majority:+.4f}")
print("-" * 66)


# ## 4. Splits — grouped by lesion, not by image

if cfg.SUBSET_N:
    meta = meta.groupby("dx", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(20, cfg.SUBSET_N // 7)), random_state=SEED)
    ).reset_index(drop=True)
    print(f"SMOKE TEST -- subset of {len(meta)} images. Set SUBSET_N=None for the real run.\n")

# group split on lesion_id so no lesion appears in two splits
lesions = meta[["lesion_id", "dx"]].drop_duplicates("lesion_id")
tr_les, tmp_les = train_test_split(
    lesions, test_size=0.30, stratify=lesions["dx"], random_state=SEED)
va_les, te_les = train_test_split(
    tmp_les, test_size=0.50, stratify=tmp_les["dx"], random_state=SEED)

split_map = {}
for lid in tr_les["lesion_id"]: split_map[lid] = "train"
for lid in va_les["lesion_id"]: split_map[lid] = "val"
for lid in te_les["lesion_id"]: split_map[lid] = "test"
meta["split"] = meta["lesion_id"].map(split_map)

train_df = meta[meta.split == "train"].reset_index(drop=True)
val_df   = meta[meta.split == "val"].reset_index(drop=True)
test_df  = meta[meta.split == "test"].reset_index(drop=True)

overlap = set(train_df.lesion_id) & set(test_df.lesion_id)
assert not overlap, f"LESION LEAK: {len(overlap)} lesions in both train and test"
print("No lesion overlap between train and test.\n")

print(f"train {len(train_df):>6}  |  val {len(val_df):>5}  |  test {len(test_df):>5}")
print(pd.crosstab(meta["split"], meta["dx"]).reindex(["train", "val", "test"]))

# ## 5. Dataset and dataloaders

import torchvision.transforms as T

IMNET_MEAN = [0.5, 0.5, 0.5]   # ViT in21k preprocessing
IMNET_STD  = [0.5, 0.5, 0.5]

train_tf = T.Compose([
    T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])
eval_tf = T.Compose([
    T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])

tokenizer = BertTokenizerFast.from_pretrained(cfg.BERT_NAME)

class SkinDataset(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        enc = tokenizer(
            r["symptom_text"], truncation=True, padding="max_length",
            max_length=cfg.MAX_TEXT_LEN, return_tensors="pt",
        )
        return {
            "pixel_values": self.tfm(img),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
        }

train_ds = SkinDataset(train_df, train_tf)
val_ds   = SkinDataset(val_df,   eval_tf)
test_ds  = SkinDataset(test_df,  eval_tf)

# Class-balanced sampling: without this the model just learns to say "nv".
counts = train_df["label"].value_counts().reindex(range(cfg.NUM_CLASSES)).fillna(0).values
w_per_class = 1.0 / np.maximum(counts, 1)
sample_w = w_per_class[train_df["label"].values]
sampler = WeightedRandomSampler(torch.DoubleTensor(sample_w), len(sample_w), replacement=True)

NW = 2 if DEVICE.type == "cuda" else 0
train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                      num_workers=NW, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=NW, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=NW, pin_memory=True)

print(f"batches -- train {len(train_dl)} | val {len(val_dl)} | test {len(test_dl)}")
b = next(iter(train_dl))
print({k: tuple(v.shape) for k, v in b.items()})

# ## 6. Models
# 
# Three architectures sharing the same encoders and the same classifier head shape, so the
# comparison is apples-to-apples:
# 
# - **ImageOnly** — ViT `[CLS]` → head
# - **TextOnly** — BERT `[CLS]` → head
# - **Multimodal** — bidirectional cross-attention (text queries image patches; image queries text tokens), concatenated → head
# 
# The fusion is bidirectional rather than text→image only, which lets visual evidence
# modulate the text representation as well. Attention weights are returned so Section 9
# can show which patches the text attended to.

class ImageOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vit = ViTModel.from_pretrained(cfg.VIT_NAME)
        d = self.vit.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d, cfg.FUSION_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.FUSION_DIM, cfg.NUM_CLASSES),
        )
    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        cls = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(cls), None


class TextOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.BERT_NAME)
        d = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d, cfg.FUSION_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.FUSION_DIM, cfg.NUM_CLASSES),
        )
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls), None


class CrossAttentionBlock(nn.Module):
    # Query stream attends to a key/value stream. Pre-norm, residual, FFN.
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.nf = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
    def forward(self, q, kv, key_padding_mask=None):
        qn, kn = self.nq(q), self.nk(kv)
        out, w = self.attn(qn, kn, kn, key_padding_mask=key_padding_mask,
                           need_weights=True, average_attn_weights=True)
        q = q + out
        q = q + self.ffn(self.nf(q))
        return q, w


class MultimodalViTBERT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vit  = ViTModel.from_pretrained(cfg.VIT_NAME)
        self.bert = BertModel.from_pretrained(cfg.BERT_NAME)
        dv = self.vit.config.hidden_size
        dt = self.bert.config.hidden_size
        D  = cfg.FUSION_DIM

        self.proj_v = nn.Linear(dv, D)
        self.proj_t = nn.Linear(dt, D)

        self.t2i = CrossAttentionBlock(D, cfg.N_HEADS, cfg.DROPOUT)  # text queries image
        self.i2t = CrossAttentionBlock(D, cfg.N_HEADS, cfg.DROPOUT)  # image queries text

        self.head = nn.Sequential(
            nn.LayerNorm(D * 2), nn.Dropout(cfg.DROPOUT),
            nn.Linear(D * 2, D), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(D, cfg.NUM_CLASSES),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        V = self.proj_v(self.vit(pixel_values=pixel_values).last_hidden_state)          # B,1+P,D
        T = self.proj_t(self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state)     # B,L,D

        pad = (attention_mask == 0)                       # True where padding

        T_att, w_t2i = self.t2i(T, V)                     # text attends over image patches
        V_att, _     = self.i2t(V, T, key_padding_mask=pad)

        t_vec = (T_att * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        v_vec = V_att[:, 0]                               # fused [CLS] of image stream

        return self.head(torch.cat([v_vec, t_vec], dim=-1)), w_t2i


def build(name):
    return {"image": ImageOnly, "text": TextOnly, "multimodal": MultimodalViTBERT}[name](cfg).to(DEVICE)

m = build("multimodal")
n_tot = sum(p.numel() for p in m.parameters())
print(f"Multimodal parameters: {n_tot/1e6:.1f}M")
del m; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

# ## 7. Training
# 
# - Discriminative LRs: pretrained backbones at `2e-5`, fresh heads at `1e-3`
# - Class-weighted loss **on top of** the balanced sampler, since imbalance is severe
# - Model selection on **validation balanced accuracy**, not accuracy — selecting on plain accuracy just picks whichever checkpoint is best at predicting `nv`
# - Early stopping, cosine schedule with warmup, mixed precision

def make_optim(model, n_steps):
    backbone, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith(("vit.", "bert.")) else head).append(p)
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": cfg.LR_BACKBONE},
        {"params": head,     "lr": cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)
    warm = int(cfg.WARMUP_FRAC * n_steps)
    def lr_lambda(s):
        if s < warm:
            return s / max(1, warm)
        prog = (s - warm) / max(1, n_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

cls_w = torch.tensor(
    len(train_df) / (cfg.NUM_CLASSES * np.maximum(counts, 1)),
    dtype=torch.float, device=DEVICE,
)
print("class weights:", dict(zip(CLASSES, cls_w.cpu().numpy().round(2))))


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    P, Y, S = [], [], []
    for b in dl:
        px = b["pixel_values"].to(DEVICE, non_blocking=True)
        ii = b["input_ids"].to(DEVICE, non_blocking=True)
        am = b["attention_mask"].to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=(cfg.USE_AMP and DEVICE.type == "cuda")):
            logits, _ = model(px, ii, am)
        prob = F.softmax(logits.float(), dim=-1)
        S.append(prob.cpu().numpy())
        P.append(prob.argmax(-1).cpu().numpy())
        Y.append(b["label"].numpy())
    return np.concatenate(Y), np.concatenate(P), np.concatenate(S)


def train_model(name):
    set_seed()
    model = build(name)
    steps = cfg.EPOCHS * len(train_dl)
    opt, sched = make_optim(model, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.USE_AMP and DEVICE.type == "cuda"))
    lossfn = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=cfg.LABEL_SMOOTH)

    hist, best, bad, best_state = [], -1.0, 0, None
    for ep in range(1, cfg.EPOCHS + 1):
        model.train(); tot = 0.0
        for b in tqdm(train_dl, desc=f"[{name}] epoch {ep}/{cfg.EPOCHS}", leave=False):
            px = b["pixel_values"].to(DEVICE, non_blocking=True)
            ii = b["input_ids"].to(DEVICE, non_blocking=True)
            am = b["attention_mask"].to(DEVICE, non_blocking=True)
            y  = b["label"].to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(cfg.USE_AMP and DEVICE.type == "cuda")):
                logits, _ = model(px, ii, am)
                loss = lossfn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()

        yv, pv, _ = evaluate(model, val_dl)
        acc = accuracy_score(yv, pv)
        bal = balanced_accuracy_score(yv, pv)
        mel = ((pv == MEL_IDX) & (yv == MEL_IDX)).sum() / max((yv == MEL_IDX).sum(), 1)
        hist.append({"epoch": ep, "train_loss": tot / len(train_dl),
                     "val_acc": acc, "val_bal_acc": bal, "val_mel_recall": mel})
        print(f"[{name}] ep{ep:>2} loss {tot/len(train_dl):.4f} | "
              f"val acc {acc:.4f} | bal acc {bal:.4f} | mel recall {mel:.4f}")

        if bal > best:                      # select on BALANCED accuracy
            best, bad = bal, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, f"{cfg.OUT_DIR}/best_{name}.pt")
        else:
            bad += 1
            if bad >= cfg.PATIENCE:
                print(f"[{name}] early stop at epoch {ep} (best val bal acc {best:.4f})")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

RESULTS, HISTORY, PREDS = {}, {}, {}

for name in ["image", "text", "multimodal"]:
    print("\n" + "=" * 66)
    print(f"TRAINING: {name}")
    print("=" * 66)
    model, hist = train_model(name)
    y, p, s = evaluate(model, test_dl)

    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    prw, rcw, f1w, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y, s, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    RESULTS[name] = {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "precision_macro": pr, "recall_macro": rc, "f1_macro": f1,
        "precision_weighted": prw, "recall_weighted": rcw, "f1_weighted": f1w,
        "auc_macro_ovr": auc,
        "melanoma_recall": ((p == MEL_IDX) & (y == MEL_IDX)).sum() / max((y == MEL_IDX).sum(), 1),
    }
    HISTORY[name] = hist
    PREDS[name] = {"y": y, "pred": p, "prob": s}

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[{name}] test: acc {RESULTS[name]['accuracy']:.4f} | "
          f"bal acc {RESULTS[name]['balanced_accuracy']:.4f} | "
          f"macro F1 {RESULTS[name]['f1_macro']:.4f} | "
          f"mel recall {RESULTS[name]['melanoma_recall']:.4f}")

# persist raw predictions so the analysis can be rerun without retraining
np.savez(f"{cfg.OUT_DIR}/test_predictions.npz",
         **{f"{k}_{f}": v[f] for k, v in PREDS.items() for f in ("y", "pred", "prob")})
print(f"\nSaved raw predictions -> {cfg.OUT_DIR}/test_predictions.npz")

# ## 8. Evaluation

summary = pd.DataFrame(RESULTS).T
summary.index = ["Image only (ViT)", "Text only (BERT)", "Multimodal (ViT+BERT)"]
summary_pct = (summary * 100).round(2)

print("TEST-SET RESULTS (%)\n")
print(summary_pct[[
    "accuracy", "balanced_accuracy", "precision_macro",
    "recall_macro", "f1_macro", "melanoma_recall", "auc_macro_ovr",
]])

mm, im = RESULTS["multimodal"], RESULTS["image"]
print("\nMultimodal vs image-only (the comparison that matters -- text alone is not a "
      "deployable diagnostic system):")
for k in ["accuracy", "balanced_accuracy", "f1_macro", "melanoma_recall"]:
    d = 100 * (mm[k] - im[k])
    print(f"  {k:<20} {100*im[k]:6.2f}  ->  {100*mm[k]:6.2f}   ({d:+.2f} pts)")

print(f"\nMajority-class baseline (always 'nv'): {100*(PREDS['image']['y']==CLS2IDX['nv']).mean():.2f}%")
print("Compare every accuracy above against that number, not against zero.")

# Per-class breakdown for the multimodal model
y, p = PREDS["multimodal"]["y"], PREDS["multimodal"]["pred"]

print("PER-CLASS REPORT -- multimodal\n")
print(classification_report(y, p, target_names=CLASSES, digits=4, zero_division=0))

rows = []
for i, c in enumerate(CLASSES):
    for name in ["image", "text", "multimodal"]:
        yy, pp = PREDS[name]["y"], PREDS[name]["pred"]
        n = (yy == i).sum()
        rec = ((pp == i) & (yy == i)).sum() / max(n, 1)
        rows.append({"class": c, "model": name, "recall": rec, "support": int(n)})
per_class = pd.DataFrame(rows).pivot_table(index="class", columns="model", values="recall")
per_class = per_class[["image", "text", "multimodal"]].reindex(CLASSES)
per_class["support"] = [int((y == i).sum()) for i in range(len(CLASSES))]
per_class["Δ (mm − img)"] = per_class["multimodal"] - per_class["image"]

print("\nPER-CLASS RECALL BY MODEL\n")
print((per_class[["image", "text", "multimodal", "Δ (mm − img)"]] * 100).round(2)
        .assign(support=per_class["support"]))
print("\nRare classes (df, vasc, akiec) have small support. Their recall is noisy --")
print("report a confidence interval or say so explicitly rather than quoting a point estimate.")

fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
for ax, name in zip(axes, ["image", "text", "multimodal"]):
    cm = confusion_matrix(PREDS[name]["y"], PREDS[name]["pred"],
                          labels=range(cfg.NUM_CLASSES))
    cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False)
    acc = RESULTS[name]["accuracy"]; bal = RESULTS[name]["balanced_accuracy"]
    ax.set_title(f"{name}\nacc {acc:.3f} | bal acc {bal:.3f}")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.suptitle("Confusion matrices (counts annotated, colour = row-normalised recall)", y=1.03)
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/confusion_matrices.png", dpi=150,
                                bbox_inches="tight"); plt.show()

print("Look at the mel row. Melanomas predicted as 'nv' are the clinically dangerous errors --")
print("a missed melanoma sent home as a benign mole. Count them and report them.")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
for ax, met, ttl in zip(
    axes,
    ["train_loss", "val_bal_acc", "val_mel_recall"],
    ["Training loss", "Val balanced accuracy", "Val melanoma recall"],
):
    for name in ["image", "text", "multimodal"]:
        h = HISTORY[name]
        ax.plot(h["epoch"], h[met], marker="o", ms=3, label=name)
    ax.set_title(ttl); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/training_curves.png", dpi=150); plt.show()

# ## 9. Ranked predictions and cross-attention

model = build("multimodal")
model.load_state_dict(torch.load(f"{cfg.OUT_DIR}/best_multimodal.pt", map_location=DEVICE))
model.eval()

def topk_table(k=5, n=8):
    rows = []
    prob, y = PREDS["multimodal"]["prob"], PREDS["multimodal"]["y"]
    idx = np.random.RandomState(SEED).choice(len(y), size=min(n, len(y)), replace=False)
    for i in idx:
        order = prob[i].argsort()[::-1][:k]
        r = {
            "symptom": test_df.iloc[i]["symptom_text"][:52] + "...",
            "true": CLASSES[y[i]],
        }
        for j, c in enumerate(order, 1):
            r[f"rank {j}"] = f"{CLASSES[c]} ({100*prob[i][c]:.1f}%)"
        r["correct"] = "yes" if order[0] == y[i] else "NO"
        rows.append(r)
    return pd.DataFrame(rows)

print("TOP-5 RANKED PREDICTIONS (random test cases -- not cherry-picked)\n")
print(topk_table())

print("\nSample honestly. If you show only the cases the model got right, the table is")
print("decoration, not evidence -- and that is exactly what the current draft does.")

# top-k accuracy: does the correct class at least appear in the top-k?
prob, y = PREDS["multimodal"]["prob"], PREDS["multimodal"]["y"]
print("TOP-K ACCURACY (multimodal)")
for k in [1, 2, 3, 5]:
    topk = np.argsort(prob, axis=1)[:, ::-1][:, :k]
    hit = np.mean([y[i] in topk[i] for i in range(len(y))])
    print(f"  top-{k}: {100*hit:.2f}%")
print("\nTop-5 over 7 classes is a weak claim -- random guessing already gets 5/7 = 71%.")
print("Report top-1 and top-3; top-5 here is close to meaningless.")

@torch.no_grad()
def show_attention(i):
    s = test_ds[i]
    px = s["pixel_values"].unsqueeze(0).to(DEVICE)
    ii = s["input_ids"].unsqueeze(0).to(DEVICE)
    am = s["attention_mask"].unsqueeze(0).to(DEVICE)
    logits, w = model(px, ii, am)          # w: B, L_text, 1+P
    prob = F.softmax(logits.float(), -1)[0].cpu().numpy()

    ntok = int(am.sum().item())
    patch = w[0, :ntok, 1:].mean(0)        # drop CLS, average over real tokens
    g = int(math.sqrt(patch.numel()))
    heat = patch.reshape(g, g).cpu().numpy()
    heat = (heat - heat.min()) / (heat.ptp() + 1e-8)

    img = Image.open(test_df.iloc[i]["path"]).convert("RGB").resize((224, 224))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].imshow(img); ax[0].set_title("dermoscopic image"); ax[0].axis("off")
    ax[1].imshow(img)
    ax[1].imshow(np.kron(heat, np.ones((224 // g, 224 // g))),
                 cmap="jet", alpha=0.45)
    ax[1].set_title("text→image cross-attention"); ax[1].axis("off")
    order = prob.argsort()[::-1][:5]
    ax[2].barh([CLASSES[c] for c in order][::-1], [prob[c] for c in order][::-1],
               color="#4C72B0")
    ax[2].set_xlim(0, 1); ax[2].set_title("top-5 confidence")
    plt.suptitle(f"true: {CLASSES[PREDS['multimodal']['y'][i]]}  |  "
                 f"pred: {CLASSES[order[0]]}  |  "
                 f"\"{test_df.iloc[i]['symptom_text'][:70]}...\"", y=1.04)
    plt.tight_layout(); plt.show()

for i in np.random.RandomState(SEED).choice(len(test_df), 3, replace=False):
    show_attention(int(i))

# ## 10. Export results

summary_pct.to_csv(f"{cfg.OUT_DIR}/results_summary.csv")
per_class.to_csv(f"{cfg.OUT_DIR}/per_class_recall.csv")
pd.concat({k: v for k, v in HISTORY.items()}).to_csv(f"{cfg.OUT_DIR}/training_history.csv")

audit = {
    "text_source": TEXT_SOURCE,
    "leakage_verdict": VERDICT,
    "text_only_probe_accuracy": float(probe_acc),
    "majority_baseline": float(majority),
    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    "lesion_grouped_split": True,
    "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in RESULTS.items()},
}
with open(f"{cfg.OUT_DIR}/run_manifest.json", "w") as f:
    json.dump(audit, f, indent=2)

print("Written to", cfg.OUT_DIR)
for f in sorted(os.listdir(cfg.OUT_DIR)):
    print("  ", f)

mm, im = RESULTS["multimodal"], RESULTS["image"]
print("=" * 70)
print(f"\nText provenance : {TEXT_SOURCE}")
print(f"Leakage verdict : {VERDICT}  (text-only probe = {100*probe_acc:.1f}%, "
      f"baseline = {100*majority:.1f}%)")
print(f"\nSplit           : lesion-grouped, {len(train_df)}/{len(val_df)}/{len(test_df)}")
print(f"\nImage-only      : acc {100*im['accuracy']:.2f} | bal {100*im['balanced_accuracy']:.2f} "
      f"| macro-F1 {100*im['f1_macro']:.2f} | mel recall {100*im['melanoma_recall']:.2f}")
print(f"Multimodal      : acc {100*mm['accuracy']:.2f} | bal {100*mm['balanced_accuracy']:.2f} "
      f"| macro-F1 {100*mm['f1_macro']:.2f} | mel recall {100*mm['melanoma_recall']:.2f}")
print(f"\nFusion gain     : {100*(mm['accuracy']-im['accuracy']):+.2f} acc, "
      f"{100*(mm['balanced_accuracy']-im['balanced_accuracy']):+.2f} balanced acc")
print("-" * 70)