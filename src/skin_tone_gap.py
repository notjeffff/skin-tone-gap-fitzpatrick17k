"""
Skin-tone generalisation gap in dermatology classifiers.
Train on light skin (FST I-III), evaluate on light AND dark (FST IV-VI).

USAGE
-----
Smoke test first (2 min, proves the pipeline runs):
    SMOKE=1 python skin_tone_gap.py

Full run, one seed:
    SEED=0 python skin_tone_gap.py

Mitigation run (oversamples dark skin in training):
    SEED=0 MITIGATE=1 python skin_tone_gap.py

In a Kaggle notebook cell:
    import os; os.environ["SMOKE"] = "1"
    %run skin_tone_gap.py

OUTPUTS (to OUT_DIR)
--------------------
    results_seed{N}.json      per-group metrics
    probs_seed{N}.csv         raw per-image probabilities  <-- keep these!
    model_seed{N}.pt          checkpoint for Grad-CAM later

The probs CSV is the important artefact: it lets you compute NEW metrics
tomorrow (ECE, bootstrapped CIs, subgroup splits) WITHOUT retraining.
"""

import os
import glob
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SEED      = int(os.environ.get("SEED", 0))
SMOKE     = os.environ.get("SMOKE", "0") == "1"
MITIGATE  = os.environ.get("MITIGATE", "0") == "1"
EVAL_ONLY = os.environ.get("EVAL_ONLY", "0") == "1"   # load checkpoint, skip training
RESUME    = os.environ.get("RESUME", "0") == "1"      # continue from checkpoint

# MIT_TARGET: which FST levels the mitigation oversamples.
#   "dark" = all of FST 4-6 (the naive version)
#   "6"    = FST 6 only -- use this if the cliff is at FST 6, not across 4-6
MIT_TARGET = os.environ.get("MIT_TARGET", "dark")

# HOLDOUT_FST: train on light + these FST levels, test on the rest.
#   e.g. HOLDOUT_FST="4,5" trains on light+FST4+FST5 and tests on FST6.
#   Isolates "never saw dark skin" from "FST 6 is intrinsically harder".
HOLDOUT_FST = os.environ.get("HOLDOUT_FST", "")

# SIZE_MATCH=1 with HOLDOUT_FST: drop light training images so the training set
#   stays the same size as the light-only baseline. Separates "saw darker skin"
#   from "simply had more training data".
SIZE_MATCH = os.environ.get("SIZE_MATCH", "0") == "1"
# CV_FOLD=k with FOLDS_CSV: all-skin-type training, k-fold cross-validation.
#   Train on every image whose fold != k (all FST), test on fold k (all FST).
CV_FOLD = os.environ.get("CV_FOLD", "")
FOLDS_CSV = os.environ.get("FOLDS_CSV", "")

# ARCH: "resnet50" (default) or "effb0" (EfficientNet-B0). Second backbone checks
#   whether the result belongs to the data or to one architecture.
ARCH = os.environ.get("ARCH", "resnet50")
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"   # 0 only for offline tests

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 2 if SMOKE else int(os.environ.get("EPOCHS", 10))
LR         = 3e-4
N_SMOKE    = 200          # images used in smoke mode

DATA_ROOT = os.environ.get("DATA_ROOT", "/kaggle/input")
IMG_ROOT  = os.environ.get("IMG_ROOT", "") or DATA_ROOT   # where images are looked up (CSV still from DATA_ROOT)
OUT_DIR   = os.environ.get("OUT_DIR", "/kaggle/working")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

tag = (f"seed{SEED}"
       + (f"_{ARCH}" if ARCH != "resnet50" else "")
       + (f"_mitigate{MIT_TARGET}" if MITIGATE else "")
       + (f"_hold{HOLDOUT_FST.replace(',', '')}" if HOLDOUT_FST else "")
       + ("_sm" if (HOLDOUT_FST and SIZE_MATCH) else "")
       + (f"_cv{CV_FOLD}" if CV_FOLD != "" else "")
       + ("_smoke" if SMOKE else ""))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# 1. FIND THE DATA
#    Kaggle mirrors differ. We sniff for the CSV and the image folder.
# ----------------------------------------------------------------------------
def find_csv():
    if os.environ.get("CSV_PATH"):
        return os.environ["CSV_PATH"]
    cands = glob.glob(f"{DATA_ROOT}/**/*.csv", recursive=True)
    if not cands:
        raise FileNotFoundError(f"No CSV under {DATA_ROOT}. Is the dataset attached?")
    # prefer one that looks like the fitzpatrick index
    for c in cands:
        if "fitz" in os.path.basename(c).lower():
            return c
    return max(cands, key=os.path.getsize)


def build_image_index():
    """Map stem -> full path for every image under IMG_ROOT."""
    idx = {}
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        for p in glob.glob(f"{IMG_ROOT}/**/*.{ext}", recursive=True):
            idx[os.path.splitext(os.path.basename(p))[0]] = p
    if not idx:
        raise FileNotFoundError(f"No images found under {IMG_ROOT}")
    print(f"[data] image root: {IMG_ROOT}  ({len(idx)} images indexed)")
    return idx


def pick_col(df, options, what):
    low = {c.lower(): c for c in df.columns}
    for o in options:
        if o in low:
            return low[o]
    raise KeyError(
        f"Could not find a {what} column. Looked for {options}.\n"
        f"Columns present: {list(df.columns)}\n"
        f"--> Edit pick_col() with the right name."
    )


print(f"[setup] device={DEVICE}  seed={SEED}  arch={ARCH}  smoke={SMOKE}  "
      f"mitigate={MITIGATE}  holdout={HOLDOUT_FST or '-'}  size_match={SIZE_MATCH}")

csv_path = find_csv()
df = pd.read_csv(csv_path)
print(f"[data] csv: {csv_path}  rows={len(df)}")
print(f"[data] columns: {list(df.columns)}")

FST_COL   = pick_col(df, ["fitzpatrick_scale", "fitzpatrick", "fitzpatrick_centaur",
                          "fst", "skin_type"], "Fitzpatrick")
LABEL_COL = pick_col(df, ["three_partition_label", "label", "diagnosis",
                          "nine_partition_label", "target"], "label")
ID_COL    = pick_col(df, ["md5hash", "image_id", "filename", "image", "id",
                          "url_alphanum"], "image id")

print(f"[data] using  fst={FST_COL}  label={LABEL_COL}  id={ID_COL}")

# ----------------------------------------------------------------------------
# 2. BUILD LABELS AND GROUPS
# ----------------------------------------------------------------------------
df[FST_COL] = pd.to_numeric(df[FST_COL], errors="coerce")
df = df[df[FST_COL].between(1, 6)].copy()          # drop -1 / unknown

df["group"] = np.where(df[FST_COL] <= 3, "light", "dark")

# Binary target: malignant vs not. Adjust the keyword if your label space differs.
lab = df[LABEL_COL].astype(str).str.lower()
df["y"] = lab.str.contains("malignant").astype(int)

if df["y"].nunique() < 2:
    raise ValueError(
        f"Binary target collapsed. Unique {LABEL_COL} values: "
        f"{df[LABEL_COL].unique()[:20]}\n--> Fix the 'malignant' rule above."
    )

img_index = build_image_index()
df["path"] = df[ID_COL].astype(str).map(img_index)
missing = df["path"].isna().sum()
df = df.dropna(subset=["path"]).reset_index(drop=True)
print(f"[data] resolved {len(df)} images ({missing} unresolved / URL-rotted)")

print("\n[data] group x class counts")
print(pd.crosstab(df["group"], df["y"]))

# ----------------------------------------------------------------------------
# 3. SPLIT  -- train on LIGHT only; test on both groups
# ----------------------------------------------------------------------------
set_seed(SEED)
light = df[df.group == "light"].sample(frac=1, random_state=SEED).reset_index(drop=True)
dark  = df[df.group == "dark"].reset_index(drop=True)

n_test = int(0.20 * len(light))
test_light  = light.iloc[:n_test]
train_light = light.iloc[n_test:]

if CV_FOLD != "":
    folds = pd.read_csv(FOLDS_CSV)[["md5hash", "fold"]]
    fmap = dict(zip(folds.md5hash, folds.fold))
    df["fold"] = df[ID_COL].map(fmap)
    if df["fold"].isna().any():
        raise ValueError(f"{int(df.fold.isna().sum())} images have no fold in {FOLDS_CSV}")
    k = int(CV_FOLD)
    test_all = df[df.fold == k]
    train = df[df.fold != k].sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_light = test_all[test_all.group == "light"].reset_index(drop=True)
    test_dark = test_all[test_all.group == "dark"].reset_index(drop=True)
    print(f"[cv] fold {k}: train on all FST ({len(train)} images, "
          f"{int((train.group == 'dark').sum())} FST IV-VI); test {len(test_all)} images")
elif HOLDOUT_FST:
    # train on light + the named FST levels; test on the FST levels NOT named
    keep = [int(v) for v in HOLDOUT_FST.split(",")]
    train_extra = dark[dark[FST_COL].isin(keep)]
    test_dark   = dark[~dark[FST_COL].isin(keep)]
    light_part  = train_light
    if SIZE_MATCH:
        n_light = len(train_light) - len(train_extra)
        if n_light <= 0:
            raise ValueError("SIZE_MATCH: more extra images than the baseline training set")
        light_part = train_light.sample(n_light, random_state=SEED)
        print(f"[holdout] size-matched: {n_light} light + {len(train_extra)} FST {keep} "
              f"= {n_light + len(train_extra)} (baseline {len(train_light)})")
    train = pd.concat([light_part, train_extra], ignore_index=True)
    print(f"[holdout] training also on FST {keep}; testing on FST "
          f"{sorted(test_dark[FST_COL].unique().astype(int))}")
elif MITIGATE:
    # hold out half the targeted data for testing, put the rest in training
    pool = dark if MIT_TARGET == "dark" else dark[dark[FST_COL] == int(MIT_TARGET)]
    rest = dark.drop(pool.index)
    pool = pool.sample(frac=1, random_state=SEED)
    cut = len(pool) // 2
    test_dark   = pd.concat([pool.iloc[:cut], rest], ignore_index=True)
    train_extra = pool.iloc[cut:]
    train = pd.concat([train_light, train_extra], ignore_index=True)
    print(f"[mitigate] oversampling target='{MIT_TARGET}', "
          f"{len(train_extra)} images added to training")
else:
    test_dark = dark
    train = train_light

if SMOKE:
    train     = train.sample(min(N_SMOKE, len(train)), random_state=SEED)
    test_light = test_light.sample(min(100, len(test_light)), random_state=SEED)
    test_dark  = test_dark.sample(min(100, len(test_dark)),  random_state=SEED)

print(f"\n[split] train={len(train)}  test_light={len(test_light)}  test_dark={len(test_dark)}")
print(f"[split] test_dark positives={int(test_dark.y.sum())}")
if len(test_dark) < 200 and not SMOKE:
    print("[WARN] dark test set is small -- expect wide CIs. "
          "Consider bootstrapping rather than relying on seed variance alone.")

# ----------------------------------------------------------------------------
# 4. DATA PIPELINE
# ----------------------------------------------------------------------------
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(), NORM,
])
# NOTE: deliberately NO colour jitter -- it would confound the skin-tone signal.

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), NORM,
])


class SkinDS(Dataset):
    def __init__(self, frame, tf):
        self.df, self.tf = frame.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["y"]), i


NW = int(os.environ.get("NUM_WORKERS", 4 if DEVICE == "cuda" else 0))


def loader(frame, tf, shuffle=False, sampler=None):
    return DataLoader(SkinDS(frame, tf), batch_size=BATCH_SIZE,
                      shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=NW, pin_memory=(DEVICE == "cuda"))


sampler = None
if MITIGATE:
    # oversample the targeted images so the model sees them near parity
    if MIT_TARGET == "dark":
        is_target = (train["group"].values == "dark")
    else:
        is_target = (train[FST_COL].values == int(MIT_TARGET))
    n_t = max(is_target.sum(), 1)
    n_o = max((~is_target).sum(), 1)
    w = np.where(is_target, len(train) / n_t, len(train) / n_o)
    sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(train), replacement=True)
    print(f"[mitigate] sampler weights: {n_t} target vs {n_o} other")

train_dl = loader(train, train_tf, shuffle=True, sampler=sampler)
light_dl = loader(test_light, eval_tf)
dark_dl  = loader(test_dark,  eval_tf)

# ----------------------------------------------------------------------------
# 5. MODEL
# ----------------------------------------------------------------------------
if ARCH == "resnet50":
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if PRETRAINED else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
elif ARCH == "effb0":
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if PRETRAINED else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
else:
    raise ValueError(f"Unknown ARCH '{ARCH}' (use resnet50 or effb0)")
print(f"[model] {ARCH}  pretrained={PRETRAINED}")
model = model.to(DEVICE)

# class-weighted loss: malignant is the minority and the one that matters
counts = train["y"].value_counts()
cw = torch.tensor([len(train) / (2 * counts.get(0, 1)),
                   len(train) / (2 * counts.get(1, 1))],
                  dtype=torch.float32, device=DEVICE)
crit = nn.CrossEntropyLoss(weight=cw)
opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

# ----------------------------------------------------------------------------
# 6. TRAIN
# ----------------------------------------------------------------------------
CKPT = f"{OUT_DIR}/model_{tag}.pt"

if EVAL_ONLY or RESUME:
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"No checkpoint at {CKPT}. The session was wiped -- retrain from scratch."
        )
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print(f"[ckpt] loaded {CKPT}")

if EVAL_ONLY:
    print("[eval] skipping training, going straight to evaluation")
else:
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for x, y, _ in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()
        print(f"[train] epoch {ep+1}/{EPOCHS}  loss={tot/len(train_dl.dataset):.4f}",
              flush=True)   # flush: Kaggle buffers stdout and looks frozen otherwise
        # checkpoint EVERY epoch -- Kaggle sessions die without warning
        torch.save(model.state_dict(), CKPT)

# ----------------------------------------------------------------------------
# 7. EVALUATE
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(dl):
    model.eval()
    P, Y, I = [], [], []
    for x, y, i in dl:
        p = torch.softmax(model(x.to(DEVICE)), 1)[:, 1]
        P.append(p.cpu().numpy()); Y.append(y.numpy()); I.append(i.numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(I)


def ece(probs, y, bins=10):
    """Expected calibration error -- keep this even if the accuracy gap is small."""
    conf = np.where(probs >= 0.5, probs, 1 - probs)
    correct = ((probs >= 0.5).astype(int) == y).astype(float)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def metrics(probs, y):
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None,
        "ece": ece(probs, y),
        "accuracy": float((pred == y).mean()),
    }


res, rows = {}, []
for name, dl, frame in [("light", light_dl, test_light), ("dark", dark_dl, test_dark)]:
    p, y, i = predict(dl)
    res[name] = metrics(p, y)
    sub = frame.reset_index(drop=True).iloc[i]
    rows.append(pd.DataFrame({
        "group": name, "prob": p, "y": y,
        "path": sub["path"].values, "fst": sub[FST_COL].values,
        # carry the diagnosis through so the disease-mix confound can be
        # checked later WITHOUT retraining
        "condition": sub[LABEL_COL].values,
    }))

if res["light"]["sensitivity"] is not None and res["dark"]["sensitivity"] is not None:
    res["gap_sensitivity"] = res["light"]["sensitivity"] - res["dark"]["sensitivity"]
if res["light"]["auc"] is not None and res["dark"]["auc"] is not None:
    res["gap_auc"] = res["light"]["auc"] - res["dark"]["auc"]
res["gap_ece"] = res["dark"]["ece"] - res["light"]["ece"]
res["config"] = {"seed": SEED, "smoke": SMOKE, "mitigate": MITIGATE,
                 "mit_target": MIT_TARGET if MITIGATE else None,
                 "holdout_fst": HOLDOUT_FST or None, "size_match": SIZE_MATCH, "cv_fold": CV_FOLD or None,
                 "arch": ARCH, "epochs": EPOCHS, "n_train": len(train), "img_root": IMG_ROOT}

pd.concat(rows).to_csv(f"{OUT_DIR}/probs_{tag}.csv", index=False)
with open(f"{OUT_DIR}/results_{tag}.json", "w") as f:
    json.dump(res, f, indent=2)

print("\n" + "=" * 60)
print(json.dumps(res, indent=2))
print("=" * 60)
print(f"saved -> {OUT_DIR}/results_{tag}.json  /  probs_{tag}.csv  /  model_{tag}.pt")
