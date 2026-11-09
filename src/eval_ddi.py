"""
External validation on DDI (Diverse Dermatology Images, Stanford AIMI).
Runs the checkpoints you already trained on biopsy-confirmed DDI images, which come
grouped as FST I-II, III-IV and V-VI. No training.

In a Kaggle cell:
    %run eval_ddi.py                 # all checkpoints found under /kaggle/input
    DDI_SMOKE=1 -> first checkpoint, 64 images, to check the setup in a minute

Inputs (attach in the sidebar):
    - your private DDI dataset (images + the metadata CSV)
    - notebook 01 output (model_seed{0-4}.pt, plus mitigation/holdout checkpoints)
    - notebook 02 output once finished (model_seed{N}_effb0.pt, model_seed{N}_hold45_sm.pt)

Outputs: probs_ddi_all.csv (every image x model) and printed tables.

Read AUC first: the 0.5 threshold was tuned on Fitzpatrick17k, not DDI, and DDI's
clinical photos differ from the background-removed atlas images the models saw.
Compare DDI's groups with EACH OTHER, not with the Fitzpatrick17k numbers.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

IN    = os.environ.get("INPUT_ROOT", "/kaggle/input")
OUT   = os.environ.get("OUT_DIR", "/kaggle/working")
SMOKE = os.environ.get("DDI_SMOKE", "0") == "1"
FAKE  = os.environ.get("FAKE_INFER", "0") == "1"      # testing hook only: random probabilities
NB    = int(os.environ.get("NBOOT", 1000))
ONLY  = [c for c in os.environ.get("CONFIGS", "").split(",") if c]
GROUPS = ["I-II", "III-IV", "V-VI"]
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200, "display.precision", 3)
rng = np.random.default_rng(0)


def pick(cols, options, what):
    low = {c.lower(): c for c in cols}
    for o in options:
        if o in low:
            return low[o]
    raise KeyError(f"No {what} column; looked for {options}. Columns: {list(cols)}")


# ------------------------------------------------------------------ metadata
meta_path = None
for c in sorted(glob.glob(f"{IN}/**/*.csv", recursive=True)):
    try:
        cols = [x.lower() for x in pd.read_csv(c, nrows=1).columns]
    except Exception:
        continue
    if "skin_tone" in cols and "malignant" in cols:
        meta_path = c
        break
if meta_path is None:
    raise FileNotFoundError(f"DDI metadata CSV (columns skin_tone, malignant) not found under {IN}. "
                            "Is the DDI dataset attached?")
meta = pd.read_csv(meta_path)
SKIN = pick(meta.columns, ["skin_tone"], "skin tone")
MAL  = pick(meta.columns, ["malignant"], "malignant")
FILE = pick(meta.columns, ["ddi_file", "file", "filename", "image", "image_name"], "file name")
print(f"[ddi] metadata: {meta_path}  rows={len(meta)}  columns={list(meta.columns)}")

GMAP = {"12": "I-II", "34": "III-IV", "56": "V-VI",
        "1": "I-II", "2": "I-II", "3": "III-IV", "4": "III-IV", "5": "V-VI", "6": "V-VI"}
meta["grp"] = meta[SKIN].map(lambda v: GMAP.get(str(v).strip().split(".")[0]))


def to01(v):
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes", "malignant"):
        return 1
    if s in ("false", "0", "0.0", "no", "benign"):
        return 0
    return np.nan


meta["y"] = meta[MAL].map(to01)

ddi_dir = os.path.dirname(meta_path)
imgs = {}
for root in (ddi_dir, os.path.dirname(ddi_dir)):
    for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
        for p in glob.glob(f"{root}/**/*.{ext}", recursive=True):
            imgs.setdefault(os.path.basename(p), p)
            imgs.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    if imgs:
        break
meta["path"] = meta[FILE].astype(str).map(
    lambda f: imgs.get(os.path.basename(f)) or imgs.get(os.path.splitext(os.path.basename(f))[0]))
bad = meta["grp"].isna() | meta["y"].isna() | meta["path"].isna()
if bad.any():
    print(f"[ddi] dropping {int(bad.sum())} rows with unmapped skin tone, label or image")
meta = meta[~bad].reset_index(drop=True)
meta["y"] = meta["y"].astype(int)
if SMOKE:
    meta = (meta.sample(frac=1, random_state=0).groupby(["grp", "y"]).head(11)
            .reset_index(drop=True))
print("\n[ddi] images by skin group x malignant")
print(pd.crosstab(meta.grp, meta.y).reindex(GROUPS).to_string())

# ------------------------------------------------------------------ checkpoints
found = {}
for p in glob.glob(f"{IN}/**/model_seed*.pt", recursive=True):
    m = re.fullmatch(r"model_seed(\d+)(.*)\.pt", os.path.basename(p))
    if not m or "smoke" in m.group(2):
        continue
    suf = m.group(2)
    cfg = ("effb0" + suf[len("_effb0"):]) if suf.startswith("_effb0") else ("resnet50" + suf)
    found.setdefault((cfg, int(m.group(1))), p)
ckpts = sorted((cfg, s, p) for (cfg, s), p in found.items() if not ONLY or cfg in ONLY)
if not ckpts:
    raise FileNotFoundError(f"No model_seed*.pt under {IN}. Attach notebooks 01 and 02 as inputs.")
if SMOKE:
    ckpts = [c for c in ckpts if c[0] == "resnet50"][:1] or ckpts[:1]
print(f"\n[ddi] {len(ckpts)} checkpoints: " +
      ", ".join(f"{c}/seed{s}" for c, s, _ in ckpts))


# ------------------------------------------------------------------ inference
def predict(cfg, path, files):
    if FAKE:
        return np.random.default_rng(abs(hash(path)) % 2**32).random(len(files))
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import models, transforms

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.startswith("effb0"):
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
    else:
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(path, map_location=dev))
    net = net.to(dev).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for i in range(0, len(files), 32):
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 32]]).to(dev)
            out.append(torch.softmax(net(x), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


rows = []
for cfg, s, p in ckpts:
    print(f"[ddi] {cfg} seed {s} ...", flush=True)
    pr = predict(cfg, p, meta["path"].tolist())
    rows.append(pd.DataFrame(dict(cfg=cfg, seed=s, file=meta[FILE].values,
                                  grp=meta.grp.values, y=meta.y.values, prob=pr)))
P = pd.concat(rows, ignore_index=True)
P.to_csv(f"{OUT}/probs_ddi_all{'_smoke' if SMOKE else ''}.csv", index=False)


# ------------------------------------------------------------------ metrics
def sens(p, y):
    m = y == 1
    return float((p[m] >= 0.5).mean()) if m.any() else np.nan


def spec(p, y):
    m = y == 0
    return float((p[m] < 0.5).mean()) if m.any() else np.nan


def auc(p, y):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


T = []
for (cfg, s, g), d in P.groupby(["cfg", "seed", "grp"]):
    T.append(dict(cfg=cfg, seed=s, grp=g, n=len(d), pos=int(d.y.sum()),
                  sens=sens(d.prob.values, d.y.values), spec=spec(d.prob.values, d.y.values),
                  auc=auc(d.prob.values, d.y.values)))
T = pd.DataFrame(T)
print("\n" + "=" * 74)
print("DDI RESULTS BY MODEL AND SKIN GROUP (mean over seeds; *_sd = SD over seeds)")
print("=" * 74)
for cfg, d in T.groupby("cfg"):
    A = d.groupby("grp").agg(seeds=("seed", "nunique"), n=("n", "first"), pos=("pos", "first"),
                             sens=("sens", "mean"), sens_sd=("sens", "std"),
                             spec=("spec", "mean"), auc=("auc", "mean"), auc_sd=("auc", "std"))
    print(f"\n{cfg}")
    print(A.reindex(GROUPS).round(3).to_string())


def boot(L, D, stat):
    pl, yl, pd_, yd = L.prob.values, L.y.values, D.prob.values, D.y.values
    point = stat(pl, yl) - stat(pd_, yd)
    g = np.empty(NB)
    for b in range(NB):
        i = rng.integers(0, len(yl), len(yl))
        j = rng.integers(0, len(yd), len(yd))
        g[b] = stat(pl[i], yl[i]) - stat(pd_[j], yd[j])
    return point, np.nanpercentile(g, 2.5), np.nanpercentile(g, 97.5)


print("\n" + "=" * 74)
print("CONTRASTS (bootstrap over DDI images, per seed)")
print("=" * 74)
for cfg, d in P.groupby("cfg"):
    for lo, hi in (("I-II", "V-VI"), ("III-IV", "V-VI"), ("I-II", "III-IV")):
        for metric, stat in (("AUC", auc), ("sensitivity", sens)):
            res = [boot(x[x.grp == lo], x[x.grp == hi], stat) for _, x in d.groupby("seed")]
            gaps = np.array([r[0] for r in res])
            sig = sum(1 for r in res if r[1] > 0 or r[2] < 0)
            print(f"{cfg:22s} {lo:>6s} minus {hi:<6s} {metric:12s} mean gap {gaps.mean():+.3f}  "
                  f"CI excludes 0 in {sig}/{len(res)} seeds")
print("""
Notes for the paper:
  - DDI merges V and VI, so it can test the darkest group but cannot isolate type VI.
  - V-VI has about 48 malignant images: expect wide intervals.
  - The models were trained on background-removed atlas images; DDI photos are clinical
    and unprocessed, so absolute numbers drop for every group. Compare groups, not datasets.
""")
