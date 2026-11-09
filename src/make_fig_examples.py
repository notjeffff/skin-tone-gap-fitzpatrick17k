"""Example-image figure for the paper (Fig. 2): one squamous cell carcinoma and one
psoriasis image per skin type I-VI, from Fitzpatrick17k images where both skin-type
annotations agree.

Kaggle inputs: fitzpatrick17k_original (Mobaswir, images) + fitzpatrick17k (Nazmus, CSV).
Outputs in /kaggle/working:
  fig_examples.pdf        the figure for the paper
  fig_examples_candidates.png   8 candidates per cell, numbered, to swap a pick
To swap a pick, set PICKS[(fst, "mal"|"non")] = candidate number and rerun.
"""
import os, glob, random
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
SEED = 7
PICKS = {}          # e.g. {(6, "mal"): 3, (2, "non"): 5}
ROWS = {"mal": "squamous cell carcinoma", "non": "psoriasis"}
N_CAND = 8

# ---------- locate CSV and images ----------
csv = os.environ.get("CSV_PATH") or next(
    c for c in glob.glob("/kaggle/input/**/*.csv", recursive=True)
    if "fitzpatrick_scale" in open(c, errors="ignore").readline())
df = pd.read_csv(csv)
root = os.environ.get("IMG_ROOT") or next(
    d for d in glob.glob("/kaggle/input/**/", recursive=True) if "original" in d.lower())
paths = {os.path.splitext(os.path.basename(p))[0]: p
         for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)}
print(f"csv={csv}\nimages={root} ({len(paths)} files)")

df = df[df.fitzpatrick_scale.between(1, 6) & (df.fitzpatrick_scale == df.fitzpatrick_centaur)]
df = df[df.md5hash.isin(paths)]
if "qc" in df:                                   # drop images flagged as wrongly labelled
    df = df[~df.qc.astype(str).str.startswith("3")]
df["mal"] = df.three_partition_label.str.contains("malignant")
print(df.groupby(["fitzpatrick_scale", "mal"]).size().unstack())

def square(p, size=320):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size))

cand = {}
for f in range(1, 7):
    for kind, flag in (("mal", True), ("non", False)):
        pool = df[(df.fitzpatrick_scale == f) & (df.label == ROWS[kind])]
        cand[(f, kind)] = pool.sample(min(N_CAND, len(pool)), random_state=SEED + f).reset_index(drop=True)

# ---------- candidate sheet ----------
fig, ax = plt.subplots(12, N_CAND, figsize=(N_CAND * 1.3, 12 * 1.45))
for r, key in enumerate([(f, k) for f in range(1, 7) for k in ("mal", "non")]):
    for c in range(N_CAND):
        a = ax[r, c]; a.axis("off")
        if c < len(cand[key]):
            row = cand[key].iloc[c]
            a.imshow(square(paths[row.md5hash], 160))
            a.set_title(f"FST {key[0]} {ROWS[key[1]][:8]} #{c}", fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_examples_candidates.png", dpi=110); plt.close(fig)

# ---------- final figure ----------
roman = ["I", "II", "III", "IV", "V", "VI"]
plt.rcParams.update({"font.family": "serif", "font.size": 6.5})
fig, ax = plt.subplots(2, 6, figsize=(3.45, 1.38))
for c, f in enumerate(range(1, 7)):
    for r, kind in enumerate(("mal", "non")):
        row = cand[(f, kind)].iloc[PICKS.get((f, kind), 0)]
        a = ax[r, c]; a.imshow(square(paths[row.md5hash])); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_linewidth(0.4)
        if r == 0: a.set_title(f"FST {roman[c]}", fontsize=7, pad=2)
        if c == 0: a.set_ylabel("Squamous cell\ncarcinoma" if kind == "mal" else "Psoriasis", fontsize=6.3, labelpad=2)
fig.subplots_adjust(left=0.105, right=0.995, top=0.90, bottom=0.01, wspace=0.05, hspace=0.06)
fig.savefig(f"{OUT}/fig_examples.pdf"); fig.savefig(f"{OUT}/fig_examples.png", dpi=300)
print("picked:", {k: v.iloc[PICKS.get(k, 0)].md5hash for k, v in cand.items()})
print("saved fig_examples.pdf / .png / fig_examples_candidates.png")
