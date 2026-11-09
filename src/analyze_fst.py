"""
Per-Fitzpatrick-level analysis across ALL seeds, plus the paired checks.
CPU only, about a minute. Reads the probs_*.csv files written by skin_tone_gap.py.

In a Kaggle cell:   %run analyze_fst.py
Needs:     probs_seed{0-4}.csv                      (main runs)
Optional:  probs_seed{0-2}_mitigate6.csv            (mitigation)
           probs_seed{0-2}_hold45.csv               (holdout)
           the Fitzpatrick17k CSV under /kaggle/input (subtype check)
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT  = os.environ.get("OUT_DIR", "/kaggle/working")
DATA = os.environ.get("DATA_ROOT", "/kaggle/input")
NB   = int(os.environ.get("NBOOT", 1000))
rng  = np.random.default_rng(0)
pd.set_option("display.width", 200, "display.precision", 3)


def sens(p, y):
    m = y == 1
    return float((p[m] >= 0.5).mean()) if m.any() else np.nan


def spec(p, y):
    m = y == 0
    return float((p[m] < 0.5).mean()) if m.any() else np.nan


def auc(p, y):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def summarize(g):
    p, y = g["prob"].values, g["y"].values
    return dict(n=len(g), pos=int(y.sum()), sens=sens(p, y), spec=spec(p, y), auc=auc(p, y))


def read_seed_files(suffix):
    found = {}
    for f in glob.glob(f"{OUT}/probs_seed*{suffix}.csv"):
        m = re.fullmatch(rf"probs_seed(\d+){re.escape(suffix)}\.csv", os.path.basename(f))
        if m:
            found[int(m.group(1))] = pd.read_csv(f)
    return dict(sorted(found.items()))


main = read_seed_files("")
mit  = read_seed_files("_mitigate6")
hold = read_seed_files("_hold45")
if not main:
    raise SystemExit(f"No probs_seed*.csv in {OUT}")
print(f"main seeds: {list(main)}   mitigate6 seeds: {list(mit)}   hold45 seeds: {list(hold)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("1. PER-FST RESULTS ACROSS SEEDS (mean, SD over seeds; threshold 0.5)")
print("=" * 74)
rows = []
for s, d in main.items():
    for lvl, g in d.groupby("fst"):
        rows.append(dict(seed=s, fst=int(lvl), **summarize(g)))
T = pd.DataFrame(rows)
A = T.groupby("fst").agg(
    n=("n", "mean"), pos=("pos", "mean"),
    sens=("sens", "mean"), sens_sd=("sens", "std"),
    spec=("spec", "mean"), spec_sd=("spec", "std"),
    auc=("auc", "mean"), auc_sd=("auc", "std"))
print(A.round(3).to_string())
print("\nsensitivity by seed")
print(T.pivot(index="fst", columns="seed", values="sens").round(3).to_string())
print("\nAUC by seed")
print(T.pivot(index="fst", columns="seed", values="auc").round(3).to_string())

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("2. PRE-SPECIFIED CONTRASTS (bootstrap over test images, per seed)")
print("=" * 74)


def boot_gap(L, D, stat):
    pl, yl = L["prob"].values, L["y"].values
    pdk, yd = D["prob"].values, D["y"].values
    point = stat(pl, yl) - stat(pdk, yd)
    gaps = np.empty(NB)
    for b in range(NB):
        i = rng.integers(0, len(yl), len(yl))
        j = rng.integers(0, len(yd), len(yd))
        gaps[b] = stat(pl[i], yl[i]) - stat(pdk[j], yd[j])
    return point, np.nanpercentile(gaps, 2.5), np.nanpercentile(gaps, 97.5)


CONTRASTS = [("FST 1-3 minus FST 6",   [1, 2, 3], [6]),
             ("FST 1-3 minus FST 4-5", [1, 2, 3], [4, 5]),
             ("FST 4-5 minus FST 6",   [4, 5],    [6])]
for name, lo, hi in CONTRASTS:
    for metric, stat in (("sensitivity", sens), ("AUC", auc)):
        res = [(s, *boot_gap(d[d.fst.isin(lo)], d[d.fst.isin(hi)], stat))
               for s, d in main.items()]
        gaps = np.array([r[1] for r in res])
        sig = sum(1 for r in res if r[2] > 0 or r[3] < 0)
        print(f"{name:24s} {metric:12s} mean gap {gaps.mean():+.3f}  "
              f"per-seed [{gaps.min():+.3f}, {gaps.max():+.3f}]  "
              f"CI excludes 0 in {sig}/{len(res)} seeds")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("3. MITIGATION, PAIRED ON THE SAME IMAGES")
print("   (baseline vs oversample-FST6; FST 6 images = the held-out half only)")
print("=" * 74)
rows = []
for s, m in mit.items():
    if s not in main:
        continue
    b = main[s]
    m6 = m[m.fst == 6]
    b6 = b[(b.fst == 6) & b.path.isin(m6.path)]
    m6 = m6[m6.path.isin(b6.path)]
    for tag, bb, mm in (("FST6 (same imgs)", b6, m6),
                        ("light test", b[b.group == "light"], m[m.group == "light"])):
        rows.append(dict(seed=s, subset=tag, model="baseline", **summarize(bb)))
        rows.append(dict(seed=s, subset=tag, model="oversample6", **summarize(mm)))
if rows:
    M = pd.DataFrame(rows)
    print(M.round(3).to_string(index=False))
    print("\nmean over seeds")
    print(M.groupby(["subset", "model"])[["sens", "spec", "auc"]].mean().round(3).to_string())
else:
    print("no mitigation files found")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("4. HOLDOUT (train on light + FST 4,5) vs BASELINE, ON FST 6")
print("   NB: holdout also adds ~4,300 training images; tone exposure and data")
print("   volume are not separated by this comparison.")
print("=" * 74)
rows = []
for s, h in hold.items():
    if s not in main:
        continue
    b = main[s]
    rows.append(dict(seed=s, subset="FST6", model="baseline", **summarize(b[b.fst == 6])))
    rows.append(dict(seed=s, subset="FST6", model="train+FST4,5", **summarize(h[h.fst == 6])))
    rows.append(dict(seed=s, subset="light test", model="baseline", **summarize(b[b.group == "light"])))
    rows.append(dict(seed=s, subset="light test", model="train+FST4,5", **summarize(h[h.group == "light"])))
if rows:
    H = pd.DataFrame(rows)
    print(H.round(3).to_string(index=False))
    print("\nmean over seeds")
    print(H.groupby(["subset", "model"])[["sens", "spec", "auc"]].mean().round(3).to_string())
else:
    print("no holdout files found")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("5. MALIGNANT SUBTYPE MIX (is FST 6 a different mix of cancers?)")
print("=" * 74)
csvs = [c for c in glob.glob(f"{DATA}/**/*.csv", recursive=True)
        if "fitz" in os.path.basename(c).lower()]
if os.environ.get("CSV_PATH"):
    csvs = [os.environ["CSV_PATH"]]
if not csvs:
    print("Fitzpatrick CSV not found under", DATA, "- skipped")
else:
    meta = (pd.read_csv(csvs[0])[["md5hash", "label"]]
            .rename(columns={"label": "dx"}).drop_duplicates("md5hash"))

    def with_dx(d):
        d = d.copy()
        d["md5hash"] = d["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
        return d.merge(meta, on="md5hash", how="left")

    d0 = with_dx(main[min(main)])
    d0 = d0[d0.y == 1].copy()
    d0["grp"] = np.where(d0.fst <= 3, "FST1-3", np.where(d0.fst <= 5, "FST4-5", "FST6"))
    ct = pd.crosstab(d0.dx, d0.grp)
    share = (ct / ct.sum() * 100).round(1).add_suffix("_%")
    comp = ct.join(share).sort_values("FST6", ascending=False)
    print("malignant cases by diagnosis (counts, and % of that group's malignant cases)")
    print(comp.head(10).to_string())

    f6 = pd.concat([d.assign(seed=s) for s, d in main.items()])
    f6 = with_dx(f6[(f6.fst == 6) & (f6.y == 1)])
    f6["hit"] = f6.prob >= 0.5
    r = (f6.groupby("dx").agg(n=("path", "nunique"), sens=("hit", "mean"))
         .sort_values("n", ascending=False))
    print("\nFST 6 malignant cases: sensitivity by diagnosis (mean over seeds)")
    print(r.head(10).round(3).to_string())
    print("-> if FST 6 cancers are dominated by subtypes that are rare in the")
    print("   light-skin training set, the cliff is partly a subtype effect.")
