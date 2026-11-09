"""
Analyses for the EfficientNet and intervention runs. CPU only, about a minute.

1. EfficientNet-B0 (5 seeds): does the FST 6 cliff replicate on a second backbone?
2. Size-matched holdout: is the FST 6 recovery from darker-skin exposure or data volume?
3. Second skin-type annotation (fitzpatrick_centaur): is the cliff robust to FST label noise?

Reads probs_*.csv from OUT_DIR (default /kaggle/working) and the Fitzpatrick17k CSV
under DATA_ROOT (default /kaggle/input).
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


def per_level(runs, col="fst"):
    rows = []
    for s, d in runs.items():
        for lvl, g in d.groupby(col):
            rows.append(dict(seed=s, fst=int(lvl), **summarize(g)))
    return pd.DataFrame(rows)


def contrasts(runs, col="fst", label=""):
    spec_ = [("FST 1-3 minus FST 6", [1, 2, 3], [6]),
             ("FST 1-3 minus FST 4-5", [1, 2, 3], [4, 5]),
             ("FST 4-5 minus FST 6", [4, 5], [6])]
    for name, lo, hi in spec_:
        for metric, stat in (("sensitivity", sens), ("AUC", auc)):
            res = []
            for s, d in runs.items():
                L, D = d[d[col].isin(lo)], d[d[col].isin(hi)]
                if L.y.sum() < 5 or D.y.sum() < 5:
                    continue
                res.append(boot_gap(L, D, stat))
            if not res:
                continue
            gaps = np.array([r[0] for r in res])
            sig = sum(1 for r in res if r[1] > 0 or r[2] < 0)
            print(f"{label}{name:24s} {metric:12s} mean gap {gaps.mean():+.3f}  "
                  f"per-seed [{gaps.min():+.3f}, {gaps.max():+.3f}]  "
                  f"CI excludes 0 in {sig}/{len(res)} seeds")


main = read_seed_files("")
eff  = read_seed_files("_effb0")
hold = read_seed_files("_hold45")
hsm  = read_seed_files("_hold45_sm")
print(f"resnet50 main: {list(main)}  effb0: {list(eff)}  hold45: {list(hold)}  hold45_sm: {list(hsm)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("1. SECOND BACKBONE: EfficientNet-B0 vs ResNet-50 (mean over seeds)")
print("=" * 74)
if eff and main:
    A = per_level(main).groupby("fst")[["sens", "auc"]].agg(["mean", "std"])
    B = per_level(eff).groupby("fst")[["sens", "auc"]].agg(["mean", "std"])
    A.columns = [f"r50_{a}_{b}" for a, b in A.columns]
    B.columns = [f"eff_{a}_{b}" for a, b in B.columns]
    print(A.join(B).round(3).to_string())
    print("\nEfficientNet-B0 contrasts")
    contrasts(eff, label="effb0  ")
    print("\nResNet-50 contrasts (for comparison)")
    contrasts(main, label="r50    ")
else:
    print("effb0 runs or resnet main runs missing - skipped")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("2. SIZE-MATCHED HOLDOUT, ON FST 6 (same images for every model)")
print("   baseline     : light only, 8,851 images")
print("   +FST4,5      : light + FST 4,5, ~13,165 images (more data AND darker skin)")
print("   +FST4,5 (sm) : fewer light + FST 4,5, 8,851 images (darker skin, same volume)")
print("=" * 74)
rows = []
for s in sorted(set(hsm) | set(hold)):
    if s not in main:
        continue
    for name, src in (("baseline", main), ("+FST4,5", hold), ("+FST4,5 (sm)", hsm)):
        if s in src:
            d = src[s]
            rows.append(dict(seed=s, subset="FST6", model=name, **summarize(d[d.fst == 6])))
            rows.append(dict(seed=s, subset="light test", model=name,
                             **summarize(d[d.group == "light"])))
if rows:
    H = pd.DataFrame(rows)
    print(H.round(3).to_string(index=False))
    print("\nmean over seeds")
    print(H.groupby(["subset", "model"], sort=False)[["sens", "spec", "auc"]].mean().round(3).to_string())
    print("""
How to read it:
  +FST4,5 (sm) close to +FST4,5, both above baseline -> darker-skin exposure helps,
                                                      not just extra volume.
  +FST4,5 (sm) close to baseline                     -> most of the gain was volume.
""")
else:
    print("no holdout runs found - skipped")

# ---------------------------------------------------------------------------
print("=" * 74)
print("3. SECOND SKIN-TYPE ANNOTATION (fitzpatrick_centaur)")
print("   Caveat: training used fitzpatrick_scale. Grouping test images by the other")
print("   annotation is a robustness check on the test side, not a clean re-run.")
print("=" * 74)
csvs = [c for c in glob.glob(f"{DATA}/**/*.csv", recursive=True)
        if "fitz" in os.path.basename(c).lower()]
if not csvs:
    print("Fitzpatrick17k CSV not found - skipped")
else:
    raw = pd.read_csv(csvs[0])
    if "fitzpatrick_centaur" not in raw.columns:
        print("no fitzpatrick_centaur column - skipped")
    else:
        meta = raw[["md5hash", "fitzpatrick_scale", "fitzpatrick_centaur"]].drop_duplicates("md5hash")
        meta["fst_c"] = pd.to_numeric(meta["fitzpatrick_centaur"], errors="coerce")
        meta.loc[~meta.fst_c.between(1, 6), "fst_c"] = np.nan

        def add_c(d):
            d = d.copy()
            d["md5hash"] = d["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
            d = d.merge(meta[["md5hash", "fst_c"]], on="md5hash", how="left")
            return d

        cm = {s: add_c(d) for s, d in main.items()}
        d0 = cm[min(cm)].dropna(subset=["fst_c"])
        print(f"test images with a valid second label (seed {min(cm)}): {len(d0)} "
              f"of {len(cm[min(cm)])}")
        ag = (d0.fst == d0.fst_c).mean()
        w1 = ((d0.fst - d0.fst_c).abs() <= 1).mean()
        print(f"exact agreement {ag:.1%}   within one level {w1:.1%}")
        print("\nrows = fitzpatrick_scale (used for training), cols = fitzpatrick_centaur")
        print(pd.crosstab(d0.fst.astype(int), d0.fst_c.astype(int)).to_string())

        print("\n(a) grouped by the SECOND annotation (mean over seeds)")
        runs_c = {s: d.dropna(subset=["fst_c"]).assign(fst_c=lambda x: x.fst_c.astype(int))
                  for s, d in cm.items()}
        Tc = per_level(runs_c, col="fst_c")
        print(Tc.groupby("fst")[["n", "pos", "sens", "auc"]].mean().round(3).to_string())
        contrasts(runs_c, col="fst_c", label="centaur ")

        print("\n(b) only images where BOTH annotators agree (mean over seeds)")
        runs_a = {s: d[d.fst == d.fst_c] for s, d in runs_c.items()}
        Ta = per_level(runs_a)
        print(Ta.groupby("fst")[["n", "pos", "sens", "auc"]].mean().round(3).to_string())
        contrasts(runs_a, label="agreed  ")
        print("""
How to read it:
  If FST 6 is still far below FST 1-3 under the second annotation and on the
  agreed subset, the cliff is not an artefact of one annotator's skin-type labels.
  Agreed-subset counts can be small; check the pos column before trusting a row.
""")
