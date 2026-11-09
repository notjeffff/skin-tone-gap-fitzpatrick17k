"""
Does the FST VI drop survive on images the background removal left intact?
Inputs (under /kaggle/input or /kaggle/working): probs_seed{0-4}.csv (notebook (a)/(b) outputs)
and bg_check_per_image.csv (notebook (c) output). CPU only, no retraining.
"""
import glob, os, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

IN = os.environ.get("INPUT_ROOT", "/kaggle/input")
W = os.environ.get("OUT_DIR", "/kaggle/working")
NB = int(os.environ.get("NBOOT", 1000))
rng = np.random.default_rng(0)
pd.set_option("display.width", 200, "display.precision", 3)


def find(pattern):
    hits = sorted(glob.glob(f"{IN}/**/{pattern}", recursive=True) + glob.glob(f"{W}/{pattern}"))
    return hits


bgf = find("bg_check_per_image.csv")
if not bgf:
    raise FileNotFoundError("bg_check_per_image.csv not found: attach notebook (c) as an input")
bg = pd.read_csv(bgf[0])[["md5hash", "fill_kind", "fg_frac"]]
main = {}
for f in find("probs_seed*.csv"):
    m = re.fullmatch(r"probs_seed(\d+)\.csv", os.path.basename(f))
    if m and int(m.group(1)) not in main:
        main[int(m.group(1))] = pd.read_csv(f)
if not main:
    raise FileNotFoundError("probs_seed*.csv not found: attach notebook (b) (or (a)) as an input")
print(f"[bg] seeds {sorted(main)}  bg rows {len(bg)}")

for s, d in main.items():
    d["md5hash"] = d["path"].map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    main[s] = d.merge(bg, on="md5hash", how="left")
miss = main[min(main)].fill_kind.isna().mean()
print(f"[bg] test images without a bg record: {miss:.1%}")

SUBSETS = {
    "all images": lambda d: d,
    "intact (background kept, >=90% kept)": lambda d: d[(d.fill_kind == "other") & (d.fg_frac >= 0.9)],
    "masked, >=50% kept": lambda d: d[(d.fill_kind == "black") & (d.fg_frac >= 0.5)],
    "masked, <50% kept": lambda d: d[(d.fill_kind == "black") & (d.fg_frac < 0.5)],
}


def sens(p, y):
    m = y == 1
    return float((p[m] >= .5).mean()) if m.any() else np.nan


def auc(p, y):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def grp(d):
    return np.where(d.fst <= 3, "I-III", np.where(d.fst <= 5, "IV-V", "VI"))


for name, fn in SUBSETS.items():
    rows = []
    for s, d in main.items():
        x = fn(d).copy()
        x["g"] = grp(x)
        for g, z in x.groupby("g"):
            rows.append(dict(seed=s, g=g, n=len(z), pos=int(z.y.sum()),
                             sens=sens(z.prob.values, z.y.values), auc=auc(z.prob.values, z.y.values)))
    T = pd.DataFrame(rows)
    A = T.groupby("g").agg(n=("n", "mean"), pos=("pos", "mean"), sens=("sens", "mean"),
                           sens_sd=("sens", "std"), auc=("auc", "mean"), auc_sd=("auc", "std"))
    print("\n" + "=" * 74 + f"\n{name.upper()}\n" + "=" * 74)
    print(A.round(3).to_string())
    for lo, hi in (("I-III", "VI"), ("IV-V", "VI")):
        for metric, stat in (("sensitivity", sens), ("AUC", auc)):
            res = []
            for s, d in main.items():
                x = fn(d).copy(); x["g"] = grp(x)
                L, D = x[x.g == lo], x[x.g == hi]
                if D.y.nunique() < 2 or L.y.nunique() < 2:
                    continue
                pt = stat(L.prob.values, L.y.values) - stat(D.prob.values, D.y.values)
                b = []
                for _ in range(NB):
                    i = rng.integers(0, len(L), len(L)); j = rng.integers(0, len(D), len(D))
                    b.append(stat(L.prob.values[i], L.y.values[i]) - stat(D.prob.values[j], D.y.values[j]))
                lo_ci, hi_ci = np.nanpercentile(b, [2.5, 97.5])
                res.append((pt, lo_ci > 0 or hi_ci < 0))
            if res:
                print(f"  {lo:>5s} minus {hi:<3s} {metric:12s} mean gap {np.mean([r[0] for r in res]):+.3f}  "
                      f"CI excludes 0 in {sum(r[1] for r in res)}/{len(res)} seeds")

# within each skin group: does losing pixels lower sensitivity?
print("\n" + "=" * 74 + "\nSENSITIVITY BY AREA KEPT (malignant images, mean over seeds)\n" + "=" * 74)
rows = []
for s, d in main.items():
    x = d[d.y == 1].copy(); x["g"] = grp(x)
    x["kept"] = pd.cut(x.fg_frac, [-.01, .5, .9, 1.01], labels=["<50%", "50-90%", ">=90%"])
    for (g, k), z in x.groupby(["g", "kept"], observed=True):
        rows.append(dict(seed=s, g=g, kept=k, n=len(z), sens=(z.prob >= .5).mean()))
T = pd.DataFrame(rows)
print(T.pivot_table(index="g", columns="kept", values="sens", aggfunc="mean", observed=True).round(3).to_string())
print("\nmalignant images per cell (mean per seed)")
print(T.pivot_table(index="g", columns="kept", values="n", aggfunc="mean", observed=True).round(1).to_string())
print("""
How to read it:
  The INTACT block is the key one. If FST VI is still well below I-III and IV-V there,
  the cliff is not an artefact of the background removal. If the gap mostly vanishes on
  intact images, the preprocessing explains much of it and the paper's claim must change.
  Check the 'pos' column: intact FST VI has few malignant images.
""")
