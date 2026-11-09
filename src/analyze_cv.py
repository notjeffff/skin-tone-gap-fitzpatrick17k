"""All-skin-type 5-fold CV: pooled out-of-fold results per FST with 95% CIs."""
import glob, os, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
OUT = os.environ.get("OUT_DIR", "/kaggle/working"); NB = int(os.environ.get("NBOOT", 2000))
SUF = os.environ.get("CV_SUFFIX", "")            # "" for ResNet-50, "_effb0" for EfficientNet
rng = np.random.default_rng(0)
fs = sorted(f for f in glob.glob(f"{OUT}/probs_seed*{SUF}_cv*.csv") if re.search(rf"{re.escape(SUF)}_cv\d+\.csv$", f))
P = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
print(f"{len(fs)} folds, {len(P)} out-of-fold predictions, {P.path.nunique()} unique images")
sens = lambda p, y: float((p[y == 1] >= .5).mean())
auc = lambda p, y: float(roc_auc_score(y, p))
rows = []
for f, d in P.groupby("fst"):
    p, y = d.prob.values, d.y.values; k = int((p[y == 1] >= .5).sum()); n = int(y.sum())
    lo, hi = wilson(k, n)
    bs = [auc(p[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(NB))]
    rows.append(dict(fst=int(f), n=len(d), malignant=n, sens=k / n, sens_lo=lo, sens_hi=hi,
                     spec=float((p[y == 0] < .5).mean()), auc=auc(p, y),
                     auc_lo=np.percentile(bs, 2.5), auc_hi=np.percentile(bs, 97.5)))
T = pd.DataFrame(rows); pd.set_option("display.width", 200)
print("\nPER-TYPE, ALL-SKIN-TYPE TRAINING (pooled out-of-fold; Wilson CI for sens, bootstrap CI for AUC)")
print(T.round(3).to_string(index=False))
T.to_csv(f"{OUT}/cv_per_type{SUF}.csv", index=False)
grp = lambda f: np.where(f <= 3, "I-III", np.where(f <= 5, "IV-V", "VI"))
P["g"] = grp(P.fst)
print("\nCONTRASTS (bootstrap over images, 95% CI)")
for lo, hi in (("I-III", "VI"), ("I-III", "IV-V"), ("IV-V", "VI")):
    A, B = P[P.g == lo], P[P.g == hi]
    for name, st in (("sensitivity", sens), ("AUC", auc)):
        pt = st(A.prob.values, A.y.values) - st(B.prob.values, B.y.values)
        bs = []
        for _ in range(NB):
            i = rng.integers(0, len(A), len(A)); j = rng.integers(0, len(B), len(B))
            bs.append(st(A.prob.values[i], A.y.values[i]) - st(B.prob.values[j], B.y.values[j]))
        l, h = np.percentile(bs, [2.5, 97.5])
        print(f"  {lo:>5s} minus {hi:<4s} {name:11s} {pt:+.3f}  [{l:+.3f}, {h:+.3f}]")
