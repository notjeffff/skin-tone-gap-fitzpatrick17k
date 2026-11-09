"""
Duplicate-leakage check using Fitzpatrick17k-C (Abhishek et al. 2025).
For each seed: a test image is 'leaked' if a duplicate of it (same cluster) was in that seed's training set.
Recompute per-FST results (a) excluding leaked test images, (b) on Fitzpatrick17k-C images only.
Usage: python analyze_dups.py <dir with probs_seed{N}{suffix}.csv> [suffix]
"""
import ast, glob, os, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

H = os.path.dirname(os.path.abspath(__file__))
P, SUF = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
NB = int(os.environ.get("NBOOT", 1000))
rng = np.random.default_rng(0)
pd.set_option("display.width", 200, "display.precision", 3)

off = pd.read_csv(f"{H}/official.csv"); off = off[off.fitzpatrick_scale.between(1, 6)]
fst = dict(zip(off.md5hash, off.fitzpatrick_scale.astype(int)))
names = pd.read_csv(f"{H}/f17k_names.csv"); n2m = dict(zip(names.filename, names.md5hash))
cl = [[n2m[n] for n in ast.literal_eval(l) if n in n2m] for l in open(f"{H}/complete_clusters.txt") if l.strip()]
cid = {h: i for i, c in enumerate(cl) for h in c}
members = {i: set(c) for i, c in enumerate(cl)}
keepC = set(pd.read_csv(f"{H}/f17k_C.csv").md5hash)

runs = {}
for f in glob.glob(f"{P}/probs_seed*{SUF}.csv"):
    m = re.fullmatch(rf"probs_seed(\d+){re.escape(SUF)}\.csv", os.path.basename(f))
    if m: runs[int(m.group(1))] = pd.read_csv(f)
if not runs: sys.exit(f"no probs_seed*{SUF}.csv in {P}")
print(f"seeds {sorted(runs)}  suffix='{SUF}'")

light_all = {h for h, f in fst.items() if f <= 3}
for s, d in runs.items():
    d["md5hash"] = d.path.map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    train = light_all - set(d.loc[d.fst <= 3, "md5hash"])          # baseline: train = light images not in test
    if "hold45" in SUF:                                              # holdout runs also trained on every FST IV-V image
        train |= {h for h, f in fst.items() if f in (4, 5)}         # (size-matched: superset of its light images -> conservative)
    def leaked(h):
        i = cid.get(h)
        return i is not None and bool((members[i] - {h}) & train)
    d["leaked"] = d.md5hash.map(leaked)
    d["inC"] = d.md5hash.isin(keepC)


def grp(f): return np.where(f <= 3, "I-III", np.where(f <= 5, "IV-V", "VI"))
def sens(p, y): return float((p[y == 1] >= .5).mean()) if (y == 1).any() else np.nan
def auc(p, y): return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


print("\nleaked test images (a duplicate was in training), mean per seed")
L = pd.concat([d.assign(seed=s) for s, d in runs.items()], ignore_index=True)
L["g"] = grp(L.fst)
L["leaked_mal"] = L.leaked & (L.y == 1)
L["mal"] = L.y == 1
print(L.groupby("g")[["leaked", "mal", "leaked_mal"]].agg(["size", "sum"]).iloc[:, [0, 1, 3, 5]]
      .set_axis(["test", "leaked", "malignant", "leaked_malignant"], axis=1).div(len(runs)).round(1).to_string())

for name, keep in (("ALL TEST IMAGES", lambda d: d), ("EXCLUDING LEAKED TEST IMAGES", lambda d: d[~d.leaked]),
                   ("FITZPATRICK17K-C IMAGES ONLY", lambda d: d[d.inC]),
                   ("FITZPATRICK17K-C AND NOT LEAKED", lambda d: d[d.inC & ~d.leaked])):
    rows = []
    for s, d in runs.items():
        x = keep(d)
        for f, z in x.groupby("fst"):
            rows.append(dict(seed=s, fst=int(f), n=len(z), pos=int(z.y.sum()),
                             sens=sens(z.prob.values, z.y.values), auc=auc(z.prob.values, z.y.values)))
    T = pd.DataFrame(rows)
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
    print(T.groupby("fst").agg(n=("n", "mean"), pos=("pos", "mean"), sens=("sens", "mean"),
                               sens_sd=("sens", "std"), auc=("auc", "mean"), auc_sd=("auc", "std")).round(3).to_string())
    pairs = (("I-III", "VI"),) if "hold45" in SUF else (("I-III", "VI"), ("I-III", "IV-V"), ("IV-V", "VI"))
    for lo, hi in pairs:
        for metric, st in (("sensitivity", sens), ("AUC", auc)):
            res = []
            for s, d in runs.items():
                x = keep(d); g = grp(x.fst)
                A, B = x[g == lo], x[g == hi]
                pt = st(A.prob.values, A.y.values) - st(B.prob.values, B.y.values)
                bs = []
                for _ in range(NB):
                    i = rng.integers(0, len(A), len(A)); j = rng.integers(0, len(B), len(B))
                    bs.append(st(A.prob.values[i], A.y.values[i]) - st(B.prob.values[j], B.y.values[j]))
                lo_, hi_ = np.nanpercentile(bs, [2.5, 97.5]); res.append((pt, lo_ > 0 or hi_ < 0))
            print(f"  {lo:>5s} minus {hi:<4s} {metric:11s} mean gap {np.mean([r[0] for r in res]):+.3f}  "
                  f"CI excludes 0 in {sum(r[1] for r in res)}/{len(res)} seeds")
