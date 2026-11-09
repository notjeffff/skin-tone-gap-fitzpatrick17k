import sys
import numpy as np, pandas as pd
P = pd.read_csv(sys.argv[1])
rng = np.random.default_rng(0)
G = ["I-II", "III-IV", "V-VI"]
def sens_at_spec(p, y, spec):
    neg = np.sort(p[y == 0]); pos = p[y == 1]
    thr = np.quantile(neg, spec)           # spec fraction of negatives fall <= thr
    return float((pos > thr).mean())
def boot_gap(L, D, spec, nb=500):
    f = lambda d: sens_at_spec(d.prob.values, d.y.values, spec)
    pt = f(L) - f(D); g = []
    for _ in range(nb):
        a = L.iloc[rng.integers(0, len(L), len(L))]; b = D.iloc[rng.integers(0, len(D), len(D))]
        g.append(f(a) - f(b))
    return pt, np.percentile(g, 2.5), np.percentile(g, 97.5)
for spec in (0.8, 0.9):
    print(f"\n=== sensitivity at {int(spec*100)}% specificity, set WITHIN each group (mean over seeds) ===")
    rows = []
    for (cfg, s, g), d in P.groupby(["cfg", "seed", "grp"]):
        rows.append(dict(cfg=cfg, seed=s, grp=g, sens=sens_at_spec(d.prob.values, d.y.values, spec)))
    T = pd.DataFrame(rows)
    print(T.pivot_table(index="cfg", columns="grp", values="sens").reindex(columns=G).round(3).to_string())
    print(f"--- contrasts at {int(spec*100)}% spec (bootstrap per seed) ---")
    for cfg, d in P.groupby("cfg"):
        for lo, hi in (("I-II", "V-VI"), ("III-IV", "V-VI")):
            res = [boot_gap(x[x.grp == lo], x[x.grp == hi], spec) for _, x in d.groupby("seed")]
            sig = sum(1 for r in res if r[1] > 0 or r[2] < 0)
            print(f"{cfg:20s} {lo:>6s} minus {hi:<6s} mean gap {np.mean([r[0] for r in res]):+.3f}  CI excludes 0 in {sig}/{len(res)}")
# ensemble over seeds (mean prob per image) for the two main configs
print("\n=== seed-ensemble AUC (mean prob per image) ===")
from sklearn.metrics import roc_auc_score
for cfg, d in P.groupby("cfg"):
    E = d.groupby(["file", "grp", "y"], as_index=False).prob.mean()
    print(cfg, {g: round(roc_auc_score(x.y, x.prob), 3) for g, x in E.groupby("grp")})
