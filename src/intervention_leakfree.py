"""Table III: baseline vs +FST IV-V vs size-matched on the same leak-free FST VI images.
Also writes leak-free probability files to lf/out for the Fig. 3 script."""
import ast, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
n2m = dict(zip(*[pd.read_csv("f17k_names.csv")[c] for c in ("filename", "md5hash")]))
o = pd.read_csv("official.csv"); o = o[o.fitzpatrick_scale.between(1, 6)]
fst = dict(zip(o.md5hash, o.fitzpatrick_scale))
cl = [[n2m[n] for n in ast.literal_eval(l) if n in n2m] for l in open("complete_clusters.txt") if l.strip()]
cid = {h: i for i, c in enumerate(cl) for h in c}; mem = {i: set(c) for i, c in enumerate(cl)}
keepC = set(pd.read_csv("f17k_C.csv").md5hash)
light = {h for h, f in fst.items() if f <= 3}; ivv = {h for h, f in fst.items() if f in (4, 5)}
os.makedirs("lf/out", exist_ok=True)
stem = lambda p: os.path.splitext(os.path.basename(str(p)))[0]
rows = []
for s in range(3):
    P = {k: pd.read_csv(f"zipin/probs_seed{s}{k}.csv") for k in ("", "_hold45", "_hold45_sm")}
    for d in P.values(): d["h"] = d.path.map(stem)
    train = (light - set(P[""].loc[P[""].fst <= 3, "h"])) | ivv
    bad = {h for h in P[""].loc[P[""].fst == 6, "h"] if (i := cid.get(h)) is not None and (mem[i] - {h}) & train}
    for k, d in P.items():
        d[~((d.fst == 6) & d.h.isin(bad))].drop(columns="h").to_csv(f"lf/out/probs_seed{s}{k}.csv", index=False)
    six = P[""][P[""].fst == 6]
    for sub, H in (("all FST VI", set(six.h)), ("leak-free FST VI", set(six.h) - bad),
                   ("F17k-C & leak-free", (set(six.h) - bad) & keepC)):
        for k, lab in (("", "baseline"), ("_hold45", "+IV-V"), ("_hold45_sm", "+IV-V same size")):
            z = P[k][P[k].h.isin(H) & (P[k].fst == 6)]
            L = P[k][P[k].fst <= 3]
            rows.append(dict(seed=s, subset=sub, model=lab, n=len(z), pos=int(z.y.sum()),
                             sens=(z.prob[z.y == 1] >= .5).mean(), auc=roc_auc_score(z.y, z.prob),
                             light_sens=(L.prob[L.y == 1] >= .5).mean(), light_spec=(L.prob[L.y == 0] < .5).mean(),
                             light_auc=roc_auc_score(L.y, L.prob)))
    print(f"seed {s}: {len(bad)} FST VI test images have a near-duplicate in the holdout training set")
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.groupby(["subset", "model"], sort=False)[["n", "pos", "sens", "auc", "light_sens", "light_spec", "light_auc"]]
      .mean().round(3).to_string())
for sub in T.subset.unique():
    x = T[T.subset == sub]
    for m in ("+IV-V", "+IV-V same size"):
        g = (x[x.model == m].set_index("seed")[["sens", "auc"]] - x[x.model == "baseline"].set_index("seed")[["sens", "auc"]])
        print(f"{sub:20s} {m:16s} per-seed gain  sens {g.sens.round(3).tolist()}  auc {g.auc.round(3).tolist()}")
