"""5-fold split over ALL skin types, stratified by FST x malignant, grouped by
Fitzpatrick17k-C duplicate clusters so near-duplicates never cross folds.
Writes folds.csv (md5hash, fold). Needs: CSV_PATH, complete_clusters.txt, f17k_names.csv."""
import ast, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
CSV = os.environ["CSV_PATH"]; K = int(os.environ.get("CV_K", 5))
d = pd.read_csv(CSV); d = d[pd.to_numeric(d.fitzpatrick_scale, errors="coerce").between(1, 6)].copy()
d["y"] = d.three_partition_label.astype(str).str.lower().str.contains("malignant").astype(int)
n2m = dict(zip(*[pd.read_csv("f17k_names.csv")[c] for c in ("filename", "md5hash")]))
grp = {}
for i, l in enumerate(open("complete_clusters.txt")):
    if l.strip():
        for n in ast.literal_eval(l):
            if n in n2m: grp[n2m[n]] = f"c{i}"
d["g"] = [grp.get(h, h) for h in d.md5hash]
strata = d.fitzpatrick_scale.astype(int) * 2 + d.y
d["fold"] = -1
for k, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=0).split(d, strata, d.g)):
    d.iloc[te, d.columns.get_loc("fold")] = k
d[["md5hash", "fold"]].to_csv("folds.csv", index=False)
print(f"{d.g.nunique()} groups ({(d.g.str.startswith('c')).sum()} images in duplicate clusters)")
print("malignant test images per fold x FST")
print(pd.crosstab(d.fold, d.fitzpatrick_scale, values=d.y, aggfunc="sum").to_string())
assert d.groupby("g").fold.nunique().max() == 1, "a duplicate cluster spans two folds"
print("OK: no duplicate cluster spans two folds")
