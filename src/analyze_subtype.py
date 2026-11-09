"""
Subtype-stratified check: is the FST 6 cliff a skin-tone effect, a cancer-subtype
effect, or both? CPU only, seconds.

In a Kaggle cell (same setup as analyze_fst.py):
    os.environ["OUT_DIR"] = "<folder holding probs_*.csv>"
    %run analyze_subtype.py
Needs the Fitzpatrick17k CSV under /kaggle/input (for the diagnosis column).

All counts are small (FST 6 has 61 malignant images in total). Read the output as
descriptive, not as a hypothesis test.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

OUT  = os.environ.get("OUT_DIR", "/kaggle/working")
DATA = os.environ.get("DATA_ROOT", "/kaggle/input")
pd.set_option("display.width", 200, "display.precision", 3)

main = {}
for f in glob.glob(f"{OUT}/probs_seed*.csv"):
    m = re.fullmatch(r"probs_seed(\d+)\.csv", os.path.basename(f))
    if m:
        main[int(m.group(1))] = pd.read_csv(f)
main = dict(sorted(main.items()))
if not main:
    raise SystemExit(f"No probs_seed*.csv in {OUT}")

csvs = [c for c in glob.glob(f"{DATA}/**/*.csv", recursive=True)
        if "fitz" in os.path.basename(c).lower()]
if os.environ.get("CSV_PATH"):
    csvs = [os.environ["CSV_PATH"]]
if not csvs:
    raise SystemExit(f"Fitzpatrick17k CSV not found under {DATA}")
meta = (pd.read_csv(csvs[0])[["md5hash", "label"]]
        .rename(columns={"label": "dx"}).drop_duplicates("md5hash"))


def family(dx):
    s = str(dx).lower()
    if "kaposi" in s:
        return "kaposi sarcoma"
    if "squamous" in s:
        return "squamous cell carcinoma"
    if "basal cell" in s:
        return "basal cell carcinoma"
    if "melanoma" in s or "lentigo maligna" in s:
        return "melanoma (all types)"
    if "mycosis" in s:
        return "mycosis fungoides"
    return "other malignant"


def prep(d):
    d = d.copy()
    d["md5hash"] = d["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    d = d.merge(meta, on="md5hash", how="left")
    d = d[d.y == 1].copy()
    d["fam"] = d["dx"].map(family)
    d["grp"] = np.where(d.fst <= 3, "FST1-3", np.where(d.fst <= 5, "FST4-5", "FST6"))
    d["hit"] = d.prob >= 0.5
    return d


per_seed = []
for s, d in main.items():
    p = prep(d)
    g = p.groupby(["fam", "grp"]).agg(n=("hit", "size"), sens=("hit", "mean")).reset_index()
    g["seed"] = s
    per_seed.append(g)
R = pd.concat(per_seed)
GRPS = ["FST1-3", "FST4-5", "FST6"]

print("=" * 74)
print("1. MALIGNANT IMAGES PER SUBTYPE AND SKIN GROUP (mean per seed)")
print("=" * 74)
N = R.pivot_table(index="fam", columns="grp", values="n", aggfunc="mean").reindex(columns=GRPS)
N = N.sort_values("FST6", ascending=False)
print(N.round(1).to_string())

print("\n" + "=" * 74)
print("2. SENSITIVITY PER SUBTYPE AND SKIN GROUP (mean over seeds, threshold 0.5)")
print("   Compare down each row: does the same subtype get worse at FST 6?")
print("=" * 74)
S = R.pivot_table(index="fam", columns="grp", values="sens", aggfunc="mean").reindex(columns=GRPS)
print(S.reindex(N.index).round(3).to_string())

print("\n" + "=" * 74)
print("3. STANDARDISED SENSITIVITY (same subtype mix on both sides)")
print("=" * 74)


def standardised(tab, mix_grp, target_grp):
    t = tab.pivot(index="fam", columns="grp", values="sens")
    n = tab.pivot(index="fam", columns="grp", values="n")
    if mix_grp not in n or target_grp not in t:
        return np.nan
    ok = t[target_grp].notna() & n[mix_grp].notna()
    if not ok.any():
        return np.nan
    return float((n.loc[ok, mix_grp] * t.loc[ok, target_grp]).sum() / n.loc[ok, mix_grp].sum())


def observed(tab, grp):
    t = tab[tab.grp == grp]
    return float((t.n * t.sens).sum() / t.n.sum()) if len(t) else np.nan


rows = []
for s, tab in R.groupby("seed"):
    rows.append(dict(
        seed=s,
        FST6_observed=observed(tab, "FST6"),
        FST6_at_FST1_3_mix=standardised(tab, "FST1-3", "FST6"),
        FST1_3_observed=observed(tab, "FST1-3"),
        FST1_3_at_FST6_mix=standardised(tab, "FST6", "FST1-3"),
    ))
Z = pd.DataFrame(rows)
print(Z.round(3).to_string(index=False))
print("\nmean over seeds")
print(Z.drop(columns="seed").mean().round(3).to_string())
print("""
How to read it:
  FST6_observed vs FST6_at_FST1_3_mix   if the second is much higher, FST 6 looks bad
                                        mainly because of WHICH cancers it contains.
  FST1_3_observed vs FST1_3_at_FST6_mix if the second is much lower, light-skin images
                                        also score badly on the FST 6 subtype mix.
  If FST6_at_FST1_3_mix is still far below FST1_3_observed, a skin-tone effect remains.
""")
