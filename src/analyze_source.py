"""
Source-site confound check: is the FST 6 cliff a skin-tone effect, or does it
come from FST 6 images being drawn from a different source site (different
photo style) than the rest?

In a Kaggle cell (same setup as analyze_fst.py):
    os.environ["OUT_DIR"] = "<folder holding probs_*.csv>"
    %run analyze_source.py
Needs the Fitzpatrick17k CSV (with its `url` column) under /kaggle/input.
CPU only, seconds. Counts for FST 6 are small, so read it as descriptive.
"""
import glob
import os
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT  = os.environ.get("OUT_DIR", "/kaggle/working")
DATA = os.environ.get("DATA_ROOT", "/kaggle/input")
MIN_IMAGES = 100          # sites with fewer test images are pooled into "other"
pd.set_option("display.width", 200, "display.precision", 3)


def sens(p, y):
    m = y == 1
    return float((p[m] >= 0.5).mean()) if m.any() else np.nan


def auc(p, y):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


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
raw = pd.read_csv(csvs[0])
if "url" not in raw.columns:
    raise SystemExit(f"No `url` column. Columns: {list(raw.columns)}")


def site_of(u):
    try:
        host = urlparse(str(u)).netloc.lower().replace("www.", "")
    except Exception:
        host = ""
    return host or "unknown"


meta = raw[["md5hash", "url"]].copy()
meta["site"] = meta["url"].map(site_of)
meta = meta[["md5hash", "site"]].drop_duplicates("md5hash")


def tag(d, keep_sites=None):
    d = d.copy()
    d["md5hash"] = d["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    d = d.merge(meta, on="md5hash", how="left")
    d["site"] = d["site"].fillna("unknown")
    d["grp"] = np.where(d.fst <= 3, "FST1-3", np.where(d.fst <= 5, "FST4-5", "FST6"))
    if keep_sites is not None:
        d.loc[~d.site.isin(keep_sites), "site"] = "other"
    return d


first = tag(main[min(main)])
counts = first.site.value_counts()
keep = list(counts[counts >= MIN_IMAGES].index)
print("sites with >=", MIN_IMAGES, "test images:", dict(counts[counts >= MIN_IMAGES]))

print("\n" + "=" * 74)
print("1. SOURCE-SITE MIX BY SKIN GROUP (% of that group's test images, seed 0)")
print("=" * 74)
first = tag(main[min(main)], keep)
ct = pd.crosstab(first.site, first.grp).reindex(columns=["FST1-3", "FST4-5", "FST6"])
print(ct.join((ct / ct.sum() * 100).round(1).add_suffix("_%")).to_string())
print("\nmalignant prevalence by site and skin group")
print(first.pivot_table(index="site", columns="grp", values="y", aggfunc="mean")
      .reindex(columns=["FST1-3", "FST4-5", "FST6"]).round(3).to_string())

rows = []
for s, d in main.items():
    d = tag(d, keep)
    for (site, grp), g in d.groupby(["site", "grp"]):
        p, y = g.prob.values, g.y.values
        rows.append(dict(seed=s, site=site, grp=grp, n=len(g), pos=int(y.sum()),
                         sens=sens(p, y), auc=auc(p, y)))
R = pd.DataFrame(rows)
G = ["FST1-3", "FST4-5", "FST6"]

print("\n" + "=" * 74)
print("2. SENSITIVITY AND AUC BY SITE AND SKIN GROUP (mean over seeds)")
print("   Does the FST 6 drop appear inside EACH site?")
print("=" * 74)
for col, label in (("pos", "malignant images (mean per seed)"),
                   ("sens", "sensitivity"), ("auc", "AUC")):
    print(f"\n{label}")
    print(R.pivot_table(index="site", columns="grp", values=col, aggfunc="mean")
          .reindex(columns=G).round(3).to_string())

print("\n" + "=" * 74)
print("3. WITHIN-SITE GAP: FST1-3 minus FST6 (mean over seeds)")
print("=" * 74)
P_s = R.pivot_table(index="site", columns="grp", values="sens", aggfunc="mean")
P_a = R.pivot_table(index="site", columns="grp", values="auc", aggfunc="mean")
gap = pd.DataFrame({"sens_gap": P_s["FST1-3"] - P_s["FST6"],
                    "auc_gap": P_a["FST1-3"] - P_a["FST6"]})
print(gap.round(3).to_string())
print("""
How to read it:
  A gap that is large inside every site that has FST 6 images means the cliff is
  not a source-site artefact. A gap that appears in only one site, or vanishes
  inside sites, means photo style / provenance is a competing explanation.
  Ignore rows where FST 6 has fewer than ~10 malignant images.
""")
