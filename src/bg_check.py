"""
Did background removal work equally well for every Fitzpatrick type?
Reads the Kaggle Fitzpatrick17k copy (images in a 'background removed' folder), no GPU.

For each image: find the fill colour from the four corners, then measure
  - fg_frac   : fraction of pixels that are NOT the fill colour (the 'kept' area)
  - fg_lum    : mean brightness of the kept pixels (sanity: should fall with darker skin)
  - corner_ok : do the four corners agree on one fill colour?
Outputs: bg_check_per_image.csv, bg_check_summary.csv, bg_contact_sheet.png (+ printed tables).
Env: DATA_ROOT (default /kaggle/input), OUT_DIR (default /kaggle/working), N_PER_FST (0 = all), TOL.
"""
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

DATA = os.environ.get("DATA_ROOT", "/kaggle/input")
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
NPER = int(os.environ.get("N_PER_FST", 0))
TOL = int(os.environ.get("TOL", 12))          # colour distance counted as 'fill'
SIDE = 160
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200, "display.precision", 3)

csv = None
for c in sorted(glob.glob(f"{DATA}/**/*.csv", recursive=True)):
    try:
        cols = pd.read_csv(c, nrows=1).columns
    except Exception:
        continue
    if {"md5hash", "fitzpatrick_scale"} <= set(cols):
        csv = c
        break
if csv is None:
    raise FileNotFoundError(f"Fitzpatrick17k CSV not found under {DATA}")
meta = pd.read_csv(csv)
meta = meta[meta.fitzpatrick_scale.between(1, 6)].copy()
root = os.path.dirname(csv)
files = {}
for ext in ("jpg", "jpeg", "png", "JPG", "PNG"):
    for p in glob.glob(f"{root}/**/*.{ext}", recursive=True):
        files.setdefault(os.path.splitext(os.path.basename(p))[0], p)
meta["path"] = meta.md5hash.map(files)
print(f"[bg] csv={csv}\n[bg] rows with valid FST={len(meta)}  images found={meta.path.notna().sum()}")
meta = meta.dropna(subset=["path"])
if NPER:
    meta = meta.groupby("fitzpatrick_scale", group_keys=False).apply(
        lambda d: d.sample(min(len(d), NPER), random_state=0))
print(f"[bg] analysing {len(meta)} images\n", flush=True)


def analyse(path):
    im = Image.open(path)
    im.draft("RGB", (SIDE * 2, SIDE * 2))
    im = im.convert("RGB").resize((SIDE, SIDE))
    a = np.asarray(im).astype(int)
    k = 6
    corners = np.stack([a[:k, :k].reshape(-1, 3), a[:k, -k:].reshape(-1, 3),
                        a[-k:, :k].reshape(-1, 3), a[-k:, -k:].reshape(-1, 3)])
    med = np.median(corners, axis=1)                       # 4 corner colours
    fill = np.median(med, axis=0)
    ok = bool((np.abs(med - fill).max(axis=1) <= TOL).all())
    is_fill = (np.abs(a - fill).max(axis=2) <= TOL)
    fg = ~is_fill
    lum = a.mean(axis=2)
    return dict(fill_r=fill[0], fill_g=fill[1], fill_b=fill[2], corner_ok=ok,
                fg_frac=float(fg.mean()), fg_lum=float(lum[fg].mean()) if fg.any() else np.nan)


rows = []
for i, (h, p) in enumerate(zip(meta.md5hash, meta.path)):
    try:
        rows.append(dict(md5hash=h, **analyse(p)))
    except Exception as e:
        rows.append(dict(md5hash=h, err=str(e)[:60]))
    if i % 2000 == 0:
        print(f"[bg] {i}/{len(meta)}", flush=True)
R = pd.DataFrame(rows).merge(meta[["md5hash", "fitzpatrick_scale", "three_partition_label", "path"]], on="md5hash")
R["fst"] = R.fitzpatrick_scale.astype(int)
R["fill_kind"] = np.where(R[["fill_r", "fill_g", "fill_b"]].max(axis=1) < 30, "black",
                 np.where(R[["fill_r", "fill_g", "fill_b"]].min(axis=1) > 225, "white", "other"))
R.to_csv(f"{OUT}/bg_check_per_image.csv", index=False)
if "err" in R:
    print("[bg] unreadable images:", int(R.err.notna().sum()))

S = R.groupby("fst").agg(n=("md5hash", "size"),
                         fill_black=("fill_kind", lambda s: (s == "black").mean()),
                         fill_white=("fill_kind", lambda s: (s == "white").mean()),
                         corners_agree=("corner_ok", "mean"),
                         fg_frac_mean=("fg_frac", "mean"), fg_frac_median=("fg_frac", "median"),
                         fg_frac_p10=("fg_frac", lambda s: s.quantile(.10)),
                         near_empty=("fg_frac", lambda s: (s < 0.10).mean()),
                         near_full=("fg_frac", lambda s: (s > 0.98).mean()),
                         fg_lum=("fg_lum", "mean"))
S.to_csv(f"{OUT}/bg_check_summary.csv")
print("=" * 78)
print("BACKGROUND REMOVAL BY FITZPATRICK TYPE")
print("  fill_black/white = share of images whose fill colour is black / white")
print("  fg_frac          = share of pixels kept (not fill). p10 = 10th percentile image")
print("  near_empty       = images where <10% of pixels were kept (cut-out may have failed)")
print("  near_full        = images where >98% kept (background probably NOT removed)")
print("=" * 78)
print(S.round(3).to_string())

# does fg_frac differ by FST after taking the fill colour into account?
print("\nfg_frac by fill colour and FST (mean)")
print(R.pivot_table(index="fill_kind", columns="fst", values="fg_frac", aggfunc="mean").round(3).to_string())
print("\nimage counts by fill colour and FST")
print(R.pivot_table(index="fill_kind", columns="fst", values="md5hash", aggfunc="count").fillna(0).astype(int).to_string())

# contact sheet: 8 random images per FST (FST VI last), plus the 8 with the smallest kept area
from PIL import ImageDraw
rng = np.random.default_rng(0)
tile, per = 112, 8
sheet = Image.new("RGB", (tile * per, (tile + 14) * 7), "white")
d = ImageDraw.Draw(sheet)
y = 0
for f in range(1, 7):
    sub = R[R.fst == f]
    pick = sub.sample(min(per, len(sub)), random_state=0)
    d.text((2, y), f"FST {f} (random)", fill="black")
    for j, p in enumerate(pick.path):
        sheet.paste(Image.open(p).convert("RGB").resize((tile, tile)), (j * tile, y + 14))
    y += tile + 14
worst = R[R.fst == 6].nsmallest(per, "fg_frac")
d.text((2, y), "FST 6: smallest kept area", fill="black")
for j, p in enumerate(worst.path):
    sheet.paste(Image.open(p).convert("RGB").resize((tile, tile)), (j * tile, y + 14))
sheet.save(f"{OUT}/bg_contact_sheet.png")
print("\nwrote bg_check_per_image.csv, bg_check_summary.csv, bg_contact_sheet.png")
