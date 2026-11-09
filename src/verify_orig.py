"""Check every attached original-image dataset, pick the best, and stop if FST VI coverage is poor."""
import glob, hashlib, os, random
import numpy as np, pandas as pd
from PIL import Image

IN = os.environ.get("INPUT_ROOT", "/kaggle/input")
pd.set_option("display.width", 200)
csvs = [c for c in sorted(glob.glob(f"{IN}/**/*.csv", recursive=True))
        if {"md5hash", "fitzpatrick_scale"} <= set(pd.read_csv(c, nrows=1).columns)]
csvs.sort(key=lambda c: "original" in c.lower())          # prefer a CSV outside the image datasets
if not csvs:
    raise FileNotFoundError("Fitzpatrick17k CSV not found: attach the Nazmus Resan dataset (for its CSV)")
meta = pd.read_csv(csvs[0])
v = meta[meta.fitzpatrick_scale.between(1, 6)].copy()
print(f"[orig] csv: {csvs[0]}")

if os.environ.get("IMG_ROOT"):
    roots = [os.environ["IMG_ROOT"]]
else:
    cand = sorted({d.rstrip("/") for d in glob.glob(f"{IN}/**/", recursive=True) if "original" in d.lower()}, key=len)
    roots = [d for d in cand if not any(d != r and d.startswith(r + "/") for r in cand)]
if not roots:
    raise FileNotFoundError("No original-image dataset found: attach fitzpatrick17k_original and/or asosenge's Fitzpatrick17K")


def check(root):
    files = {}
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        for p in glob.glob(f"{root}/**/*.{ext}", recursive=True):
            files[os.path.splitext(os.path.basename(p))[0]] = p
    x = v.assign(found=v.md5hash.isin(files))
    cov = x.groupby("fitzpatrick_scale").agg(rows=("found", "size"), found=("found", "sum"))
    mal = x[x.three_partition_label == "malignant"]
    cov["malignant_rows"] = mal.groupby("fitzpatrick_scale").size()
    cov["malignant_found"] = mal[mal.found].groupby("fitzpatrick_scale").size()
    cov = cov.fillna(0).astype(int)
    cov["coverage"] = (cov.found / cov.rows).round(4)
    samp = random.Random(0).sample(list(files.items()), min(300, len(files))) if files else []
    match = sum(hashlib.md5(open(p, "rb").read()).hexdigest() == h for h, p in samp)
    black = 0
    for h, p in samp:
        a = np.asarray(Image.open(p).convert("RGB").resize((64, 64))).astype(int)
        c = np.concatenate([a[:4, :4], a[:4, -4:], a[-4:, :4], a[-4:, -4:]]).reshape(-1, 3)
        black += int(np.median(c.max(axis=1)) < 30)
    print("\n" + "=" * 70 + f"\n{root}\n" + "=" * 70)
    print(f"images: {len(files)}")
    print(cov.to_string())
    print(f"file fingerprint matches name: {match}/{len(samp)}   (near {len(samp)} = untouched originals)")
    print(f"black-corner images in sample: {black}/{len(samp)}   (originals ~6% from dark photo borders; masked copy ~54%)")
    return dict(root=root, cov6=cov.coverage.get(6, 0), covmin=cov.coverage.min(),
                found=int(cov.found.sum()), match=match / max(len(samp), 1), black=black / max(len(samp), 1))


R = [check(r) for r in roots]
best = max(R, key=lambda r: (r["cov6"] >= .95, r["match"] >= .95, r["cov6"], r["match"], r["found"]))
print("\n" + "=" * 70)
print(pd.DataFrame(R).round(3).to_string(index=False))
print(f"\nchosen: {best['root']}")
if best["cov6"] < 0.95 or best["covmin"] < 0.90 or best["match"] < 0.95:
    raise RuntimeError("No attached dataset is clean and complete enough - stopping before training. Check the attached datasets.")
os.environ["IMG_ROOT"] = best["root"]
os.environ["CSV_PATH"] = csvs[0]
print("OK - IMG_ROOT set. Training cells will use only these images.")
