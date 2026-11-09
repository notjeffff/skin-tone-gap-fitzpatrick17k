"""Gather everything the post-hoc analyses need into the working folder (CPU, ~1 min, internet on)."""
import glob, json, os, shutil, urllib.request
IN = os.environ.get("INPUT_ROOT", "/kaggle/input")
W = os.environ.get("WORK", "/kaggle/working")
os.makedirs(f"{W}/zipin", exist_ok=True)
RAW = ("https://raw.githubusercontent.com/kakumarabhishek/Corrected-Skin-Image-Datasets/main/"
       "Fitzpatrick17k/Fitzpatrick17k_Analysis/")
files = {"complete_clusters.txt": "AnalysesOutputFiles/complete_clusters.txt",
         "f17k_names.csv": "DatasetSplits/SimThresh_F_A2_F_0.99_0.70_FC_F_KeepOne_Out_F_OutThresh_None_0FST_F.csv",
         "f17k_C.csv": "DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv"}
for name, path in files.items():
    if not os.path.exists(f"{W}/{name}"):
        urllib.request.urlretrieve(RAW + path, f"{W}/{name}")
    print(f"[setup] {name}: {os.path.getsize(f'{W}/{name}'):,} bytes")
csv = [c for c in glob.glob(f"{IN}/**/*.csv", recursive=True) if os.path.basename(c) == "fitzpatrick17k (1).csv"]
if not csv:
    raise FileNotFoundError("Attach the Nazmus Resan fitzpatrick17k dataset (for its CSV)")
shutil.copy(csv[0], f"{W}/official.csv")
n = skipped = 0
for jf in glob.glob(f"{IN}/**/results_seed*.json", recursive=True):
    if "smoke" in jf:
        continue
    if "original" not in str(json.load(open(jf)).get("config", {}).get("img_root", "")).lower():
        skipped += 1; continue
    pf = jf.replace("results_", "probs_").replace(".json", ".csv")
    if os.path.exists(pf) and not os.path.exists(f"{W}/zipin/{os.path.basename(pf)}"):
        shutil.copy(pf, f"{W}/zipin/"); n += 1
print(f"[setup] copied {n} original-image probability files (skipped {skipped} masked-image runs)")
print(sorted(os.listdir(f"{W}/zipin")))
ddi = glob.glob(f"{IN}/**/probs_ddi_all.csv", recursive=True)
if ddi:
    shutil.copy(ddi[0], f"{W}/probs_ddi_all.csv"); print("[setup] probs_ddi_all.csv copied")
else:
    print("[setup] probs_ddi_all.csv not found: attach notebook 03 (DDI) to run the DDI analysis")
