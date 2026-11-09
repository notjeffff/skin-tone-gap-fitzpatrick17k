"""
Paper figures at IEEE width (column 3.5 in, page 7.16 in), as vector PDF + 300 dpi PNG.
Fonts are embedded as TrueType (pdf.fonttype 42), which IEEE PDF checks require.

In a Kaggle cell:
    os.environ["OUT_DIR"] = "<folder holding probs_*.csv>"
    %run make_figures.py
Writes to FIG_DIR (default /kaggle/working/figs). Download that folder.

fig1_per_fst        sensitivity and AUC by Fitzpatrick level, mean +/- SD, dots = seeds
fig2_subtype        sensitivity by cancer subtype and skin group (needs the CSV; descriptive)
fig3_recovery       FST 6 recovery: baseline vs +FST 4,5 (holdout) vs oversample FST 6,
                    each compared on exactly the same test images
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT  = os.environ.get("OUT_DIR", "/kaggle/working")
DATA = os.environ.get("DATA_ROOT", "/kaggle/input")
FIG  = os.environ.get("FIG_DIR", "/kaggle/working/figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
BASE, HI, REF = "#7A9CC6", "#D55E00", "#333333"      # FST 6 gets colour AND hatching
GRP_COL = {"FST1-3": "#7A9CC6", "FST4-5": "#E69F00", "FST6": "#D55E00"}
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def sens(p, y):
    m = y == 1
    return float((p[m] >= 0.5).mean()) if m.any() else np.nan


def auc(p, y):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def read(suffix):
    found = {}
    for f in glob.glob(f"{OUT}/probs_seed*{suffix}.csv"):
        m = re.fullmatch(rf"probs_seed(\d+){re.escape(suffix)}\.csv", os.path.basename(f))
        if m:
            found[int(m.group(1))] = pd.read_csv(f)
    return dict(sorted(found.items()))


main, mit, hold = read(""), read("_mitigate6"), read("_hold45")
hold_sm = read("_hold45_sm")
if not main:
    raise SystemExit(f"No probs_seed*.csv in {OUT}")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{FIG}/{name}.pdf / .png")


def dots(ax, x, vals, color="black"):
    vals = np.asarray(vals, float)
    off = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else np.zeros(1)
    ax.scatter(x + off, vals, s=5, color=color, zorder=3, linewidths=0)


# ---------------------------------------------------------------- Figure 1
rows = []
for s, d in main.items():
    for lvl, g in d.groupby("fst"):
        p, y = g.prob.values, g.y.values
        rows.append(dict(seed=s, fst=int(lvl), sens=sens(p, y), auc=auc(p, y)))
    L = d[d.fst <= 3]
    rows.append(dict(seed=s, fst=0, sens=sens(L.prob.values, L.y.values),
                     auc=auc(L.prob.values, L.y.values)))      # fst 0 = pooled FST 1-3
T = pd.DataFrame(rows)
levels = [l for l in sorted(T.fst.unique()) if l > 0]

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
for ax, col, ylab in ((axes[0], "sens", "Sensitivity"), (axes[1], "auc", "AUC")):
    for i, l in enumerate(levels):
        v = T[T.fst == l][col].values
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=2,
               color=HI if l == 6 else BASE, hatch="///" if l == 6 else None,
               edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.6))
        dots(ax, i, v)
    ref = T[T.fst == 0][col].mean()
    ax.axhline(ref, ls="--", lw=0.7, color=REF)
    ax.text(len(levels) - 0.5, ref + 0.02, "FST I–III pooled", ha="right", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([ROMAN[l] for l in levels])
    ax.set_xlabel("Fitzpatrick skin type")
    ax.set_ylabel(ylab)
    ax.set_ylim(0, 1)
axes[0].set_title("(a) Sensitivity at threshold 0.5", loc="left")
axes[1].set_title("(b) AUC", loc="left")
fig.tight_layout()
save(fig, "fig1_per_fst")

# ---------------------------------------------------------------- Figure 2
csvs = [c for c in glob.glob(f"{DATA}/**/*.csv", recursive=True)
        if "fitz" in os.path.basename(c).lower()]
if os.environ.get("CSV_PATH"):
    csvs = [os.environ["CSV_PATH"]]
if csvs:
    meta = (pd.read_csv(csvs[0])[["md5hash", "label"]]
            .rename(columns={"label": "dx"}).drop_duplicates("md5hash"))

    def family(dx):
        s = str(dx).lower()
        if "kaposi" in s:
            return "Kaposi\nsarcoma"
        if "squamous" in s:
            return "Squamous\ncell"
        if "basal cell" in s:
            return "Basal\ncell"
        if "melanoma" in s or "lentigo maligna" in s:
            return "Melanoma"
        return None

    per = []
    for s, d in main.items():
        d = d.copy()
        d["md5hash"] = d["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
        d = d.merge(meta, on="md5hash", how="left")
        d = d[d.y == 1]
        d["fam"] = d["dx"].map(family)
        d = d.dropna(subset=["fam"])
        d["grp"] = np.where(d.fst <= 3, "FST1-3", np.where(d.fst <= 5, "FST4-5", "FST6"))
        d["hit"] = d.prob >= 0.5
        g = d.groupby(["fam", "grp"]).agg(n=("hit", "size"), sens=("hit", "mean")).reset_index()
        g["seed"] = s
        per.append(g)
    R = pd.concat(per)
    S = R.pivot_table(index="fam", columns="grp", values="sens", aggfunc="mean")
    N = R.pivot_table(index="fam", columns="grp", values="n", aggfunc="mean")
    order = [f for f in N.sort_values("FST6", ascending=False).index if N.loc[f, "FST6"] >= 5]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    w = 0.26
    for j, grp in enumerate(["FST1-3", "FST4-5", "FST6"]):
        xs = np.arange(len(order)) + (j - 1) * w
        ax.bar(xs, [S.loc[f, grp] for f in order], w, color=GRP_COL[grp],
               hatch="///" if grp == "FST6" else None, edgecolor="black", linewidth=0.5,
               label={"FST1-3": "FST I–III", "FST4-5": "FST IV–V", "FST6": "FST VI"}[grp])
    for i, f in enumerate(order):
        ax.text(i + w, S.loc[f, "FST6"] + 0.02, f"n={int(round(N.loc[f, 'FST6']))}",
                ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_ylabel("Sensitivity")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save(fig, "fig2_subtype")
else:
    print("Fitzpatrick17k CSV not found - skipped fig2_subtype")

# ---------------------------------------------------------------- Figure 3
pairs = []     # (label, [per-seed sens], [per-seed auc], colour, hatch)
if hold or mit:
    def collect(src, subset_fn):
        out = []
        for s, m in src.items():
            if s not in main:
                continue
            b, t = subset_fn(main[s], m)
            out.append((sens(b.prob.values, b.y.values), auc(b.prob.values, b.y.values),
                        sens(t.prob.values, t.y.values), auc(t.prob.values, t.y.values)))
        return np.array(out)

    def hold_fn(b, h):
        return b[b.fst == 6], h[h.fst == 6]

    def mit_fn(b, m):
        m6 = m[m.fst == 6]
        return b[(b.fst == 6) & b.path.isin(m6.path)], m6[m6.path.isin(b.path)]

    xs, labels, sv, av, cols, centres = [], [], [], [], [], []
    x = 0.0
    groups = (("all FST VI images", hold_fn,
               (("+ IV–V\nmore data", hold), ("+ IV–V\nsame size", hold_sm))),
              ("held-out half of FST VI", mit_fn, (("oversample\nVI", mit),)))
    for name, fn, treats in groups:
        arms = [(t, collect(src, fn)) for t, src in treats if src]
        arms = [(t, r) for t, r in arms if len(r)]
        if not arms:
            continue
        # baseline uses the seeds of the first arm; every arm is tested on the same images
        r0 = arms[0][1]
        bars = [("baseline", r0[:, 0], r0[:, 1], BASE)]
        for t, r in arms:
            bars.append((t, r[:, 2], r[:, 3], HI))
        for k, (lab, sv_, av_, c) in enumerate(bars):
            xs.append(x + k); labels.append(lab); sv.append(sv_); av.append(av_); cols.append(c)
        centres.append((x + (len(bars) - 1) / 2, "tested on\n" + name))
        x += len(bars) + 0.8
    ref_s = np.mean([sens(d[d.fst <= 3].prob.values, d[d.fst <= 3].y.values) for d in main.values()])
    ref_a = np.mean([auc(d[d.fst <= 3].prob.values, d[d.fst <= 3].y.values) for d in main.values()])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    for ax, vals, ref, ylab in ((axes[0], sv, ref_s, "FST VI sensitivity"), (axes[1], av, ref_a, "FST VI AUC")):
        for xi, v, c in zip(xs, vals, cols):
            ax.bar(xi, v.mean(), 0.8, yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=2, color=c,
                   hatch="///" if c == HI else None, edgecolor="black", linewidth=0.5,
                   error_kw=dict(lw=0.6))
            dots(ax, xi, v)
        ax.axhline(ref, ls="--", lw=0.7, color=REF)
        ax.text(xs[-1] + 0.5, ref + 0.02, "FST I–III", ha="right", va="bottom", fontsize=6.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6)
        for cx, cap in centres:
            ax.annotate(cap, xy=(cx, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -30), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, style="italic", linespacing=1.15)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
    axes[0].set_title("(a) Sensitivity", loc="left")
    axes[1].set_title("(b) AUC", loc="left")
    fig.tight_layout()
    save(fig, "fig3_recovery")
else:
    print("no holdout/mitigation files - skipped fig3_recovery")
