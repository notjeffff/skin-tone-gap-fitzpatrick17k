"""Extra paper figures: study design, grouping comparison, robustness forest plot."""
import ast, os, glob, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
BASE, HI, REF = "#7A9CC6", "#D55E00", "#333333"
OUT = "figs_extra"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
NB = int(os.environ.get("NBOOT", 1000))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- data
off = pd.read_csv("official.csv"); off = off[off.fitzpatrick_scale.between(1, 6)]
cent = dict(zip(off.md5hash, off.fitzpatrick_centaur))
n2m = dict(zip(*[pd.read_csv("f17k_names.csv")[c] for c in ("filename", "md5hash")]))
cl = [[n2m[n] for n in ast.literal_eval(l) if n in n2m] for l in open("complete_clusters.txt") if l.strip()]
cid = {h: i for i, c in enumerate(cl) for h in c}; mem = {i: set(c) for i, c in enumerate(cl)}
keepC = set(pd.read_csv("f17k_C.csv").md5hash)
light_all = set(off.md5hash[off.fitzpatrick_scale <= 3])


def load(suf):
    R = {}
    for f in glob.glob(f"zipin/probs_seed*{suf}.csv"):
        m = re.fullmatch(rf"probs_seed(\d+){re.escape(suf)}\.csv", os.path.basename(f))
        if not m: continue
        d = pd.read_csv(f); d["h"] = d.path.map(lambda p: os.path.splitext(os.path.basename(p))[0])
        train = light_all - set(d.h[d.fst <= 3])
        d["leaked"] = d.h.map(lambda h: (i := cid.get(h)) is not None and bool((mem[i] - {h}) & train))
        d["inC"] = d.h.isin(keepC); d["fc"] = d.h.map(cent)
        R[int(m.group(1))] = d
    return R


main, eff = load(""), load("_effb0")
sens = lambda p, y: float((p[y == 1] >= .5).mean())
auc = lambda p, y: float(roc_auc_score(y, p))

# ---------------------------------------------------------------- Fig: grouping comparison
groups = [("Light / dark", [("I–III", [1, 2, 3]), ("IV–VI", [4, 5, 6])]),
          ("Pairs (as DDI)", [("I–II", [1, 2]), ("III–IV", [3, 4]), ("V–VI", [5, 6])]),
          ("Per type", [(r, [i]) for i, r in enumerate(["I", "II", "III", "IV", "V", "VI"], 1)])]
fig, ax = plt.subplots(figsize=(3.5, 1.95))
x, ticks, labs, centres = 0.0, [], [], []
for gname, bars in groups:
    start = x
    for lab, types in bars:
        v = np.array([sens(d.prob[d.fst.isin(types)].values, d.y[d.fst.isin(types)].values) for d in main.values()])
        isvi = types == [6]
        has6 = 6 in types
        ax.bar(x, v.mean(), 0.8, color=HI if isvi else BASE, hatch="////" if isvi else None,
               edgecolor="black", linewidth=0.5, alpha=1 if (isvi or not has6) else 0.55)
        ax.errorbar(x, v.mean(), v.std(ddof=1), color="black", capsize=1.5, lw=0.6)
        ax.text(x, v.mean() + v.std(ddof=1) + 0.02, f"{v.mean():.2f}", ha="center", fontsize=5.5)
        ticks.append(x); labs.append(lab); x += 1.3 if len(bars) < 6 else 1.0
    centres.append(((start + x - (1.3 if len(bars) < 6 else 1.0)) / 2, gname)); x += 0.9
ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=5.6)
ax.set_ylabel("Sensitivity"); ax.set_ylim(0, 0.95)
for c, g in centres:
    ax.text(c, -0.19, g, ha="center", va="top", fontsize=6.2, style="italic", transform=ax.get_xaxis_transform())
save(fig, "fig_grouping")

# ---------------------------------------------------------------- Fig: robustness forest plot
def gap_ci(runs, filt, lo_types, hi_types, stat, fcol="fst"):
    pts, los, his = [], [], []
    for d in runs.values():
        x = filt(d); x = x[x[fcol].notna()]
        A, B = x[x[fcol].isin(lo_types)], x[x[fcol].isin(hi_types)]
        pa, ya, pb, yb = A.prob.values, A.y.values, B.prob.values, B.y.values
        pts.append(stat(pa, ya) - stat(pb, yb))
        bs = [stat(pa[i], ya[i]) - stat(pb[j], yb[j]) for i, j in
              ((rng.integers(0, len(ya), len(ya)), rng.integers(0, len(yb), len(yb))) for _ in range(NB))]
        l, h = np.percentile(bs, [2.5, 97.5]); los.append(l); his.append(h)
    return np.mean(pts), np.mean(los), np.mean(his)


checks = [("All test images", main, lambda d: d, "fst"),
          ("EfficientNet-B0", eff, lambda d: d, "fst"),
          ("Second annotator", main, lambda d: d, "fc"),
          ("Both annotators agree", main, lambda d: d[d.fst == d.fc], "fst"),
          ("Duplicates removed", main, lambda d: d[~d.leaked], "fst"),
          ("Fitzpatrick17k-C only", main, lambda d: d[d.inC], "fst")]
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
for ax, (title, stat) in zip(axes, (("(a) Sensitivity gap", sens), ("(b) AUC gap", auc))):
    for k, (name, runs, filt, col) in enumerate(checks):
        y = len(checks) - 1 - k
        for off_, (lo_t, hi_t, mk, colr) in zip((0.15, -0.15), (([1, 2, 3], [6], "o", HI), ([4, 5], [6], "s", BASE))):
            p, l, h = gap_ci(runs, filt, lo_t, hi_t, stat, col)
            ax.errorbar(p, y + off_, xerr=[[p - l], [h - p]], fmt=mk, ms=3, color=colr, mec="black", mew=0.4,
                        ecolor="black", elinewidth=0.6, capsize=1.2)
    ax.axvline(0, color=REF, lw=0.6, ls="--")
    ax.set_title(title, loc="left", fontsize=7.5); ax.set_xlim(left=-0.05)
    ax.set_yticks(range(len(checks))); ax.set_yticklabels([c[0] for c in checks][::-1], fontsize=6.3)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=HI, mec="black", label="FST I–III minus VI"),
                        Line2D([], [], marker="s", ls="", color=BASE, mec="black", label="FST IV–V minus VI")],
               loc="upper center", bbox_to_anchor=(1.0, -0.13), ncol=2, frameon=False, fontsize=6)
save(fig, "fig_forest")

# ---------------------------------------------------------------- Fig: study design
from matplotlib.patches import Rectangle
LB, LF, DB, DF = "#4C72B0", "#EEF3FA", "#D55E00", "#FCEEE6"        # light-path / dark-path edge + fill
TONES = ["#F3D9C4", "#E6BE9C", "#CC9C74", "#A06D48", "#704630", "#3E271C"]   # FST I..VI swatches
fig, ax = plt.subplots(figsize=(7.16, 2.15)); ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis("off")
def box(x, y, w, h, title, body="", ec=LB, fc=LF, bfs=6.8, tfs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.9", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=tfs, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 4.8, body, ha="center", va="top", fontsize=bfs, linespacing=1.3)
def arrow(pts, color=REF, ls="-"):
    xs, ys = zip(*pts)
    if len(pts) > 2: ax.plot(xs[:-1], ys[:-1], color=color, lw=0.8, ls=ls)
    ax.annotate("", pts[-1], pts[-2], arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, ls=ls,
                                                       shrinkA=0, shrinkB=0, mutation_scale=7))
box(0.4, 9, 15.2, 15, "Fitzpatrick17k", "16,012 images\nFST I–VI\n2,160 malignant", ec=REF, fc="#F5F5F5", tfs=7.0)
box(18.8, 18.2, 15.8, 10.5, "FST I–III", "train 8,851\ntest 2,212")
box(18.8, 1.2, 15.8, 13.3, "FST IV–VI", "test only\nIV 2,781 · V 1,533\nVI 635", ec=DB, fc=DF, bfs=6.3)
box(37.5, 13, 15, 13, "Train", "ResNet-50\nEfficientNet-B0\n5 seeds")
box(55.2, 9, 19.5, 17, "Evaluate per type")
for i, c in enumerate(TONES):                                    # six skin-type swatches
    x0 = 56.8 + i * 2.85
    ax.add_patch(Rectangle((x0, 15.4), 2.45, 4.2, fc=c, ec=DB if i == 5 else "white", lw=1.4 if i == 5 else 0.4))
    ax.text(x0 + 1.22, 14.6, ["I", "II", "III", "IV", "V", "VI"][i], ha="center", va="top", fontsize=6.3,
            color=DB if i == 5 else "black", weight="bold" if i == 5 else "normal")
ax.text(64.95, 11.6, "sensitivity · AUC", ha="center", va="top", fontsize=6.8)
box(78.0, 22.4, 21.6, 9.6, "External check", "DDI, biopsy-confirmed\nno retraining")
box(78.0, 11.6, 21.6, 9.6, "Robustness", "subtype · source site\nannotator · duplicates")
box(78.0, 0.8, 21.6, 9.6, "Interventions", "+FST IV–V in training\nsize-matched · 3 seeds")
arrow([(15.9, 19), (18.5, 23.4)]); arrow([(15.9, 14), (18.5, 7.8)])
arrow([(34.9, 23.4), (37.2, 21.5)], color=LB)
arrow([(52.8, 19.5), (54.9, 19.5)])
arrow([(34.9, 5.0), (64.95, 5.0), (64.95, 8.7)], color=DB, ls="--")
ax.text(50.0, 4.3, "held out: never trained on", fontsize=6.2, style="italic", ha="center", va="top", color=DB)
arrow([(45.0, 26.3), (45.0, 31.2), (76.2, 31.2), (76.2, 27.2), (77.7, 27.2)])
arrow([(75.0, 17.5), (77.7, 16.4)]); arrow([(75.0, 14.0), (77.7, 5.6)])
save(fig, "fig_design")
