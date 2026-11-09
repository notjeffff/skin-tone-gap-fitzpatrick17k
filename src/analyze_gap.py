"""
Analyse the skin-tone gap from saved probabilities. No GPU, no retraining.

    python analyze_gap.py

Tests the key hypothesis: is the sensitivity gap a THRESHOLD artefact
(same ranking ability, different operating point) rather than a
representational failure? AUC ~equal + sensitivity down + specificity up
is the signature of exactly that.
"""

import glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

OUT = "/kaggle/working"
files = sorted(glob.glob(f"{OUT}/probs_seed*.csv"))
files = [f for f in files if "smoke" not in f and "mitigate" not in f]
print(f"found {len(files)} seed(s): {[f.split('/')[-1] for f in files]}\n")


def sens_spec(p, y, thr):
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return (tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)


def thr_at_spec(p, y, target):
    """Threshold on this group that achieves a target specificity."""
    fpr, tpr, thr = roc_curve(y, p)
    i = np.argmin(np.abs((1 - fpr) - target))
    return thr[i]


def boot_gap(pl, yl, pd_, yd, thr=0.5, n=2000, seed=0):
    """Bootstrap CI for the sensitivity gap (light - dark)."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n):
        il = rng.integers(0, len(yl), len(yl))
        id_ = rng.integers(0, len(yd), len(yd))
        sl, _ = sens_spec(pl[il], yl[il], thr)
        sd, _ = sens_spec(pd_[id_], yd[id_], thr)
        if not (np.isnan(sl) or np.isnan(sd)):
            gaps.append(sl - sd)
    g = np.array(gaps)
    return g.mean(), np.percentile(g, 2.5), np.percentile(g, 97.5)


rows = []
for f in files:
    seed = f.split("seed")[-1].split(".")[0]
    d = pd.read_csv(f)
    L = d[d.group == "light"]; D = d[d.group == "dark"]
    pl, yl = L["prob"].values, L["y"].values
    pdk, ydk = D["prob"].values, D["y"].values

    # --- at the fixed 0.5 threshold ---
    sl, spl = sens_spec(pl, yl, 0.5)
    sd, spd = sens_spec(pdk, ydk, 0.5)

    # --- matched operating point: force dark to light's specificity ---
    t_match = thr_at_spec(pdk, ydk, spl)
    sd_m, spd_m = sens_spec(pdk, ydk, t_match)

    rows.append(dict(
        seed=seed,
        auc_light=roc_auc_score(yl, pl), auc_dark=roc_auc_score(ydk, pdk),
        sens_light=sl, sens_dark=sd, spec_light=spl, spec_dark=spd,
        gap_raw=sl - sd,
        thr_matched=t_match, sens_dark_matched=sd_m, spec_dark_matched=spd_m,
        gap_matched=sl - sd_m,
    ))

r = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.precision", 4)
print(r.to_string(index=False), "\n")

print("ACROSS SEEDS (mean +/- std)")
for c in ["auc_light", "auc_dark", "sens_light", "sens_dark",
          "gap_raw", "gap_matched"]:
    print(f"  {c:20s} {r[c].mean():+.4f} +/- {r[c].std():.4f}")

print("\nINTERPRETATION")
auc_gap = r.auc_light.mean() - r.auc_dark.mean()
print(f"  AUC gap            {auc_gap:+.4f}  "
      f"({'negligible -> ranking ability is equal' if abs(auc_gap) < 0.02 else 'real ranking difference'})")
print(f"  gap at thr=0.5     {r.gap_raw.mean():+.4f}")
print(f"  gap spec-matched   {r.gap_matched.mean():+.4f}")
shrink = 1 - abs(r.gap_matched.mean()) / max(abs(r.gap_raw.mean()), 1e-9)
print(f"  -> matching the operating point removes {shrink*100:.0f}% of the gap")
if shrink > 0.6 and abs(auc_gap) < 0.02:
    print("  => THRESHOLD ARTEFACT, not a representational failure.")
    print("     This is your paper's claim. Per-group thresholds are the remedy.")
else:
    print("  => gap survives threshold matching: a genuine representational gap.")

# --- bootstrap CI on seed 0 (is the raw gap even real?) ---
d = pd.read_csv(files[0])
L = d[d.group == "light"]; D = d[d.group == "dark"]
m, lo, hi = boot_gap(L["prob"].values, L["y"].values,
                     D["prob"].values, D["y"].values)
print(f"\nBOOTSTRAP (seed {files[0].split('seed')[-1].split('.')[0]}, 2000 resamples)")
print(f"  sensitivity gap {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print("  " + ("CI excludes zero -> gap is statistically detectable."
               if lo > 0 or hi < 0 else
               "CI INCLUDES ZERO -> cannot claim a significant gap at this n."))

# --- per-FST-level breakdown: AUC is the decisive column ---
# A threshold artefact shows: AUC flat, sens down, spec UP.
# A real failure shows:       AUC down, sens down, spec down.
if "fst" in d.columns:
    print("\nPER-FITZPATRICK-LEVEL (seed 0, thr=0.5)")
    print(f"  {'FST':>4} {'n':>6} {'pos':>5} {'sens':>7} {'spec':>7} "
          f"{'AUC':>7} {'sens_se':>8} {'verdict':>22}")
    ref_auc = None
    fst_rows = []
    for lvl in sorted(d["fst"].dropna().unique()):
        s = d[d.fst == lvl]
        y, p = s["y"].values, s["prob"].values
        if y.sum() < 10 or len(np.unique(y)) < 2:
            continue
        se_, sp_ = sens_spec(p, y, 0.5)
        auc_ = roc_auc_score(y, p)
        stderr = np.sqrt(se_ * (1 - se_) / y.sum())   # SE of sensitivity
        if ref_auc is None:
            ref_auc = auc_
        if auc_ < ref_auc - 0.05 and sp_ < 0.90:
            verdict = "genuine degradation"
        elif se_ < 0.6 and sp_ > 0.92:
            verdict = "operating-point shift"
        else:
            verdict = "-"
        print(f"  {int(lvl):>4} {len(s):>6} {int(y.sum()):>5} {se_:>7.3f} "
              f"{sp_:>7.3f} {auc_:>7.3f} {stderr:>8.3f} {verdict:>22}")
        fst_rows.append((int(lvl), se_, stderr, auc_))

    # pairwise test: worst level vs best level
    if len(fst_rows) >= 2:
        best = max(fst_rows, key=lambda r: r[1])
        worst = min(fst_rows, key=lambda r: r[1])
        gap = best[1] - worst[1]
        gse = np.sqrt(best[2] ** 2 + worst[2] ** 2)
        print(f"\n  FST {best[0]} vs FST {worst[0]}: "
              f"sensitivity gap {gap:+.3f} +/- {gse:.3f}  "
              f"({gap/gse:.1f} SE)")
        print("  " + ("SIGNIFICANT -- this, not the light/dark split, is the finding."
                      if abs(gap) > 2 * gse else
                      "not significant at this n."))
        print(f"  AUC at FST {best[0]} = {best[3]:.3f}, at FST {worst[0]} = {worst[3]:.3f}")
        print("  -> AUC also down  => genuine representational failure at that level."
              if worst[3] < best[3] - 0.05 else
              "  -> AUC holds up  => still an operating-point effect.")
    print("\n  NOTE: binarising FST 1-3 vs 4-6 averages FST 4/5 (fine) together")
    print("        with FST 6 (poor), which is how the aggregate gap loses significance.")
