"""Reviewer-response analyses on existing light-only predictions (no training).
Pooled estimates with image-level CIs, site x FST, site-adjusted regression,
matched specificity, trend test, subtype CIs, intervention differences, negative mix.
Run in a folder holding zipin/probs_seed*.csv, official.csv, complete_clusters.txt, f17k_names.csv."""
import ast, glob, os, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf

NB = int(os.environ.get("NBOOT", 2000)); rng = np.random.default_rng(0)
pd.set_option("display.width", 220, "display.precision", 3)
stem = lambda p: os.path.splitext(os.path.basename(str(p)))[0]
off = pd.read_csv("official.csv"); off = off[off.fitzpatrick_scale.between(1, 6)].copy()
off["site"] = off.url.astype(str).str.extract(r"https?://(?:www\.)?([^/]+)")[0]
off["site"] = np.where(off.site.isin(["dermaamin.com", "atlasdermatologico.com.br"]), off.site, "other")
lab = off.label.str.lower()
off["fam"] = np.select([lab.str.contains("melanoma") | lab.str.contains("lentigo maligna"),
                        lab.str.contains("squamous"), lab.str.contains("basal"),
                        lab.str.contains("kaposi"), lab.str.contains("mycosis")],
                       ["melanoma", "squamous cell carcinoma", "basal cell carcinoma",
                        "kaposi sarcoma", "mycosis fungoides"], "other")
meta = off.set_index("md5hash")[["site", "fam", "nine_partition_label"]]


def wilson(k, n, z=1.959964):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(suf):
    R = []
    for f in glob.glob(f"zipin/probs_seed*{suf}.csv"):
        m = re.fullmatch(rf"probs_seed(\d+){re.escape(suf)}\.csv", os.path.basename(f))
        if m:
            d = pd.read_csv(f); d["h"] = d.path.map(stem); d["seed"] = int(m.group(1)); R.append(d)
    A = pd.concat(R)
    # pool: average each image's score over the seeds in which it was tested
    P = A.groupby("h").agg(prob=("prob", "mean"), y=("y", "first"), fst=("fst", "first"),
                           nseeds=("seed", "nunique")).reset_index()
    return P.join(meta, on="h")


sens = lambda p, y: float((p[y == 1] >= .5).mean())
auc = lambda p, y: float(roc_auc_score(y, p))
grp = lambda f: np.where(f <= 3, "I-III", np.where(f <= 5, "IV-V", "VI"))


def boot_ci(fn, d, nb=NB):
    v = [fn(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(nb)]
    return np.nanpercentile(v, [2.5, 97.5])


for suf, name in (("", "RESNET-50"), ("_effb0", "EFFICIENTNET-B0")):
    P = load(suf); P["g"] = grp(P.fst)
    print("=" * 90 + f"\n{name}: POOLED OVER SEEDS (mean score per image), image-level 95% CIs\n" + "=" * 90)
    rows = []
    for f, d in P.groupby("fst"):
        k, n = int((d.prob[d.y == 1] >= .5).sum()), int(d.y.sum()); lo, hi = wilson(k, n)
        al, ah = boot_ci(lambda x: auc(x.prob.values, x.y.values), d)
        rows.append(dict(fst=int(f), images=len(d), malignant=n, sens=k / n, sens_ci=f"[{lo:.2f}, {hi:.2f}]",
                         spec=float((d.prob[d.y == 0] < .5).mean()), auc=auc(d.prob.values, d.y.values),
                         auc_ci=f"[{al:.2f}, {ah:.2f}]"))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for lo_, hi_ in (("I-III", "VI"), ("I-III", "IV-V"), ("IV-V", "VI")):
        A, B = P[P.g == lo_], P[P.g == hi_]
        for mname, st in (("sensitivity", sens), ("AUC", auc)):
            pt = st(A.prob.values, A.y.values) - st(B.prob.values, B.y.values)
            bs = [st(A.prob.values[i], A.y.values[i]) - st(B.prob.values[j], B.y.values[j])
                  for i, j in ((rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B))) for _ in range(NB))]
            l, h = np.percentile(bs, [2.5, 97.5])
            print(f"  {lo_:>5s} minus {hi_:<4s} {mname:11s} {pt:+.3f}  95% CI [{l:+.3f}, {h:+.3f}]")
    if suf: continue

    # ---------------------------------------------------------------- matched specificity (common threshold)
    print("\nMATCHED SPECIFICITY: one threshold set on FST I-III negatives, applied to every type")
    L = P[(P.g == "I-III") & (P.y == 0)].prob.values
    for sp in (0.80, 0.90):
        t = np.quantile(L, sp)
        out = {int(f): f"{(d.prob[d.y == 1] > t).mean():.3f} (spec {(d.prob[d.y == 0] <= t).mean():.2f})"
               for f, d in P.groupby("fst")}
        print(f"  spec {sp:.0%} on I-III: sensitivity by FST {out}")
    print("  within-type matched specificity (threshold set inside each type):")
    for sp in (0.80, 0.90):
        out = {}
        for f, d in P.groupby("fst"):
            t = np.quantile(d.prob[d.y == 0].values, sp); out[int(f)] = round(float((d.prob[d.y == 1] > t).mean()), 3)
        print(f"    spec {sp:.0%}: {out}")

    # ---------------------------------------------------------------- trend test
    print("\nTREND TEST: linear fit of sensitivity on FST I-V, extrapolated to VI (bootstrap over images)")
    def trend(d):
        s = [sens(x.prob.values, x.y.values) for _, x in d[d.fst <= 5].groupby("fst")]
        b = np.polyfit(np.arange(1, 6), s, 1); pred6 = np.polyval(b, 6)
        return pred6, sens(d[d.fst == 6].prob.values, d[d.fst == 6].y.values)
    p6, o6 = trend(P)
    bs = []
    for _ in range(NB):
        d = pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for _, x in P.groupby("fst")])
        a, b = trend(d); bs.append(b - a)
    l, h = np.percentile(bs, [2.5, 97.5])
    print(f"  predicted FST VI {p6:.3f}, observed {o6:.3f}, observed minus predicted {o6 - p6:+.3f} [{l:+.3f}, {h:+.3f}]")

    # ---------------------------------------------------------------- site x FST
    print("\nSITE x SKIN GROUP (pooled): malignant n, sensitivity [Wilson CI], AUC")
    rows = []
    for (s, g), d in P[P.site != "other"].groupby(["site", "g"]):
        k, n = int((d.prob[d.y == 1] >= .5).sum()), int(d.y.sum()); lo, hi = wilson(k, n)
        rows.append(dict(site=s, group=g, images=len(d), malignant=n, sens=round(k / n, 3),
                         sens_ci=f"[{lo:.2f}, {hi:.2f}]", auc=round(auc(d.prob.values, d.y.values), 3)))
    print(pd.DataFrame(rows).to_string(index=False))
    M = P[(P.y == 1) & (P.site != "other")].copy()
    M["det"] = (M.prob >= .5).astype(int)
    M["g"] = pd.Categorical(M.g, ["I-III", "IV-V", "VI"])
    for form in ("det ~ C(g)", "det ~ C(g) + C(site)", "det ~ C(g) + C(site) + C(fam)"):
        r = smf.logit(form, M).fit(disp=0)
        orr = np.exp(r.params["C(g)[T.VI]"]); ci = np.exp(r.conf_int().loc["C(g)[T.VI]"])
        print(f"  {form:32s} odds ratio of detection, FST VI vs I-III: {orr:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  p={r.pvalues['C(g)[T.VI]']:.4f}")
    D = P[P.site == "dermaamin.com"]
    k6 = D[(D.fst == 6) & (D.y == 1)]
    print(f"  within dermaamin.com: FST VI sens {sens(k6.prob.values, k6.y.values):.3f} (n={len(k6)}) "
          f"vs FST I-III {sens(D[D.g == 'I-III'].prob.values, D[D.g == 'I-III'].y.values):.3f}")

    # ---------------------------------------------------------------- subtype CIs
    print("\nSUBTYPE x GROUP sensitivity [Wilson CI] (malignant n)")
    rows = []
    for (fam, g), d in P[P.y == 1].groupby(["fam", "g"]):
        k, n = int((d.prob >= .5).sum()), len(d); lo, hi = wilson(k, n)
        rows.append(dict(subtype=fam, group=g, n=n, sens=f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]"))
    print(pd.DataFrame(rows).pivot(index="subtype", columns="group", values="sens").to_string())
    print(pd.DataFrame(rows).pivot(index="subtype", columns="group", values="n").to_string())

    # ---------------------------------------------------------------- negative class mix
    print("\nNON-MALIGNANT CLASS MIX by group (nine-partition label, % of negatives)")
    Ng = P[P.y == 0]
    print((pd.crosstab(Ng.nine_partition_label, Ng.g, normalize="columns") * 100).round(1).to_string())
    print("  malignant prevalence:", {g: round(d.y.mean(), 3) for g, d in P.groupby("g")})
    ref = Ng[Ng.g == "I-III"].nine_partition_label.value_counts(normalize=True)
    def std_auc(d):
        neg = d[d.y == 0]; w_neg = neg.nine_partition_label.map(ref / neg.nine_partition_label.value_counts(normalize=True)).fillna(0)
        w = pd.concat([pd.Series(1.0, index=d[d.y == 1].index), w_neg])
        x = d.loc[w.index]
        return roc_auc_score(x.y, x.prob, sample_weight=w.values)
    for g in ("IV-V", "VI"):
        d = P[P.g == g]
        print(f"  FST {g}: AUC {auc(d.prob.values, d.y.values):.3f}; with negatives reweighted to the I-III mix {std_auc(d):.3f}")

# ---------------------------------------------------------------- interventions: paired differences
print("\n" + "=" * 90 + "\nINTERVENTIONS: paired differences on the same leak-free FST VI images (seed-averaged scores)\n" + "=" * 90)
n2m = dict(zip(*[pd.read_csv("f17k_names.csv")[c] for c in ("filename", "md5hash")]))
cl = [[n2m[n] for n in ast.literal_eval(l) if n in n2m] for l in open("complete_clusters.txt") if l.strip()]
cid = {h: i for i, c in enumerate(cl) for h in c}; mem = {i: set(c) for i, c in enumerate(cl)}
ivv = set(off.md5hash[off.fitzpatrick_scale.isin([4, 5])])
leak = {h for h in off.md5hash[off.fitzpatrick_scale == 6] if (i := cid.get(h)) is not None and (mem[i] - {h}) & ivv}
S = {}
for suf in ("", "_hold45", "_hold45_sm"):
    R = pd.concat([pd.read_csv(f"zipin/probs_seed{s}{suf}.csv").assign(seed=s) for s in range(3)])
    R = R[R.fst == 6]; R["h"] = R.path.map(stem)
    S[suf] = R.groupby("h").agg(prob=("prob", "mean"), y=("y", "first"))
idx = sorted(set(S[""].index) - leak)
base = S[""].loc[idx]
print(f"leak-free FST VI images: {len(idx)} ({int(base.y.sum())} malignant)")
for suf, nm in (("_hold45", "+FST IV-V"), ("_hold45_sm", "+FST IV-V size-matched")):
    X = S[suf].loc[idx]
    for mname, st in (("sensitivity", sens), ("AUC", auc)):
        pt = st(X.prob.values, X.y.values) - st(base.prob.values, base.y.values)
        bs = []
        for _ in range(NB):
            i = rng.integers(0, len(idx), len(idx))
            bs.append(st(X.prob.values[i], X.y.values[i]) - st(base.prob.values[i], base.y.values[i]))
        l, h = np.percentile(bs, [2.5, 97.5])
        print(f"  {nm:24s} minus baseline  {mname:11s} {pt:+.3f}  95% CI [{l:+.3f}, {h:+.3f}]")
Xa, Xb = S["_hold45"].loc[idx], S["_hold45_sm"].loc[idx]
for mname, st in (("sensitivity", sens), ("AUC", auc)):
    pt = st(Xa.prob.values, Xa.y.values) - st(Xb.prob.values, Xb.y.values)
    bs = [st(Xa.prob.values[i], Xa.y.values[i]) - st(Xb.prob.values[i], Xb.y.values[i])
          for i in (rng.integers(0, len(idx), len(idx)) for _ in range(NB))]
    l, h = np.percentile(bs, [2.5, 97.5])
    print(f"  full minus size-matched             {mname:11s} {pt:+.3f}  95% CI [{l:+.3f}, {h:+.3f}]")
