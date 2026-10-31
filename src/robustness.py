"""Robustness checks for the Storm Boris analysis (added after review round 1). Run after analyze.py.

Inputs : results/kaggle_out/*, results/summary.json, results/boris_per_gauge.csv, results/diag/event_compare.csv
Outputs: results/robustness.json, results/numbers_rob.tex (LaTeX macros used verbatim by the paper)
"""
import sys, json, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, binomtest, rankdata

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KO = os.path.join(ROOT, "results", "kaggle_out"); RES = os.path.join(ROOT, "results")
R = np.load(os.path.join(KO, "eval_ref.npz"), allow_pickle=True)
dates = pd.to_datetime(R["dates"]); gauges = R["gauges"].astype(str); Q = R["Q"]; N = len(gauges)
qmax_b, q99 = R["qmax_train_base"], R["q99_train_base"]
summ = json.load(open(os.path.join(RES, "summary.json")))
B = pd.read_csv(os.path.join(RES, "boris_per_gauge.csv"), dtype={"gauge": str})

models = {"HBV": R["SIM_HBV"], "LSTMbm": R["SIM_LSTM"]}
for cfg, t in [("base_nse", "BaseNse"), ("base_peak", "BasePeak"), ("long_nse", "LongNse"), ("long_peak", "LongPeak"),
               ("early_nse", "EarlyNse"), ("long3x_nse", "LongThreeNse")]:
    fs = sorted(glob.glob(os.path.join(KO, f"sim_lstm_{cfg}_s*.npy")))
    if fs:
        models[t] = np.median([np.load(f) for f in fs], 0)
models["LGBM"] = np.median([np.load(f) for f in sorted(glob.glob(os.path.join(KO, "sim_lgbm_s*.npy")))], 0)
REF = "BaseNse"
out, mac = {}, []


def m(name, val, fmt="{:.1f}"):
    v = fmt.format(val)
    if re.fullmatch(r"-0(\.0*)?", v):  # avoid printing a negative zero
        v = v[1:]
    mac.append("\\newcommand{\\%s}{%s}\n" % (re.sub(r"[^A-Za-z]", "", name), v))


def pstr(p):
    return "$p<0.001$" if p < 0.001 else "$p=%.3f$" % p


def sprho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    r = spearmanr(x[ok], y[ok]); return float(r[0]), float(r[1]), int(ok.sum())


def boris(win=("2024-09-10", "2024-09-30"), flood_mult=1.0, min_p10=None):
    """Per-gauge Boris peak errors for each model under a given event definition."""
    w = np.asarray((dates >= win[0]) & (dates <= win[1])); ob = np.where(w[None], Q, np.nan)
    ok = np.isfinite(ob).sum(1) >= 15
    pk = np.full(N, np.nan); pk[ok] = np.nanmax(ob[ok], 1)
    err = {k: 100 * (np.nanmax(np.where(w[None], s, np.nan), 1) - pk) / pk for k, s in models.items()}
    flood = ok & (pk > flood_mult * q99)
    if min_p10 is not None:
        flood &= P10PK >= min_p10  # 10-day rain ending on the observed peak day: excludes routed flood waves
    return pk, err, flood, flood & (pk > qmax_b)


# ---- 1. Is the error--OOD correlation built in? Control: flood annual maxima 2016-2023 (also out-of-sample). ----
rows = []
for y in range(2016, 2024):
    ix = np.where(dates.year == y)[0]
    for i in range(N):
        o = Q[i, ix]
        if np.isfinite(o).sum() < 300:
            continue
        t = ix[np.nanargmax(o)]; pk = Q[i, t]
        if not pk > q99[i]:
            continue
        r = dict(gauge=gauges[i], year=y, ratio=pk / qmax_b[i])
        for k, s in models.items():  # simulated peak within +-3 days of the observed annual maximum
            r[k] = 100 * (np.nanmax(s[i, max(t - 3, 0):t + 4]) - pk) / pk
        rows.append(r)
A = pd.DataFrame(rows); A.to_csv(os.path.join(RES, "annual_max_2016_2023.csv"), index=False)
out["annmax"] = {"n": len(A), "n_ood": int((A.ratio > 1).sum()), "n_gauges": int(A.gauge.nunique())}
m("NAnnMax", len(A), "{:d}"); m("NAnnMaxOod", int((A.ratio > 1).sum()), "{:d}")
for k in models:
    rho, p, n = sprho(A.ratio, A[k]); ood_med = float(A[A.ratio > 1][k].median())
    out["annmax"][k] = dict(rho=rho, p=p, n=n, err_ood=ood_med, err_in=float(A[A.ratio <= 1][k].median()))
    m(k + "AnnRho", rho, "{:.2f}"); m(k + "AnnErrOod", ood_med); m(k + "AnnErrIn", out["annmax"][k]["err_in"])
rhos = [out["annmax"][k]["rho"] for k in models]
m("AnnRhoMin", min(rhos), "{:.2f}"); m("AnnRhoMax", max(rhos), "{:.2f}")

# ---- 2. Precipitation-based OOD ratio (no observed discharge involved), from the Kaggle diagnostic kernel ----
E = pd.read_csv(os.path.join(RES, "diag", "event_compare.csv"), dtype={"gauge": str}).set_index("gauge")
HAS_P = "p10_boris_max" in E.columns
P10B = np.array([E.p10_boris_max.get(g, np.nan) if HAS_P else np.nan for g in gauges])
P10PK = np.array([E.boris_p10.get(g, np.nan) for g in gauges])
pk0, err0, flood0, ood0 = boris()
if HAS_P:
    rp = P10B / np.array([E.p10_max_base.get(g, np.nan) for g in gauges])
    out["precip_ratio"] = {}
    for k in models:
        rho, p, n = sprho(rp[flood0], err0[k][flood0]); out["precip_ratio"][k] = dict(rho=rho, p=p, n=n)
        m(k + "PRho", rho, "{:.2f}"); mac.append("\\newcommand{\\%sPRhoP}{%s}\n" % (k, pstr(p)))
    m("NPRatioAboveOne", int((rp[flood0] > 1).sum()), "{:d}")
    lp = [out["precip_ratio"][k]["rho"] for k in ["LSTMbm", "BaseNse", "BasePeak", "LongNse", "LongPeak"]]
    m("LstmPRhoMin", min(lp), "{:.2f}"); m("LstmPRhoMax", max(lp), "{:.2f}")

# ---- 3. Multiple comparisons: Holm correction over all Wilcoxon and Spearman tests of the main analysis ----
fam = []
MAIN = ["HBV (benchmark)", "LSTM (benchmark)", "LSTM base_peak", "LSTM long_nse", "LSTM long_peak", "LightGBM", "LSTM base_nse"]
for k, d in summ["wilcoxon"].items():
    if k.split(" vs ")[0] not in MAIN:  # the family is the pre-planned main comparisons only
        continue
    for metric, v in d.items():
        if v:
            fam.append((k + "|" + metric, v["p"]))
tag = {"HBV (benchmark)": "HBV", "LSTM (benchmark)": "LSTMbm", "LSTM base_nse": "BaseNse", "LSTM base_peak": "BasePeak",
       "LSTM long_nse": "LongNse", "LSTM long_peak": "LongPeak", "LightGBM": "LGBM",
       "LSTM early_nse": "EarlyNse", "LSTM long3x_nse": "LongThreeNse"}
for k, d in summ["models"].items():
    if k not in MAIN:
        continue
    fam.append((k + "|rho", d["rho_p"]))
order = np.argsort([p for _, p in fam]); nf = len(fam); adj = {}; run = 0.0
for r, j in enumerate(order):
    run = max(run, min(1.0, (nf - r) * fam[j][1])); adj[fam[j][0]] = run
out["holm"] = adj; m("NHolmTests", nf, "{:d}")
for key, p in adj.items():
    name, metric = key.split("|")
    t = tag[name.split(" vs ")[0]]
    mac.append("\\newcommand{\\%s}{%s}\n" % (re.sub(r"[^A-Za-z]", "", t + "H" + metric), pstr(p)))

# ---- 4. Spatial clustering: cluster bootstrap over river systems ----
# The IMGW gauge code encodes a 1x1 degree grid cell, not a basin, so clusters are river systems assigned by hand
# from gauge name and water body (src/river_systems.csv): the Odra main stem is one cluster, and each tributary system
# draining to it (e.g. Nysa Klodzka incl. Scinawka and Biala Ladecka; Bobr incl. Kwisa) is another.
RS = pd.read_csv(os.path.join(ROOT, "src", "river_systems.csv"), dtype=str).set_index("gauge").river_system
assert all(g in RS.index for g in gauges[flood0]), "every flood gauge needs a river system"
clus = np.array([RS.get(g, g) for g in gauges]); grid = np.array([g[:6] for g in gauges]); rng = np.random.default_rng(0)


def cboot(a, b, sel, nb=4000, cl=None):
    cl = clus if cl is None else cl
    d = (a - b)[sel]; c = cl[sel]; ok = np.isfinite(d); d, c = d[ok], c[ok]
    uc = np.unique(c); grp = [d[c == u] for u in uc]
    meds = [np.median(np.concatenate([grp[j] for j in rng.integers(0, len(uc), len(uc))])) for _ in range(nb)]
    cm = np.array([np.median(g) for g in grp])  # sign test on cluster medians: one vote per river system
    ps = binomtest(int((cm > 0).sum()), int((cm != 0).sum())).pvalue
    return float(np.median(d)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)), len(uc), float(ps)


out["cluster_boot"] = {}
for nm, a, b, sel in ([("LongAbs", np.abs(err0["LongNse"]), np.abs(err0[REF]), flood0),
                      ("LongOod", err0["LongNse"], err0[REF], ood0),
                      ("HbvOod", err0["HBV"], err0[REF], ood0),
                      ("BmHbvOod", err0["LSTMbm"], err0["HBV"], ood0),
                      ("LgbmOod", err0["LGBM"], err0[REF], ood0)]
                     + ([("LongThreeAbs", np.abs(err0["LongThreeNse"]), np.abs(err0[REF]), flood0)] if "LongThreeNse" in err0 else [])
                     + ([("EarlyAbs", np.abs(err0["EarlyNse"]), np.abs(err0[REF]), flood0)] if "EarlyNse" in err0 else [])):
    med, lo, hi, nc, ps = cboot(a, b, sel); _, glo, ghi, gnc, _ = cboot(a, b, sel, cl=grid)
    out["cluster_boot"][nm] = dict(med=med, lo=lo, hi=hi, n_clusters=nc, sign_p=ps, grid_lo=glo, grid_hi=ghi, grid_nc=gnc)
    m("CB" + nm + "Med", med); m("CB" + nm + "Lo", lo); m("CB" + nm + "Hi", hi); m("CB" + nm + "NC", nc, "{:d}")
    mac.append("\\newcommand{\\CB%sSignP}{%s}\n" % (nm, pstr(ps)))
m("NRiverSysFlood", len(np.unique(clus[flood0])), "{:d}"); m("NRiverSysOod", len(np.unique(clus[ood0])), "{:d}")
out["ood_river_systems"] = pd.Series(clus[ood0]).value_counts().to_dict()
m("NOodNysaKlodzka", out["ood_river_systems"].get("Nysa Klodzka", 0), "{:d}")

# ---- 5. Where does the long-record degradation come from? ----
inr = flood0 & ~ood0
for k in ["BaseNse", "LongNse"] + (["LongThreeNse"] if "LongThreeNse" in models else []) + (["EarlyNse"] if "EarlyNse" in models else []):
    e = err0[k]
    out.setdefault("decomp", {})[k] = dict(err_inrange=float(np.nanmedian(e[inr])), n_below50=int(np.sum(e[flood0] < -50)),
                                            n_above50=int(np.sum(e[flood0] > 50)))
    m(k + "ErrIn", out["decomp"][k]["err_inrange"]); m(k + "NBelowFifty", out["decomp"][k]["n_below50"], "{:d}")
    m(k + "NAboveFifty", out["decomp"][k]["n_above50"], "{:d}")
m("NInRange", int(inr.sum()), "{:d}")

# ---- 6. Do the models exceed each gauge's own training maximum at the OOD gauges? ----
for k, s in models.items():
    spk = np.nanmax(np.where(np.asarray((dates >= "2024-09-10") & (dates <= "2024-09-30"))[None], s, np.nan), 1)
    n = int(np.sum(spk[ood0] > qmax_b[ood0])); out.setdefault("exceed_own_max", {})[k] = n; m(k + "NExceed", n, "{:d}")
allq = R["qmax_train_base"]
m("PooledMax", float(np.nanmax(allq))); m("BorisMaxPeak", float(np.nanmax(pk0[flood0])))

# ---- 7. Paired comparison of Boris with the 1997 (long-record maximum) event at the OOD gauges ----
Eo = E.loc[[g for g in gauges[ood0] if g in E.index]]
m("NLongMaxAboveBoris", int((Eo.long_max > Eo.boris_peak).sum()), "{:d}")
m("LongMaxOverBoris", float((Eo.long_max / Eo.boris_peak).median()), "{:.2f}")
m("PairedRunoffRatio", float(((Eo.long_max / Eo.long_p10) / (Eo.boris_peak / Eo.boris_p10)).median()), "{:.2f}")
small_rain = Eo[Eo.boris_p10 < 50]; m("NOodLowRain", len(small_rain), "{:d}")

# ---- 8. Sensitivity of the main results to the event definition ----
variants = {"Main": {}, "Strict": dict(flood_mult=1.5), "Wide": dict(win=("2024-09-01", "2024-10-15"))}
if HAS_P:
    variants["Rain"] = dict(min_p10=50.0)
out["sensitivity"] = {}
for vn, kw in variants.items():
    pk, err, fl, oo = boris(**kw)
    d = dict(n_flood=int(fl.sum()), n_ood=int(oo.sum()))
    for k in ["HBV", "LSTMbm", "BaseNse", "LongNse", "LGBM"]:
        d[k] = float(np.nanmedian(err[k][oo]))
        m("S" + vn + k, d[k])
    a, b = np.abs(err["LongNse"][fl]), np.abs(err[REF][fl]); ok = np.isfinite(a) & np.isfinite(b)
    d["long_abs_diff"] = float(np.median(a[ok] - b[ok])); d["long_abs_p"] = float(wilcoxon(a[ok], b[ok]).pvalue)
    d["rho_base"] = sprho(pk[fl] / qmax_b[fl], err[REF][fl])[0]
    out["sensitivity"][vn] = d
    m("S" + vn + "NFlood", d["n_flood"], "{:d}"); m("S" + vn + "NOod", d["n_ood"], "{:d}")
    m("S" + vn + "LongAbsD", d["long_abs_diff"]); mac.append("\\newcommand{\\S%sLongAbsP}{%s}\n" % (vn, pstr(d["long_abs_p"])))
    m("S" + vn + "RhoBase", d["rho_base"], "{:.2f}")

# ---- 9. Added after review round 2: what the OOD ratio does and does not show ----
ratio0 = pk0 / qmax_b; MK = ["HBV", "LSTMbm", "BaseNse", "BasePeak", "LongNse", "LongPeak", "LGBM"]
# 9a. Within the OOD gauges only: does the error keep growing with the ratio?
out["within_ood"] = {}
bins = [("BinA", 1.0, 1.5), ("BinB", 1.5, 2.0), ("BinC", 2.0, np.inf)]
for k in MK:
    rho, p, n = sprho(ratio0[ood0], err0[k][ood0]); d = dict(rho=rho, p=p, n=n)
    m(k + "RhoOod", rho, "{:.2f}"); mac.append("\\newcommand{\\%sRhoOodP}{%s}\n" % (k, pstr(p)))
    for bn, lo, hi in bins:
        s = ood0 & (ratio0 > lo) & (ratio0 <= hi); d[bn] = float(np.nanmedian(err0[k][s])); m(k + "Err" + bn, d[bn])
    out["within_ood"][k] = d
for bn, lo, hi in bins:
    m("N" + bn, int((ood0 & (ratio0 > lo) & (ratio0 <= hi)).sum()), "{:d}")
r_ood = [out["within_ood"][k]["rho"] for k in MK]
m("RhoOodMin", min(r_ood), "{:.2f}"); m("RhoOodMax", max(r_ood), "{:.2f}")
# 9b. The long LSTM's own range: at OOD gauges where Boris stayed inside the 1981-2015 maximum
qmax_l = R["qmax_train_long"]; inl = ood0 & (pk0 <= qmax_l)
m("NOodInLong", int(inl.sum()), "{:d}"); m("NOodOutLong", int((ood0 & ~inl).sum()), "{:d}")
for k in ["LongNse", "LongPeak", "BaseNse"]:
    m(k + "ErrOodInLong", float(np.nanmedian(err0[k][inl])))
out["long_own_range"] = {k: float(np.nanmedian(err0[k][inl])) for k in ["LongNse", "LongPeak", "BaseNse"]}
# 9c. Annual-max control per year and in a matched ratio range
out["annmax_year"] = {}
for k in ["BaseNse", "LSTMbm", "HBV"]:
    ys = [sprho(A[A.year == y].ratio, A[A.year == y][k])[0] for y in sorted(A.year.unique())]
    out["annmax_year"][k] = ys; m(k + "AnnYearMin", min(ys), "{:.2f}"); m(k + "AnnYearMax", max(ys), "{:.2f}")
m("AnnRatioMax", float(A.ratio.max()), "{:.2f}"); m("NAnnAboveOnePointFive", int((A.ratio > 1.5).sum()), "{:d}")
m("NAnnAboveTwo", int((A.ratio > 2).sum()), "{:d}")
m("NBorisAboveOnePointFive", int((flood0 & (ratio0 > 1.5)).sum()), "{:d}"); m("NBorisAboveTwo", int((flood0 & (ratio0 > 2)).sum()), "{:d}")
mb = flood0 & (ratio0 >= 0.3) & (ratio0 <= 1.0); ma = (A.ratio >= 0.3) & (A.ratio <= 1.0)
m("NMatchBoris", int(mb.sum()), "{:d}"); m("NMatchAnn", int(ma.sum()), "{:d}")
out["matched"] = {}
for k in MK:
    rb, pb, _ = sprho(ratio0[mb], err0[k][mb]); ra, pa, _ = sprho(A.ratio[ma], A[k][ma])
    out["matched"][k] = dict(boris=rb, boris_p=pb, ann=ra, ann_p=pa)
    m(k + "MatchBoris", rb, "{:.2f}"); m(k + "MatchAnn", ra, "{:.2f}")
# 9d. Forcing: event runoff coefficient = quickflow 10 Sep-15 Oct (above the 1-9 Sep mean) / E-OBS rain 5-30 Sep.
# Values near one are implausible for a 200 mm event and point to under-estimated basin rainfall (or reservoir releases).
P = R["P"]
pre = np.asarray((dates >= "2024-09-01") & (dates <= "2024-09-09")); ev = np.asarray((dates >= "2024-09-10") & (dates <= "2024-10-15"))
pw = np.asarray((dates >= "2024-09-05") & (dates <= "2024-09-30"))
qb = np.nanmean(np.where(pre[None], Q, np.nan), 1)
qf = np.nansum(np.clip(np.where(ev[None], Q, np.nan) - qb[:, None], 0, None), 1); psum = np.nansum(np.where(pw[None], P, np.nan), 1)
rc = qf / psum; inr0 = flood0 & ~ood0
m("RcOod", float(np.nanmedian(rc[ood0])), "{:.2f}"); m("RcIn", float(np.nanmedian(rc[inr0])), "{:.2f}")
m("RcMax", float(np.nanmax(rc[ood0])), "{:.2f}"); m("NRcAboveSeven", int(np.sum(rc[ood0] > 0.7)), "{:d}")
m("POod", float(np.nanmedian(psum[ood0])), "{:.0f}"); m("PIn", float(np.nanmedian(psum[inr0])), "{:.0f}")
rr, rp_, _ = sprho(rc[flood0], ratio0[flood0]); m("RcRhoRatio", rr, "{:.2f}")


def partial(x, y, z):
    """Spearman correlation of x and y after removing the linear effect of rank(z) from both ranks."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


out["runoff_coef"] = dict(ood=float(np.nanmedian(rc[ood0])), inrange=float(np.nanmedian(rc[inr0])), rho_ratio=rr)
for k in MK:
    rho, p, n = sprho(rc[flood0], err0[k][flood0]); pr = partial(ratio0[flood0], err0[k][flood0], rc[flood0])
    out["runoff_coef"][k] = dict(rho=rho, p=p, partial_ratio=pr)
    m(k + "RcRho", rho, "{:.2f}"); m(k + "PartialRho", pr, "{:.2f}")
pd.DataFrame(dict(gauge=gauges, river_system=clus, ratio=ratio0, runoff_coef=rc, p_event=psum, flood=flood0, ood=ood0)
             )[flood0].to_csv(os.path.join(RES, "boris_runoff_coef.csv"), index=False)
# 9e. Seed spread: the ensemble median can damp peaks, so report each seed's OOD median error too
out["seed_ood"] = {}
for cfg, t in [("base_nse", "BaseNse"), ("long_nse", "LongNse")]:
    w = np.asarray((dates >= "2024-09-10") & (dates <= "2024-09-30"))
    e = [float(np.nanmedian((100 * (np.nanmax(np.where(w[None], np.load(f), np.nan), 1) - pk0) / pk0)[ood0]))
         for f in sorted(glob.glob(os.path.join(KO, f"sim_lstm_{cfg}_s*.npy")))]
    out["seed_ood"][t] = e  # the macros (SeedOodMin/Max) are written by analyze.py; this is a cross-check
# 9f. The precipitation ratio is not independent of the discharge ratio
if HAS_P:
    m("PRatioRhoR", sprho(rp[flood0], ratio0[flood0])[0], "{:.2f}")

json.dump(out, open(os.path.join(RES, "robustness.json"), "w"), indent=1, default=float)
open(os.path.join(RES, "numbers_rob.tex"), "w").write("".join(mac).replace("{-", r"{\ensuremath{-}"))
print(json.dumps(out, indent=1, default=float))

# ---- 10. Paragraph with the control-run results (kernel C), read by the paper as \KCParagraph ----
# The reading was fixed in the Methods before the runs finished; the wording below states the outcome
# that the asserts check, so a change in the results breaks the build instead of the text.
if {"LongThreeAbs", "EarlyAbs"} <= set(out["cluster_boot"]):
    cb3, cbe = out["cluster_boot"]["LongThreeAbs"], out["cluster_boot"]["EarlyAbs"]
    assert cb3["lo"] > 0, "long-3x no longer worse than base: rewrite the control paragraph"
    assert cbe["lo"] > 0, "early no longer worse than base: rewrite the control paragraph"
    kc = (r"\textbf{Control runs.} Long-3$\times$, which sees each sample as often as base, did not bring the error back "
          r"to the base level: its median absolute Boris error on the flood gauges was \LongThreeNsePkAbs\% (seeds "
          r"\LongThreeNseSeedAbsMin--\LongThreeNseSeedAbsMax), against \BaseNsePkAbs\% for base and \LongNsePkAbs\% for long. "
          r"The paired increase over base was \CBLongThreeAbsMed{} points (river-system CI [\CBLongThreeAbsLo, \CBLongThreeAbsHi], "
          r"sign test \CBLongThreeAbsSignP), as large as for long. Early, trained on 1981--2000 only, was worse still: "
          r"absolute error \EarlyNsePkAbs\% (+\CBEarlyAbsMed{} [\CBEarlyAbsLo, \CBEarlyAbsHi]), OOD error \EarlyNsePkErrOOD\%, "
          r"and test NSE \EarlyNseNSE{} against \BaseNseNSE{} for base. By the reading fixed in advance, reduced exposure does "
          r"not explain the degradation, and the older data do. Early, however, also widens the gap between training and test "
          r"period, so this control cannot separate a property of the 1981--2000 data, such as the changing station network "
          r"behind the forcing, from that gap. The controls are reported without multiplicity adjustment.")
    open(os.path.join(RES, "numbers_c.tex"), "w").write("\\newcommand{\\KCParagraph}{%s}\n" % kc)
