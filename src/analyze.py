"""Evaluate test-period simulations (2016-01-01..2024-10-31) and the Storm Boris event (Sept 2024).

Inputs : results/kaggle_out/{eval_ref.npz, sim_*.npy, train_logs.json}, results/kaggle_prep/topographic_attributes.csv
Outputs: results/metrics_per_gauge.csv, results/boris_per_gauge.csv, results/summary.json,
         results/numbers.tex (LaTeX macros used verbatim by the paper), figures/*.pdf
"""
import sys, json, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KO = os.path.join(ROOT, "results", "kaggle_out")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

R = np.load(os.path.join(KO, "eval_ref.npz"), allow_pickle=True)
dates = pd.to_datetime(R["dates"]); gauges = R["gauges"].astype(str)
Q = R["Q"]; N = len(gauges)
BORIS = np.asarray((dates >= "2024-09-10") & (dates <= "2024-09-30"))

# ---- models: benchmark sims + ensemble means of our seeds ----
models = {"HBV (benchmark)": R["SIM_HBV"], "LSTM (benchmark)": R["SIM_LSTM"]}
seeds = {}
for cfg in ["base_nse", "base_peak", "long_nse", "long_peak", "early_nse", "long3x_nse"]:
    fs = sorted(glob.glob(os.path.join(KO, f"sim_lstm_{cfg}_s*.npy")))
    if fs:
        seeds[f"LSTM {cfg}"] = [np.load(f) for f in fs]
fs = sorted(glob.glob(os.path.join(KO, "sim_lgbm_s*.npy")))
if fs:
    seeds["LightGBM"] = [np.load(f) for f in fs]
for k, v in seeds.items():
    models[k] = np.median(v, 0)  # ensemble median, matching the CAMELS-PL benchmark files


def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 365:
        return np.nan
    o, s = o[m], s[m]
    return 1 - ((o - s) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def kge(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 365:
        return np.nan
    o, s = o[m], s[m]
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def fhv(o, s, h=0.02):
    """Percent bias of the top-h flow-duration segment (Yilmaz et al. 2008)."""
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 365:
        return np.nan
    o, s = np.sort(o[m])[::-1], np.sort(s[m])[::-1]
    n = max(1, int(h * len(o)))
    return 100 * (s[:n].sum() - o[:n].sum()) / o[:n].sum()


rows = []
for name, S in models.items():
    for i in range(N):
        rows.append(dict(model=name, gauge=gauges[i], nse=nse(Q[i], S[i]), kge=kge(Q[i], S[i]), fhv=fhv(Q[i], S[i])))
M = pd.DataFrame(rows)
M.to_csv(os.path.join(RES, "metrics_per_gauge.csv"), index=False)

# per-seed medians (spread across seeds)
seed_stats = {}
for k, v in seeds.items():
    med = [np.nanmedian([nse(Q[i], s[i]) for i in range(N)]) for s in v]
    seed_stats[k] = dict(n=len(v), mean=float(np.mean(med)), std=float(np.std(med, ddof=1)) if len(v) > 1 else 0.0)

# ---- Storm Boris event ----
ob = np.where(BORIS[None], Q, np.nan)
nvalid = np.isfinite(ob).sum(1)
obs_pk = np.nanmax(np.where(nvalid[:, None] >= 15, ob, np.nan), 1)
obs_t = np.array([np.nanargmax(ob[i]) if nvalid[i] >= 15 else -1 for i in range(N)])
ood = obs_pk / R["qmax_train_base"]
rec = obs_pk > R["qmax_pre2016"]
brow = []
for name, S in models.items():
    sb = np.where(BORIS[None], S, np.nan)
    for i in range(N):
        if nvalid[i] < 15 or not np.isfinite(sb[i]).any():
            continue
        sp = np.nanmax(sb[i]); st = np.nanargmax(sb[i])
        brow.append(dict(model=name, gauge=gauges[i], obs_peak=obs_pk[i], sim_peak=sp,
                         rel_err=100 * (sp - obs_pk[i]) / obs_pk[i], timing=int(st - obs_t[i]),
                         ood_ratio=ood[i], record=bool(rec[i]), q99=R["q99_train_base"][i]))
B = pd.DataFrame(brow)
B.to_csv(os.path.join(RES, "boris_per_gauge.csv"), index=False)

# Event subset: gauges where the Boris peak exceeded the base-training 99th percentile (i.e., a real flood response).
flood = B[B.model == "HBV (benchmark)"].query("obs_peak > q99").gauge.tolist()
oodg = B[B.model == "HBV (benchmark)"].query("ood_ratio > 1").gauge.tolist()
summ = {"n_gauges": int(N), "n_boris_valid": int((nvalid >= 15).sum()), "n_boris_flood": len(flood),
        "n_boris_ood": len(oodg), "n_boris_record": int(B[(B.model == "HBV (benchmark)") & B.record].shape[0]),
        "seed_stats": seed_stats, "models": {}}
for name in models:
    m = M[M.model == name]; b = B[B.model == name]
    bf = b[b.gauge.isin(flood)]; bo = b[b.gauge.isin(oodg)]
    rho = spearmanr(bf.ood_ratio, bf.rel_err) if len(bf) > 3 else (np.nan, np.nan)
    summ["models"][name] = dict(
        nse_med=float(m.nse.median()), kge_med=float(m.kge.median()), fhv_med=float(m.fhv.median()),
        nse_neg=int((m.nse < 0).sum()),
        pk_err_med_flood=float(bf.rel_err.median()), pk_abs_med_flood=float(bf.rel_err.abs().median()),
        pk_err_med_ood=float(bo.rel_err.median()) if len(bo) else None,
        under_frac_flood=float((bf.rel_err < 0).mean()),
        timing_abs_med_flood=float(bf.timing.abs().median()),
        rho_ood_err=float(rho[0]), rho_p=float(rho[1]))


def wil(a, b, col, df, subset=None):
    x = df[df.model == a].set_index("gauge")[col]; y = df[df.model == b].set_index("gauge")[col]
    j = x.index.intersection(y.index)
    if subset is not None:
        j = j.intersection(subset)
    x, y = x[j], y[j]; ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6:
        return None
    st = wilcoxon(x[ok], y[ok])
    return dict(n=int(ok.sum()), med_diff=float((x[ok] - y[ok]).median()), p=float(st.pvalue))


B["abs_err"] = B.rel_err.abs()
ref = "LSTM base_nse" if "LSTM base_nse" in models else "LSTM (benchmark)"
tests = {}
for other in models:
    if other == ref:
        continue
    tests[f"{other} vs {ref}"] = dict(nse=wil(other, ref, "nse", M), abs_peak=wil(other, ref, "abs_err", B, flood),
                                      ood_err=wil(other, ref, "rel_err", B, oodg))
summ["wilcoxon"] = tests

# Seed-level Boris statistics: is a difference between configurations larger than the spread across seeds?
pk_i = {g: i for i, g in enumerate(gauges)}
fi = np.array([pk_i[g] for g in flood]); oi = np.array([pk_i[g] for g in oodg])
seed_boris = {}
for k, v in seeds.items():
    ab, so = [], []
    for s in v:
        sp = np.nanmax(np.where(BORIS[None], s, np.nan), 1)
        e = 100 * (sp - obs_pk) / obs_pk
        ab.append(float(np.nanmedian(np.abs(e[fi])))); so.append(float(np.nanmedian(e[oi])))
    seed_boris[k] = dict(abs_flood=ab, err_ood=so)
summ["seed_boris"] = seed_boris
# How many base-OOD gauges are still OOD relative to the long (1981-2015) training maximum?
summ["n_ood_long"] = int(np.sum(obs_pk[oi] > R["qmax_train_long"][oi]))
json.dump(summ, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))

# ---- LaTeX macros: every number in the paper comes from here ----
def mac(name, val, fmt="{:.2f}"):
    name = re.sub(r"[^A-Za-z]", "", name)
    s = "--" if val is None or (isinstance(val, float) and np.isnan(val)) else fmt.format(val)
    return "\\newcommand{\\%s}{%s}\n" % (name, re.sub(r"^-", r"\\ensuremath{-}", s))  # typographic minus


def pstr(p):
    """Full relation for the text, e.g. $p<0.001$ or $p=0.068$."""
    return "$p<0.001$" if p < 0.001 else "$p=%.3f$" % p


tag = {"HBV (benchmark)": "HBV", "LSTM (benchmark)": "LSTMbm", "LSTM base_nse": "BaseNse", "LSTM base_peak": "BasePeak",
       "LSTM long_nse": "LongNse", "LSTM long_peak": "LongPeak", "LightGBM": "LGBM",
       "LSTM early_nse": "EarlyNse", "LSTM long3x_nse": "LongThreeNse"}  # last two: controls added after review round 1
with open(os.path.join(RES, "numbers.tex"), "w") as f:
    for k in ["n_gauges", "n_boris_valid", "n_boris_flood", "n_boris_ood", "n_boris_record"]:
        f.write(mac("N" + k.title().replace("_", ""), summ[k], "{:d}"))
    for name, d in summ["models"].items():
        t = tag[name]
        f.write(mac(t + "NSE", d["nse_med"], "{:.3f}") + mac(t + "KGE", d["kge_med"], "{:.3f}") + mac(t + "FHV", d["fhv_med"], "{:.1f}"))
        f.write(mac(t + "NegNSE", d["nse_neg"], "{:d}"))
        f.write(mac(t + "PkErr", d["pk_err_med_flood"], "{:.1f}") + mac(t + "PkAbs", d["pk_abs_med_flood"], "{:.1f}"))
        f.write(mac(t + "PkErrOOD", d["pk_err_med_ood"], "{:.1f}") + mac(t + "Under", 100 * d["under_frac_flood"], "{:.0f}"))
        f.write(mac(t + "Timing", d["timing_abs_med_flood"], "{:.1f}") + mac(t + "Rho", d["rho_ood_err"], "{:.2f}"))
        f.write("\\newcommand{\\%sRhoP}{%s}\n" % (t, pstr(d["rho_p"])))
    for k, d in seed_stats.items():
        t = tag[k]; f.write(mac(t + "SeedMean", d["mean"], "{:.3f}") + mac(t + "SeedSd", d["std"], "{:.3f}"))
        sb = seed_boris[k]
        f.write(mac(t + "SeedAbsMin", min(sb["abs_flood"]), "{:.1f}") + mac(t + "SeedAbsMax", max(sb["abs_flood"]), "{:.1f}"))
        f.write(mac(t + "SeedOodMin", min(sb["err_ood"]), "{:.1f}") + mac(t + "SeedOodMax", max(sb["err_ood"]), "{:.1f}"))
    f.write(mac("NNOodLong", summ["n_ood_long"], "{:d}"))
    # Best validation (2000-2005) median NSE per LSTM configuration, averaged over seeds (from the Kaggle training logs).
    logs = json.load(open(os.path.join(KO, "train_logs.json")))
    for cfg in ["base_nse", "base_peak", "long_nse", "long_peak", "early_nse", "long3x_nse"]:
        best = [max(h["val_median_nse"] for h in v) for k, v in logs.items() if k.startswith(f"lstm_{cfg}_s")]
        if best:
            f.write(mac(tag[f"LSTM {cfg}"] + "Val", float(np.mean(best)), "{:.3f}"))
    # Event comparison from the Kaggle diagnostic kernel (kaggle/diag): Boris vs the event that set the 1981-2015 maximum.
    ev = os.path.join(RES, "diag", "event_compare.csv")
    if os.path.exists(ev):
        E = pd.read_csv(ev, dtype={"gauge": str})
        E = E[E.gauge.isin(oodg)]
        f.write(mac("NNOodMaxNinetySeven", int((E.long_max_date.str[:4] == "1997").sum()), "{:d}"))
        f.write(mac("RatioBoris", float((E.boris_peak / E.boris_p10).median()), "{:.2f}"))
        f.write(mac("RatioLongMax", float((E.long_max / E.long_p10).median()), "{:.2f}"))
        f.write(mac("PTenBoris", float(E.boris_p10.median()), "{:.0f}") + mac("PTenLongMax", float(E.long_p10.median()), "{:.0f}"))
        summ["event_compare_n"] = len(E)
    for k, d in tests.items():
        t = tag[k.split(" vs ")[0]]
        for metric in ("nse", "abs_peak", "ood_err"):
            if d[metric]:
                p = d[metric]["p"]
                f.write("\\newcommand{\\%s}{%s}\n" % (re.sub(r"[^A-Za-z]", "", t + "W" + metric + "P"), pstr(p)))
                f.write(mac(t + "W" + metric + "D", d[metric]["med_diff"], "{:.3f}" if metric == "nse" else "{:.1f}"))

# ---- figures ----
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
order = [m for m in list(tag)[:7] if m in models]  # figures show the main models only
PRETTY = {"LSTM base_nse": "LSTM base", "LSTM base_peak": "LSTM base, peak", "LSTM long_nse": "LSTM long",
          "LSTM long_peak": "LSTM long, peak"}  # paper names for the configurations
lab = lambda m: PRETTY.get(m, m)
fig, ax = plt.subplots(figsize=(4.6, 3))
for name in order:
    v = np.sort(M[M.model == name].nse.dropna().values)
    ax.plot(v, np.linspace(0, 1, len(v)), label=lab(name), lw=1.2)
ax.set_xlim(-0.5, 1); ax.set_xlabel("NSE (test 2016-2024)"); ax.set_ylabel("CDF over gauges"); ax.legend(fontsize=6); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "nse_cdf.pdf"))

fig, ax = plt.subplots(figsize=(4.6, 3))
for name, mk in [("HBV (benchmark)", "s"), ("LSTM base_nse", "o"), ("LSTM long_nse", "^"), ("LightGBM", "x")]:
    b = B[(B.model == name) & B.gauge.isin(flood)]
    ax.scatter(b.ood_ratio, b.rel_err, s=10, marker=mk, label=lab(name), alpha=.75)
ax.axhline(0, c="k", lw=.6); ax.axvline(1, c="gray", ls="--", lw=.6)
ax.axvspan(1, ax.get_xlim()[1], color="gray", alpha=.08, lw=0)  # out-of-distribution region
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=100, linscale=2); ax.set_xlabel("Boris peak / max. training-period discharge")
ax.set_ylabel("Peak error (%; log scale above 100)")
ax.set_yticks([-100, -50, 0, 50, 100, 200]); ax.set_yticklabels(["-100", "-50", "0", "50", "100", "200"])
ax.legend(fontsize=6); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(FIG, "peak_vs_ood.pdf"))

# hydrographs of the three gauges with the largest OOD ratio
top = B[B.model == "HBV (benchmark)"].sort_values("ood_ratio", ascending=False).gauge.head(3).tolist()
win = np.asarray((dates >= "2024-09-05") & (dates <= "2024-10-05"))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
topo = glob.glob(os.path.join(RES, "kaggle_prep", "topographic_attributes.csv"))
names = pd.read_csv(topo[0], dtype={"gauge_id": str}).set_index("gauge_id") if topo else None
for ax, g in zip(axs, top):
    i = list(gauges).index(g)
    ax.plot(dates[win], Q[i, win], "k", lw=1.6, label="Observed")
    for name in order:
        ax.plot(dates[win], models[name][i, win], lw=.9, label=lab(name))
    ttl = g if names is None or g not in names.index else f"{names.loc[g, 'gauge_name']} ({names.loc[g, 'water_body_name']})"
    ax.set_title(ttl, fontsize=7); ax.tick_params(axis="x", labelsize=6)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=[5, 15, 25])); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
axs[0].set_ylabel("Q (mm/d)")
h, l = axs[0].get_legend_handles_labels()
leg = fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
for ln in leg.get_lines():
    ln.set_linewidth(2.2)
fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(os.path.join(FIG, "boris_hydrographs.pdf"))
print("figures + numbers.tex written")
