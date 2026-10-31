# Diagnostic: for gauges where the Boris peak exceeded the 2005-2015 training maximum, compare the
# event that set the 1981-2015 maximum with Boris in terms of peak discharge and antecedent precipitation.
import glob
import numpy as np, pandas as pd

D = np.load(glob.glob("/kaggle/input/**/camels_pl.npz", recursive=True)[0], allow_pickle=True)
dates = pd.to_datetime(D["dates"]); gauges = D["gauges"]; Q = D["Q"]; P = D["X"][:, :, 0]


def mask(a, b):
    return np.asarray((dates >= a) & (dates <= b))


base = mask("2005-11-01", "2015-12-31")
long_ = mask("1981-11-01", "2000-10-31") | base
boris = mask("2024-09-10", "2024-09-30")
P10 = pd.DataFrame(P.T).rolling(10).sum().to_numpy().T  # 10-day precipitation ending on each day
rows = []
for i in range(len(gauges)):
    qb = np.where(boris, Q[i], np.nan)
    if np.isfinite(qb).sum() < 15:
        continue
    pk = np.nanmax(qb); tb = int(np.nanargmax(qb))
    qmax_base = np.nanmax(np.where(base, Q[i], np.nan))
    ql = np.where(long_, Q[i], np.nan); tl = int(np.nanargmax(ql))
    p5 = lambda t: float(P[i, t - 4:t + 1].sum())  # 5-day precipitation ending on the peak day
    p10 = lambda t: float(P[i, t - 9:t + 1].sum())
    rows.append(dict(gauge=gauges[i], boris_peak=pk, boris_date=str(dates[tb].date()), boris_p5=p5(tb), boris_p10=p10(tb),
                     qmax_base=qmax_base, long_max=float(ql[tl]), long_max_date=str(dates[tl].date()),
                     long_p5=p5(tl), long_p10=p10(tl),
                     # precipitation-based OOD ratio (independent of observed discharge): Boris vs training maxima
                     p10_boris_max=float(np.nanmax(P10[i, boris])), p10_max_base=float(np.nanmax(P10[i, base])),
                     p10_max_long=float(np.nanmax(P10[i, long_]))))
df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/event_compare.csv", index=False)
o = df[df.boris_peak > df.qmax_base]
print(o.to_string())
print("OOD gauges", len(o), "long max year counts", o.long_max_date.str[:4].value_counts().to_dict())
print("median peak/P10: boris", float((o.boris_peak / o.boris_p10).median()), "long-max event", float((o.long_max / o.long_p10).median()))
print("DONE")
