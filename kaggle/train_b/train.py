# Train regional LSTMs (2x2: training record x loss) and LightGBM on CAMELS-PL; save test-period simulations.
# Splits follow the CAMELS-PL benchmark: val 2000-11-01..2005-10-31, train 2005-11-01..2015-12-31, test 2016-01-01..2024-10-31.
import os, glob, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn

SMOKE = False
PART = "B"  # A: base_* LSTMs + LightGBM, B: long_* LSTMs (split to stay under the 12 h Kaggle session limit)
OUT = "/kaggle/working"
src = glob.glob("/kaggle/input/**/camels_pl.npz", recursive=True)[0]
D = np.load(src, allow_pickle=True)
dates = pd.to_datetime(D["dates"]); gauges = D["gauges"]
X, Q, S = D["X"], D["Q"], D["S"]
N, T, F = X.shape
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", "torch", torch.__version__, flush=True)


def mask(a, b):
    return np.asarray((dates >= a) & (dates <= b))


VAL = mask("2000-11-01", "2005-10-31")
TEST = mask("2016-01-01", "2024-10-31")
TRAIN = {"base": mask("2005-11-01", "2015-12-31"),
         "long": mask("1981-11-01", "2000-10-31") | mask("2005-11-01", "2015-12-31")}

# Normalisation statistics from the base training period only (shared by all configs).
tb = TRAIN["base"]
x_mu = X[:, tb].reshape(-1, F).mean(0); x_sd = X[:, tb].reshape(-1, F).std(0)
s_mu = S.mean(0); s_sd = S.std(0) + 1e-6
q_mu = float(np.nanmean(Q[:, tb])); q_sd = float(np.nanstd(Q[:, tb]))
Xn = ((X - x_mu) / x_sd).astype(np.float32)
Sn = ((S - s_mu) / s_sd).astype(np.float32)
SEQ, LAST, BATCH, HID = 365, 30, 256, 256
EPOCHS, STEPS = (2, 30) if SMOKE else (30, 500)
SEEDS = [0] if SMOKE else [0, 1, 2]

Xg = torch.tensor(np.concatenate([Xn, np.repeat(Sn[:, None, :], T, 1)], 2), device=dev)  # N,T,F+S


class LSTM(nn.Module):
    def __init__(self, nin, hid=HID, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(nin, hid, batch_first=True)
        self.drop = nn.Dropout(drop); self.head = nn.Linear(hid, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(self.drop(h[:, -LAST:])).squeeze(-1) * q_sd + q_mu


def windows(maskT, step):
    """End indices of windows whose last LAST days lie in maskT (inference)."""
    idx = np.where(maskT)[0]
    ends = list(range(idx[0] + LAST - 1, idx[-1] + 1, step))
    if ends[-1] != idx[-1]:
        ends.append(idx[-1])
    return ends


@torch.no_grad()
def simulate(model, maskT):
    model.eval()
    out = np.full((N, T), np.nan, np.float32)
    for e in windows(maskT, LAST):
        p = model(Xg[:, e - SEQ + 1:e + 1]).clamp_min(0).cpu().numpy()
        out[:, e - LAST + 1:e + 1] = p
    out[:, ~maskT] = np.nan
    return out


def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 30:
        return np.nan
    o, s = o[m], s[m]
    return 1 - ((o - s) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def train_lstm(cfg, seed):
    split, loss = cfg.split("_")
    torch.manual_seed(seed); np.random.seed(seed)
    tm = TRAIN[split]
    Qtr = np.where(tm[None], Q, np.nan)
    sig = torch.tensor(np.nanstd(Qtr, 1), device=dev)
    q99 = torch.tensor(np.nanpercentile(Qtr, 99, axis=1), device=dev)
    Qtr_g = torch.tensor(Qtr, device=dev)
    # candidate (basin, end) pairs with >=1 valid target among the last LAST days
    valid = np.isfinite(Qtr)
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(valid, 1)], 1)
    ends = np.arange(SEQ - 1, T)
    cnt = cs[:, ends + 1] - cs[:, ends + 1 - LAST]
    bi, ei = np.nonzero(cnt > 0); ei = ends[ei]
    print(cfg, seed, "train windows", len(bi), flush=True)
    model = LSTM(Xg.shape[2]).to(dev)
    with torch.no_grad():  # forget-gate bias init = 3 (Kratzert et al.)
        model.lstm.bias_hh_l0[HID:2 * HID] = 3.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    off = torch.arange(-SEQ + 1, 1, device=dev)
    best, best_state, hist = -np.inf, None, []
    for ep in range(EPOCHS):
        if ep == 20:
            for g in opt.param_groups:
                g["lr"] = 5e-4
        model.train(); t0 = time.time(); tot = 0.0
        for _ in range(STEPS):
            k = np.random.randint(0, len(bi), BATCH)
            b = torch.tensor(bi[k], device=dev); e = torch.tensor(ei[k], device=dev)
            tt = e[:, None] + off[None]
            x = Xg[b[:, None], tt]
            y = Qtr_g[b[:, None], tt[:, -LAST:]]
            yh = model(x)
            w = 1.0 / (sig[b, None] + 0.1) ** 2
            if loss == "peak":
                w = w * (1 + 4.0 * (y > q99[b, None]).float())
            m = torch.isfinite(y)
            l = (w * (yh - torch.nan_to_num(y)) ** 2)[m].mean()
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += l.item()
        sv = simulate(model, VAL)
        med = float(np.nanmedian([nse(Q[i, VAL], sv[i, VAL]) for i in range(N)]))
        hist.append({"epoch": ep + 1, "loss": tot / STEPS, "val_median_nse": med, "sec": round(time.time() - t0, 1)})
        print(cfg, seed, hist[-1], flush=True)
        if med > best:
            best, best_state = med, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return simulate(model, TEST), hist


# ---- LightGBM baseline (base training period) ----
def lgb_features():
    P = X[:, :, 0]; Tm = X[:, :, 1]

    def roll(a, w, f="sum"):
        r = pd.DataFrame(a.T).rolling(w, min_periods=1)
        return (r.sum() if f == "sum" else r.mean()).to_numpy().T

    feats = [P] + [np.roll(P, l, 1) for l in range(1, 7)]
    feats += [roll(P, w) for w in (3, 7, 14, 30, 60, 90, 180, 365)]
    feats += [Tm, roll(Tm, 7, "mean"), roll(Tm, 30, "mean"), roll(Tm, 90, "mean"),
              X[:, :, 2], X[:, :, 3], X[:, :, 4], X[:, :, 5], roll(X[:, :, 5], 30, "mean")]
    feats += [roll(np.where(Tm < 0, P, 0), 90)]  # snow proxy: precipitation on sub-zero days, last 90 d
    doy = dates.dayofyear.to_numpy()
    feats += [np.tile(np.sin(2 * np.pi * doy / 365.25), (N, 1)), np.tile(np.cos(2 * np.pi * doy / 365.25), (N, 1))]
    dyn = np.stack(feats, 2).astype(np.float32)
    return np.concatenate([dyn, np.repeat(S[:, None, :].astype(np.float32), T, 1)], 2)


def train_lgb(seed, Fm):
    import lightgbm as lgb
    tm = TRAIN["base"].copy()

    def rows(mk):
        bi, ti_ = np.nonzero(mk[None] & np.isfinite(Q))
        return Fm[bi, ti_], Q[bi, ti_]

    xtr, ytr = rows(tm); xva, yva = rows(VAL)
    params = dict(objective="l2", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, seed=seed, verbose=-1, num_threads=4)
    bst = lgb.train(params, lgb.Dataset(xtr, ytr), 200 if SMOKE else 3000, valid_sets=[lgb.Dataset(xva, yva)],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    out = np.full((N, T), np.nan, np.float32)
    ti_ = np.where(TEST)[0]
    for i in range(N):
        out[i, ti_] = np.clip(bst.predict(Fm[i, ti_], num_iteration=bst.best_iteration), 0, None)
    return out, {"best_iter": int(bst.best_iteration)}


ti = np.where(TEST)[0]
logs = {}
CFGS = {"A": ["base_nse", "base_peak"], "B": ["long_nse", "long_peak"]}[PART]
for cfg in (["base_nse"] if SMOKE else CFGS):
    for seed in SEEDS:
        sim, hist = train_lstm(cfg, seed)
        np.save(f"{OUT}/sim_lstm_{cfg}_s{seed}.npy", sim[:, ti])
        logs[f"lstm_{cfg}_s{seed}"] = hist
        json.dump(logs, open(f"{OUT}/train_logs.json", "w"), indent=1)
Fm = lgb_features() if PART == "A" else None
for seed in (SEEDS if PART == "A" else []):
    sim, info = train_lgb(seed, Fm)
    np.save(f"{OUT}/sim_lgbm_s{seed}.npy", sim[:, ti]); logs[f"lgbm_s{seed}"] = info
    json.dump(logs, open(f"{OUT}/train_logs.json", "w"), indent=1)

# Reference arrays for local analysis (test period + pre-test record maxima).
pre = np.asarray(dates < pd.Timestamp("2016-01-01"))
np.savez_compressed(f"{OUT}/eval_ref.npz", gauges=gauges, dates=D["dates"][ti], Q=Q[:, ti], QV=D["QV"][:, ti],
                    WL=D["WL"][:, ti], P=X[:, ti, 0], SIM_LSTM=D["SIM_LSTM"][:, ti], SIM_HBV=D["SIM_HBV"][:, ti],
                    qmax_pre2016=np.nanmax(np.where(pre[None], Q, np.nan), 1),
                    qmax_train_base=np.nanmax(np.where(TRAIN["base"][None], Q, np.nan), 1),
                    qmax_train_long=np.nanmax(np.where(TRAIN["long"][None], Q, np.nan), 1),
                    q99_train_base=np.nanpercentile(np.where(TRAIN["base"][None], Q, np.nan), 99, axis=1))
print("DONE", flush=True)
