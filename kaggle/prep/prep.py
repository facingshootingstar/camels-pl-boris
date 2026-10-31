# Build compact arrays from CAMELS-PL v1.0.0 (Zenodo 10.5281/zenodo.20133183).
# Output: /kaggle/working/camels_pl.npz + static/benchmark CSVs. Raw data is extracted to /tmp (not kept).
import os, zipfile, urllib.request, hashlib, json
import numpy as np, pandas as pd

URL = "https://zenodo.org/records/20133183/files/CAMELS-PL.zip?download=1"
MD5 = "0ff7d9635c941f4aac49c191376502c9"
ZIP, RAW, OUT = "/tmp/CAMELS-PL.zip", "/tmp/raw", "/kaggle/working"
urllib.request.urlretrieve(URL, ZIP)
assert hashlib.md5(open(ZIP, "rb").read()).hexdigest() == MD5
zipfile.ZipFile(ZIP).extractall(RAW)
R = os.path.join(RAW, "CAMELS-PL")

DYN = ["precipitation_mean", "temperature_mean", "maximum_temperature_mean", "minimum_temperature_mean",
       "relative_humidity_mean", "radiation_global_mean"]   # wind excluded: missing before 1980
topo = pd.read_csv(f"{R}/CAMELS_PL_topographic_attributes.csv", dtype={"gauge_id": str})
gauges = sorted(topo.gauge_id)
dates = pd.date_range("1951-01-01", "2024-10-31", freq="D")
N, T = len(gauges), len(dates)
X = np.full((N, T, len(DYN)), np.nan, np.float32)
Q = np.full((N, T), np.nan, np.float32)        # specific discharge, mm/d
QV = np.full((N, T), np.nan, np.float32)       # volumetric discharge, m3/s
WL = np.full((N, T), np.nan, np.float32)       # water level, cm
SIM_LSTM = np.full((N, T), np.nan, np.float32)
SIM_HBV = np.full((N, T), np.nan, np.float32)
for i, g in enumerate(gauges):
    d = pd.read_csv(f"{R}/timeseries/CAMELS_PL_hydromet_timeseries_{g}.csv", parse_dates=["date"]).set_index("date").reindex(dates)
    X[i] = d[DYN].to_numpy(np.float32)
    Q[i] = d.discharge_spec_obs.to_numpy(np.float32)
    QV[i] = d.discharge_vol_obs.to_numpy(np.float32)
    WL[i] = d.water_level_obs.to_numpy(np.float32)
    s = pd.read_csv(f"{R}/timeseries_simulated/CAMELS_PL_discharge_sim_{g}.csv", parse_dates=["date"]).set_index("date").reindex(dates)
    SIM_LSTM[i] = s.discharge_spec_sim_lstm.to_numpy(np.float32)
    SIM_HBV[i] = s.discharge_spec_sim_hbv.to_numpy(np.float32)
    if i == 0:
        assert np.allclose(np.nan_to_num(s.discharge_spec_obs.to_numpy()), np.nan_to_num(Q[i]), atol=1e-3)

# Input gaps (RH sporadic, radiation 2021-2024): linear interpolation in time, then edge fill.
nan_frac = {c: float(np.isnan(X[:, :, k]).mean()) for k, c in enumerate(DYN)}
for i in range(N):
    X[i] = pd.DataFrame(X[i]).interpolate(limit_direction="both").to_numpy(np.float32)
assert not np.isnan(X).any()

clim = pd.read_csv(f"{R}/CAMELS_PL_climatic_attributes.csv", dtype={"gauge_id": str})
land = pd.read_csv(f"{R}/CAMELS_PL_landcover_attributes.csv", dtype={"gauge_id": str})
soil = pd.read_csv(f"{R}/CAMELS_PL_soil_attributes.csv", dtype={"gauge_id": str})
# Hydrologic signatures are excluded: they are computed from discharge that includes the test period.
ST = {"topo": ["area_metadata", "elev_mean", "elev_std", "elev_min", "elev_max"],
      "clim": ["p_mean", "p_seasonality", "frac_snow", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"],
      "land": ["artificial_surfaces_perc", "agricultural_areas_perc", "forests_and_seminatural_areas_perc", "wetlands_perc", "water_bodies_perc"],
      "soil": ["clay_0_30cm_mean", "sand_0_30cm_mean", "silt_0_30cm_mean", "bulk_density_0_30cm_mean", "soil_organic_carbon_0_30cm_mean"]}
S = topo[["gauge_id"]]
for df, cols in [(topo, ST["topo"]), (clim, ST["clim"]), (land, ST["land"]), (soil, ST["soil"])]:
    S = S.merge(df[["gauge_id"] + cols], on="gauge_id", how="left")
S = S.set_index("gauge_id").loc[gauges]
S["area_metadata"] = np.log10(S["area_metadata"])
S = S.fillna(S.median())
static_cols = list(S.columns)

np.savez_compressed(f"{OUT}/camels_pl.npz", gauges=np.array(gauges), dates=dates.strftime("%Y-%m-%d").to_numpy(),
                    X=X, Q=Q, QV=QV, WL=WL, SIM_LSTM=SIM_LSTM, SIM_HBV=SIM_HBV,
                    S=S.to_numpy(np.float32), dyn_cols=np.array(DYN), static_cols=np.array(static_cols))
topo.to_csv(f"{OUT}/topographic_attributes.csv", index=False)
pd.read_csv(f"{R}/CAMELS_PL_simulation_benchmark.csv", dtype={"gauge_id": str}).to_csv(f"{OUT}/simulation_benchmark.csv", index=False)
pd.read_csv(f"{R}/CAMELS_PL_hydrologic_attributes.csv", dtype={"gauge_id": str}).to_csv(f"{OUT}/hydrologic_attributes.csv", index=False)
info = {"md5": MD5, "N": N, "T": T, "dyn": DYN, "static": static_cols, "input_nan_frac_before_fill": nan_frac,
        "q_obs_frac_available": float(np.isfinite(Q).mean())}
json.dump(info, open(f"{OUT}/prep_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
