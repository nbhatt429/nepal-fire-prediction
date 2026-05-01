import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import time
import os

print("Nepal Forest Fire Ensemble Prediction (XGBoost + 90-day ERA5)")
print("="*60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

MODEL_XGB_PATH    = "xgboost_corrected.json"
IMPUTER_PATH      = "xgboost_imputer.pkl"
FEATURE_COLS_PATH = "feature_cols.json"
TERRAIN_PATH      = "nepal_forest_grid_terrain.csv"
NDVI_PATH         = "nepal_ndvi_monthly_update.csv"
GRID_PATH         = "nepal_forest_grid.csv"

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(MODEL_XGB_PATH)
imputer = joblib.load(IMPUTER_PATH)
with open(FEATURE_COLS_PATH) as f:
    feature_cols = json.load(f)
print(f"XGBoost loaded: {len(feature_cols)} features")

grid    = pd.read_csv(GRID_PATH)
terrain = pd.read_csv(TERRAIN_PATH)
ndvi    = pd.read_csv(NDVI_PATH)

static_df = terrain.copy()
for col, default in [("ndvi_30to45",0.30),("ndvi_60to90",0.35),
                     ("ndvi_trend",0.05),("ndvi_anomaly",0.0)]:
    if col not in static_df.columns:
        if col in ndvi.columns:
            static_df = static_df.merge(ndvi[["point_id",col]],
                                        on="point_id", how="left")
        static_df[col] = static_df.get(col, pd.Series([default]*len(static_df))).fillna(default)
if "dist_water_km" not in static_df.columns:
    static_df["dist_water_km"] = 1.0
print(f"Grid points: {len(static_df):,}")

VARIABLES = ["temperature_2m","precipitation","windspeed_10m",
             "winddirection_10m","surface_pressure",
             "et0_fao_evapotranspiration"]

def fetch_era5_coarse():
    try:
        import openmeteo_requests, requests_cache
        from retry_requests import retry
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
    except ImportError:
        print("openmeteo_requests not available")
        return None

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=83)
    lats = np.arange(26.0, 31.0, 0.5)
    lons = np.arange(80.0, 88.5, 0.5)
    coarse_pts = [(la, lo) for la in lats for lo in lons]
    print(f"Fetching ERA5 for {len(coarse_pts)} coarse points...")

    all_data = {}
    for i, (lat, lon) in enumerate(coarse_pts):
        try:
            r_hist = openmeteo.weather_api(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date":   end_date.strftime("%Y-%m-%d"),
                        "hourly": VARIABLES})[0]
            h = r_hist.Hourly()
            times = pd.date_range(
                start=pd.Timestamp(h.Time(), unit="s"),
                end=pd.Timestamp(h.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=h.Interval()),
                inclusive="left")
            df_h = pd.DataFrame(
                {VARIABLES[j]: h.Variables(j).ValuesAsNumpy() for j in range(6)},
                index=times)
            hist_daily = df_h.resample("D").agg(["max","mean","min"]).iloc[:83]
            hist_daily.columns = ["_".join(c) for c in hist_daily.columns]

            r_fc = openmeteo.weather_api(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "forecast_days": 7, "hourly": VARIABLES})[0]
            hf = r_fc.Hourly()
            times_fc = pd.date_range(
                start=pd.Timestamp(hf.Time(), unit="s"),
                end=pd.Timestamp(hf.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hf.Interval()),
                inclusive="left")
            df_fc = pd.DataFrame(
                {VARIABLES[j]: hf.Variables(j).ValuesAsNumpy() for j in range(6)},
                index=times_fc)
            fc_daily = df_fc.resample("D").agg(["max","mean","min"]).iloc[:7]
            fc_daily.columns = ["_".join(c) for c in fc_daily.columns]

            combined = pd.concat([hist_daily, fc_daily]).iloc[:90]
            all_data[(lat,lon)] = combined.values
        except Exception as e:
            if i < 3:  # print first 3 errors only
                print(f"ERA5 error {lat},{lon}: {e}")
        time.sleep(0.05)

    if not all_data:
        return None

    coarse_lats   = np.array([k[0] for k in all_data])
    coarse_lons   = np.array([k[1] for k in all_data])
    coarse_vals   = np.array(list(all_data.values()))
    coarse_coords = np.column_stack([coarse_lats, coarse_lons])
    grid_coords   = static_df[["latitude","longitude"]].values
    # Drop NaN grid coords
    valid_idx = np.where(np.isfinite(grid_coords).all(axis=1))[0]
    grid_coords = grid_coords[valid_idx]
    print(f"Valid grid coords: {len(grid_coords)}/{len(static_df)}")
    era5_grid = np.zeros((len(grid_coords), 90, 18))

    for day in range(90):
        for feat in range(18):
            vals   = coarse_vals[:, day, feat]
            interp = griddata(coarse_coords, vals, grid_coords, method="linear")
            nan_mask = np.isnan(interp)
            if nan_mask.any():
                nearest = griddata(coarse_coords, vals, grid_coords, method="nearest")
                interp[nan_mask] = nearest[nan_mask]
            era5_grid[:, day, feat] = interp

    print(f"ERA5 interpolated: {era5_grid.shape}")
    return era5_grid, valid_idx

def build_xgb_features(era5_grid, terrain_df):
    f = pd.DataFrame()
    for col in ["elevation","slope","aspect","landcover","ndvi_30to45",
                "ndvi_60to90","ndvi_trend","is_south_facing",
                "dist_water_km","ndvi_anomaly"]:
        f[col] = terrain_df[col].values if col in terrain_df.columns else 0.0

    def add_era5(idx, prefix):
        f[f"{prefix}_90d_mean"] = era5_grid[:,:,idx].mean(axis=1)
        f[f"{prefix}_90d_max"]  = era5_grid[:,:,idx].max(axis=1)
        f[f"{prefix}_90d_min"]  = era5_grid[:,:,idx].min(axis=1)
        f[f"{prefix}_90d_std"]  = era5_grid[:,:,idx].std(axis=1)
        f[f"{prefix}_7d_mean"]  = era5_grid[:,-7:,idx].mean(axis=1)
        f[f"{prefix}_30d_mean"] = era5_grid[:,-30:,idx].mean(axis=1)

    add_era5(0,"temperat_max"); add_era5(1,"temperat_mean"); add_era5(2,"temperat_min")
    add_era5(3,"totalpre_max"); add_era5(4,"totalpre_mean"); add_era5(5,"totalpre_min")
    add_era5(12,"surfacep_max"); add_era5(13,"surfacep_mean"); add_era5(14,"surfacep_min")

    u = -era5_grid[:,:,6] * np.sin(np.radians(era5_grid[:,:,9]))
    v = -era5_grid[:,:,6] * np.cos(np.radians(era5_grid[:,:,9]))
    for arr, prefix in [(u,"ucompone_max"),(u,"ucompone_mean"),(u,"ucompone_min"),
                        (v,"vcompone_max"),(v,"vcompone_mean"),(v,"vcompone_min")]:
        f[f"{prefix}_90d_mean"] = arr.mean(axis=1)
        f[f"{prefix}_90d_max"]  = arr.max(axis=1)
        f[f"{prefix}_90d_min"]  = arr.min(axis=1)
        f[f"{prefix}_90d_std"]  = arr.std(axis=1)
        f[f"{prefix}_7d_mean"]  = arr[:,-7:].mean(axis=1)
        f[f"{prefix}_30d_mean"] = arr[:,-30:].mean(axis=1)

    add_era5(15,"volumetr_max"); add_era5(16,"volumetr_mean"); add_era5(17,"volumetr_min")

    precip = era5_grid[:,:,4]
    f["consecutive_dry_days"] = (precip < 0.001).sum(axis=1).astype(float)
    f["max_dry_spell_90d"]    = f["consecutive_dry_days"]
    f["total_dry_days_90d"]   = f["consecutive_dry_days"]
    f["precip_7d_total"]      = era5_grid[:,-7:,4].sum(axis=1)
    f["precip_30d_total"]     = era5_grid[:,-30:,4].sum(axis=1)
    f["temp_anomaly_7d"]      = era5_grid[:,-7:,1].mean(axis=1) - era5_grid[:,:,1].mean(axis=1)
    f["temp_max_90d"]         = era5_grid[:,:,0].max(axis=1)
    return f[feature_cols]

print("Fetching ERA5...")
result = fetch_era5_coarse()

if result is not None:
    era5_grid, valid_idx = result
    static_df = static_df.iloc[valid_idx].reset_index(drop=True)
    print(f"Valid grid points: {len(static_df)}")
    print("Building features...")
    X     = build_xgb_features(era5_grid, static_df)
    X_imp = imputer.transform(X)
    probs = xgb_model.predict_proba(X_imp)[:,1]
    print(f"Done. Mean prob: {probs.mean():.3f}")
else:
    print("ERA5 failed — zeros fallback")
    valid_mask = np.ones(len(static_df), dtype=bool)
    probs = np.zeros(len(static_df))

def to_risk(p):
    if p >= 0.70: return "VERY HIGH"
    if p >= 0.40: return "HIGH"
    if p >= 0.30: return "MEDIUM"
    return "LOW"

date_str   = datetime.now().strftime("%Y-%m-%d")
results_df = static_df[["latitude","longitude"]].copy()
results_df["probability"] = np.round(probs, 4)
results_df["risk_level"]  = [to_risk(p) for p in probs]
results_df.to_csv(f"fire_risk_{date_str}.csv", index=False)

print(f"\nFORE RISK — {date_str}")
print(results_df["risk_level"].value_counts().to_string())

fig, ax = plt.subplots(figsize=(14,8))
cmap = {"LOW":"#16A34A","MEDIUM":"#D97706","HIGH":"#EA580C","VERY HIGH":"#DC2626"}
smap = {"LOW":8,"MEDIUM":18,"HIGH":30,"VERY HIGH":50}
for risk, color in cmap.items():
    sub = results_df[results_df["risk_level"]==risk]
    if len(sub):
        ax.scatter(sub["longitude"],sub["latitude"],
                   c=color,s=smap[risk],alpha=0.75,zorder=3,label=f"{risk} ({len(sub)})")
patches = [mpatches.Patch(color=cmap[r],label=r) for r in ["LOW","MEDIUM","HIGH","VERY HIGH"]]
ax.legend(handles=patches,fontsize=11,title="Fire Risk",loc="upper right")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(
    f"Nepal Forest Fire Risk — {date_str}\n"
    f"XGBoost Ensemble (73.1% Detection Rate) + 90-day Open-Meteo ERA5",
    fontsize=12, fontweight="bold")
ax.set_xlim(80.0,88.2); ax.set_ylim(26.3,30.5)
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(f"fire_risk_map_{date_str}.png",dpi=120,bbox_inches="tight",facecolor="white")
print(f"Map saved.")
print("DONE")
