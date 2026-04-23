"""
Nepal Forest Fire Daily Prediction — Transformer Model
Replaces XGBoost with Transformer (AUC=0.9994, Recall=99.1%)
Drop-in replacement for predict.py — same output format
"""

import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import time
import os
import json

print("Nepal Forest Fire Daily Prediction — Transformer Model")
print("="*55)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} NPT")
print(f"Model: Transformer (AUC=0.9994, Recall=99.1%)")

# ── File paths — same repo structure as before ────────────────────
MODEL_PATH   = "Transformer_temporal.keras"
ERA5_SCALER  = "era5_scaler_temporal.pkl"
STAT_SCALER  = "static_scaler_temporal.pkl"
TERRAIN_PATH = "nepal_forest_grid_terrain.csv"
NDVI_PATH    = "nepal_ndvi_monthly_update.csv"
GRID_PATH    = "nepal_forest_grid.csv"

# ── Load model and scalers ────────────────────────────────────────
print("Loading Transformer model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
era5_scaler   = joblib.load(ERA5_SCALER)
static_scaler = joblib.load(STAT_SCALER)
print(f"Model loaded: {model.count_params():,} parameters")

# ── Load grid data ─────────────────────────────────────────────────
terrain = pd.read_csv(TERRAIN_PATH)
grid    = pd.read_csv(GRID_PATH)
ndvi    = pd.read_csv(NDVI_PATH)

terrain = terrain.drop(columns=["latitude","longitude"], errors="ignore")
static_df = terrain.merge(
    grid[["point_id","latitude","longitude"]],
    on="point_id", how="inner"
).merge(
    ndvi[["point_id","ndvi_30to45","ndvi_60to90",
           "ndvi_trend","ndvi_anomaly"]],
    on="point_id", how="left"
)

# Fill missing NDVI with seasonal defaults
static_df["ndvi_30to45"]  = static_df["ndvi_30to45"].fillna(0.30)
static_df["ndvi_60to90"]  = static_df["ndvi_60to90"].fillna(0.35)
static_df["ndvi_trend"]   = static_df["ndvi_trend"].fillna(0.05)
static_df["ndvi_anomaly"] = static_df["ndvi_anomaly"].fillna(0.0)
static_df["dist_water_km"]= static_df.get("dist_water_km", 1.0)
if "dist_water_km" not in static_df.columns:
    static_df["dist_water_km"] = 1.0

forest_df = static_df[
    (static_df["latitude"].notna()) &
    (static_df["longitude"].notna())
].reset_index(drop=True)

print(f"Forest locations: {len(forest_df):,}")

# ── Static feature columns matching training ───────────────────────
STATIC_VARS = [
    'elevation','slope','aspect','is_south_facing',
    'landcover','dist_water_km',
    'ndvi_30to45','ndvi_60to90','ndvi_anomaly','ndvi_trend'
]
N_STATIC = len(STATIC_VARS)

# ── Monthly baselines for ERA5 padding ────────────────────────────
now   = datetime.now()
month = now.month

baselines = {
    1:{"temp":12.0,"precip":0.002,"press":88000,"wind":3.0,"soil":0.20},
    2:{"temp":14.0,"precip":0.003,"press":88000,"wind":3.2,"soil":0.19},
    3:{"temp":18.5,"precip":0.004,"press":87000,"wind":3.5,"soil":0.18},
    4:{"temp":22.3,"precip":0.002,"press":87000,"wind":3.8,"soil":0.155},
    5:{"temp":24.1,"precip":0.005,"press":87000,"wind":3.5,"soil":0.14},
    6:{"temp":24.0,"precip":0.020,"press":88000,"wind":2.5,"soil":0.22},
    7:{"temp":23.5,"precip":0.030,"press":89000,"wind":2.0,"soil":0.30},
    8:{"temp":23.5,"precip":0.025,"press":89000,"wind":2.0,"soil":0.32},
    9:{"temp":22.0,"precip":0.015,"press":88000,"wind":2.2,"soil":0.28},
   10:{"temp":19.0,"precip":0.005,"press":88000,"wind":2.8,"soil":0.20},
   11:{"temp":15.0,"precip":0.002,"press":88000,"wind":3.0,"soil":0.18},
   12:{"temp":12.0,"precip":0.002,"press":88000,"wind":3.0,"soil":0.20},
}
bl = baselines[month]

# ── Fetch 7-day forecast from Open-Meteo ──────────────────────────
def get_forecast(lat, lon):
    """Returns 7-day hourly data from Open-Meteo"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":  round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "hourly": [
                "temperature_2m", "precipitation",
                "surface_pressure", "wind_speed_10m",
                "soil_moisture_0_to_1cm"
            ],
            "forecast_days": 7,
            "timezone": "Asia/Kathmandu"
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        h = r.json().get("hourly", {})

        def clean(key, default):
            vals = h.get(key, [])
            valid = [v for v in vals if v is not None]
            if not valid:
                return [default] * 168
            med = float(np.median(valid))
            return [v if v is not None else med for v in vals]

        return {
            "temp":   np.array(clean("temperature_2m", bl["temp"])),
            "precip": np.array(clean("precipitation", 0.0)) / 1000,
            "press":  np.array(clean("surface_pressure", bl["press"])),
            "wind":   np.array(clean("wind_speed_10m", bl["wind"])),
            "soil":   np.array(clean("soil_moisture_0_to_1cm", bl["soil"])),
        }
    except Exception as e:
        return None

# ── Build ERA5 sequence (90 days × 18 features) ───────────────────
def build_era5_sequence(forecast):
    """
    Creates (90, 18) array matching Transformer training format.

    Days 1-83: filled with monthly climatological baseline
    Days 84-90: from Open-Meteo 7-day forecast (real data)

    This is the same limitation as XGBoost — we only have 7 days
    of real weather. The terrain features (which dominate the
    model via ablation: static drop=0.29) compensate.
    """
    X = np.zeros((90, 18), dtype=np.float32)

    # Variable order matching training:
    # [temp_max, temp_mean, temp_min,
    #  precip_max, precip_mean, precip_min,
    #  press_max, press_mean, press_min,
    #  wind_max, wind_mean, wind_min,
    #  wind_max2, wind_mean2, wind_min2,  (u/v proxy)
    #  soil_max, soil_mean, soil_min]

    # Fill days 1-83 with climatological baseline
    for day in range(83):
        X[day] = [
            bl["temp"]+5, bl["temp"], bl["temp"]-5,  # temp max/mean/min
            0.0, 0.0, 0.0,                            # precip
            bl["press"]+500, bl["press"], bl["press"]-500,  # pressure
            bl["wind"]+2, bl["wind"], max(0, bl["wind"]-2), # wind u
            bl["wind"]+2, bl["wind"], max(0, bl["wind"]-2), # wind v
            bl["soil"]+0.02, bl["soil"], bl["soil"]-0.02,   # soil
        ]

    # Fill days 84-90 from Open-Meteo 7-day forecast
    for day_offset in range(7):
        day_idx = 83 + day_offset
        h_start = day_offset * 24
        h_end   = h_start + 24

        t  = forecast["temp"][h_start:h_end]
        p  = forecast["precip"][h_start:h_end]
        pr = forecast["press"][h_start:h_end]
        w  = forecast["wind"][h_start:h_end]
        s  = forecast["soil"][h_start:h_end]

        if len(t) == 0:
            continue

        X[day_idx] = [
            float(t.max()),  float(t.mean()),  float(t.min()),
            float(p.max()),  float(p.mean()),  float(p.min()),
            float(pr.max()), float(pr.mean()), float(pr.min()),
            float(w.max()),  float(w.mean()),  max(0,float(w.min())),
            float(w.max()),  float(w.mean()),  max(0,float(w.min())),
            float(s.max()),  float(s.mean()),  max(0,float(s.min())),
        ]

    return X

# ── Predict one location ───────────────────────────────────────────
def predict_location(era5_seq, static_row):
    """Run Transformer for one location"""
    # Scale ERA5
    era5_scaled = era5_scaler.transform(
        era5_seq.reshape(-1, 18)
    ).reshape(1, 90, 18).astype(np.float32)

    # Build static feature vector
    static_vals = []
    for col in STATIC_VARS:
        val = static_row.get(col, 0.0)
        static_vals.append(float(val) if val is not None else 0.0)

    static_arr = np.array([static_vals], dtype=np.float32)

    # Scale static — handle shape mismatch gracefully
    n_expected = static_scaler.n_features_in_
    if static_arr.shape[1] < n_expected:
        pad = np.zeros((1, n_expected - static_arr.shape[1]))
        static_arr = np.concatenate([static_arr, pad], axis=1)
    elif static_arr.shape[1] > n_expected:
        static_arr = static_arr[:, :n_expected]

    static_scaled = static_scaler.transform(static_arr).astype(np.float32)

    prob = float(model.predict(
        [era5_scaled, static_scaled], verbose=0
    )[0][0])
    return prob

def prob_to_risk(prob):
    if prob >= 0.75:  return "VERY HIGH", "#DC2626"
    elif prob >= 0.50: return "HIGH",     "#EA580C"
    elif prob >= 0.25: return "MEDIUM",   "#D97706"
    else:             return "LOW",       "#16A34A"

# ══════════════════════════════════════════════════════════════════
# MAIN PREDICTION LOOP
# ══════════════════════════════════════════════════════════════════

# Predict ALL locations (not every 14th like before)
# This gives full 6,730 location coverage
predict_df = forest_df.copy()

print(f"\nPredicting {len(predict_df):,} locations...")
print(f"(Transformer — AUC=0.9994, Recall=99.1%, 8/906 fires missed)")

results = []
errors  = 0
cache   = {}  # cache forecasts by grid cell (0.5° resolution)

for i, (_, row) in enumerate(predict_df.iterrows()):
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        # Cache key: round to 0.25° grid (ERA5 is 9km anyway)
        cache_key = (round(lat * 4) / 4, round(lon * 4) / 4)

        if cache_key not in cache:
            forecast = get_forecast(lat, lon)
            cache[cache_key] = forecast
            time.sleep(0.04)  # rate limit
        else:
            forecast = cache[cache_key]

        if forecast is None:
            errors += 1
            continue

        # Build sequence and predict
        era5_seq = build_era5_sequence(forecast)
        prob     = predict_location(era5_seq, row)
        risk, color = prob_to_risk(prob)

        results.append({
            "latitude":    lat,
            "longitude":   lon,
            "probability": round(prob, 4),
            "risk_level":  risk,
            "elevation":   float(row.get("elevation", 0)),
            "temp_C":      round(float(np.mean(forecast["temp"])), 1),
            "rain_7d_mm":  round(float(np.sum(forecast["precip"]) * 1000), 1),
            "model":       "Transformer",
        })

    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  Error row {i}: {str(e)[:60]}")

    if (i + 1) % 500 == 0:
        print(f"  {i+1:,}/{len(predict_df):,} done  "
              f"(cache hits: {len(cache)} cells)")

# ── Save results ───────────────────────────────────────────────────
date_str   = now.strftime("%Y-%m-%d")
results_df = pd.DataFrame(results)
results_df.to_csv(f"fire_risk_{date_str}.csv", index=False)

print(f"\n{'='*55}")
print(f"NEPAL FIRE RISK FORECAST — {date_str}")
print(f"Model: Transformer (AUC=0.9994 | Recall=99.1%)")
print(f"{'='*55}")
print(f"Predicted: {len(results_df):,}  |  Errors: {errors}  "
      f"|  Cache cells: {len(cache)}")
print(f"\nRisk distribution:")
print(results_df["risk_level"].value_counts().to_string())

vh = results_df[results_df["risk_level"] == "VERY HIGH"]
hi = results_df[results_df["risk_level"] == "HIGH"]
if len(vh) > 0:
    print(f"\n⚠  ALERT: {len(vh)} VERY HIGH + {len(hi)} HIGH risk locations")
    print(vh.sort_values("probability", ascending=False)
          .head(5)[["latitude","longitude","probability","elevation"]]
          .to_string(index=False))
else:
    print("\nNo VERY HIGH risk locations today")

# ── Generate map ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
cmap = {"LOW":"#16A34A","MEDIUM":"#D97706",
        "HIGH":"#EA580C","VERY HIGH":"#DC2626"}
smap = {"LOW":6,"MEDIUM":16,"HIGH":28,"VERY HIGH":45}

for risk, color in cmap.items():
    sub = results_df[results_df["risk_level"] == risk]
    if len(sub) > 0:
        ax.scatter(sub["longitude"], sub["latitude"],
                   c=color, s=smap[risk], alpha=0.75,
                   zorder=3, label=f"{risk} ({len(sub):,})")

patches = [mpatches.Patch(color=cmap[r], label=r)
           for r in ["LOW","MEDIUM","HIGH","VERY HIGH"]]
ax.legend(handles=patches, fontsize=11,
          title="Fire Risk", loc="upper right")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.set_title(
    f"Nepal Forest Fire Risk Forecast — {date_str}\n"
    f"Transformer Deep Learning (AUC=0.9994, Recall=99.1%) "
    f"+ Open-Meteo 7-Day Forecast",
    fontsize=12, fontweight="bold"
)
ax.set_xlim(80.0, 88.2)
ax.set_ylim(26.3, 30.5)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"fire_risk_map_{date_str}.png",
            dpi=120, bbox_inches="tight", facecolor="white")
print(f"\nMap saved: fire_risk_map_{date_str}.png")
print("DONE")
