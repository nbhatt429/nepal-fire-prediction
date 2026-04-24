"""
Nepal Forest Fire Daily Prediction — Transformer (ONNX)
AUC=0.9983 | No TensorFlow needed | Fast deployment
"""

import requests
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import time
import os

print("Nepal Forest Fire Daily Prediction — Transformer (ONNX)")
print("="*55)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} NPT")

# ── File paths ────────────────────────────────────────────────────
ONNX_PATH    = "transformer_relu.onnx"
ERA5_SCALER  = "era5_scaler_temporal.pkl"
STAT_SCALER  = "static_scaler_temporal.pkl"
TERRAIN_PATH = "nepal_forest_grid_terrain.csv"
NDVI_PATH    = "nepal_ndvi_monthly_update.csv"
GRID_PATH    = "nepal_forest_grid.csv"

# ── Load ONNX model ───────────────────────────────────────────────
print("Loading Transformer ONNX model...")
sess = ort.InferenceSession(ONNX_PATH)
print(f"Model loaded — inputs: "
      f"{[i.name for i in sess.get_inputs()]}")

# ── Load scalers ──────────────────────────────────────────────────
era5_scaler   = joblib.load(ERA5_SCALER)
static_scaler = joblib.load(STAT_SCALER)
print("Scalers loaded")

# ── Load grid ─────────────────────────────────────────────────────
terrain = pd.read_csv(TERRAIN_PATH)
grid    = pd.read_csv(GRID_PATH)
ndvi    = pd.read_csv(NDVI_PATH)

terrain = terrain.drop(
    columns=["latitude","longitude"], errors="ignore")

static_df = terrain.merge(
    grid[["point_id","latitude","longitude"]],
    on="point_id", how="inner"
).merge(
    ndvi[["point_id","ndvi_30to45","ndvi_60to90",
           "ndvi_trend","ndvi_anomaly"]],
    on="point_id", how="left"
)

for col, val in [("ndvi_30to45",0.30),("ndvi_60to90",0.35),
                  ("ndvi_trend",0.05),("ndvi_anomaly",0.0)]:
    static_df[col] = static_df[col].fillna(val)

if "dist_water_km" not in static_df.columns:
    static_df["dist_water_km"] = 1.0

forest_df = static_df[
    static_df["latitude"].notna() &
    static_df["longitude"].notna()
].reset_index(drop=True)

print(f"Forest locations: {len(forest_df):,}")

STATIC_VARS = [
    'elevation','slope','aspect','is_south_facing',
    'landcover','dist_water_km',
    'ndvi_30to45','ndvi_60to90','ndvi_anomaly','ndvi_trend'
]

# ── Monthly baselines ─────────────────────────────────────────────
now   = datetime.now()
month = now.month

baselines = {
    1:{"temp":12.0,"press":88000,"wind":3.0,"soil":0.20},
    2:{"temp":14.0,"press":88000,"wind":3.2,"soil":0.19},
    3:{"temp":18.5,"press":87000,"wind":3.5,"soil":0.18},
    4:{"temp":22.3,"press":87000,"wind":3.8,"soil":0.155},
    5:{"temp":24.1,"press":87000,"wind":3.5,"soil":0.14},
    6:{"temp":24.0,"press":88000,"wind":2.5,"soil":0.22},
    7:{"temp":23.5,"press":89000,"wind":2.0,"soil":0.30},
    8:{"temp":23.5,"press":89000,"wind":2.0,"soil":0.32},
    9:{"temp":22.0,"press":88000,"wind":2.2,"soil":0.28},
   10:{"temp":19.0,"press":88000,"wind":2.8,"soil":0.20},
   11:{"temp":15.0,"press":88000,"wind":3.0,"soil":0.18},
   12:{"temp":12.0,"press":88000,"wind":3.0,"soil":0.20},
}
bl = baselines[month]

# ── Open-Meteo forecast ───────────────────────────────────────────
def get_forecast(lat, lon):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "hourly": ["temperature_2m","precipitation",
                           "surface_pressure","wind_speed_10m",
                           "soil_moisture_0_to_1cm"],
                "forecast_days": 7,
                "timezone": "Asia/Kathmandu"
            }, timeout=15
        )
        h = r.json().get("hourly", {})

        def clean(key, default):
            vals  = h.get(key, [])
            valid = [v for v in vals if v is not None]
            if not valid: return [default]*168
            med = float(np.median(valid))
            return [v if v is not None else med for v in vals]

        return {
            "temp":   np.array(clean("temperature_2m",   bl["temp"])),
            "precip": np.array(clean("precipitation",    0.0))/1000,
            "press":  np.array(clean("surface_pressure", bl["press"])),
            "wind":   np.array(clean("wind_speed_10m",   bl["wind"])),
            "soil":   np.array(clean("soil_moisture_0_to_1cm",
                                      bl["soil"])),
        }
    except:
        return None

# ── Build ERA5 sequence ───────────────────────────────────────────
def build_era5_sequence(fc):
    X = np.zeros((90, 18), dtype=np.float32)
    for d in range(83):
        t=bl["temp"]; p=bl["press"]
        w=bl["wind"]; s=bl["soil"]
        X[d] = [t+5,t,t-5, 0,0,0,
                p+500,p,p-500,
                w+2,w,max(0,w-2),
                w+2,w,max(0,w-2),
                s+0.02,s,max(0,s-0.02)]
    for day in range(7):
        idx = 83+day
        sh  = day*24; eh = sh+24
        t=fc["temp"][sh:eh];   p=fc["precip"][sh:eh]
        pr=fc["press"][sh:eh]; w=fc["wind"][sh:eh]
        sl=fc["soil"][sh:eh]
        if len(t)==0: continue
        X[idx]=[
            float(t.max()),  float(t.mean()),  float(t.min()),
            float(p.max()),  float(p.mean()),  float(p.min()),
            float(pr.max()), float(pr.mean()), float(pr.min()),
            float(w.max()),  float(w.mean()),  max(0,float(w.min())),
            float(w.max()),  float(w.mean()),  max(0,float(w.min())),
            float(sl.max()), float(sl.mean()), max(0,float(sl.min())),
        ]
    return X

# ── Predict using ONNX ────────────────────────────────────────────
def predict_onnx(era5_seq, static_row):
    # Scale ERA5
    era5_sc = era5_scaler.transform(
        era5_seq.reshape(-1,18)
    ).reshape(1,90,18).astype(np.float32)

    # Build static vector
    sv = np.array([[float(static_row.get(c,0) or 0)
                    for c in STATIC_VARS]],
                   dtype=np.float32)

    # Pad/trim to scaler size
    n = static_scaler.n_features_in_
    if sv.shape[1] < n:
        sv = np.concatenate(
            [sv, np.zeros((1,n-sv.shape[1]))], axis=1)
    elif sv.shape[1] > n:
        sv = sv[:,:n]
    stat_sc = static_scaler.transform(sv).astype(np.float32)

    # Pad/trim to ONNX model input size (90)
    n_onnx = sess.get_inputs()[1].shape[1]
    if stat_sc.shape[1] < n_onnx:
        stat_sc = np.concatenate(
            [stat_sc,
             np.zeros((1,n_onnx-stat_sc.shape[1]))], axis=1)
    elif stat_sc.shape[1] > n_onnx:
        stat_sc = stat_sc[:,:n_onnx]

    # Run ONNX inference
    result = sess.run(None, {
        'era5_input':   era5_sc,
        'static_input': stat_sc
    })
    return float(result[0][0][0])

def prob_to_risk(prob):
    if prob>=0.75:   return "VERY HIGH","#DC2626"
    elif prob>=0.50: return "HIGH",     "#EA580C"
    elif prob>=0.25: return "MEDIUM",   "#D97706"
    else:            return "LOW",      "#16A34A"

# ══════════════════════════════════════════════════════════════════
# MAIN PREDICTION LOOP
# ══════════════════════════════════════════════════════════════════
print(f"\nPredicting {len(forest_df):,} locations...")

results = []
errors  = 0
cache   = {}

for i, (_, row) in enumerate(forest_df.iterrows()):
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        ck  = (round(lat*4)/4, round(lon*4)/4)

        if ck not in cache:
            cache[ck] = get_forecast(lat, lon)
            time.sleep(0.04)
        fc = cache[ck]
        if fc is None:
            errors += 1
            continue

        era5_seq    = build_era5_sequence(fc)
        prob        = predict_onnx(era5_seq, row)
        risk, color = prob_to_risk(prob)

        results.append({
            "latitude":    lat,
            "longitude":   lon,
            "probability": round(prob, 4),
            "risk_level":  risk,
            "elevation":   float(row.get("elevation",0) or 0),
            "temp_C":      round(float(np.mean(fc["temp"])),1),
            "rain_7d_mm":  round(
                float(np.sum(fc["precip"])*1000),1),
        })

    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  Error row {i}: {str(e)[:80]}")

    if (i+1) % 500 == 0:
        print(f"  {i+1:,}/{len(forest_df):,} done  "
              f"| cache: {len(cache)} cells")

# ── Save results ───────────────────────────────────────────────────
date_str   = now.strftime("%Y-%m-%d")
results_df = pd.DataFrame(results)
results_df.to_csv(f"fire_risk_{date_str}.csv", index=False)

print(f"\n{'='*55}")
print(f"NEPAL FIRE RISK — {date_str}")
print(f"Model: Transformer ONNX (AUC=0.9983)")
print(f"{'='*55}")
print(f"Predicted: {len(results_df):,}  |  Errors: {errors}")
for risk in ["VERY HIGH","HIGH","MEDIUM","LOW"]:
    n   = (results_df["risk_level"]==risk).sum()
    bar = "█" * (n//50)
    print(f"  {risk:<12}: {n:>5}  {bar}")

# Alert
vh = results_df[results_df["risk_level"]=="VERY HIGH"]
hi = results_df[results_df["risk_level"]=="HIGH"]
if len(vh)>0:
    print(f"\n⚠  ALERT: {len(vh)} VERY HIGH + {len(hi)} HIGH")
    print(vh.nlargest(5,'probability')[
        ["latitude","longitude","probability","elevation"]
    ].to_string(index=False))

# Save alert JSON
alert = {
    "date":       date_str,
    "model":      "Transformer ONNX AUC=0.9983",
    "total":      len(results_df),
    "very_high":  int((results_df["risk_level"]=="VERY HIGH").sum()),
    "high":       int((results_df["risk_level"]=="HIGH").sum()),
    "medium":     int((results_df["risk_level"]=="MEDIUM").sum()),
    "low":        int((results_df["risk_level"]=="LOW").sum()),
    "top10": results_df.nlargest(10,'probability')[
        ["latitude","longitude","probability",
         "risk_level","elevation"]
    ].to_dict('records')
}
with open(f"fire_alert_{date_str}.json",'w') as f:
    json.dump(alert, f, indent=2)
print(f"Alert saved: fire_alert_{date_str}.json")

# ── Map ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14,8))
cmap = {"LOW":"#16A34A","MEDIUM":"#D97706",
        "HIGH":"#EA580C","VERY HIGH":"#DC2626"}
smap = {"LOW":5,"MEDIUM":14,"HIGH":26,"VERY HIGH":42}

for risk,color in cmap.items():
    sub = results_df[results_df["risk_level"]==risk]
    if len(sub)>0:
        ax.scatter(sub["longitude"],sub["latitude"],
                   c=color,s=smap[risk],alpha=0.75,zorder=3)

patches = [
    mpatches.Patch(
        color=cmap[r],
        label=f"{r} ({(results_df['risk_level']==r).sum():,})")
    for r in ["VERY HIGH","HIGH","MEDIUM","LOW"]
]
ax.legend(handles=patches, fontsize=10,
          title="Fire Risk", loc="upper right")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude",  fontsize=12)
ax.set_title(
    f"Nepal Forest Fire Risk — {date_str}\n"
    f"Transformer ONNX (AUC=0.9983) "
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
print(f"Map saved: fire_risk_map_{date_str}.png")
print("DONE")
