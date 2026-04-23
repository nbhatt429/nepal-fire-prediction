"""
Nepal Forest Fire Daily Prediction — Transformer Model
With local level (municipality) names and district alerts
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

# ── File paths ────────────────────────────────────────────────────
MODEL_PATH   = "Transformer_temporal.keras"
ERA5_SCALER  = "era5_scaler_temporal.pkl"
STAT_SCALER  = "static_scaler_temporal.pkl"
TERRAIN_PATH = "nepal_forest_grid_terrain.csv"
NDVI_PATH    = "nepal_ndvi_monthly_update.csv"
# Use the new admin-enriched grid
GRID_PATH    = "nepal_forest_grid_admin.csv"

# ── Load model ────────────────────────────────────────────────────
model         = tf.keras.models.load_model(MODEL_PATH, compile=False)
era5_scaler   = joblib.load(ERA5_SCALER)
static_scaler = joblib.load(STAT_SCALER)
print(f"Model loaded")

# ── Load grid with admin names ────────────────────────────────────
terrain = pd.read_csv(TERRAIN_PATH)
grid    = pd.read_csv(GRID_PATH)  # now has province/district/local_level
ndvi    = pd.read_csv(NDVI_PATH)

terrain = terrain.drop(columns=["latitude","longitude"], errors="ignore")
static_df = terrain.merge(
    grid[["point_id","latitude","longitude",
          "province","district","local_level"]],
    on="point_id", how="inner"
).merge(
    ndvi[["point_id","ndvi_30to45","ndvi_60to90",
           "ndvi_trend","ndvi_anomaly"]],
    on="point_id", how="left"
)

for col in ["ndvi_30to45","ndvi_60to90","ndvi_trend","ndvi_anomaly"]:
    static_df[col] = static_df[col].fillna(
        {"ndvi_30to45":0.30,"ndvi_60to90":0.35,
         "ndvi_trend":0.05,"ndvi_anomaly":0.0}[col]
    )
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

def get_forecast(lat, lon):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": round(float(lat),4),
                "longitude": round(float(lon),4),
                "hourly": ["temperature_2m","precipitation",
                           "surface_pressure","wind_speed_10m",
                           "soil_moisture_0_to_1cm"],
                "forecast_days": 7,
                "timezone": "Asia/Kathmandu"
            }, timeout=15
        )
        h = r.json().get("hourly",{})
        def clean(key, default):
            vals = h.get(key,[])
            valid = [v for v in vals if v is not None]
            if not valid: return [default]*168
            med = float(np.median(valid))
            return [v if v is not None else med for v in vals]
        return {
            "temp":   np.array(clean("temperature_2m", bl["temp"])),
            "precip": np.array(clean("precipitation", 0.0))/1000,
            "press":  np.array(clean("surface_pressure", bl["press"])),
            "wind":   np.array(clean("wind_speed_10m", bl["wind"])),
            "soil":   np.array(clean("soil_moisture_0_to_1cm", bl["soil"])),
        }
    except:
        return None

def build_era5_sequence(forecast):
    X = np.zeros((90, 18), dtype=np.float32)
    for day in range(83):
        X[day] = [
            bl["temp"]+5, bl["temp"], bl["temp"]-5,
            0.0, 0.0, 0.0,
            bl["press"]+500, bl["press"], bl["press"]-500,
            bl["wind"]+2, bl["wind"], max(0,bl["wind"]-2),
            bl["wind"]+2, bl["wind"], max(0,bl["wind"]-2),
            bl["soil"]+0.02, bl["soil"], bl["soil"]-0.02,
        ]
    for d in range(7):
        idx = 83+d
        s,e = d*24, d*24+24
        t=forecast["temp"][s:e]; p=forecast["precip"][s:e]
        pr=forecast["press"][s:e]; w=forecast["wind"][s:e]
        sl=forecast["soil"][s:e]
        if len(t)==0: continue
        X[idx]=[float(t.max()),float(t.mean()),float(t.min()),
                float(p.max()),float(p.mean()),float(p.min()),
                float(pr.max()),float(pr.mean()),float(pr.min()),
                float(w.max()),float(w.mean()),max(0,float(w.min())),
                float(w.max()),float(w.mean()),max(0,float(w.min())),
                float(sl.max()),float(sl.mean()),max(0,float(sl.min()))]
    return X

def predict_location(era5_seq, static_row):
    era5_sc = era5_scaler.transform(
        era5_seq.reshape(-1,18)).reshape(1,90,18).astype(np.float32)
    sv = np.array([[float(static_row.get(c,0) or 0)
                    for c in STATIC_VARS]], dtype=np.float32)
    n = static_scaler.n_features_in_
    if sv.shape[1] < n:
        sv = np.concatenate([sv, np.zeros((1,n-sv.shape[1]))], axis=1)
    elif sv.shape[1] > n:
        sv = sv[:,:n]
    stat_sc = static_scaler.transform(sv).astype(np.float32)
    return float(model.predict([era5_sc, stat_sc], verbose=0)[0][0])

def prob_to_risk(prob):
    if prob>=0.75:  return "VERY HIGH","#DC2626"
    elif prob>=0.50: return "HIGH","#EA580C"
    elif prob>=0.25: return "MEDIUM","#D97706"
    else:           return "LOW","#16A34A"

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

        era5_seq = build_era5_sequence(fc)
        prob     = predict_location(era5_seq, row)
        risk, color = prob_to_risk(prob)

        results.append({
            "latitude":    lat,
            "longitude":   lon,
            "probability": round(prob, 4),
            "risk_level":  risk,
            "province":    str(row.get("province","Unknown")),
            "district":    str(row.get("district","Unknown")),
            "local_level": str(row.get("local_level","Unknown")),
            "elevation":   float(row.get("elevation",0)),
            "temp_C":      round(float(np.mean(fc["temp"])),1),
            "rain_7d_mm":  round(float(np.sum(fc["precip"])*1000),1),
        })
    except Exception as e:
        errors += 1
    if (i+1) % 500 == 0:
        print(f"  {i+1:,}/{len(forest_df):,} done")

# ── Save results ───────────────────────────────────────────────────
date_str   = now.strftime("%Y-%m-%d")
results_df = pd.DataFrame(results)
results_df.to_csv(f"fire_risk_{date_str}.csv", index=False)

print(f"\n{'='*55}")
print(f"NEPAL FIRE RISK — {date_str}")
print(f"{'='*55}")
vc = results_df["risk_level"].value_counts()
for risk in ["VERY HIGH","HIGH","MEDIUM","LOW"]:
    print(f"  {risk:<12}: {vc.get(risk,0):>5}")

# ══════════════════════════════════════════════════════════════════
# ALERT REPORT — by district and local level
# ══════════════════════════════════════════════════════════════════
alert_df = results_df[
    results_df["risk_level"].isin(["VERY HIGH","HIGH"])
].copy()

print(f"\n{'='*55}")
print(f"FIRE ALERT REPORT — {date_str}")
print(f"{'='*55}")
print(f"Total HIGH/VERY HIGH locations: {len(alert_df)}")

if len(alert_df) > 0:

    # ── By district ───────────────────────────────────────────────
    print(f"\nALERT BY DISTRICT:")
    print(f"{'District':<25} {'Province':<20} {'VH':>4} {'H':>4} {'Max Prob':>9}")
    print("-"*65)

    dist_summary = alert_df.groupby(['district','province']).agg(
        very_high=('risk_level', lambda x: (x=='VERY HIGH').sum()),
        high=('risk_level', lambda x: (x=='HIGH').sum()),
        max_prob=('probability','max'),
        local_levels=('local_level', lambda x: list(x.unique()))
    ).reset_index().sort_values('max_prob', ascending=False)

    for _, row in dist_summary.head(20).iterrows():
        print(f"{str(row['district']):<25} "
              f"{str(row['province']):<20} "
              f"{int(row['very_high']):>4} "
              f"{int(row['high']):>4} "
              f"{row['max_prob']:>9.3f}")

    # ── By local level (municipality) ─────────────────────────────
    print(f"\nALERT BY LOCAL LEVEL (Palika/Municipality):")
    print(f"{'Local Level':<30} {'District':<22} {'Risk':<12} {'Prob':>6}")
    print("-"*72)

    local_summary = alert_df.groupby(
        ['local_level','district','province']
    ).agg(
        max_prob=('probability','max'),
        max_risk=('risk_level', lambda x:
                  'VERY HIGH' if 'VERY HIGH' in x.values else 'HIGH'),
        n_pixels=('probability','count')
    ).reset_index().sort_values('max_prob', ascending=False)

    for _, row in local_summary.head(30).iterrows():
        flag = "🔴" if row['max_risk']=='VERY HIGH' else "🟠"
        print(f"{flag} {str(row['local_level']):<28} "
              f"{str(row['district']):<22} "
              f"{row['max_risk']:<12} "
              f"{row['max_prob']:>6.3f}")

    # ── Save alert report ─────────────────────────────────────────
    alert_report = {
        "date":        date_str,
        "model":       "Transformer AUC=0.9994",
        "total_alerts": len(alert_df),
        "by_district": dist_summary.to_dict('records'),
        "by_local_level": local_summary.head(50).to_dict('records'),
        "top_locations": alert_df.nlargest(20,'probability')[
            ['latitude','longitude','probability','risk_level',
             'province','district','local_level','elevation',
             'temp_C','rain_7d_mm']
        ].to_dict('records')
    }

    with open(f"fire_alert_{date_str}.json", 'w') as f:
        json.dump(alert_report, f, indent=2, default=str)
    print(f"\nAlert report saved: fire_alert_{date_str}.json")

    # ── Province summary ──────────────────────────────────────────
    print(f"\nSUMMARY BY PROVINCE:")
    prov = alert_df.groupby('province').agg(
        districts=('district','nunique'),
        local_levels=('local_level','nunique'),
        very_high=('risk_level', lambda x: (x=='VERY HIGH').sum()),
        high=('risk_level', lambda x: (x=='HIGH').sum()),
    ).reset_index().sort_values('very_high', ascending=False)
    print(prov.to_string(index=False))

else:
    print("No HIGH or VERY HIGH risk locations today")
    with open(f"fire_alert_{date_str}.json",'w') as f:
        json.dump({"date":date_str,"total_alerts":0,
                   "message":"All clear"}, f)

# ── Generate map ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
cmap = {"LOW":"#16A34A","MEDIUM":"#D97706",
        "HIGH":"#EA580C","VERY HIGH":"#DC2626"}
smap = {"LOW":5,"MEDIUM":14,"HIGH":26,"VERY HIGH":42}

for risk, color in cmap.items():
    sub = results_df[results_df["risk_level"]==risk]
    if len(sub)>0:
        ax.scatter(sub["longitude"],sub["latitude"],
                   c=color,s=smap[risk],alpha=0.75,zorder=3)

# Annotate top 5 VERY HIGH locations with local level name
if len(alert_df)>0:
    top5 = alert_df.nlargest(5,'probability')
    for _, r in top5.iterrows():
        name = str(r.get('local_level',''))
        if name and name != 'Unknown' and name != 'nan':
            ax.annotate(
                name,
                (float(r['longitude']), float(r['latitude'])),
                fontsize=7.5, fontweight='bold', color='#DC2626',
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2',
                         facecolor='white', alpha=0.7, edgecolor='#DC2626')
            )

patches = [mpatches.Patch(color=cmap[r],label=r)
           for r in ["LOW","MEDIUM","HIGH","VERY HIGH"]]
ax.legend(handles=patches, fontsize=11,
          title="Fire Risk", loc="upper right")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.set_title(
    f"Nepal Forest Fire Risk — {date_str}\n"
    f"Transformer (AUC=0.9994) + Open-Meteo 7-Day Forecast",
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
print(f"\nMap saved — top VERY HIGH locations labelled by name")
print("DONE")
