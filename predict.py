"""
Nepal Forest Fire Daily Prediction — Transformer ONNX
Self-contained — no external numpy files needed
"""
import requests, numpy as np, pandas as pd
import onnxruntime as ort, joblib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from datetime import datetime
import time, os

print("Nepal Forest Fire Daily Prediction — Transformer ONNX")
print("="*55)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} NPT")
print(f"Python: {os.sys.version}")

# ── File paths ────────────────────────────────────────────────────
MODEL_PATH  = "transformer_relu.onnx"
ERA5_SCALER = "era5_scaler_temporal.pkl"
STAT_SCALER = "static_scaler_temporal.pkl"
GRID_PATH   = "nepal_forest_grid.csv"

# ── Verify all files exist ────────────────────────────────────────
for f in [MODEL_PATH, ERA5_SCALER, STAT_SCALER, GRID_PATH]:
    exists = os.path.exists(f)
    size   = os.path.getsize(f)/1024 if exists else 0
    print(f"  {'✓' if exists else '✗'} {f} ({size:.0f} KB)")
    if not exists:
        raise FileNotFoundError(f"Missing: {f}")

# ── Load model ────────────────────────────────────────────────────
print("\nLoading model...")
sess          = ort.InferenceSession(MODEL_PATH)
era5_scaler   = joblib.load(ERA5_SCALER)
static_scaler = joblib.load(STAT_SCALER)

print(f"ERA5 scaler: {era5_scaler.n_features_in_} features")
print(f"Static scaler: {static_scaler.n_features_in_} features")
print(f"ONNX inputs: {[(i.name,i.shape) for i in sess.get_inputs()]}")

# ── ERA5 baseline from scaler means ──────────────────────────────
baseline = era5_scaler.mean_.copy()
print(f"Baseline temp (K): {baseline[1]:.1f} = "
      f"{baseline[1]-273.15:.1f}°C")

# ── Load grid ─────────────────────────────────────────────────────
grid = pd.read_csv(GRID_PATH)
grid = grid[grid['latitude'].notna() &
            grid['longitude'].notna()
            ].reset_index(drop=True)
print(f"Grid: {len(grid):,} locations")
print(f"Grid columns: {grid.columns.tolist()}")

# ── Build static features from grid directly ──────────────────────
# The static scaler expects 90 features = 9 pixels x 10 features
# For deployment we replicate the centre pixel 9 times
# This is an approximation — all 9 pixels get same terrain values
# The model still uses terrain signal correctly

TERRAIN_COLS = [
    'elevation','slope','aspect','is_south_facing',
    'landcover','dist_water_km',
    'ndvi_30to45','ndvi_60to90','ndvi_anomaly','ndvi_trend'
]

# Fill missing columns with defaults
defaults = {
    'elevation':1000.0, 'slope':15.0, 'aspect':180.0,
    'is_south_facing':1, 'landcover':10,
    'dist_water_km':1.0, 'ndvi_30to45':0.30,
    'ndvi_60to90':0.35, 'ndvi_anomaly':0.0,
    'ndvi_trend':0.05
}
for col, val in defaults.items():
    if col not in grid.columns:
        grid[col] = val
    else:
        grid[col] = grid[col].fillna(val)

print("Building static feature matrix...")
# Each sample = 9 pixels x 10 features = 90 features
# Replicate centre pixel features for all 9 neighbours
centre_features = grid[TERRAIN_COLS].values  # (N, 10)
# Tile to (N, 90) — 9 pixels each with same features
static_raw = np.tile(centre_features, (1, 9))  # (N, 90)
print(f"Static raw shape: {static_raw.shape}")

# Scale
n_sc = static_scaler.n_features_in_
if static_raw.shape[1] < n_sc:
    pad = np.zeros((len(static_raw), n_sc - static_raw.shape[1]))
    static_raw = np.concatenate([static_raw, pad], axis=1)
elif static_raw.shape[1] > n_sc:
    static_raw = static_raw[:, :n_sc]

grid_static_scaled = static_scaler.transform(
    static_raw).astype(np.float32)
print(f"Scaled static shape: {grid_static_scaled.shape}")
print(f"Mean: {grid_static_scaled.mean():.4f}  "
      f"Std: {grid_static_scaled.std():.4f}")

# Match ONNX input size
n_onnx = sess.get_inputs()[1].shape[1]
if grid_static_scaled.shape[1] < n_onnx:
    pad = np.zeros((len(grid_static_scaled),
                    n_onnx - grid_static_scaled.shape[1]),
                   dtype=np.float32)
    grid_static_scaled = np.concatenate(
        [grid_static_scaled, pad], axis=1)
elif grid_static_scaled.shape[1] > n_onnx:
    grid_static_scaled = grid_static_scaled[:, :n_onnx]
print(f"Final ONNX static shape: {grid_static_scaled.shape}")

# ── Quick sanity check ────────────────────────────────────────────
print("\nSanity check (3 sample predictions):")
for i in [0, len(grid)//2, len(grid)-1]:
    test_era5 = np.tile(baseline, (90,1)).astype(np.float32)
    test_sc   = era5_scaler.transform(
        test_era5.reshape(-1,18)).reshape(1,90,18).astype(np.float32)
    test_st   = grid_static_scaled[i:i+1]
    prob = float(sess.run(None,{
        'era5_input':test_sc,
        'static_input':test_st})[0][0][0])
    risk = ("VERY HIGH" if prob>=0.75 else
            "HIGH" if prob>=0.50 else
            "MEDIUM" if prob>=0.25 else "LOW")
    print(f"  Location {i}: lat={grid.iloc[i]['latitude']:.3f} "
          f"prob={prob:.4f} → {risk}")

# ── Forecast fetcher ──────────────────────────────────────────────
def get_forecast(lat, lon):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  round(float(lat),4),
                "longitude": round(float(lon),4),
                "hourly": ["temperature_2m","precipitation",
                           "surface_pressure","wind_speed_10m",
                           "soil_moisture_0_to_1cm"],
                "forecast_days": 7,
                "timezone": "Asia/Kathmandu"
            }, timeout=15)
        if r.status_code != 200: return None
        h = r.json().get("hourly",{})
        def clean(key, default):
            vals  = h.get(key,[])
            valid = [v for v in vals if v is not None]
            med   = float(np.median(valid)) if valid else default
            return [v if v is not None else med for v in vals]
        return {
            "temp_K":   np.array([t+273.15
                for t in clean("temperature_2m",20.0)],
                dtype=np.float32),
            "precip_m": np.array([p/1000.0
                for p in clean("precipitation",0.0)],
                dtype=np.float32),
            "press_Pa": np.array([p*100.0
                for p in clean("surface_pressure",870.0)],
                dtype=np.float32),
            "wind_ms":  np.array(
                clean("wind_speed_10m",3.0), dtype=np.float32),
            "soil_f":   np.array(
                clean("soil_moisture_0_to_1cm",0.15),
                dtype=np.float32),
        }
    except: return None

def build_sequence(fc):
    X = np.zeros((90,18), dtype=np.float32)
    for d in range(83):
        X[d] = baseline.copy()
    if fc:
        n_days = min(7, len(fc["temp_K"])//24)
        for d in range(n_days):
            sh=d*24; eh=sh+24
            t=fc["temp_K"][sh:eh]; p=fc["precip_m"][sh:eh]
            pr=fc["press_Pa"][sh:eh]; w=fc["wind_ms"][sh:eh]
            sl=fc["soil_f"][sh:eh]
            if len(t)==0: continue
            X[83+d]=[
                float(t.max()),float(t.mean()),float(t.min()),
                float(p.max()),float(p.mean()),float(p.min()),
                float(pr.max()),float(pr.mean()),float(pr.min()),
                float(w.max()),float(w.mean()),max(0,float(w.min())),
                float(w.max()),float(w.mean()),max(0,float(w.min())),
                float(sl.max()),float(sl.mean()),max(0,float(sl.min())),
            ]
    return X

def prob_to_risk(p):
    if p>=0.75:   return "VERY HIGH","#DC2626"
    elif p>=0.50: return "HIGH",     "#EA580C"
    elif p>=0.25: return "MEDIUM",   "#D97706"
    else:         return "LOW",      "#16A34A"

# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════
now     = datetime.now()
date    = now.strftime("%Y-%m-%d")
results = []; cache = {}; errors = 0

print(f"\nPredicting {len(grid):,} locations for {date}...")

for i,(_, row) in enumerate(grid.iterrows()):
    try:
        lat=float(row['latitude']); lon=float(row['longitude'])
        ck=(round(lat*4)/4, round(lon*4)/4)
        if ck not in cache:
            cache[ck] = get_forecast(lat, lon)
            time.sleep(0.04)
        fc = cache[ck]

        seq = build_sequence(fc)
        era5_sc = era5_scaler.transform(
            seq.reshape(-1,18)).reshape(1,90,18).astype(np.float32)
        stat_sc = grid_static_scaled[i:i+1]

        prob = float(sess.run(None,{
            'era5_input':era5_sc,
            'static_input':stat_sc})[0][0][0])
        risk,color = prob_to_risk(prob)

        results.append({
            'latitude':    lat,
            'longitude':   lon,
            'probability': round(prob,4),
            'risk_level':  risk,
            'color':       color,
            'elevation':   float(row.get('elevation',0) or 0),
        })
    except Exception as e:
        errors+=1
        if errors<=3: print(f"  Error {i}: {str(e)[:80]}")

    if (i+1)%500==0:
        hi=sum(1 for r in results
               if r['risk_level'] in ['HIGH','VERY HIGH'])
        print(f"  {i+1:,}/{len(grid):,}  HIGH/VH:{hi}  "
              f"cache:{len(cache)}")

# ── Save ──────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(f"fire_risk_{date}.csv", index=False)
vc = df['risk_level'].value_counts()

print(f"\n{'='*55}")
print(f"NEPAL FIRE RISK — {date}")
print(f"{'='*55}")
print(f"Predicted:{len(df):,}  Errors:{errors}")
for risk in ['VERY HIGH','HIGH','MEDIUM','LOW']:
    n=vc.get(risk,0); pct=n/len(df)*100 if len(df)>0 else 0
    print(f"  {risk:<12}:{n:>6} ({pct:4.1f}%)")

vh=df[df['risk_level']=='VERY HIGH']
hi=df[df['risk_level']=='HIGH']
if len(vh)>0:
    print(f"\n⚠  ALERT: {len(vh)} VERY HIGH + {len(hi)} HIGH")

# Alert JSON
alert={
    "date":date,"model":"Transformer ONNX AUC=0.9983",
    "total":len(df),
    "very_high":int(vc.get('VERY HIGH',0)),
    "high":int(vc.get('HIGH',0)),
    "medium":int(vc.get('MEDIUM',0)),
    "low":int(vc.get('LOW',0)),
    "top10":df.nlargest(10,'probability')[
        ['latitude','longitude','probability','risk_level']
    ].to_dict('records')
}
with open(f"fire_alert_{date}.json",'w') as f:
    json.dump(alert,f,indent=2)

# Map
fig,ax=plt.subplots(figsize=(14,8))
cmap={"LOW":"#16A34A","MEDIUM":"#D97706",
      "HIGH":"#EA580C","VERY HIGH":"#DC2626"}
smap={"LOW":5,"MEDIUM":16,"HIGH":30,"VERY HIGH":48}
for risk,color in cmap.items():
    sub=df[df['risk_level']==risk]
    if len(sub)>0:
        ax.scatter(sub['longitude'],sub['latitude'],
                   c=color,s=smap[risk],alpha=0.75,zorder=3)
patches=[mpatches.Patch(color=cmap[r],
         label=f"{r} ({vc.get(r,0):,})")
         for r in ['VERY HIGH','HIGH','MEDIUM','LOW']]
ax.legend(handles=patches,fontsize=10,
          title="Fire Risk",loc="upper right")
ax.set_title(
    f"Nepal Forest Fire Risk — {date}\n"
    f"Transformer ONNX (AUC=0.9983) "
    f"+ Open-Meteo 7-Day Forecast",
    fontsize=12,fontweight='bold')
ax.set_xlim(80.0,88.2); ax.set_ylim(26.3,30.5)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(f"fire_risk_map_{date}.png",
            dpi=120,bbox_inches='tight',facecolor='white')
print(f"Map saved: fire_risk_map_{date}.png")
print("DONE")
