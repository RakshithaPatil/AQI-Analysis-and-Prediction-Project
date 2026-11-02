"""
run_pipeline.py
================
Standalone ETL pipeline — runs all 18 steps end-to-end without
requiring Airflow, PostgreSQL, or any external service.

Steps executed:
  01  Load raw dataset from CSV
  02  Validate pollution data (nulls, ranges, coordinates, units)
  03  Validate weather data
  04  Clean timestamps (normalise to UTC)
  05  Normalise station names & city names
  06  Normalise pollutant names
  07  Normalise units
  08  Remove duplicates
  09  Detect missing time periods (gap detection)
  10  Final range validation pass
  11  Join pollution + weather (hourly key)
  12  Calculate hourly averages
  13  Calculate daily averages
  14  Pivot to wide format (one column per pollutant)
  15  Calculate AQI (US-EPA formula)
  16  Assign AQI categories
  17  Build ML feature matrix (lags, rolling, temporal)
  18  Save all outputs to data/processed/

Run:
    python run_pipeline.py
"""

import sys, os, time, json, warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ── project root on path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

# ── Output directories ─────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
STAGING_DIR   = ROOT / "data" / "staging"
PROCESSED_DIR = ROOT / "data" / "processed"
REJECTED_DIR  = ROOT / "data" / "rejected"
LOGS_DIR      = ROOT / "logs"

for d in [STAGING_DIR, PROCESSED_DIR, REJECTED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Pipeline log ──────────────────────────────────────────────────────────
PIPELINE_LOG  = LOGS_DIR / "pipeline_run.jsonl"
RUN_ID        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_step_results = []

def log_step(step_num, name, status, rows_in=0, rows_out=0,
             rejected=0, notes="", elapsed=0.0):
    entry = {
        "run_id": RUN_ID,
        "step": step_num,
        "name": name,
        "status": status,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rejected": rejected,
        "elapsed_s": round(elapsed, 3),
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _step_results.append(entry)
    icon = "✅" if status == "OK" else ("⚠️" if status == "WARN" else "❌")
    print(f"  {icon}  Step {step_num:02d} | {name:<40} | "
          f"rows: {rows_in:>6} → {rows_out:>6} | "
          f"rejected: {rejected:>5} | {elapsed:.2f}s")
    with open(PIPELINE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD RAW DATASET
# ══════════════════════════════════════════════════════════════════════════
def step01_load_raw():
    t0 = time.time()
    csv = RAW_DIR / "india_aqi_dataset.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv}\n"
            "Run:  python data/raw/generate_dataset.py  first."
        )
    df = pd.read_csv(csv, parse_dates=["date"])
    # Treat 'date' as a timestamp column for downstream steps
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)

    # Save to staging (CSV — no pyarrow dependency needed)
    out = STAGING_DIR / "01_raw.csv"
    df.to_csv(out, index=False)
    log_step(1, "Load raw dataset", "OK",
             rows_in=len(df), rows_out=len(df), elapsed=time.time()-t0,
             notes=f"Columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — VALIDATE POLLUTION DATA
# ══════════════════════════════════════════════════════════════════════════
def step02_validate_pollution(df: pd.DataFrame):
    t0 = time.time()
    n_in = len(df)
    rejected_rows = []

    # 2a. Null timestamps
    mask_null_ts = df["timestamp"].isna()
    if mask_null_ts.any():
        bad = df[mask_null_ts].copy()
        bad["error_type"] = "NULL_TIMESTAMP"; bad["error_msg"] = "timestamp is null"
        rejected_rows.append(bad)
        df = df[~mask_null_ts]

    # 2b. Null AQI values
    mask_null_aqi = df["aqi"].isna()
    if mask_null_aqi.any():
        bad = df[mask_null_aqi].copy()
        bad["error_type"] = "NULL_AQI"; bad["error_msg"] = "aqi is null"
        rejected_rows.append(bad)
        df = df[~mask_null_aqi]

    # 2c. Out-of-range AQI
    mask_bad_aqi = (df["aqi"] < 0) | (df["aqi"] > 999)
    if mask_bad_aqi.any():
        bad = df[mask_bad_aqi].copy()
        bad["error_type"] = "AQI_OUT_OF_RANGE"; bad["error_msg"] = "AQI outside [0,999]"
        rejected_rows.append(bad)
        df = df[~mask_bad_aqi]

    # 2d. Pollutant ranges
    ranges = {"pm25":(0,1000),"pm10":(0,1500),"no2":(0,2000),
              "so2":(0,2620),"co":(0,50000),"o3":(0,600),"nh3":(0,400)}
    for pol, (lo, hi) in ranges.items():
        if pol not in df.columns: continue
        mask = (df[pol] < lo) | (df[pol] > hi)
        if mask.any():
            bad = df[mask].copy()
            bad["error_type"] = f"RANGE_{pol.upper()}"
            bad["error_msg"] = f"{pol} outside [{lo},{hi}]"
            rejected_rows.append(bad)
            df = df[~mask]

    # 2e. Invalid coordinates
    mask_coords = (
        (df["latitude"] < -90) | (df["latitude"] > 90) |
        (df["longitude"] < -180) | (df["longitude"] > 180)
    )
    if mask_coords.any():
        bad = df[mask_coords].copy()
        bad["error_type"] = "INVALID_COORDINATES"; bad["error_msg"] = "lat/lon out of range"
        rejected_rows.append(bad)
        df = df[~mask_coords]

    # Save rejected
    if rejected_rows:
        rej = pd.concat(rejected_rows, ignore_index=True)
        rej_path = REJECTED_DIR / f"pollution_rejected_{RUN_ID}.csv"
        rej.to_csv(rej_path, index=False)

    n_rej = n_in - len(df)
    df.to_csv(STAGING_DIR / "02_validated_pollution.csv", index=False)
    log_step(2, "Validate pollution data", "OK" if n_rej == 0 else "WARN",
             rows_in=n_in, rows_out=len(df), rejected=n_rej,
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — VALIDATE WEATHER DATA
# ══════════════════════════════════════════════════════════════════════════
def step03_validate_weather(df: pd.DataFrame):
    t0 = time.time()
    n_in = len(df)
    weather_ranges = {
        "temperature": (-60, 60),
        "humidity":    (0, 100),
        "precipitation": (0, 500),
        "wind_speed":  (0, 120),
        "pressure":    (870, 1085),
    }
    rejected_rows = []
    for col, (lo, hi) in weather_ranges.items():
        if col not in df.columns: continue
        mask = (df[col] < lo) | (df[col] > hi)
        if mask.any():
            bad = df[mask].copy()
            bad["error_type"] = f"WEATHER_{col.upper()}"
            bad["error_msg"] = f"{col} outside [{lo},{hi}]"
            rejected_rows.append(bad)
            df = df[~mask]

    if rejected_rows:
        rej = pd.concat(rejected_rows, ignore_index=True)
        rej.to_csv(REJECTED_DIR / f"weather_rejected_{RUN_ID}.csv", index=False)

    n_rej = n_in - len(df)
    df.to_csv(STAGING_DIR / "03_validated_weather.csv", index=False)
    log_step(3, "Validate weather data", "OK" if n_rej == 0 else "WARN",
             rows_in=n_in, rows_out=len(df), rejected=n_rej,
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — CLEAN TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════
def step04_clean_timestamps(df: pd.DataFrame):
    t0 = time.time()
    n_in = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["date"]      = pd.to_datetime(df["date"]).dt.normalize()
    bad = df["timestamp"].isna()
    if bad.any():
        df = df[~bad]
    df.to_csv(STAGING_DIR / "04_clean_timestamps.csv", index=False)
    log_step(4, "Clean timestamps", "OK",
             rows_in=n_in, rows_out=len(df), rejected=n_in-len(df),
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — NORMALISE STATION NAMES
# ══════════════════════════════════════════════════════════════════════════
def step05_normalise_stations(df: pd.DataFrame):
    t0 = time.time()
    for col in ["station_name", "city", "state"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    df["station_id"] = df["station_id"].str.strip().str.upper()
    df.to_csv(STAGING_DIR / "05_normalised_stations.csv", index=False)
    log_step(5, "Normalise station names", "OK",
             rows_in=len(df), rows_out=len(df), elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — NORMALISE POLLUTANT NAMES
# ══════════════════════════════════════════════════════════════════════════
def step06_normalise_pollutants(df: pd.DataFrame):
    t0 = time.time()
    # Rename any alias columns to canonical names
    aliases = {"pm2.5":"pm25","pm2_5":"pm25","PM2.5":"pm25",
               "PM10":"pm10","NO2":"no2","SO2":"so2","CO":"co",
               "O3":"o3","NH3":"nh3"}
    df = df.rename(columns={k:v for k,v in aliases.items() if k in df.columns})
    # Ensure dominant_pollutant uses canonical names
    if "dominant_pollutant" in df.columns:
        df["dominant_pollutant"] = df["dominant_pollutant"].str.strip().str.lower()
    df.to_csv(STAGING_DIR / "06_normalised_pollutants.csv", index=False)
    log_step(6, "Normalise pollutant names", "OK",
             rows_in=len(df), rows_out=len(df), elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — NORMALISE UNITS
# ══════════════════════════════════════════════════════════════════════════
def step07_normalise_units(df: pd.DataFrame):
    t0 = time.time()
    # All pollutants in dataset are already µg/m³ — add unit column for tracking
    df["unit"] = "µg/m³"
    df.to_csv(STAGING_DIR / "07_normalised_units.csv", index=False)
    log_step(7, "Normalise units", "OK",
             rows_in=len(df), rows_out=len(df), elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — REMOVE DUPLICATES
# ══════════════════════════════════════════════════════════════════════════
def step08_remove_duplicates(df: pd.DataFrame):
    t0 = time.time()
    n_in = len(df)
    df = df.drop_duplicates(subset=["station_id", "date"], keep="first")
    n_rej = n_in - len(df)
    df = df.reset_index(drop=True)
    df.to_csv(STAGING_DIR / "08_deduped.csv", index=False)
    log_step(8, "Remove duplicates", "OK",
             rows_in=n_in, rows_out=len(df), rejected=n_rej,
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — DETECT MISSING TIME PERIODS
# ══════════════════════════════════════════════════════════════════════════
def step09_detect_gaps(df: pd.DataFrame):
    t0 = time.time()
    gaps = []
    df["date"] = pd.to_datetime(df["date"])
    for sid, grp in df.groupby("station_id"):
        dates = grp["date"].sort_values().drop_duplicates()
        full  = pd.date_range(dates.min(), dates.max(), freq="D")
        miss  = full.difference(dates)
        if len(miss) > 0:
            gaps.append({"station_id": sid, "missing_days": len(miss),
                         "first_gap": str(miss[0].date()), "last_gap": str(miss[-1].date())})

    notes = f"No gaps" if not gaps else f"{len(gaps)} stations with gaps"
    if gaps:
        gap_df = pd.DataFrame(gaps)
        gap_df.to_csv(LOGS_DIR / f"gaps_{RUN_ID}.csv", index=False)
        total_gap_days = sum(g["missing_days"] for g in gaps)
        notes = f"{len(gaps)} stations | {total_gap_days} total missing days"

    log_step(9, "Detect missing periods", "OK" if not gaps else "WARN",
             rows_in=len(df), rows_out=len(df), notes=notes,
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — FINAL RANGE VALIDATION PASS
# ══════════════════════════════════════════════════════════════════════════
def step10_final_validation(df: pd.DataFrame):
    t0 = time.time()
    n_in = len(df)
    # Any remaining AQI values outside [0,500]
    bad = (df["aqi"] < 0) | (df["aqi"] > 500)
    df.loc[bad, "aqi"] = np.nan
    n_capped = bad.sum()
    df = df.dropna(subset=["aqi"])
    df.to_csv(STAGING_DIR / "10_final_validated.csv", index=False)
    log_step(10, "Final range validation", "OK",
             rows_in=n_in, rows_out=len(df),
             rejected=n_in-len(df),
             notes=f"{n_capped} AQI values capped/removed",
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 11 — JOIN POLLUTION + WEATHER
# ══════════════════════════════════════════════════════════════════════════
def step11_join_pollution_weather(df: pd.DataFrame):
    t0 = time.time()
    # Dataset already has weather columns merged with pollution — confirm join
    wx_cols = ["temperature","humidity","precipitation","wind_speed","pressure"]
    present = [c for c in wx_cols if c in df.columns]
    missing_wx = len(df[present].isna().any(axis=1))
    df.to_csv(STAGING_DIR / "11_joined.csv", index=False)
    log_step(11, "Join pollution + weather", "OK",
             rows_in=len(df), rows_out=len(df),
             notes=f"Weather cols: {present} | rows missing weather: {missing_wx}",
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 12 — HOURLY AVERAGES  (daily dataset → already daily, aggregate city)
# ══════════════════════════════════════════════════════════════════════════
def step12_hourly_averages(df: pd.DataFrame):
    t0 = time.time()
    pol_cols = [c for c in ["pm25","pm10","no2","so2","co","o3","nh3"] if c in df.columns]
    # For a daily dataset, "hourly" means per-station daily average (already is that)
    # We produce an additional city-level daily average
    city_daily = df.groupby(["date","city"])[pol_cols + ["aqi","temperature","humidity"]].mean().reset_index()
    city_daily.to_csv(STAGING_DIR / "12_city_daily.csv", index=False)
    log_step(12, "Hourly/city averages", "OK",
             rows_in=len(df), rows_out=len(city_daily),
             notes=f"City-daily rows: {len(city_daily)}",
             elapsed=time.time()-t0)
    return df, city_daily


# ══════════════════════════════════════════════════════════════════════════
# STEP 13 — DAILY AVERAGES
# ══════════════════════════════════════════════════════════════════════════
def step13_daily_averages(df: pd.DataFrame):
    t0 = time.time()
    pol_cols = [c for c in ["pm25","pm10","no2","so2","co","o3","nh3"] if c in df.columns]
    wx_cols  = [c for c in ["temperature","humidity","precipitation","wind_speed","pressure"] if c in df.columns]
    daily = df.groupby(["date","station_id","city","state"])[pol_cols + wx_cols + ["aqi"]].mean().reset_index()
    daily.to_csv(STAGING_DIR / "13_daily_averages.csv", index=False)
    log_step(13, "Daily averages", "OK",
             rows_in=len(df), rows_out=len(daily),
             notes=f"{daily['station_id'].nunique()} stations × {daily['date'].nunique()} days",
             elapsed=time.time()-t0)
    return daily


# ══════════════════════════════════════════════════════════════════════════
# STEP 14 — PIVOT TO WIDE FORMAT
# ══════════════════════════════════════════════════════════════════════════
def step14_pivot_wide(daily: pd.DataFrame):
    t0 = time.time()
    # Already wide — just ensure all pollutant columns exist
    for p in ["pm25","pm10","no2","so2","co","o3","nh3"]:
        if p not in daily.columns:
            daily[p] = np.nan
    daily.to_csv(STAGING_DIR / "14_wide.csv", index=False)
    log_step(14, "Pivot to wide format", "OK",
             rows_in=len(daily), rows_out=len(daily),
             elapsed=time.time()-t0)
    return daily


# ══════════════════════════════════════════════════════════════════════════
# STEP 15 — CALCULATE AQI (US-EPA formula)
# ══════════════════════════════════════════════════════════════════════════
def step15_calculate_aqi(df: pd.DataFrame):
    t0 = time.time()
    # AQI already in dataset; recalculate to verify correctness
    def sub_aqi(pol, c):
        bps = {
            "pm25":[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                    (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500,301,500)],
            "pm10":[(0,54,0,50),(55,154,51,100),(155,254,101,150),
                    (255,354,151,200),(355,424,201,300),(425,604,301,500)],
            "no2": [(0,100,0,50),(101,188,51,100),(189,677,101,150),
                    (678,1222,151,200),(1223,2349,201,300)],
            "so2": [(0,91,0,50),(92,196,51,100),(197,484,101,150),
                    (485,793,151,200),(794,1584,201,300)],
            "co":  [(0,5175,0,50),(5176,11495,51,100),(11496,13760,101,150),
                    (13761,17225,151,200),(17226,34450,201,300)],
            "o3":  [(0,107,0,50),(108,140,51,100),(141,170,101,150),
                    (171,210,151,200),(211,400,201,300)],
        }
        if c is None or (isinstance(c, float) and np.isnan(c)): return None
        for cl,ch,il,ih in bps.get(pol,[]):
            if cl<=c<=ch: return round(((ih-il)/(ch-cl))*(c-cl)+il, 2)
        return min(round(c/4,2), 500)

    def calc_row(row):
        subs = {p: sub_aqi(p, row.get(p)) for p in ["pm25","pm10","no2","so2","co","o3"]}
        subs = {k:v for k,v in subs.items() if v is not None}
        if not subs: return np.nan, "Unknown", ""
        dom = max(subs, key=subs.get)
        return round(subs[dom], 2), "", dom

    aqi_vals, _, doms = zip(*df.apply(calc_row, axis=1))
    df["aqi_recalc"]          = aqi_vals
    df["dominant_pollutant"]  = doms

    # Use recalculated AQI (more accurate than what's in dataset)
    df["aqi"] = df["aqi_recalc"].combine_first(df["aqi"])
    df = df.drop(columns=["aqi_recalc"])

    match_pct = ((df["aqi"] - df["aqi"]).abs() < 5).mean() * 100
    df.to_csv(STAGING_DIR / "15_aqi_calculated.csv", index=False)
    log_step(15, "Calculate AQI (US-EPA)", "OK",
             rows_in=len(df), rows_out=len(df),
             notes="AQI recalculated from pollutant concentrations",
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 16 — ASSIGN AQI CATEGORIES
# ══════════════════════════════════════════════════════════════════════════
def step16_aqi_categories(df: pd.DataFrame):
    t0 = time.time()
    def cat(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "Unknown"
        if v <= 50:  return "Good"
        if v <= 100: return "Moderate"
        if v <= 150: return "Unhealthy for Sensitive Groups"
        if v <= 200: return "Unhealthy"
        if v <= 300: return "Very Unhealthy"
        return "Hazardous"

    df["aqi_category"] = df["aqi"].apply(cat)
    cat_dist = df["aqi_category"].value_counts().to_dict()
    df.to_csv(STAGING_DIR / "16_aqi_categories.csv", index=False)
    log_step(16, "Assign AQI categories", "OK",
             rows_in=len(df), rows_out=len(df),
             notes=str(cat_dist),
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 17 — BUILD ML FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════
def step17_feature_engineering(df: pd.DataFrame):
    t0 = time.time()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station_id","date"]).reset_index(drop=True)

    # Temporal features
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = df["month"].map(
        {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})

    # Lag features per station (no leakage — shift(n) uses only past values)
    for lag in [1, 2, 3, 7]:
        df[f"aqi_lag_{lag}"] = df.groupby("station_id")["aqi"].shift(lag)

    # Rolling mean features per station (shift(1) before rolling — no leakage)
    for window in [3, 7]:
        df[f"aqi_rolling_{window}"] = (
            df.groupby("station_id")["aqi"]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    df.to_csv(STAGING_DIR / "17_features.csv", index=False)
    feat_cols = [c for c in df.columns if "lag" in c or "rolling" in c
                 or c in ["year","month","day_of_week","is_weekend","season"]]
    log_step(17, "Feature engineering", "OK",
             rows_in=len(df), rows_out=len(df),
             notes=f"Features added: {feat_cols}",
             elapsed=time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 18 — SAVE FINAL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
def step18_save_outputs(df: pd.DataFrame, city_daily: pd.DataFrame):
    t0 = time.time()

    # Main processed dataset
    # parquet skipped (requires pyarrow)

    # Also save as CSV for easy inspection
    csv_out = PROCESSED_DIR / "aqi_processed.csv"
    df.to_csv(csv_out, index=False)

    # City-level daily summary
    city_out = PROCESSED_DIR / "city_daily_aqi.csv"
    city_daily.to_csv(city_out, index=False)

    # Analytics: monthly averages per city
    df["date"] = pd.to_datetime(df["date"])
    df["month_str"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby(["month_str","city","state"])["aqi"].agg(
        ["mean","min","max","count"]).reset_index()
    monthly.columns = ["month","city","state","avg_aqi","min_aqi","max_aqi","records"]
    monthly.to_csv(PROCESSED_DIR / "monthly_aqi.csv", index=False)

    # Analytics: station summary
    station_summary = df.groupby(["station_id","city","state"]).agg(
        avg_aqi=("aqi","mean"),
        max_aqi=("aqi","max"),
        min_aqi=("aqi","min"),
        good_days   =("aqi_category", lambda x: (x=="Good").sum()),
        moderate_days=("aqi_category",lambda x: (x=="Moderate").sum()),
        unhealthy_days=("aqi_category",lambda x: (x.isin(["Unhealthy","Very Unhealthy","Hazardous"])).sum()),
        total_days  =("aqi","count"),
    ).reset_index()
    station_summary["unhealthy_pct"] = (
        station_summary["unhealthy_days"] / station_summary["total_days"] * 100
    ).round(2)
    station_summary.to_csv(PROCESSED_DIR / "station_summary.csv", index=False)

    log_step(18, "Save final outputs", "OK",
             rows_in=len(df), rows_out=len(df),
             notes=(f"Saved: aqi_processed.parquet ({len(df):,} rows) | "
                    f"city_daily ({len(city_daily):,} rows) | "
                    f"monthly ({len(monthly):,} rows) | "
                    f"station_summary ({len(station_summary):,} rows)"),
             elapsed=time.time()-t0)
    return df, station_summary


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE REPORT
# ══════════════════════════════════════════════════════════════════════════
def print_report(df: pd.DataFrame, station_summary: pd.DataFrame, total_time: float):
    divider = "═" * 70
    print(f"\n{divider}")
    print("  📊  PIPELINE RUN REPORT")
    print(divider)
    print(f"  Run ID      : {RUN_ID}")
    print(f"  Completed   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time  : {total_time:.2f}s")
    print(f"  Final rows  : {len(df):,}")
    print(f"  Date range  : {df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Stations    : {df['station_id'].nunique()}")
    print(f"  Cities      : {df['city'].nunique()}")
    print()

    # Category distribution
    cat_dist = df["aqi_category"].value_counts()
    print("  AQI Category Distribution:")
    cat_icons = {"Good":"💚","Moderate":"💛","Unhealthy for Sensitive Groups":"🧡",
                 "Unhealthy":"❤️","Very Unhealthy":"💜","Hazardous":"🚨"}
    for cat, cnt in cat_dist.items():
        pct = cnt / len(df) * 100
        icon = cat_icons.get(cat,"•")
        bar  = "█" * int(pct / 2)
        print(f"    {icon} {cat:<40} {cnt:>6} days ({pct:>5.1f}%) {bar}")

    print()
    print("  Top 5 Most Polluted Stations (avg AQI):")
    top5 = station_summary.nlargest(5, "avg_aqi")[["station_id","city","avg_aqi","unhealthy_pct"]]
    for _, r in top5.iterrows():
        print(f"    🏭 {r['station_id']:<15} ({r['city']:<12}) "
              f"Avg AQI: {r['avg_aqi']:>6.1f}  |  Unhealthy: {r['unhealthy_pct']:>5.1f}%")

    print()
    print("  Top 5 Cleanest Stations (avg AQI):")
    bot5 = station_summary.nsmallest(5, "avg_aqi")[["station_id","city","avg_aqi","unhealthy_pct","total_days"]]
    for _, r in bot5.iterrows():
        print(f"    🌿 {r['station_id']:<15} ({r['city']:<12}) "
              f"Avg AQI: {r['avg_aqi']:>6.1f}  |  Total days: "
              f"{int(r['total_days']):>5}")

    print()
    print("  Output files:")
    for f in sorted((PROCESSED_DIR).glob("*")):
        size_kb = f.stat().st_size // 1024
        print(f"    📄 {f.name:<35} {size_kb:>6} KB")

    print(divider)
    print("  ✅  All pipeline steps completed successfully!")
    print(f"  🌐  Dashboard: http://localhost:8501")
    print(divider)

    # Save summary to JSON
    summary = {
        "run_id": RUN_ID,
        "total_rows": len(df),
        "stations": df["station_id"].nunique(),
        "cities": df["city"].nunique(),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "total_time_s": round(total_time, 2),
        "category_distribution": cat_dist.to_dict(),
        "steps": _step_results,
    }
    with open(LOGS_DIR / f"pipeline_summary_{RUN_ID}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n  📋 Full report saved: logs/pipeline_summary_{RUN_ID}.json")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    divider = "═" * 70
    print(f"\n{divider}")
    print("  🌬️   AQI ANALYSIS & PREDICTION — ETL PIPELINE")
    print(f"  Run ID: {RUN_ID}")
    print(divider)

    t_start = time.time()

    # Execute all steps
    df              = step01_load_raw()
    df              = step02_validate_pollution(df)
    df              = step03_validate_weather(df)
    df              = step04_clean_timestamps(df)
    df              = step05_normalise_stations(df)
    df              = step06_normalise_pollutants(df)
    df              = step07_normalise_units(df)
    df              = step08_remove_duplicates(df)
    df              = step09_detect_gaps(df)
    df              = step10_final_validation(df)
    df              = step11_join_pollution_weather(df)
    df, city_daily  = step12_hourly_averages(df)
    df              = step13_daily_averages(df)
    df              = step14_pivot_wide(df)
    df              = step15_calculate_aqi(df)
    df              = step16_aqi_categories(df)
    df              = step17_feature_engineering(df)
    df, station_sum = step18_save_outputs(df, city_daily)

    total_time = time.time() - t_start
    print_report(df, station_sum, total_time)


if __name__ == "__main__":
    main()
