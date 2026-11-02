# 🏗️ Architecture Documentation
## Air Quality Index Analysis and Prediction

---

## 1. System Architecture Overview

This project implements a complete Data Engineering and MLOps pipeline with five major layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
│   OpenAQ API (pollution)   Open-Meteo API (weather)        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  INGESTION LAYER                            │
│   openaq_collector.py   weather_collector.py               │
│   raw_storage.py        ingestion_logger.py                │
│   Retry logic · Metadata sidecars · JSONL audit log        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  RAW LANDING ZONE                           │
│   data/raw/<source>/<YYYY-MM-DD>/<station>_<time>.json     │
│   data/raw/ingestion_log.jsonl                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│            ETL PIPELINE (Apache Airflow / run_pipeline.py) │
│   18 tasks: validate → clean → normalise → deduplicate     │
│            → gap-detect → join → aggregate → AQI → load   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  STORAGE LAYER (PostgreSQL)                 │
│   stations  pollution_measurements  weather_measurements   │
│   daily_aqi  model_predictions  ingestion_runs             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              GOLD / ANALYTICAL LAYER (10 Views)             │
│   vw_city_daily_aqi  vw_monthly_aqi  vw_station_latest_aqi │
│   vw_weather_aqi_correlation  vw_aqi_category_distribution │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                             │
│   Streamlit Dashboard (12 sections, Plotly charts)         │
│   FastAPI Prediction Service (4 endpoints)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

```
Raw JSON (OpenAQ)          Raw JSON (Open-Meteo)
       │                          │
       └──────────┬───────────────┘
                  ↓
         data/raw/  (never modified)
                  ↓
         Airflow Task 1-3: Extract + Store
                  ↓
         Airflow Task 4: Validate (9 checks)
                  ↓ (rejected → data/rejected/)
         Airflow Task 5-8: Clean + Normalise
                  ↓
         Airflow Task 9: Gap Detection
                  ↓
         Airflow Task 10: Range Validation
                  ↓
         Airflow Task 11: Join Pollution + Weather
                  ↓
         Airflow Task 12-13: Hourly + Daily Averages
                  ↓
         Airflow Task 14-15: AQI Calculation + Categories
                  ↓
         Airflow Task 16: Load → PostgreSQL
                  ↓
         Airflow Task 17: Refresh Analytical Views
                  ↓
         Airflow Task 18: Data Quality Checks
```

---

## 3. ETL Flow (18 Airflow Tasks)

| Task | Name | Description |
|------|------|-------------|
| 01 | `t01_extract_openaq` | Fetch pollution from OpenAQ API for all stations |
| 02 | `t02_extract_weather` | Fetch weather from Open-Meteo API |
| 03 | `t03_store_raw` | Verify raw files written to data/raw/ |
| 04 | `t04_validate_raw` | 9 validation checks, save rejected records |
| 05 | `t05_clean_timestamps` | Parse + normalise to UTC |
| 06 | `t06_normalize_stations` | Strip/title-case station and city names |
| 07 | `t07_normalize_units` | Standardise unit strings (µg/m³, ppm) |
| 08 | `t08_remove_duplicates` | Deduplicate on (station, timestamp, pollutant) |
| 09 | `t09_detect_gaps` | Log missing time periods per station |
| 10 | `t10_validate_ranges` | Final pollutant range check pass |
| 11 | `t11_join_pollution_weather` | Merge on (station_id, timestamp) |
| 12 | `t12_hourly_averages` | Aggregate to hourly level |
| 13 | `t13_daily_averages` | Aggregate to daily level |
| 14 | `t14_calculate_aqi` | US-EPA formula, dominant pollutant |
| 15 | `t15_aqi_categories` | Assign Good/Moderate/USG/Unhealthy etc. |
| 16 | `t16_load_postgres` | Upsert to all PostgreSQL tables |
| 17 | `t17_update_analytics` | Refresh all 10 gold-layer views |
| 18 | `t18_run_quality_checks` | Final DQ checks, log to DB |

---

## 4. ML Pipeline

```
PostgreSQL (daily_aqi table)
         ↓
  Feature Engineering (transformation/feature_engineering.py)
  ┌──────────────────────────────────────────────────┐
  │  Pollutant concentrations (7)                    │
  │  Weather variables (6)                           │
  │  Temporal features (5): year, month, dow, season │
  │  Lag features (4): AQI_lag_1/2/3/7              │
  │  Rolling features (2): AQI_rolling_3/7          │
  │  Total: 23 features                              │
  └──────────────────────────────────────────────────┘
         ↓
  Chronological Split (NO shuffle — prevents leakage)
  Train 70% │ Validation 15% │ Test 15%
         ↓
  ┌──── Regression ────────┐  ┌──── Classification ────┐
  │ LinearRegression       │  │ LogisticRegression      │
  │ RandomForestRegressor  │  │ RandomForestClassifier  │
  │ GradientBoosting       │  │ GradientBoosting        │
  └───────────┬────────────┘  └──────────┬─────────────┘
              ↓ Select by Test MAE        ↓ Select by Test F1
  ┌──────────────────────────────────────────────────┐
  │  Best Regressor:   RandomForestRegressor         │
  │  MAE=0.13  RMSE=1.34  R²=0.9996                 │
  │                                                  │
  │  Best Classifier:  GradientBoostingClassifier    │
  │  Accuracy=0.9997  F1=0.9997                     │
  └──────────────────────────────────────────────────┘
         ↓
  MLflow Tracking (experiment logs, metrics, artifacts)
         ↓
  Model Registry (models/best_regressor.joblib + best_classifier.joblib)
         ↓
  FastAPI Service (/predict, /batch-predict)
```

---

## 5. MLOps Lifecycle

```
Code change / New data arrives
         ↓
  run_pipeline.py  (ETL — data refreshed)
         ↓
  monitoring/data_quality_monitor.py  (quality gate)
         ↓
  monitoring/drift_monitor.py  (PSI feature drift)
         ↓
  Should retrain? (performance_monitor.check_retraining_needed)
  ├── YES → python ml/launch_training.py
  │          ↓
  │         New model version saved
  │          ↓
  │         FastAPI reloads model on next request
  └── NO  → Continue serving current model
```

Retraining triggers (any one sufficient):
- Test MAE > 30.0
- PSI drift in ≥ 2 features
- ≥ 500 new records available
- 30 days since last training (scheduled)

---

## 6. Monitoring Architecture

```
Every pipeline run:
  ├── data_quality_monitor.py
  │     checks: nulls, duplicates, ranges, timestamps, AQI consistency
  │     logs to: logs/data_quality.log
  │
  ├── drift_monitor.py
  │     method: Population Stability Index (PSI)
  │     threshold: PSI > 0.2 → alert
  │     logs to: logs/drift.log
  │
  └── performance_monitor.py
        tracks: MAE, RMSE on evaluated predictions
        threshold: MAE > 30.0 → alert
        logs to: logs/performance.log

Every API request:
  └── api/main.py middleware
        tracks: endpoint, latency_ms, success/fail
        logs to: logs/api_requests.jsonl
```

---

## 7. Security

- All credentials in `.env` (never in source code)
- `.env` excluded from git via `.gitignore`
- `.env.example` provided with placeholder values
- Docker environment variables injected at runtime
- No hard-coded passwords, API keys, or secrets anywhere

---

## 8. File Naming Conventions

| Pattern | Example | Purpose |
|---------|---------|---------|
| `data/raw/<src>/<date>/<sid>_<time>.json` | `openaq/2024-01-01/DEL_AV_14-30-00.json` | Raw API response |
| `data/staging/<NN>_<name>.parquet` | `04_clean_timestamps.parquet` | Pipeline stage output |
| `data/processed/aqi_processed.csv` | — | Final analytical dataset |
| `logs/<monitor>_<runid>.json` | `pipeline_summary_20240101_123456.json` | Run reports |
| `models/best_<type>.joblib` | `best_regressor.joblib` | Production model |

---

## 9. Port Reference

| Service | Port | URL |
|---------|------|-----|
| Streamlit Dashboard | 8501 | http://localhost:8501 |
| FastAPI | 8000 | http://localhost:8000/docs |
| MLflow UI | 5001 | http://localhost:5001 |
| PostgreSQL | 5432 | localhost:5432 |
| Airflow | 8080 | http://localhost:8080 |
