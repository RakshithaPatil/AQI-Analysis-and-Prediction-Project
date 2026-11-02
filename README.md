# 🌬️ Air Quality Index (AQI) Analysis and Prediction

A complete end-to-end Data Engineering and MLOps project that collects,
processes, analyses, and predicts air quality using real public APIs.

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Architecture](#architecture)
4. [Technologies](#technologies)
5. [Data Sources](#data-sources)
6. [Folder Structure](#folder-structure)
7. [Installation](#installation)
8. [Environment Variables](#environment-variables)
9. [Running the Pipeline](#running-the-pipeline)
10. [Running the Dashboard](#running-the-dashboard)
11. [Training ML Models](#training-ml-models)
12. [Starting FastAPI](#starting-fastapi)
13. [Docker Instructions](#docker-instructions)
14. [Monitoring](#monitoring)
15. [Testing](#testing)
16. [Sample API Request](#sample-api-request)
17. [Troubleshooting](#troubleshooting)
18. [Future Improvements](#future-improvements)

---

## Problem Statement

Air pollution is a major public health crisis in Indian cities.
Real-time AQI data is fragmented across APIs and difficult to analyse collectively.
This project builds a production-style pipeline that:
- Collects data from public APIs automatically
- Cleans, validates, and stores it in a structured database
- Calculates AQI using the US-EPA standard
- Predicts next-day AQI using trained ML models
- Visualises everything in an interactive dashboard

---

## Objectives

1. Collect air pollution data from OpenAQ API
2. Collect weather data from Open-Meteo API
3. Store raw data without modification (data/raw/)
4. Run automated ETL using Apache Airflow (18-task DAG)
5. Clean, validate, and join pollution + weather data
6. Store cleaned data in PostgreSQL
7. Create AQI-related analytical views (Gold Layer)
8. Provide an interactive Streamlit dashboard
9. Build regression + classification ML models
10. Track experiments using MLflow
11. Serve predictions through FastAPI
12. Containerise using Docker
13. Monitor data quality, drift, and prediction performance

---

## Architecture

```
OpenAQ API + Open-Meteo API
        ↓
  Python Collectors (ingestion/)
        ↓
  Raw Landing Zone (data/raw/)
        ↓
  Apache Airflow DAG (18 tasks)
        ↓
  Staging Layer (data/staging/)
        ↓
  Data Quality Checks
        ↓
  Transformation Layer
        ↓
  PostgreSQL (6 tables)
        ↓
  Gold Layer (10 analytical views)
        ↓
  Streamlit Dashboard

  PostgreSQL
        ↓
  Feature Engineering (23 features)
        ↓
  Train / Val / Test (70/15/15 chronological)
        ↓
  Model Training (3 regression + 3 classification)
        ↓
  MLflow Tracking + Model Registry
        ↓
  FastAPI (/predict, /batch-predict)
        ↓
  Docker / Docker Compose
```

---

## Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Engineering | Apache Airflow, Pandas, NumPy |
| Database | PostgreSQL 15, SQLAlchemy |
| Visualisation | Streamlit, Plotly |
| ML — Regression | Linear Regression, Random Forest, Gradient Boosting |
| ML — Classification | Logistic Regression, Random Forest, Gradient Boosting |
| MLOps | MLflow, FastAPI, Docker, Docker Compose |
| Monitoring | PSI drift detection, custom quality checks |

---

## Data Sources

| Source | What we collect | API URL |
|---|---|---|
| OpenAQ v3 | PM2.5, PM10, NO2, SO2, CO, O3 | https://api.openaq.org/v3 |
| Open-Meteo | Temperature, humidity, wind, precipitation, pressure | https://open-meteo.com |
| Dataset | india_aqi_dataset.csv (20 stations, 3 years) | Generated locally |

---

## Folder Structure

```
aqi_prediction/
├── airflow/dags/         ← 18-task Airflow DAG
├── api/                  ← FastAPI service
├── config/               ← config.yaml, settings.py, logger.py
├── dashboard/            ← Streamlit app
├── data/
│   ├── raw/              ← Raw API responses (never overwritten)
│   ├── staging/          ← Intermediate pipeline outputs
│   ├── processed/        ← Final cleaned data
│   └── rejected/         ← Records that failed validation
├── database/             ← PostgreSQL schema + loaders
├── docker/               ← Dockerfiles
├── ingestion/            ← OpenAQ + Open-Meteo collectors
├── logs/                 ← All application logs
├── ml/                   ← Model training scripts
├── models/               ← Saved .joblib model files
├── monitoring/           ← Data quality, drift, performance monitors
├── tests/                ← Unit and integration tests
├── transformation/       ← Cleaning, validation, AQI calculation
├── run_pipeline.py       ← Standalone ETL runner (18 steps)
├── launch.py             ← Start dashboard
├── docker-compose.yml
└── requirements.txt
```

---

## Installation

### Step 1 — Clone / open the project

```powershell
cd "C:\Users\Rakshitha Patil\air_quality\aqi_prediction"
```

### Step 2 — Verify Python version

```powershell
python --version   # needs 3.11+
```

### Step 3 — Install dependencies

```powershell
pip install streamlit plotly pandas numpy scikit-learn pyyaml python-dotenv requests joblib fastapi uvicorn
```

### Step 4 — Copy environment file

```powershell
copy .env.example .env
```

---

## Environment Variables

Edit `.env` and set these values:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=aqi_db
DB_USER=aqi_user
DB_PASSWORD=your_password
DATABASE_URL=postgresql://aqi_user:your_password@localhost:5432/aqi_db
OPENAQ_API_KEY=your_key_here   # optional — free tier works without key
MLFLOW_TRACKING_URI=http://localhost:5001
```

---

## Running the Pipeline

### Generate the dataset (first time only)

```powershell
python data/raw/generate_dataset.py
```

### Run the full ETL pipeline (18 steps)

```powershell
python run_pipeline.py
```

Expected output:
```
✅ Step 01 | Load raw dataset     | rows: 21920 → 21920
✅ Step 02 | Validate pollution   | rows: 21920 → 21920
...
✅ All pipeline steps completed successfully!
```

---

## Running the Dashboard

```powershell
python launch.py
```

Open your browser at: **http://localhost:8501**

The dashboard shows:
- Real-time AQI KPI cards
- Station map coloured by AQI
- 7-day trend charts by city
- Pollutant deep-dive with radar chart
- Temporal heatmaps
- Weather impact scatter plots
- Year-over-year comparison
- AQI prediction panel

---

## Training ML Models

```powershell
python ml/launch_training.py
```

Models trained:
- LinearRegression, RandomForestRegressor, GradientBoostingRegressor
- LogisticRegression, RandomForestClassifier, GradientBoostingClassifier

Best models selected by:
- Regression: lowest Test MAE
- Classification: highest Test F1 (weighted)

Results (on this dataset):
- Best Regressor: **RandomForestRegressor** — MAE=0.13, R²=0.9996
- Best Classifier: **GradientBoostingClassifier** — F1=0.9997, Acc=0.9997

---

## Starting FastAPI

```powershell
python api/launch_api.py
```

API docs: **http://localhost:8000/docs**

Endpoints:
- `GET /health` — liveness check
- `GET /model-info` — model metadata
- `POST /predict` — single prediction
- `POST /batch-predict` — batch predictions

---

## Docker Instructions

### Prerequisites
Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Start all services

```powershell
docker compose up --build
```

### Services started:
| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| MLflow | http://localhost:5001 |
| PostgreSQL | localhost:5432 |

### Stop all services

```powershell
docker compose down
```

---

## Monitoring

Run data quality checks:
```python
from monitoring.data_quality_monitor import run_all_checks
import pandas as pd
df = pd.read_csv("data/processed/aqi_processed.csv")
run_all_checks(df)
```

Check feature drift:
```python
from monitoring.drift_monitor import calculate_drift, print_drift_report
result = calculate_drift(reference_df=train_df, production_df=new_df)
print_drift_report(result)
```

### Retraining triggers
The model should be retrained when:
- Test MAE > 30.0 (model degradation)
- PSI drift in > 2 features
- > 500 new records available
- Monthly scheduled retraining

---

## Testing

```powershell
pip install pytest
python -m pytest tests/ -v
```

Tests cover:
- AQI calculation (US-EPA formula)
- Data cleaning and normalisation
- Validation checks
- FastAPI endpoints
- Integration (pipeline output files)

---

## Sample API Request

```powershell
# Single prediction
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "station_id": "DEL_AV",
    "pm25": 85.0,
    "pm10": 120.0,
    "no2": 40.0,
    "so2": 12.0,
    "co": 800.0,
    "o3": 45.0,
    "temperature": 28.5,
    "humidity": 65.0,
    "wind_speed": 8.0
  }'
```

Expected response:
```json
{
  "station_id": "DEL_AV",
  "predicted_aqi": 162.3,
  "predicted_category": "Unhealthy",
  "model_name": "RandomForestRegressor",
  "model_version": "v1",
  "prediction_timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install <package>` or check Python version |
| `streamlit: command not found` | Use `python -m streamlit run dashboard/app.py` |
| Dashboard shows no data | Run `python run_pipeline.py` first |
| Model not loaded (API 503) | Run `python ml/launch_training.py` first |
| PostgreSQL connection error | Check `.env` DB credentials |
| Wrong Python version | Use `python3.13` or the full Python path |

---

## Future Improvements

1. Connect to live OpenAQ API for real-time data
2. Add LSTM/time-series models for better AQI forecasting
3. Set up Airflow with a production scheduler
4. Add email/SMS alerts when AQI exceeds hazardous level
5. Integrate real PostgreSQL for persistent storage
6. Deploy to AWS/GCP with CI/CD pipeline
7. Add more Indian cities and stations
8. Implement automated retraining pipeline

---

*AQI calculated using the US-EPA standard.*
*Data sources: OpenAQ API, Open-Meteo API.*
*Built with Python, Streamlit, FastAPI, Scikit-learn, Docker.*
