"""
api/main.py
============
FastAPI prediction service for AQI Analysis and Prediction.

Endpoints:
  GET  /health          — liveness check
  GET  /model-info      — model metadata and metrics
  POST /predict         — single AQI prediction
  POST /batch-predict   — batch predictions

Run:
  uvicorn api.main:app --reload --port 8000

Or via launch script:
  python api/launch_api.py
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Project path ──────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.model_loader import (get_feature_cols, get_model_info, get_uptime_s,
                               is_loaded, load_models, predict)
from api.schemas import (BatchPredictionInput, BatchPredictionOutput,
                          HealthResponse, ModelInfoResponse, PredictionInput,
                          PredictionOutput)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("aqi_api")

# ── API log file ──────────────────────────────────────────────────────────
API_LOG = ROOT / "logs" / "api_requests.jsonl"
API_LOG.parent.mkdir(exist_ok=True)

def _log_request(endpoint: str, latency_ms: float, success: bool,
                 detail: str = ""):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "latency_ms": round(latency_ms, 2),
        "success": success,
        "detail": detail,
    }
    with open(API_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AQI Prediction API",
    description=(
        "Predicts Air Quality Index (AQI) and AQI category using "
        "trained ML models (RandomForestRegressor + GradientBoostingClassifier). "
        "AQI standard: US-EPA."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_VERSION = "v1"


# ── Startup: load models ──────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    logger.info("Loading ML models …")
    ok = load_models()
    if ok:
        info = get_model_info()
        logger.info(
            "Models loaded — regressor: %s | classifier: %s",
            info.get("best_regressor", {}).get("name", "?"),
            info.get("best_classifier", {}).get("name", "?"),
        )
    else:
        logger.warning(
            "Models NOT loaded. Run the training pipeline first: "
            "python ml/train_models.py"
        )


# ── Request latency middleware ────────────────────────────────────────────
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - t0) * 1000
    response.headers["X-Latency-Ms"] = str(round(latency_ms, 2))
    return response


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness check — returns model load status and uptime."""
    info = get_model_info()
    name = info.get("best_regressor", {}).get("name", "not loaded")
    return HealthResponse(
        status="ok" if is_loaded() else "degraded",
        model_loaded=is_loaded(),
        model_name=name,
        uptime_s=round(get_uptime_s(), 1),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return trained model names, metrics, and feature list."""
    if not is_loaded():
        raise HTTPException(503, "Models not loaded")
    info = get_model_info()
    reg  = info.get("best_regressor",  {})
    cls  = info.get("best_classifier", {})
    return ModelInfoResponse(
        regressor_name       = reg.get("name", "?"),
        classifier_name      = cls.get("name", "?"),
        regressor_test_mae   = reg.get("test_mae", 0.0),
        regressor_test_r2    = reg.get("test_r2",  0.0),
        classifier_test_f1   = cls.get("test_f1",  0.0),
        classifier_test_acc  = cls.get("test_accuracy", 0.0),
        feature_count        = len(get_feature_cols()),
        feature_names        = get_feature_cols(),
        label_classes        = info.get("label_classes", []),
        model_version        = MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_single(body: PredictionInput):
    """
    Predict AQI and AQI category for a single observation.

    The model uses 23 features: pollutant concentrations, weather variables,
    temporal features, and lag/rolling AQI values.
    If lag/rolling values are not provided, the imputer fills them with
    training-set medians.
    """
    if not is_loaded():
        raise HTTPException(503, "Models not loaded. Run: python ml/train_models.py")

    t0 = time.time()

    # Auto-fill temporal features from current time if not provided
    now = datetime.now(timezone.utc)
    features = {
        "pm25": body.pm25, "pm10": body.pm10, "no2": body.no2,
        "so2":  body.so2,  "co":   body.co,   "o3":  body.o3,
        "nh3":  body.nh3,
        "temperature":   body.temperature,
        "humidity":      body.humidity,
        "precipitation": body.precipitation,
        "wind_speed":    body.wind_speed,
        "pressure":      body.pressure,
        "year":        body.year        or now.year,
        "month":       body.month       or now.month,
        "day_of_week": body.day_of_week if body.day_of_week is not None else now.weekday(),
        "is_weekend":  body.is_weekend  if body.is_weekend  is not None
                       else int(now.weekday() >= 5),
        "season": body.season if body.season is not None
                  else {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2}.get(now.month, 3),
        "aqi_lag_1":     body.aqi_lag_1,
        "aqi_lag_2":     body.aqi_lag_2,
        "aqi_lag_3":     body.aqi_lag_3,
        "aqi_lag_7":     body.aqi_lag_7,
        "aqi_rolling_3": body.aqi_rolling_3,
        "aqi_rolling_7": body.aqi_rolling_7,
    }

    try:
        aqi_val, category = predict(features)
        latency = (time.time() - t0) * 1000
        logger.info("Prediction: aqi=%.1f cat=%s latency=%.1fms",
                    aqi_val, category, latency)
        _log_request("/predict", latency, True,
                     f"aqi={aqi_val} cat={category}")
    except Exception as e:
        latency = (time.time() - t0) * 1000
        _log_request("/predict", latency, False, str(e))
        logger.error("Prediction failed: %s", e)
        raise HTTPException(500, f"Prediction error: {e}")

    info = get_model_info()
    reg_name = info.get("best_regressor", {}).get("name", "model")

    return PredictionOutput(
        station_id          = body.station_id,
        predicted_aqi       = aqi_val,
        predicted_category  = category,
        model_name          = reg_name,
        model_version       = MODEL_VERSION,
        prediction_timestamp= now.isoformat(),
        input_features      = features,
    )


@app.post("/batch-predict", response_model=BatchPredictionOutput, tags=["Prediction"])
def predict_batch(body: BatchPredictionInput):
    """
    Predict AQI for multiple observations in one request.
    Maximum 500 records per batch.
    """
    if not is_loaded():
        raise HTTPException(503, "Models not loaded")
    if len(body.records) > 500:
        raise HTTPException(400, "Maximum 500 records per batch")

    t0 = time.time()
    results = []
    info = get_model_info()
    reg_name = info.get("best_regressor", {}).get("name", "model")
    now = datetime.now(timezone.utc)

    for rec in body.records:
        features = {
            "pm25": rec.pm25, "pm10": rec.pm10, "no2": rec.no2,
            "so2":  rec.so2,  "co":   rec.co,   "o3":  rec.o3,
            "nh3":  rec.nh3,
            "temperature": rec.temperature, "humidity": rec.humidity,
            "precipitation": rec.precipitation, "wind_speed": rec.wind_speed,
            "pressure": rec.pressure,
            "year": rec.year or now.year,
            "month": rec.month or now.month,
            "day_of_week": rec.day_of_week if rec.day_of_week is not None else now.weekday(),
            "is_weekend": rec.is_weekend if rec.is_weekend is not None else int(now.weekday()>=5),
            "season": rec.season if rec.season is not None
                      else {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2}.get(now.month, 3),
            "aqi_lag_1": rec.aqi_lag_1, "aqi_lag_2": rec.aqi_lag_2,
            "aqi_lag_3": rec.aqi_lag_3, "aqi_lag_7": rec.aqi_lag_7,
            "aqi_rolling_3": rec.aqi_rolling_3, "aqi_rolling_7": rec.aqi_rolling_7,
        }
        try:
            aqi_val, category = predict(features)
        except Exception as e:
            aqi_val, category = -1.0, "Error"
            logger.error("Batch prediction error for %s: %s", rec.station_id, e)

        results.append(PredictionOutput(
            station_id=rec.station_id,
            predicted_aqi=aqi_val,
            predicted_category=category,
            model_name=reg_name,
            model_version=MODEL_VERSION,
            prediction_timestamp=now.isoformat(),
            input_features=features,
        ))

    latency = (time.time() - t0) * 1000
    _log_request("/batch-predict", latency, True, f"n={len(results)}")
    logger.info("Batch prediction: %d records in %.1fms", len(results), latency)

    return BatchPredictionOutput(
        predictions=results,
        total=len(results),
        model_name=reg_name,
    )


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "AQI Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
