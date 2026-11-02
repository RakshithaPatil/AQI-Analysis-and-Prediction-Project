"""
ml/predict.py
==============
Standalone prediction script.

Uses the trained best_regressor.joblib and best_classifier.joblib
to predict AQI and AQI category for new input data.

Can be used:
  1. As a CLI tool:     python ml/predict.py
  2. As a module:       from ml.predict import predict_aqi, predict_batch_from_csv

This module is intentionally kept simple so it can be explained during a review.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.model_loader import load_models, predict, get_model_info, get_feature_cols

# ── Ensure models are loaded ──────────────────────────────────────────────
_loaded = load_models()
if not _loaded:
    raise RuntimeError(
        "Models not found. Run the training pipeline first:\n"
        "  python ml/launch_training.py"
    )


def predict_aqi(
    pm25: float, pm10: float, no2: float, so2: float,
    co: float, o3: float, nh3: float = 5.0,
    temperature: float = 28.0, humidity: float = 65.0,
    precipitation: float = 0.0, wind_speed: float = 10.0,
    pressure: float = 1005.0,
    station_id: str = "unknown",
    **kwargs,
) -> dict:
    """
    Predict AQI and AQI category for a single observation.

    Parameters
    ----------
    pm25, pm10, no2, so2, co, o3, nh3 : Pollutant concentrations (µg/m³)
    temperature  : °C
    humidity     : %
    precipitation: mm
    wind_speed   : km/h
    pressure     : hPa
    station_id   : Station identifier (for logging)
    **kwargs     : Optional lag/rolling/temporal features

    Returns
    -------
    dict with predicted_aqi, predicted_category, model_name,
         model_version, prediction_timestamp
    """
    now = datetime.now(timezone.utc)

    features = {
        "pm25": pm25, "pm10": pm10, "no2": no2, "so2": so2,
        "co": co, "o3": o3, "nh3": nh3,
        "temperature": temperature, "humidity": humidity,
        "precipitation": precipitation, "wind_speed": wind_speed,
        "pressure": pressure,
        # Temporal defaults
        "year":        kwargs.get("year",        now.year),
        "month":       kwargs.get("month",       now.month),
        "day_of_week": kwargs.get("day_of_week", now.weekday()),
        "is_weekend":  kwargs.get("is_weekend",  int(now.weekday() >= 5)),
        "season":      kwargs.get("season",
                        {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2}.get(now.month, 3)),
        # Lag / rolling (filled by imputer if None)
        "aqi_lag_1":     kwargs.get("aqi_lag_1"),
        "aqi_lag_2":     kwargs.get("aqi_lag_2"),
        "aqi_lag_3":     kwargs.get("aqi_lag_3"),
        "aqi_lag_7":     kwargs.get("aqi_lag_7"),
        "aqi_rolling_3": kwargs.get("aqi_rolling_3"),
        "aqi_rolling_7": kwargs.get("aqi_rolling_7"),
    }

    aqi_val, category = predict(features)
    info = get_model_info()

    return {
        "station_id":           station_id,
        "predicted_aqi":        aqi_val,
        "predicted_category":   category,
        "model_name":           info.get("best_regressor", {}).get("name", "unknown"),
        "model_version":        "v1",
        "prediction_timestamp": now.isoformat(),
        "input_features":       features,
    }


def predict_batch_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Run predictions on every row of a CSV file.

    The CSV must contain at minimum:
      pm25, pm10, no2, so2, co, o3, temperature, humidity, wind_speed

    Returns
    -------
    DataFrame with added columns: predicted_aqi, predicted_category
    """
    df = pd.read_csv(csv_path)
    predictions = []
    for _, row in df.iterrows():
        result = predict_aqi(**{k: row[k] for k in get_feature_cols() if k in row})
        predictions.append({
            "predicted_aqi":      result["predicted_aqi"],
            "predicted_category": result["predicted_category"],
        })
    pred_df = pd.DataFrame(predictions)
    return pd.concat([df.reset_index(drop=True), pred_df], axis=1)


# ── CLI demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AQI PREDICTION DEMO")
    print("="*60)

    info = get_model_info()
    print(f"  Regressor  : {info['best_regressor']['name']}")
    print(f"  Classifier : {info['best_classifier']['name']}")
    print(f"  Features   : {len(get_feature_cols())}")

    # Example 1 — Delhi-like conditions (high pollution)
    print("\n  Example 1 — Delhi (high pollution):")
    r1 = predict_aqi(
        pm25=145.0, pm10=220.0, no2=85.0, so2=25.0, co=1500.0, o3=35.0,
        temperature=22.0, humidity=75.0, wind_speed=5.0,
        station_id="delhi_av",
    )
    print(f"    Predicted AQI      : {r1['predicted_aqi']:.1f}")
    print(f"    Predicted Category : {r1['predicted_category']}")

    # Example 2 — Bengaluru-like conditions (moderate)
    print("\n  Example 2 — Bengaluru (moderate pollution):")
    r2 = predict_aqi(
        pm25=45.0, pm10=72.0, no2=28.0, so2=8.0, co=600.0, o3=50.0,
        temperature=26.0, humidity=60.0, wind_speed=12.0,
        station_id="bengaluru_slk",
    )
    print(f"    Predicted AQI      : {r2['predicted_aqi']:.1f}")
    print(f"    Predicted Category : {r2['predicted_category']}")

    # Example 3 — Good air quality
    print("\n  Example 3 — Clean air conditions:")
    r3 = predict_aqi(
        pm25=10.0, pm10=18.0, no2=12.0, so2=3.0, co=200.0, o3=40.0,
        temperature=24.0, humidity=55.0, wind_speed=15.0,
        station_id="coastal_clean",
    )
    print(f"    Predicted AQI      : {r3['predicted_aqi']:.1f}")
    print(f"    Predicted Category : {r3['predicted_category']}")

    print("\n" + "="*60)
    print("  Model info saved to: models/model_info.json")
    print("  API endpoint: POST http://localhost:8000/predict")
    print("="*60)
