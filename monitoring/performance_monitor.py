"""
monitoring/performance_monitor.py
====================================
Tracks model prediction performance over time and defines
the retraining strategy.

Retraining triggers:
  1. MAE > MAE_THRESHOLD (default 30.0)
  2. PSI drift detected in > 2 features
  3. More than MIN_NEW_RECORDS new records available
  4. Scheduled monthly retraining
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "performance.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# Retraining thresholds (configurable)
MAE_THRESHOLD      = 30.0
MIN_NEW_RECORDS    = 500
MAX_DRIFT_FEATURES = 2


def log_prediction(
    predicted_aqi: float,
    actual_aqi: float | None,
    station_id: str,
    model_name: str,
    model_version: str,
):
    """Append one prediction record to the performance log."""
    error = abs(predicted_aqi - actual_aqi) if actual_aqi is not None else None
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "station_id": station_id,
        "predicted_aqi": predicted_aqi,
        "actual_aqi": actual_aqi,
        "abs_error": round(error, 4) if error is not None else None,
        "model_name": model_name,
        "model_version": model_version,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_performance_log() -> pd.DataFrame:
    """Load all prediction log entries as a DataFrame."""
    if not LOG_FILE.exists():
        return pd.DataFrame()
    records = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def compute_performance_metrics() -> dict:
    """Compute MAE/RMSE from evaluated predictions."""
    df = read_performance_log()
    evaluated = df.dropna(subset=["abs_error"]) if not df.empty else pd.DataFrame()

    if evaluated.empty:
        return {
            "total_predictions": len(df),
            "evaluated_predictions": 0,
            "mae": None, "rmse": None,
            "alert": False,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    mae  = evaluated["abs_error"].mean()
    rmse = np.sqrt((evaluated["abs_error"]**2).mean())

    result = {
        "total_predictions": len(df),
        "evaluated_predictions": len(evaluated),
        "mae":  round(float(mae),  4),
        "rmse": round(float(rmse), 4),
        "alert": mae > MAE_THRESHOLD,
        "mae_threshold": MAE_THRESHOLD,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    return result


def check_retraining_needed(
    new_record_count: int = 0,
    drifted_feature_count: int = 0,
    current_mae: float | None = None,
    days_since_last_training: int = 0,
) -> dict:
    """
    Evaluate all retraining triggers and return a decision.

    Retraining Strategy:
      Trigger 1 — MAE exceeds threshold (model degrading on live data)
      Trigger 2 — Feature drift detected in multiple features (distribution shift)
      Trigger 3 — Enough new data has accumulated to improve the model
      Trigger 4 — Scheduled retraining (monthly = every 30 days)

    Returns dict with: should_retrain, triggers_fired, recommendation
    """
    triggers = []

    if current_mae is not None and current_mae > MAE_THRESHOLD:
        triggers.append(f"MAE={current_mae:.2f} exceeds threshold {MAE_THRESHOLD}")

    if drifted_feature_count >= MAX_DRIFT_FEATURES:
        triggers.append(f"{drifted_feature_count} features drifted "
                        f"(threshold: {MAX_DRIFT_FEATURES})")

    if new_record_count >= MIN_NEW_RECORDS:
        triggers.append(f"{new_record_count:,} new records available "
                        f"(threshold: {MIN_NEW_RECORDS:,})")

    if days_since_last_training >= 30:
        triggers.append(f"Scheduled monthly retrain "
                        f"({days_since_last_training} days since last training)")

    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "should_retrain": len(triggers) > 0,
        "triggers_fired": triggers,
        "inputs": {
            "current_mae": current_mae,
            "new_record_count": new_record_count,
            "drifted_feature_count": drifted_feature_count,
            "days_since_last_training": days_since_last_training,
        },
        "recommendation": (
            "Run: python ml/train_models.py"
            if triggers else
            "No retraining needed at this time."
        ),
    }
    return result


def print_performance_report():
    metrics = compute_performance_metrics()
    print("\n  📈 Model Performance Monitor")
    print(f"  Total predictions logged  : {metrics['total_predictions']:,}")
    print(f"  Evaluated (actual known)  : {metrics['evaluated_predictions']:,}")
    if metrics["mae"] is not None:
        alert = " 🚨 ALERT" if metrics["alert"] else " ✅"
        print(f"  MAE                       : {metrics['mae']:.4f}{alert}")
        print(f"  RMSE                      : {metrics['rmse']:.4f}")
    else:
        print("  No evaluated predictions yet (actual AQI not available)")
    return metrics
