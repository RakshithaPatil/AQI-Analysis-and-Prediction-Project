"""
ml/features.py
===============
Loads the daily AQI dataset (from PostgreSQL or mock) and prepares the
complete feature matrix for ML training.

This module is the single source of truth for:
  - Which columns are features vs targets
  - How the data is split (chronological, never shuffled)
  - How missing features are handled (median imputation)

Separation of concerns:
  - Feature *creation* logic lives in transformation/feature_engineering.py
  - Feature *loading + splitting* for ML lives here
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from config.logger import get_logger
from config.settings import get_project_root, settings
from transformation.feature_engineering import (
    build_feature_matrix,
    chronological_split,
    get_feature_columns,
)

logger = get_logger(__name__, log_file="ml.log")

_ML_CFG = settings["ml"]
_REGRESSION_TARGET = _ML_CFG["regression_target"]       # "aqi"
_CLASSIFICATION_TARGET = _ML_CFG["classification_target"]  # "aqi_category"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_daily_aqi_from_db() -> pd.DataFrame:
    """
    Load the daily_aqi table from PostgreSQL including weather columns.
    Raises on connection failure – callers should catch and fall back to mock.
    """
    from database.connection import get_engine
    from sqlalchemy import text

    query = """
        SELECT
            d.date, d.station_id,
            d.pm25, d.pm10, d.no2, d.so2, d.co, d.o3,
            d.aqi, d.aqi_category,
            d.avg_temperature   AS temperature,
            d.avg_humidity      AS humidity,
            d.total_precipitation AS precipitation,
            d.avg_wind_speed    AS wind_speed,
            d.avg_pressure      AS pressure
        FROM daily_aqi d
        ORDER BY d.station_id, d.date
    """
    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d rows from daily_aqi table", len(df))
    return df


def load_daily_aqi_mock() -> pd.DataFrame:
    """
    Generate a realistic mock daily AQI dataset for training when
    PostgreSQL is unavailable.

    ⚠️  This is synthetic data for testing the ML pipeline only.
    Real models should be trained on data from the live pipeline.
    """
    import numpy as np
    from transformation.aqi_calculation import calculate_overall_aqi

    logger.warning("⚠️  Using MOCK data for ML training. NOT real measurements.")
    rng = np.random.default_rng(42)

    stations = [
        ("delhi_anand_vihar", 120),
        ("delhi_ito",          110),
        ("mumbai_bandra",       60),
        ("bengaluru_silk_board", 45),
        ("chennai_alandur",     50),
    ]

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=365 * 2)   # 2 years of data
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rows = []
    for sid, base_pm25 in stations:
        for d in dates:
            seasonal = 1 + 0.5 * np.sin((d.dayofyear / 365) * 2 * np.pi - np.pi)
            pm25 = max(5,  float(rng.normal(base_pm25 * seasonal, 25)))
            pm10 = max(10, float(rng.normal(pm25 * 1.6, 20)))
            no2  = max(5,  float(rng.normal(40 * seasonal, 10)))
            so2  = max(1,  float(rng.normal(12, 4)))
            co   = max(200, float(rng.normal(800, 150)))
            o3   = max(10, float(rng.normal(45, 15)))
            temp = float(rng.normal(28, 6))
            hum  = float(rng.uniform(30, 90))
            prec = max(0.0, float(rng.exponential(0.5)))
            ws   = max(0.0, float(rng.normal(10, 4)))
            pres = float(rng.normal(1005, 8))
            aqi, cat, _ = calculate_overall_aqi(pm25, pm10, o3, co, so2, no2)
            rows.append({
                "date": d, "station_id": sid,
                "pm25": round(pm25, 2), "pm10": round(pm10, 2),
                "no2": round(no2, 2), "so2": round(so2, 2),
                "co": round(co, 2), "o3": round(o3, 2),
                "aqi": round(aqi, 2) if aqi else None,
                "aqi_category": cat or "Unknown",
                "temperature": round(temp, 1), "humidity": round(hum, 1),
                "precipitation": round(prec, 2), "wind_speed": round(ws, 1),
                "pressure": round(pres, 1),
            })

    df = pd.DataFrame(rows)
    logger.info("Mock data generated: %d rows", len(df))
    return df


def load_data(use_mock_on_failure: bool = True) -> pd.DataFrame:
    """Try DB first; fall back to mock if unavailable."""
    try:
        return load_daily_aqi_from_db()
    except Exception as exc:
        logger.warning("DB unavailable (%s) – using mock data", exc)
        if use_mock_on_failure:
            return load_daily_aqi_mock()
        raise


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Build the feature matrix, split chronologically, and impute missing values.

    Parameters
    ----------
    df : Raw daily AQI DataFrame (from load_data)

    Returns
    -------
    (train_df, val_df, test_df, label_encoder)
      - Each DataFrame has all feature columns + both target columns
      - label_encoder : fitted on training categories (for classification)
    """
    # Build features (lags, rolling, temporal)
    feature_df = build_feature_matrix(df, drop_no_history=True)
    feature_df = feature_df.dropna(subset=[_REGRESSION_TARGET])

    logger.info("Feature matrix: %d rows × %d cols", *feature_df.shape)

    # Chronological split (no shuffle – prevents temporal leakage)
    train_df, val_df, test_df = chronological_split(
        feature_df,
        train_ratio=_ML_CFG["train_ratio"],
        val_ratio=_ML_CFG["val_ratio"],
    )

    # Impute missing feature values with the TRAINING median only
    # (fit on train, transform all splits)
    feature_cols = get_feature_columns()
    available_features = [c for c in feature_cols if c in feature_df.columns]

    imputer = SimpleImputer(strategy="median")
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[available_features] = imputer.fit_transform(train_df[available_features])
    val_df[available_features]   = imputer.transform(val_df[available_features])
    test_df[available_features]  = imputer.transform(test_df[available_features])

    # Encode classification target
    le = LabelEncoder()
    train_df["aqi_category_label"] = le.fit_transform(train_df[_CLASSIFICATION_TARGET].fillna("Unknown"))
    val_df["aqi_category_label"]   = le.transform(
        val_df[_CLASSIFICATION_TARGET].fillna("Unknown").map(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
    )
    test_df["aqi_category_label"]  = le.transform(
        test_df[_CLASSIFICATION_TARGET].fillna("Unknown").map(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
    )

    logger.info(
        "Split sizes → train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df)
    )
    logger.info("AQI category classes: %s", list(le.classes_))

    return train_df, val_df, test_df, le


def get_X_y(
    df: pd.DataFrame,
    task: str = "regression",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features (X) and target (y) from a prepared DataFrame.

    Parameters
    ----------
    df   : One of (train_df, val_df, test_df) from prepare_features
    task : "regression" → target = aqi (float)
           "classification" → target = aqi_category_label (int)

    Returns
    -------
    (X, y)
    """
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()

    if task == "regression":
        y = df[_REGRESSION_TARGET].copy()
    elif task == "classification":
        y = df["aqi_category_label"].copy()
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'regression' or 'classification'.")

    return X, y
