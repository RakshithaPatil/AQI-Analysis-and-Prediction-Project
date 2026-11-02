"""
api/model_loader.py
====================
Loads trained models and metadata from the models/ directory.
Singleton pattern — models are loaded once at startup.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_regressor   = None
_classifier  = None
_imputer     = None
_label_enc   = None
_feature_cols: list[str] = []
_model_info: dict = {}
_load_time: Optional[datetime] = None


def load_models() -> bool:
    """Load all model artefacts. Returns True if successful."""
    global _regressor, _classifier, _imputer, _label_enc
    global _feature_cols, _model_info, _load_time

    info_path = MODELS_DIR / "model_info.json"
    if not info_path.exists():
        return False

    with open(info_path) as f:
        _model_info = json.load(f)

    _feature_cols = _model_info.get("feature_cols", [])

    reg_path = MODELS_DIR / "best_regressor.joblib"
    cls_path = MODELS_DIR / "best_classifier.joblib"
    imp_path = MODELS_DIR / "imputer.joblib"
    le_path  = MODELS_DIR / "label_encoder.joblib"

    if reg_path.exists():
        _regressor = joblib.load(reg_path)
    if cls_path.exists():
        _classifier = joblib.load(cls_path)
    if imp_path.exists():
        _imputer = joblib.load(imp_path)
    if le_path.exists():
        _label_enc = joblib.load(le_path)

    _load_time = datetime.now(timezone.utc)
    return _regressor is not None and _classifier is not None


def is_loaded() -> bool:
    return _regressor is not None and _classifier is not None


def get_model_info() -> dict:
    return _model_info


def get_feature_cols() -> list[str]:
    return _feature_cols


def predict(features: dict) -> tuple[float, str]:
    """
    Run both regressor and classifier on a feature dict.

    Parameters
    ----------
    features : dict mapping feature_name → value

    Returns
    -------
    (predicted_aqi, predicted_category)
    """
    if not is_loaded():
        raise RuntimeError("Models not loaded. Call load_models() first.")

    # Build feature vector in correct order
    row = np.array([[features.get(c, np.nan) for c in _feature_cols]])

    # Impute missing values using training median
    if _imputer is not None:
        row = _imputer.transform(row)

    # Regression → numerical AQI
    aqi_pred = float(_regressor.predict(row)[0])
    aqi_pred = round(max(0.0, min(500.0, aqi_pred)), 2)

    # Classification → category
    cat_encoded = _classifier.predict(row)[0]
    if _label_enc is not None:
        category = _label_enc.inverse_transform([cat_encoded])[0]
    else:
        category = str(cat_encoded)

    return aqi_pred, category


def get_uptime_s() -> float:
    if _load_time is None:
        return 0.0
    return (datetime.now(timezone.utc) - _load_time).total_seconds()
