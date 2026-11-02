"""
monitoring/drift_monitor.py
=============================
Detects feature drift using Population Stability Index (PSI).

PSI < 0.1  → No drift (stable)
PSI 0.1–0.2 → Moderate drift (monitor)
PSI > 0.2  → Significant drift (alert — consider retraining)

Reference: https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "drift.log"
LOG_FILE.parent.mkdir(exist_ok=True)

PSI_ALERT_THRESHOLD = 0.20
FEATURES_TO_MONITOR = [
    "pm25","pm10","no2","so2","co","o3",
    "temperature","humidity","wind_speed","pressure",
]


def _psi_single(expected: np.ndarray, actual: np.ndarray,
                buckets: int = 10) -> float:
    """Calculate PSI for one feature."""
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    def bucket_pcts(data):
        counts, _ = np.histogram(data, bins=breakpoints)
        pcts = counts / len(data)
        pcts = np.where(pcts == 0, 1e-4, pcts)  # avoid log(0)
        return pcts

    e_pct = bucket_pcts(expected)
    a_pct = bucket_pcts(actual)
    # Align lengths (in case histogram sizes differ)
    min_len = min(len(e_pct), len(a_pct))
    e_pct, a_pct = e_pct[:min_len], a_pct[:min_len]

    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return round(float(psi), 4)


def calculate_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    features: list[str] = None,
) -> dict:
    """
    Compare production data distribution against reference (training) data.

    Parameters
    ----------
    reference_df  : Training/reference dataset
    production_df : Recent production/incoming data
    features      : Feature columns to check (default: FEATURES_TO_MONITOR)

    Returns
    -------
    dict with PSI scores and drift status per feature
    """
    if features is None:
        features = FEATURES_TO_MONITOR

    psi_scores = {}
    drifted    = []

    for feat in features:
        if feat not in reference_df.columns or feat not in production_df.columns:
            continue
        psi = _psi_single(
            reference_df[feat].values,
            production_df[feat].values,
        )
        psi_scores[feat] = psi
        if psi > PSI_ALERT_THRESHOLD:
            drifted.append(feat)

    result = {
        "check": "feature_drift",
        "ts": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(reference_df),
        "production_rows": len(production_df),
        "psi_scores": psi_scores,
        "drifted_features": drifted,
        "drift_detected": len(drifted) > 0,
        "alert": len(drifted) > 0,
        "threshold": PSI_ALERT_THRESHOLD,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


def print_drift_report(result: dict):
    print("\n  📊 Feature Drift Report (PSI)")
    print(f"  Reference: {result['reference_rows']:,} rows | "
          f"Production: {result['production_rows']:,} rows")
    print(f"  {'Feature':<20} {'PSI':>8}  Status")
    print("  " + "-"*40)
    for feat, psi in result["psi_scores"].items():
        if psi < 0.1:
            status = "✅ Stable"
        elif psi < 0.2:
            status = "⚠️  Monitor"
        else:
            status = "🚨 DRIFT!"
        print(f"  {feat:<20} {psi:>8.4f}  {status}")
    if result["drift_detected"]:
        print(f"\n  🚨 ALERT: Drift detected in: {result['drifted_features']}")
        print("  Recommendation: consider retraining the model.")
    else:
        print("\n  ✅ No significant drift detected.")
