"""
monitoring/data_quality_monitor.py
====================================
Checks the processed dataset for data-quality issues.
Runs after every pipeline execution.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "data_quality.log"
LOG_FILE.parent.mkdir(exist_ok=True)

POLLUTANTS = ["pm25","pm10","no2","so2","co","o3","nh3"]
RANGES = {
    "pm25":(0,1000),"pm10":(0,1500),"no2":(0,2000),
    "so2":(0,2620),"co":(0,50000),"o3":(0,600),"nh3":(0,400),
    "aqi":(0,500),"temperature":(-60,60),"humidity":(0,100),
}


def _log(entry: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def check_missing_values(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    critical = missing_pct[missing_pct > 10].to_dict()
    result = {
        "check": "missing_values",
        "passed": len(critical) == 0,
        "missing_counts": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "critical_columns": critical,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _log(result)
    return result


def check_duplicates(df: pd.DataFrame) -> dict:
    dup_count = df.duplicated(subset=["station_id","date"], keep=False).sum()
    result = {
        "check": "duplicates",
        "passed": dup_count == 0,
        "duplicate_rows": int(dup_count),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _log(result)
    return result


def check_value_ranges(df: pd.DataFrame) -> dict:
    violations = {}
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns: continue
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        if bad > 0:
            violations[col] = int(bad)
    result = {
        "check": "value_ranges",
        "passed": len(violations) == 0,
        "violations": violations,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _log(result)
    return result


def check_missing_timestamps(df: pd.DataFrame) -> dict:
    null_ts = df["date"].isna().sum() if "date" in df.columns else 0
    result = {
        "check": "missing_timestamps",
        "passed": null_ts == 0,
        "null_timestamps": int(null_ts),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _log(result)
    return result


def check_aqi_category_consistency(df: pd.DataFrame) -> dict:
    """Verify AQI numeric value is consistent with AQI category label."""
    if "aqi" not in df.columns or "aqi_category" not in df.columns:
        return {"check": "aqi_consistency", "passed": True, "note": "columns missing"}

    def expected_cat(v):
        if v <= 50:  return "Good"
        if v <= 100: return "Moderate"
        if v <= 150: return "Unhealthy for Sensitive Groups"
        if v <= 200: return "Unhealthy"
        if v <= 300: return "Very Unhealthy"
        return "Hazardous"

    tmp = df.dropna(subset=["aqi","aqi_category"]).copy()
    tmp["expected"] = tmp["aqi"].apply(expected_cat)
    mismatches = (tmp["aqi_category"] != tmp["expected"]).sum()
    result = {
        "check": "aqi_category_consistency",
        "passed": int(mismatches) == 0,
        "mismatches": int(mismatches),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _log(result)
    return result


def run_all_checks(df: pd.DataFrame) -> list[dict]:
    """Run all data quality checks and return list of results."""
    print("\n  🔍 Running data quality checks …")
    results = [
        check_missing_values(df),
        check_duplicates(df),
        check_value_ranges(df),
        check_missing_timestamps(df),
        check_aqi_category_consistency(df),
    ]
    passed = sum(1 for r in results if r.get("passed"))
    print(f"  Data quality: {passed}/{len(results)} checks passed")
    for r in results:
        icon = "✅" if r.get("passed") else "⚠️"
        print(f"    {icon}  {r['check']}")
    return results
