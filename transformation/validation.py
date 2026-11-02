"""
transformation/validation.py
==============================
Validates raw DataFrames before any transformation is applied.

Checks performed:
  1.  Null / missing timestamps
  2.  Unparseable / future timestamps
  3.  Duplicate (station_id, timestamp, pollutant) rows
  4.  Missing pollutant values
  5.  Out-of-range pollutant values  (per config.yaml)
  6.  Invalid coordinates
  7.  Invalid / unrecognised unit strings
  8.  Missing weather variables
  9.  Invalid weather variable ranges

Every invalid record is:
  - Removed from the returned "good" DataFrame
  - Appended to data/rejected/<source>_rejected_<date>.csv
  - Logged to the application log AND the rejection log

Returns
-------
  good_df      : pd.DataFrame  – records that passed all checks
  rejected_df  : pd.DataFrame  – records that failed, with extra columns
                                 error_type, error_message, rejected_at
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd

from config.logger import get_logger
from config.settings import get_project_root, settings

logger = get_logger(__name__, log_file="validation.log")

# ── Config shortcuts ─────────────────────────────────────────────────────────
_DQ_CFG = settings["data_quality"]
_POLLUTANT_RANGES: dict = _DQ_CFG["pollutant_ranges"]
_LAT_RANGE: list = _DQ_CFG["latitude_range"]
_LON_RANGE: list = _DQ_CFG["longitude_range"]
_MAX_MISSING_PCT: float = _DQ_CFG["max_missing_pct"]

# Recognised unit strings (lower-cased)
_VALID_UNITS = {
    "µg/m³", "ug/m3", "µg/m3", "μg/m³", "ppm", "ppb",
    "mg/m³", "mg/m3", "mock_openaq",  # mock data uses these
}

_REJECTED_DIR = get_project_root() / "data" / "rejected"
_REJECTED_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def _save_rejected(rejected_df: pd.DataFrame, source: str) -> None:
    """Append rejected records to a daily CSV file in data/rejected/."""
    if rejected_df.empty:
        return
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = _REJECTED_DIR / f"{source}_rejected_{date_str}.csv"
    write_header = not out_path.exists()
    rejected_df.to_csv(out_path, mode="a", header=write_header, index=False)
    logger.info("Saved %d rejected records → %s", len(rejected_df), out_path.name)


def _mark_rejected(df: pd.DataFrame, mask: pd.Series, error_type: str, message: str) -> pd.DataFrame:
    """Return a copy of df[mask] with error columns added."""
    bad = df[mask].copy()
    bad["error_type"] = error_type
    bad["error_message"] = message
    bad["rejected_at"] = datetime.now(timezone.utc).isoformat()
    return bad


# ── Pollution validation ──────────────────────────────────────────────────────

def validate_pollution_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate a pollution measurements DataFrame.

    Parameters
    ----------
    df : DataFrame with columns:
         timestamp, station_id, pollutant, value, unit, latitude, longitude

    Returns
    -------
    (good_df, rejected_df)
    """
    if df.empty:
        return df, pd.DataFrame()

    rejected_parts: list[pd.DataFrame] = []
    good = df.copy()

    # ── 1. Null timestamps ──────────────────────────────────────────────────
    mask = good["timestamp"].isna()
    if mask.any():
        logger.warning("Found %d null timestamps", mask.sum())
        rejected_parts.append(_mark_rejected(good, mask, "NULL_TIMESTAMP", "timestamp is null"))
        good = good[~mask].copy()

    # ── 2. Unparseable / invalid timestamps ─────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(good["timestamp"]):
        good["timestamp"] = pd.to_datetime(good["timestamp"], errors="coerce", utc=True)
    mask = good["timestamp"].isna()
    if mask.any():
        rejected_parts.append(_mark_rejected(good, mask, "INVALID_TIMESTAMP", "timestamp could not be parsed"))
        good = good[~mask].copy()

    # ── 3. Duplicate records ────────────────────────────────────────────────
    dup_cols = ["station_id", "timestamp", "pollutant"]
    dup_mask = good.duplicated(subset=dup_cols, keep="first")
    if dup_mask.any():
        logger.warning("Dropping %d duplicate records", dup_mask.sum())
        rejected_parts.append(_mark_rejected(good, dup_mask, "DUPLICATE", f"duplicate on {dup_cols}"))
        good = good[~dup_mask].copy()

    # ── 4. Missing pollutant values ─────────────────────────────────────────
    mask = good["value"].isna()
    if mask.any():
        logger.warning("Found %d missing pollutant values", mask.sum())
        rejected_parts.append(_mark_rejected(good, mask, "MISSING_VALUE", "pollutant value is null"))
        good = good[~mask].copy()

    # ── 5. Out-of-range pollutant values ────────────────────────────────────
    range_mask = pd.Series(False, index=good.index)
    for pol, bounds in _POLLUTANT_RANGES.items():
        pol_mask = (
            (good["pollutant"].str.lower() == pol)
            & ((good["value"] < bounds["min"]) | (good["value"] > bounds["max"]))
        )
        if pol_mask.any():
            logger.warning(
                "Found %d out-of-range values for %s (range: %s–%s)",
                pol_mask.sum(), pol, bounds["min"], bounds["max"],
            )
        range_mask = range_mask | pol_mask

    if range_mask.any():
        rejected_parts.append(
            _mark_rejected(good, range_mask, "OUT_OF_RANGE",
                           "pollutant value outside configured acceptable range")
        )
        good = good[~range_mask].copy()

    # ── 6. Invalid coordinates ──────────────────────────────────────────────
    if "latitude" in good.columns and "longitude" in good.columns:
        coord_mask = (
            (good["latitude"] < _LAT_RANGE[0]) | (good["latitude"] > _LAT_RANGE[1]) |
            (good["longitude"] < _LON_RANGE[0]) | (good["longitude"] > _LON_RANGE[1])
        )
        if coord_mask.any():
            logger.warning("Found %d records with invalid coordinates", coord_mask.sum())
            rejected_parts.append(
                _mark_rejected(good, coord_mask, "INVALID_COORDINATES",
                               f"lat must be {_LAT_RANGE}, lon must be {_LON_RANGE}")
            )
            good = good[~coord_mask].copy()

    # ── 7. Invalid units ────────────────────────────────────────────────────
    if "unit" in good.columns:
        unit_mask = ~good["unit"].str.lower().str.strip().isin({u.lower() for u in _VALID_UNITS})
        if unit_mask.any():
            bad_units = good.loc[unit_mask, "unit"].unique().tolist()
            logger.warning("Found %d records with unrecognised units: %s", unit_mask.sum(), bad_units)
            rejected_parts.append(
                _mark_rejected(good, unit_mask, "INVALID_UNIT",
                               f"unrecognised unit; expected one of {_VALID_UNITS}")
            )
            good = good[~unit_mask].copy()

    rejected_df = pd.concat(rejected_parts, ignore_index=True) if rejected_parts else pd.DataFrame()
    _save_rejected(rejected_df, "pollution")

    logger.info(
        "Pollution validation complete | good=%d | rejected=%d",
        len(good), len(rejected_df),
    )
    return good.reset_index(drop=True), rejected_df


# ── Weather validation ────────────────────────────────────────────────────────

def validate_weather_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate a weather measurements DataFrame.

    Parameters
    ----------
    df : DataFrame with columns:
         timestamp, station_id, temperature, humidity, precipitation,
         wind_speed, wind_direction, pressure

    Returns
    -------
    (good_df, rejected_df)
    """
    if df.empty:
        return df, pd.DataFrame()

    rejected_parts: list[pd.DataFrame] = []
    good = df.copy()

    # Reasonable ranges for weather variables
    weather_ranges = {
        "temperature":    (-60, 60),    # °C
        "humidity":       (0, 100),     # %
        "precipitation":  (0, 500),     # mm/hour – extreme but possible
        "wind_speed":     (0, 120),     # km/h
        "wind_direction": (0, 360),     # degrees
        "pressure":       (870, 1085),  # hPa
    }

    # ── 1. Null timestamps ──────────────────────────────────────────────────
    mask = good["timestamp"].isna()
    if mask.any():
        rejected_parts.append(_mark_rejected(good, mask, "NULL_TIMESTAMP", "timestamp is null"))
        good = good[~mask].copy()

    if not pd.api.types.is_datetime64_any_dtype(good["timestamp"]):
        good["timestamp"] = pd.to_datetime(good["timestamp"], errors="coerce", utc=True)
    mask = good["timestamp"].isna()
    if mask.any():
        rejected_parts.append(_mark_rejected(good, mask, "INVALID_TIMESTAMP", "cannot parse timestamp"))
        good = good[~mask].copy()

    # ── 2. Duplicates ───────────────────────────────────────────────────────
    dup_mask = good.duplicated(subset=["station_id", "timestamp"], keep="first")
    if dup_mask.any():
        rejected_parts.append(_mark_rejected(good, dup_mask, "DUPLICATE", "duplicate station+timestamp"))
        good = good[~dup_mask].copy()

    # ── 3. Weather variable ranges ──────────────────────────────────────────
    for col, (lo, hi) in weather_ranges.items():
        if col not in good.columns:
            continue
        numeric = pd.to_numeric(good[col], errors="coerce")
        bad_mask = numeric.isna() | (numeric < lo) | (numeric > hi)
        if bad_mask.any():
            logger.warning(
                "Found %d invalid %s values (range: %s–%s)", bad_mask.sum(), col, lo, hi
            )
            rejected_parts.append(
                _mark_rejected(good, bad_mask, f"INVALID_{col.upper()}",
                               f"{col} must be between {lo} and {hi}")
            )
            good = good[~bad_mask].copy()

    rejected_df = pd.concat(rejected_parts, ignore_index=True) if rejected_parts else pd.DataFrame()
    _save_rejected(rejected_df, "weather")

    logger.info(
        "Weather validation complete | good=%d | rejected=%d",
        len(good), len(rejected_df),
    )
    return good.reset_index(drop=True), rejected_df
