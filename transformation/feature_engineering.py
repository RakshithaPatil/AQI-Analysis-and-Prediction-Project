"""
transformation/feature_engineering.py
=======================================
Creates ML-ready features from the cleaned analytical dataset.

Features created:
  Temporal:
    hour, day_of_week, month, is_weekend, season

  Lag features (per station, sorted by date – NO future leakage):
    AQI_lag_1, AQI_lag_2, AQI_lag_3, AQI_lag_7

  Rolling averages (per station, using only past data):
    AQI_rolling_3  (3-day rolling mean)
    AQI_rolling_7  (7-day rolling mean)

  Pollutant concentrations:
    pm25, pm10, no2, so2, co, o3

  Weather:
    temperature, humidity, precipitation, wind_speed, wind_direction, pressure

  AQI category encoding (for regression target check):
    aqi_category_encoded  (ordinal: Good=0 … Hazardous=5)

IMPORTANT – No data leakage:
  All lag and rolling features use shift(n) so that the feature for day T
  only uses data from day T-n and earlier. The DataFrame is always sorted
  by (station_id, date) before feature creation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__, log_file="features.log")

# ── Config ────────────────────────────────────────────────────────────────────
_ML_CFG = settings["ml"]
_LAG_DAYS: list[int] = _ML_CFG.get("lag_days", [1, 2, 3, 7])
_ROLLING_WINDOWS: list[int] = _ML_CFG.get("rolling_windows", [3, 7])

# Ordinal encoding for AQI categories
_CATEGORY_ENCODING: dict[str, int] = {
    "Good": 0,
    "Moderate": 1,
    "Unhealthy for Sensitive Groups": 2,
    "Unhealthy": 3,
    "Very Unhealthy": 4,
    "Hazardous": 5,
    "Beyond Index": 6,
    "Unknown": -1,
}


# ── Temporal features ─────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add time-based features from the date column.

    Added columns: hour (0 if date-only), day_of_week, month, year,
                   is_weekend, season (0=Winter,1=Spring,2=Summer,3=Autumn)
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df["hour"] = dates.dt.hour if hasattr(dates.dt, "hour") else 0
    df["day_of_week"] = dates.dt.dayofweek        # 0=Monday
    df["month"] = dates.dt.month
    df["year"] = dates.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Season: meteorological (Northern Hemisphere)
    def _month_to_season(m: int) -> int:
        if m in (12, 1, 2):
            return 0   # Winter
        elif m in (3, 4, 5):
            return 1   # Spring
        elif m in (6, 7, 8):
            return 2   # Summer
        else:
            return 3   # Autumn

    df["season"] = df["month"].map(_month_to_season)
    return df


# ── Lag features ──────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "aqi",
    lag_days: Optional[list[int]] = None,
    group_col: str = "station_id",
) -> pd.DataFrame:
    """
    Add AQI lag features per station, sorted chronologically.

    Each lag feature AQI_lag_N is the value of `target_col` N rows earlier
    within the same station group (after sorting by date).

    Parameters
    ----------
    df         : DataFrame sorted by (station_id, date)
    target_col : Column to create lags for (default: 'aqi')
    lag_days   : List of lag periods (default from config)
    group_col  : Group column for per-station lags

    Returns
    -------
    DataFrame with AQI_lag_1, AQI_lag_2, ... columns added
    """
    if lag_days is None:
        lag_days = _LAG_DAYS

    df = df.sort_values([group_col, "date"]).copy()

    for lag in lag_days:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        n_null = df[col_name].isna().sum()
        logger.debug("Lag feature %s: %d NaN (expected for first %d rows per station)", col_name, n_null, lag)

    return df


# ── Rolling average features ──────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "aqi",
    windows: Optional[list[int]] = None,
    group_col: str = "station_id",
) -> pd.DataFrame:
    """
    Add rolling-mean AQI features per station.

    Uses shift(1) inside the rolling window so the feature for day T
    uses only data from T-window to T-1 (no leakage from current day).

    Parameters
    ----------
    df         : DataFrame sorted by (station_id, date)
    target_col : Column to create rolling averages for
    windows    : List of window sizes in days (default from config)
    group_col  : Group column for per-station rolling

    Returns
    -------
    DataFrame with AQI_rolling_3, AQI_rolling_7, ... columns added
    """
    if windows is None:
        windows = _ROLLING_WINDOWS

    df = df.sort_values([group_col, "date"]).copy()

    for window in windows:
        col_name = f"{target_col}_rolling_{window}"
        df[col_name] = (
            df.groupby(group_col)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
        df[col_name] = df[col_name].round(4)

    return df


# ── Category encoding ─────────────────────────────────────────────────────────

def encode_aqi_category(df: pd.DataFrame, cat_col: str = "aqi_category") -> pd.DataFrame:
    """
    Add an ordinal-encoded version of the AQI category column.

    Added column: aqi_category_encoded  (int)
    """
    df = df.copy()
    if cat_col not in df.columns:
        logger.warning("encode_aqi_category: column '%s' not found", cat_col)
        return df
    df["aqi_category_encoded"] = df[cat_col].map(_CATEGORY_ENCODING).fillna(-1).astype(int)
    return df


# ── Drop rows without enough history ─────────────────────────────────────────

def drop_insufficient_history(
    df: pd.DataFrame,
    min_lag: int = 7,
    target_col: str = "aqi",
    group_col: str = "station_id",
) -> pd.DataFrame:
    """
    Drop rows at the start of each station's time series that don't have
    enough historical data for the largest lag feature.

    Rows where AQI_lag_<min_lag> is NaN are removed.
    """
    lag_col = f"{target_col}_lag_{min_lag}"
    if lag_col not in df.columns:
        logger.warning(
            "drop_insufficient_history: lag column %s not found – skipping drop", lag_col
        )
        return df
    before = len(df)
    df = df.dropna(subset=[lag_col]).copy()
    logger.info(
        "drop_insufficient_history: dropped %d rows without %d-day history",
        before - len(df), min_lag,
    )
    return df.reset_index(drop=True)


# ── Master feature builder ─────────────────────────────────────────────────────

def build_feature_matrix(
    daily_aqi_df: pd.DataFrame,
    drop_no_history: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature matrix from the daily AQI table.

    Expected input columns:
        date, station_id,
        pm25, pm10, no2, so2, co, o3,
        aqi, aqi_category,
        temperature, humidity, precipitation,
        wind_speed, wind_direction, pressure   (optional but preferred)

    Output:
        All of the above + temporal + lag + rolling + encoded category

    Parameters
    ----------
    daily_aqi_df    : Daily wide-format AQI DataFrame
    drop_no_history : Remove rows without full lag history (recommended for ML)

    Returns
    -------
    Feature-enriched DataFrame ready for train/val/test splitting
    """
    logger.info("build_feature_matrix: input rows=%d", len(daily_aqi_df))

    df = daily_aqi_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    # Temporal features
    df = add_temporal_features(df, date_col="date")

    # Lag features
    df = add_lag_features(df, target_col="aqi", lag_days=_LAG_DAYS)

    # Rolling features
    df = add_rolling_features(df, target_col="aqi", windows=_ROLLING_WINDOWS)

    # Encode category
    df = encode_aqi_category(df)

    if drop_no_history:
        max_lag = max(_LAG_DAYS)
        df = drop_insufficient_history(df, min_lag=max_lag)

    logger.info("build_feature_matrix: output rows=%d, columns=%d", len(df), len(df.columns))
    return df


def get_feature_columns() -> list[str]:
    """
    Return the list of feature column names used for ML training.
    This is the canonical feature set – must match what the model was trained on.
    """
    lag_cols = [f"aqi_lag_{n}" for n in _LAG_DAYS]
    rolling_cols = [f"aqi_rolling_{n}" for n in _ROLLING_WINDOWS]
    return [
        # Pollutant concentrations
        "pm25", "pm10", "no2", "so2", "co", "o3",
        # Weather
        "temperature", "humidity", "precipitation",
        "wind_speed", "wind_direction", "pressure",
        # Temporal
        "hour", "day_of_week", "month", "is_weekend", "season",
        # Lag AQI
        *lag_cols,
        # Rolling AQI
        *rolling_cols,
    ]


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train / validation / test sets
    WITHOUT shuffling, to prevent temporal data leakage.

    Parameters
    ----------
    df          : Feature DataFrame sorted by date
    train_ratio : Fraction of data for training (default 0.70)
    val_ratio   : Fraction of data for validation (default 0.15)

    Returns
    -------
    (train_df, val_df, test_df)
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(
        "chronological_split: train=%d | val=%d | test=%d (total=%d)",
        len(train), len(val), len(test), n,
    )
    return train, val, test
