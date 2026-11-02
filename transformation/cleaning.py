"""
transformation/cleaning.py
============================
Cleans and normalises validated DataFrames before loading into PostgreSQL.

Operations performed:
  1.  Parse and normalise timestamps to UTC
  2.  Standardise pollutant names  (pm2.5 → pm25, etc.)
  3.  Standardise unit strings     (ug/m3 → µg/m³)
  4.  Normalise station/location names  (strip, title-case)
  5.  Convert pollutant values to float
  6.  Drop duplicate observations  (final pass after validation)
  7.  Detect and flag missing time periods (gap detection)
  8.  Calculate hourly and daily averages
  9.  Join pollution and weather DataFrames on station + timestamp (hourly)
"""

from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config.logger import get_logger

logger = get_logger(__name__, log_file="cleaning.log")


# ── Lookup tables ─────────────────────────────────────────────────────────────

# Map various raw pollutant name strings → canonical name
_POLLUTANT_ALIASES: dict[str, str] = {
    "pm2.5": "pm25",
    "pm 2.5": "pm25",
    "pm-2.5": "pm25",
    "pm25": "pm25",
    "pm10": "pm10",
    "pm 10": "pm10",
    "no2": "no2",
    "nitrogen dioxide": "no2",
    "so2": "so2",
    "sulphur dioxide": "so2",
    "sulfur dioxide": "so2",
    "co": "co",
    "carbon monoxide": "co",
    "o3": "o3",
    "ozone": "o3",
}

# Map raw unit strings → canonical unit
_UNIT_ALIASES: dict[str, str] = {
    "ug/m3": "µg/m³",
    "µg/m3": "µg/m³",
    "μg/m³": "µg/m³",
    "µg/m³": "µg/m³",
    "ug/m³": "µg/m³",
    "ppm": "ppm",
    "ppb": "ppb",
    "mg/m3": "mg/m³",
    "mg/m³": "mg/m³",
    "mock_openaq": "µg/m³",   # normalise mock unit label
}


# ── 1. Timestamp normalisation ────────────────────────────────────────────────

def normalise_timestamps(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """
    Parse the timestamp column to timezone-aware UTC datetime64.

    Parameters
    ----------
    df  : Input DataFrame
    col : Name of the timestamp column (default "timestamp")

    Returns
    -------
    DataFrame with col converted to datetime64[ns, UTC]
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    null_count = df[col].isna().sum()
    if null_count:
        logger.warning("normalise_timestamps: %d rows have unparseable timestamps – dropped", null_count)
        df = df.dropna(subset=[col])
    return df


# ── 2. Pollutant name standardisation ────────────────────────────────────────

def normalise_pollutant_names(df: pd.DataFrame, col: str = "pollutant") -> pd.DataFrame:
    """
    Map raw pollutant name strings to canonical lower-case names.

    Unknown names are kept as-is and logged.
    """
    df = df.copy()
    original = df[col].copy()
    df[col] = (
        df[col]
        .str.strip()
        .str.lower()
        .map(lambda x: _POLLUTANT_ALIASES.get(x, x))
    )
    # Log anything not in the lookup table
    unknown = df.loc[~df[col].isin(_POLLUTANT_ALIASES.values()), col].unique()
    if len(unknown):
        logger.warning("Unknown pollutant names kept as-is: %s", unknown.tolist())
    return df


# ── 3. Unit standardisation ───────────────────────────────────────────────────

def normalise_units(df: pd.DataFrame, col: str = "unit") -> pd.DataFrame:
    """Standardise unit strings to canonical form (µg/m³, ppm, etc.)."""
    df = df.copy()
    if col not in df.columns:
        return df
    df[col] = (
        df[col]
        .str.strip()
        .str.lower()
        .map(lambda x: _UNIT_ALIASES.get(x, x))
    )
    return df


# ── 4. Station name normalisation ─────────────────────────────────────────────

def normalise_station_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and apply title-case to station_name and city columns."""
    df = df.copy()
    for col in ["station_name", "city"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].str.strip().str.lower()
    return df


# ── 5. Value type coercion ────────────────────────────────────────────────────

def coerce_numeric_values(df: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """Convert value column to float, coercing errors to NaN."""
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    n_bad = df[col].isna().sum()
    if n_bad:
        logger.warning("coerce_numeric_values: %d non-numeric values set to NaN", n_bad)
    return df


# ── 6. Deduplication ──────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
    """
    Remove duplicate rows. For pollution data the natural key is
    (station_id, timestamp, pollutant); for weather it is (station_id, timestamp).
    """
    if subset is None:
        subset = ["station_id", "timestamp"]
        if "pollutant" in df.columns:
            subset.append("pollutant")

    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").copy()
    removed = before - len(df)
    if removed:
        logger.info("deduplicate: removed %d duplicate rows", removed)
    return df.reset_index(drop=True)


# ── 7. Gap detection ──────────────────────────────────────────────────────────

def detect_missing_periods(
    df: pd.DataFrame,
    freq: str = "h",
    timestamp_col: str = "timestamp",
    group_col: str = "station_id",
) -> pd.DataFrame:
    """
    Detect missing time periods per station.

    Parameters
    ----------
    df            : Cleaned DataFrame with a timestamp column
    freq          : Expected frequency ('h' for hourly, 'D' for daily)
    timestamp_col : Name of the timestamp column
    group_col     : Column to group by (usually station_id)

    Returns
    -------
    DataFrame with columns: station_id, missing_from, missing_to, gap_hours
    Logged but not used to drop data – callers decide what to do.
    """
    gaps = []
    for station, grp in df.groupby(group_col):
        times = grp[timestamp_col].sort_values().drop_duplicates()
        full_range = pd.date_range(start=times.min(), end=times.max(), freq=freq, tz="UTC")
        missing = full_range.difference(times)
        if len(missing) == 0:
            continue
        # Group consecutive missing timestamps into contiguous gaps
        diff = missing.to_series().diff()
        break_points = diff[diff > pd.Timedelta(freq)].index
        starts = [missing[0]] + list(break_points)
        ends = list(break_points - pd.Timedelta(freq)) + [missing[-1]]
        for s, e in zip(starts, ends):
            gap_hours = int((e - s).total_seconds() / 3600) + 1
            gaps.append({
                "station_id": station,
                "missing_from": s,
                "missing_to": e,
                "gap_hours": gap_hours,
            })

    if gaps:
        gap_df = pd.DataFrame(gaps)
        logger.warning(
            "detect_missing_periods: found %d gaps across %d stations",
            len(gap_df), gap_df["station_id"].nunique(),
        )
        return gap_df
    return pd.DataFrame(columns=["station_id", "missing_from", "missing_to", "gap_hours"])


# ── 8. Hourly and daily aggregation ──────────────────────────────────────────

def calculate_hourly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pollution measurements to hourly averages per station+pollutant.

    Input columns : timestamp (UTC), station_id, pollutant, value
    Output columns: timestamp (floored to hour), station_id, pollutant, value (mean)
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.floor("h")
    hourly = (
        df.groupby(["timestamp", "station_id", "pollutant"], as_index=False)["value"]
        .mean()
    )
    hourly["value"] = hourly["value"].round(4)
    logger.info("Hourly aggregation → %d rows", len(hourly))
    return hourly


def calculate_daily_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pollution measurements to daily averages per station+pollutant.

    Input  : hourly-level DataFrame (timestamp, station_id, pollutant, value)
    Output : date, station_id, pollutant, value (daily mean)
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    daily = (
        df.groupby(["date", "station_id", "pollutant"], as_index=False)["value"]
        .mean()
    )
    daily["value"] = daily["value"].round(4)
    logger.info("Daily aggregation → %d rows", len(daily))
    return daily


# ── 9. Join pollution + weather ───────────────────────────────────────────────

def join_pollution_weather(
    pollution_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    how: str = "left",
) -> pd.DataFrame:
    """
    Join hourly pollution data with hourly weather data on
    (station_id, timestamp) floored to the hour.

    Parameters
    ----------
    pollution_df : Hourly pollution DataFrame
    weather_df   : Hourly weather DataFrame
    how          : Join type ('left' keeps all pollution rows)

    Returns
    -------
    Merged DataFrame with both pollution and weather columns.
    Rows that could not be matched get NaN for weather columns.
    """
    if pollution_df.empty:
        logger.warning("join_pollution_weather: pollution_df is empty – returning empty")
        return pd.DataFrame()

    p = pollution_df.copy()
    w = weather_df.copy()

    # Floor timestamps to hour for consistent join keys
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True).dt.floor("h")
    w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True).dt.floor("h")

    # Keep only the columns needed from weather
    weather_cols = [
        "timestamp", "station_id",
        "temperature", "humidity", "precipitation",
        "wind_speed", "wind_direction", "pressure",
    ]
    w = w[[c for c in weather_cols if c in w.columns]]

    merged = pd.merge(p, w, on=["timestamp", "station_id"], how=how)

    matched = merged["temperature"].notna().sum() if "temperature" in merged.columns else 0
    logger.info(
        "join_pollution_weather: %d pollution rows → %d joined rows (%d with weather)",
        len(p), len(merged), matched,
    )
    return merged.reset_index(drop=True)


# ── Master clean pipeline ────────────────────────────────────────────────────

def clean_pollution_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on a pollution DataFrame.
    Validation (range checks etc.) is done separately in validation.py.

    Steps: timestamp normalise → pollutant names → units → station names →
           coerce values → deduplicate
    """
    logger.info("clean_pollution_dataframe: input rows=%d", len(df))
    df = normalise_timestamps(df)
    df = normalise_pollutant_names(df)
    df = normalise_units(df)
    df = normalise_station_names(df)
    df = coerce_numeric_values(df)
    df = deduplicate(df, subset=["station_id", "timestamp", "pollutant"])
    logger.info("clean_pollution_dataframe: output rows=%d", len(df))
    return df


def clean_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on a weather DataFrame.
    """
    logger.info("clean_weather_dataframe: input rows=%d", len(df))
    df = normalise_timestamps(df)
    df = normalise_station_names(df)
    df = deduplicate(df, subset=["station_id", "timestamp"])
    for col in ["temperature", "humidity", "precipitation", "wind_speed", "wind_direction", "pressure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("clean_weather_dataframe: output rows=%d", len(df))
    return df
