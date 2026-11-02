"""
database/load_data.py
======================
Loads cleaned DataFrames into PostgreSQL tables using upsert logic
(INSERT … ON CONFLICT DO UPDATE) so the pipeline is safe to re-run.

Functions
---------
upsert_stations            – load station metadata
upsert_pollution           – load cleaned hourly pollution measurements
upsert_weather             – load cleaned hourly weather measurements
upsert_daily_aqi           – load computed daily AQI rows
insert_model_prediction    – insert one prediction record
log_ingestion_run_to_db    – mirror JSONL log entry to PostgreSQL
log_data_quality_check     – record quality check result
"""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import text

from config.logger import get_logger
from database.connection import get_session

logger = get_logger(__name__, log_file="database.log")


# ── Helper ────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None if it cannot be converted."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Stations ─────────────────────────────────────────────────────────────────

def upsert_stations(stations: list[dict]) -> int:
    """
    Insert or update station records.

    Parameters
    ----------
    stations : list of dicts with keys:
               station_id, station_name, city, latitude, longitude

    Returns
    -------
    Number of rows upserted
    """
    if not stations:
        return 0

    sql = text("""
        INSERT INTO stations (station_id, station_name, city, latitude, longitude, updated_at)
        VALUES (:station_id, :station_name, :city, :latitude, :longitude, NOW())
        ON CONFLICT (station_id) DO UPDATE SET
            station_name = EXCLUDED.station_name,
            city         = EXCLUDED.city,
            latitude     = EXCLUDED.latitude,
            longitude    = EXCLUDED.longitude,
            updated_at   = NOW()
    """)

    with get_session() as session:
        session.execute(sql, stations)

    logger.info("upsert_stations: %d rows upserted", len(stations))
    return len(stations)


def upsert_stations_from_config() -> int:
    """Load station records directly from config.yaml."""
    from config.settings import settings
    return upsert_stations(settings["ingestion"]["stations"])


# ── Pollution measurements ────────────────────────────────────────────────────

def upsert_pollution(df: pd.DataFrame) -> int:
    """
    Upsert hourly pollution measurements.

    Required DataFrame columns:
        station_id, timestamp, pollutant, value, unit, source
    """
    if df.empty:
        logger.warning("upsert_pollution: empty DataFrame – nothing to load")
        return 0

    required = {"station_id", "timestamp", "pollutant", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"upsert_pollution: missing columns {missing}")

    sql = text("""
        INSERT INTO pollution_measurements
            (station_id, timestamp, pollutant, value, unit, source)
        VALUES
            (:station_id, :timestamp, :pollutant, :value, :unit, :source)
        ON CONFLICT (station_id, timestamp, pollutant) DO UPDATE SET
            value  = EXCLUDED.value,
            unit   = EXCLUDED.unit,
            source = EXCLUDED.source
    """)

    records = []
    for _, row in df.iterrows():
        records.append({
            "station_id": str(row["station_id"]),
            "timestamp":  row["timestamp"],
            "pollutant":  str(row["pollutant"]),
            "value":      _safe_float(row["value"]),
            "unit":       row.get("unit", ""),
            "source":     row.get("source", ""),
        })

    with get_session() as session:
        session.execute(sql, records)

    logger.info("upsert_pollution: %d rows upserted", len(records))
    return len(records)


# ── Weather measurements ──────────────────────────────────────────────────────

def upsert_weather(df: pd.DataFrame) -> int:
    """
    Upsert hourly weather measurements.

    Required DataFrame columns:
        station_id, timestamp, temperature, humidity, precipitation,
        wind_speed, wind_direction, pressure
    """
    if df.empty:
        logger.warning("upsert_weather: empty DataFrame – nothing to load")
        return 0

    sql = text("""
        INSERT INTO weather_measurements
            (station_id, timestamp, temperature, humidity,
             precipitation, wind_speed, wind_direction, pressure, source)
        VALUES
            (:station_id, :timestamp, :temperature, :humidity,
             :precipitation, :wind_speed, :wind_direction, :pressure, :source)
        ON CONFLICT (station_id, timestamp) DO UPDATE SET
            temperature    = EXCLUDED.temperature,
            humidity       = EXCLUDED.humidity,
            precipitation  = EXCLUDED.precipitation,
            wind_speed     = EXCLUDED.wind_speed,
            wind_direction = EXCLUDED.wind_direction,
            pressure       = EXCLUDED.pressure,
            source         = EXCLUDED.source
    """)

    records = []
    for _, row in df.iterrows():
        records.append({
            "station_id":    str(row["station_id"]),
            "timestamp":     row["timestamp"],
            "temperature":   _safe_float(row.get("temperature")),
            "humidity":      _safe_float(row.get("humidity")),
            "precipitation": _safe_float(row.get("precipitation")),
            "wind_speed":    _safe_float(row.get("wind_speed")),
            "wind_direction":_safe_float(row.get("wind_direction")),
            "pressure":      _safe_float(row.get("pressure")),
            "source":        row.get("source", ""),
        })

    with get_session() as session:
        session.execute(sql, records)

    logger.info("upsert_weather: %d rows upserted", len(records))
    return len(records)


# ── Daily AQI ─────────────────────────────────────────────────────────────────

def upsert_daily_aqi(df: pd.DataFrame) -> int:
    """
    Upsert daily AQI records.

    Required DataFrame columns:
        station_id, date, pm25, pm10, no2, so2, co, o3,
        aqi, aqi_category, dominant_pollutant
    Optional:
        avg_temperature, avg_humidity, total_precipitation,
        avg_wind_speed, avg_pressure
    """
    if df.empty:
        logger.warning("upsert_daily_aqi: empty DataFrame – nothing to load")
        return 0

    sql = text("""
        INSERT INTO daily_aqi (
            station_id, date, pm25, pm10, no2, so2, co, o3,
            aqi, aqi_category, dominant_pollutant,
            avg_temperature, avg_humidity, total_precipitation,
            avg_wind_speed, avg_pressure, updated_at
        ) VALUES (
            :station_id, :date, :pm25, :pm10, :no2, :so2, :co, :o3,
            :aqi, :aqi_category, :dominant_pollutant,
            :avg_temperature, :avg_humidity, :total_precipitation,
            :avg_wind_speed, :avg_pressure, NOW()
        )
        ON CONFLICT (station_id, date) DO UPDATE SET
            pm25                = EXCLUDED.pm25,
            pm10                = EXCLUDED.pm10,
            no2                 = EXCLUDED.no2,
            so2                 = EXCLUDED.so2,
            co                  = EXCLUDED.co,
            o3                  = EXCLUDED.o3,
            aqi                 = EXCLUDED.aqi,
            aqi_category        = EXCLUDED.aqi_category,
            dominant_pollutant  = EXCLUDED.dominant_pollutant,
            avg_temperature     = EXCLUDED.avg_temperature,
            avg_humidity        = EXCLUDED.avg_humidity,
            total_precipitation = EXCLUDED.total_precipitation,
            avg_wind_speed      = EXCLUDED.avg_wind_speed,
            avg_pressure        = EXCLUDED.avg_pressure,
            updated_at          = NOW()
    """)

    records = []
    for _, row in df.iterrows():
        records.append({
            "station_id":          str(row["station_id"]),
            "date":                row["date"],
            "pm25":                _safe_float(row.get("pm25")),
            "pm10":                _safe_float(row.get("pm10")),
            "no2":                 _safe_float(row.get("no2")),
            "so2":                 _safe_float(row.get("so2")),
            "co":                  _safe_float(row.get("co")),
            "o3":                  _safe_float(row.get("o3")),
            "aqi":                 _safe_float(row.get("aqi")),
            "aqi_category":        row.get("aqi_category", ""),
            "dominant_pollutant":  row.get("dominant_pollutant", ""),
            "avg_temperature":     _safe_float(row.get("avg_temperature")),
            "avg_humidity":        _safe_float(row.get("avg_humidity")),
            "total_precipitation": _safe_float(row.get("total_precipitation")),
            "avg_wind_speed":      _safe_float(row.get("avg_wind_speed")),
            "avg_pressure":        _safe_float(row.get("avg_pressure")),
        })

    with get_session() as session:
        session.execute(sql, records)

    logger.info("upsert_daily_aqi: %d rows upserted", len(records))
    return len(records)


# ── Model predictions ─────────────────────────────────────────────────────────

def insert_model_prediction(
    station_id: str,
    predicted_aqi: float,
    predicted_category: str,
    model_name: str,
    model_version: str,
    input_features: dict,
) -> int:
    """Insert a single model prediction record. Returns the new row id."""
    import json

    sql = text("""
        INSERT INTO model_predictions
            (station_id, predicted_aqi, predicted_category,
             model_name, model_version, input_features)
        VALUES
            (:station_id, :predicted_aqi, :predicted_category,
             :model_name, :model_version, :input_features)
        RETURNING id
    """)

    with get_session() as session:
        result = session.execute(sql, {
            "station_id":         station_id,
            "predicted_aqi":      predicted_aqi,
            "predicted_category": predicted_category,
            "model_name":         model_name,
            "model_version":      model_version,
            "input_features":     json.dumps(input_features),
        })
        row_id = result.scalar()

    logger.info(
        "insert_model_prediction: id=%s | aqi=%.2f | category=%s",
        row_id, predicted_aqi, predicted_category,
    )
    return row_id


# ── Ingestion run mirror ──────────────────────────────────────────────────────

def log_ingestion_run_to_db(
    run_id: str,
    source: str,
    station_id: str,
    request_url: str,
    api_status_code: int,
    record_count: int,
    success: bool,
    raw_file_path: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Mirror an ingestion run entry from the JSONL log into PostgreSQL."""
    sql = text("""
        INSERT INTO ingestion_runs
            (run_id, source, station_id, request_url, api_status_code,
             record_count, success, raw_file_path, error_type, error_message)
        VALUES
            (:run_id, :source, :station_id, :request_url, :api_status_code,
             :record_count, :success, :raw_file_path, :error_type, :error_message)
        ON CONFLICT (run_id) DO NOTHING
    """)

    with get_session() as session:
        session.execute(sql, {
            "run_id": run_id, "source": source, "station_id": station_id,
            "request_url": request_url, "api_status_code": api_status_code,
            "record_count": record_count, "success": success,
            "raw_file_path": raw_file_path, "error_type": error_type,
            "error_message": error_message,
        })


# ── Data quality log ──────────────────────────────────────────────────────────

def log_data_quality_check(
    check_name: str,
    source: str,
    passed: bool,
    total_records: int,
    failed_records: int,
    details: Optional[dict] = None,
) -> None:
    """Record the result of a data quality check."""
    import json

    failure_pct = round((failed_records / total_records * 100) if total_records > 0 else 0, 2)
    sql = text("""
        INSERT INTO data_quality_log
            (check_name, source, passed, total_records, failed_records, failure_pct, details)
        VALUES
            (:check_name, :source, :passed, :total_records, :failed_records,
             :failure_pct, :details)
    """)

    with get_session() as session:
        session.execute(sql, {
            "check_name": check_name, "source": source, "passed": passed,
            "total_records": total_records, "failed_records": failed_records,
            "failure_pct": failure_pct,
            "details": json.dumps(details or {}),
        })

    logger.info(
        "DQ check '%s' | source=%s | passed=%s | failed=%d/%d (%.1f%%)",
        check_name, source, passed, failed_records, total_records, failure_pct,
    )
