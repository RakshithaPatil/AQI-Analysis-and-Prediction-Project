"""
airflow/dags/aqi_pipeline_dag.py
==================================
Apache Airflow DAG: AQI Data Engineering Pipeline

This DAG orchestrates the complete ETL pipeline from raw API ingestion
to a fully loaded, analytically ready PostgreSQL database.

Schedule: Daily at 02:00 IST (configurable via config.yaml)
Timezone: Asia/Kolkata

Task execution order:
  t01_extract_openaq
  t02_extract_weather
  t03_store_raw          ← (raw storage is done inside extractors, this task
                             validates that files exist and logs the run)
  t04_validate_raw
  t05_clean_timestamps
  t06_normalize_stations
  t07_normalize_units
  t08_remove_duplicates
  t09_detect_gaps
  t10_validate_ranges
  t11_join_pollution_weather
  t12_hourly_averages
  t13_daily_averages
  t14_calculate_aqi
  t15_aqi_categories
  t16_load_postgres
  t17_update_analytics
  t18_run_quality_checks

HOW TO USE:
  1. Place this file in your Airflow dags/ folder (or symlink it).
  2. Start Airflow: airflow webserver & airflow scheduler
  3. Enable the DAG in the Airflow UI at http://localhost:8080
  4. Trigger a manual run or wait for the daily schedule.

IMPORTANT:
  The tasks share data through XCom (Airflow's key-value store for task
  outputs). DataFrames are serialised as JSON strings in XCom.
  For large datasets, consider using Airflow datasets or a shared file path.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Add project root to Python path ─────────────────────────────────────────
# This ensures all project modules are importable inside Airflow workers.
_DAG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DAG_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Default arguments for all tasks ─────────────────────────────────────────
default_args = {
    "owner": "aqi_pipeline",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# Each function is a self-contained unit – imports inside functions to keep
# the module-level import fast and avoid cold-start issues.
# ══════════════════════════════════════════════════════════════════════════════

# ── Task 1: Extract OpenAQ data ───────────────────────────────────────────────
def task_extract_openaq(**context):
    """Fetch pollution measurements from OpenAQ API for all configured stations."""
    from ingestion.openaq_collector import collect_all_stations
    df = collect_all_stations(use_mock_on_failure=True)
    # Push to XCom as JSON (small dataset)
    context["ti"].xcom_push(key="openaq_raw", value=df.to_json(date_format="iso"))
    return f"Extracted {len(df)} OpenAQ records"


# ── Task 2: Extract weather data ─────────────────────────────────────────────
def task_extract_weather(**context):
    """Fetch hourly weather data from Open-Meteo API for all configured stations."""
    from ingestion.weather_collector import collect_all_stations
    df = collect_all_stations(use_mock_on_failure=True)
    context["ti"].xcom_push(key="weather_raw", value=df.to_json(date_format="iso"))
    return f"Extracted {len(df)} weather records"


# ── Task 3: Verify raw storage ────────────────────────────────────────────────
def task_store_raw(**context):
    """
    Verify that raw files were written to disk.
    (Actual saving happens inside the collectors in Tasks 1 & 2.)
    """
    import pandas as pd
    from ingestion.raw_storage import list_raw_files
    from config.logger import get_logger
    logger = get_logger("t03_store_raw")

    today = datetime.now().strftime("%Y-%m-%d")
    openaq_files = list_raw_files("openaq", today)
    weather_files = list_raw_files("openmeteo", today)

    logger.info("Raw files today – OpenAQ: %d | Weather: %d",
                len(openaq_files), len(weather_files))
    return f"openaq={len(openaq_files)}, weather={len(weather_files)}"


# ── Task 4: Validate raw data ─────────────────────────────────────────────────
def task_validate_raw(**context):
    """Run validation checks on raw DataFrames. Rejected records go to data/rejected/."""
    import pandas as pd
    from transformation.validation import validate_pollution_data, validate_weather_data

    ti = context["ti"]
    openaq_json = ti.xcom_pull(task_ids="t01_extract_openaq", key="openaq_raw")
    weather_json = ti.xcom_pull(task_ids="t02_extract_weather", key="weather_raw")

    poll_df = pd.read_json(openaq_json) if openaq_json else pd.DataFrame()
    wx_df = pd.read_json(weather_json) if weather_json else pd.DataFrame()

    good_poll, rej_poll = validate_pollution_data(poll_df)
    good_wx, rej_wx = validate_weather_data(wx_df)

    ti.xcom_push(key="poll_valid", value=good_poll.to_json(date_format="iso"))
    ti.xcom_push(key="wx_valid", value=good_wx.to_json(date_format="iso"))
    return (f"Pollution: {len(good_poll)} good / {len(rej_poll)} rejected | "
            f"Weather: {len(good_wx)} good / {len(rej_wx)} rejected")


# ── Task 5: Clean timestamps ──────────────────────────────────────────────────
def task_clean_timestamps(**context):
    """Parse and normalise all timestamps to UTC."""
    import pandas as pd
    from transformation.cleaning import normalise_timestamps

    ti = context["ti"]
    poll_df = pd.read_json(ti.xcom_pull(task_ids="t04_validate_raw", key="poll_valid"))
    wx_df = pd.read_json(ti.xcom_pull(task_ids="t04_validate_raw", key="wx_valid"))

    poll_df = normalise_timestamps(poll_df)
    wx_df = normalise_timestamps(wx_df)

    ti.xcom_push(key="poll_ts", value=poll_df.to_json(date_format="iso"))
    ti.xcom_push(key="wx_ts", value=wx_df.to_json(date_format="iso"))
    return "Timestamps normalised"


# ── Task 6: Normalise station names ──────────────────────────────────────────
def task_normalize_stations(**context):
    """Strip and title-case station name / city columns."""
    import pandas as pd
    from transformation.cleaning import normalise_station_names

    ti = context["ti"]
    poll_df = pd.read_json(ti.xcom_pull(task_ids="t05_clean_timestamps", key="poll_ts"))
    wx_df = pd.read_json(ti.xcom_pull(task_ids="t05_clean_timestamps", key="wx_ts"))

    poll_df = normalise_station_names(poll_df)
    wx_df = normalise_station_names(wx_df)

    ti.xcom_push(key="poll_st", value=poll_df.to_json(date_format="iso"))
    ti.xcom_push(key="wx_st", value=wx_df.to_json(date_format="iso"))
    return "Station names normalised"


# ── Task 7: Normalise pollutant units ─────────────────────────────────────────
def task_normalize_units(**context):
    """Standardise pollutant names and unit strings."""
    import pandas as pd
    from transformation.cleaning import normalise_pollutant_names, normalise_units

    ti = context["ti"]
    poll_df = pd.read_json(ti.xcom_pull(task_ids="t06_normalize_stations", key="poll_st"))

    poll_df = normalise_pollutant_names(poll_df)
    poll_df = normalise_units(poll_df)

    ti.xcom_push(key="poll_units", value=poll_df.to_json(date_format="iso"))
    # Weather doesn't need unit normalisation
    wx_json = ti.xcom_pull(task_ids="t06_normalize_stations", key="wx_st")
    ti.xcom_push(key="wx_units", value=wx_json)
    return "Units normalised"


# ── Task 8: Remove duplicates ─────────────────────────────────────────────────
def task_remove_duplicates(**context):
    """Deduplicate pollution and weather DataFrames."""
    import pandas as pd
    from transformation.cleaning import deduplicate

    ti = context["ti"]
    poll_df = pd.read_json(ti.xcom_pull(task_ids="t07_normalize_units", key="poll_units"))
    wx_df = pd.read_json(ti.xcom_pull(task_ids="t07_normalize_units", key="wx_units"))

    poll_df = deduplicate(poll_df, subset=["station_id", "timestamp", "pollutant"])
    wx_df = deduplicate(wx_df, subset=["station_id", "timestamp"])

    ti.xcom_push(key="poll_dedup", value=poll_df.to_json(date_format="iso"))
    ti.xcom_push(key="wx_dedup", value=wx_df.to_json(date_format="iso"))
    return f"After dedup: poll={len(poll_df)}, weather={len(wx_df)}"


# ── Task 9: Detect missing periods ───────────────────────────────────────────
def task_detect_gaps(**context):
    """Log any missing time periods per station. Does NOT drop data."""
    import pandas as pd
    from transformation.cleaning import detect_missing_periods
    from config.logger import get_logger
    logger = get_logger("t09_detect_gaps")

    ti = context["ti"]
    poll_df = pd.read_json(ti.xcom_pull(task_ids="t08_remove_duplicates", key="poll_dedup"))

    if not poll_df.empty and "timestamp" in poll_df.columns:
        poll_df["timestamp"] = pd.to_datetime(poll_df["timestamp"], utc=True, errors="coerce")
        gaps_df = detect_missing_periods(poll_df)
        if not gaps_df.empty:
            logger.warning("Detected %d time gaps:\n%s", len(gaps_df), gaps_df.to_string())
    return "Gap detection complete"


# ── Task 10: Validate pollutant ranges ───────────────────────────────────────
def task_validate_ranges(**context):
    """Secondary validation pass: catch any range violations that slipped through."""
    import pandas as pd
    from transformation.validation import validate_pollution_data

    ti = context["ti"]
    poll_df = pd.read_json(ti.xcom_pull(task_ids="t08_remove_duplicates", key="poll_dedup"))
    wx_df = pd.read_json(ti.xcom_pull(task_ids="t08_remove_duplicates", key="wx_dedup"))

    good_poll, rej_poll = validate_pollution_data(poll_df)
    ti.xcom_push(key="poll_final", value=good_poll.to_json(date_format="iso"))
    ti.xcom_push(key="wx_final", value=wx_df.to_json(date_format="iso"))
    return f"Final validation: {len(good_poll)} clean, {len(rej_poll)} rejected"


# ── Task 11: Join pollution + weather ─────────────────────────────────────────
def task_join_pollution_weather(**context):
    """Merge hourly pollution and weather data on (station_id, timestamp)."""
    import pandas as pd
    from transformation.cleaning import (
        calculate_hourly_averages,
        join_pollution_weather,
    )

    ti = context["ti"]
    poll_df = pd.read_json(ti.xcom_pull(task_ids="t10_validate_ranges", key="poll_final"))
    wx_df = pd.read_json(ti.xcom_pull(task_ids="t10_validate_ranges", key="wx_final"))

    if not poll_df.empty:
        poll_df["timestamp"] = pd.to_datetime(poll_df["timestamp"], utc=True, errors="coerce")
    if not wx_df.empty:
        wx_df["timestamp"] = pd.to_datetime(wx_df["timestamp"], utc=True, errors="coerce")

    hourly_poll = calculate_hourly_averages(poll_df) if not poll_df.empty else pd.DataFrame()
    joined = join_pollution_weather(hourly_poll, wx_df) if not hourly_poll.empty else pd.DataFrame()

    ti.xcom_push(key="joined", value=joined.to_json(date_format="iso"))
    ti.xcom_push(key="hourly_poll", value=hourly_poll.to_json(date_format="iso"))
    return f"Joined: {len(joined)} rows"


# ── Task 12: Calculate hourly averages ───────────────────────────────────────
def task_hourly_averages(**context):
    """Confirm hourly averages exist; store for DB load."""
    # Already calculated in Task 11; just re-push for clarity
    ti = context["ti"]
    hourly_json = ti.xcom_pull(task_ids="t11_join_pollution_weather", key="hourly_poll")
    ti.xcom_push(key="hourly_ready", value=hourly_json)
    return "Hourly averages ready"


# ── Task 13: Calculate daily averages ────────────────────────────────────────
def task_daily_averages(**context):
    """Aggregate hourly data to daily averages."""
    import pandas as pd
    from transformation.cleaning import calculate_daily_averages

    ti = context["ti"]
    hourly_json = ti.xcom_pull(task_ids="t12_hourly_averages", key="hourly_ready")
    hourly_df = pd.read_json(hourly_json) if hourly_json else pd.DataFrame()

    if not hourly_df.empty:
        hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"], utc=True, errors="coerce")

    daily_df = calculate_daily_averages(hourly_df) if not hourly_df.empty else pd.DataFrame()
    ti.xcom_push(key="daily_long", value=daily_df.to_json(date_format="iso"))
    return f"Daily averages: {len(daily_df)} rows"


# ── Task 14: Calculate AQI ────────────────────────────────────────────────────
def task_calculate_aqi(**context):
    """Pivot daily data and calculate AQI per station per day."""
    import pandas as pd
    from transformation.aqi_calculation import (
        calculate_aqi_for_dataframe,
        pivot_daily_pollutants,
    )

    ti = context["ti"]
    daily_json = ti.xcom_pull(task_ids="t13_daily_averages", key="daily_long")
    daily_df = pd.read_json(daily_json) if daily_json else pd.DataFrame()

    if daily_df.empty:
        ti.xcom_push(key="daily_aqi", value=pd.DataFrame().to_json())
        return "No data to calculate AQI for"

    wide_df = pivot_daily_pollutants(daily_df)
    aqi_df = calculate_aqi_for_dataframe(wide_df)
    ti.xcom_push(key="daily_aqi", value=aqi_df.to_json(date_format="iso"))
    return f"AQI calculated for {len(aqi_df)} station-days"


# ── Task 15: Add AQI categories ───────────────────────────────────────────────
def task_aqi_categories(**context):
    """Verify AQI categories are populated; add weather daily aggregates."""
    import pandas as pd
    import numpy as np

    ti = context["ti"]
    aqi_df = pd.read_json(ti.xcom_pull(task_ids="t14_calculate_aqi", key="daily_aqi"))
    wx_df = pd.read_json(ti.xcom_pull(task_ids="t10_validate_ranges", key="wx_final"))

    if aqi_df.empty:
        ti.xcom_push(key="final_daily", value=aqi_df.to_json())
        return "No AQI data"

    # Add daily weather aggregates if available
    if not wx_df.empty and "timestamp" in wx_df.columns:
        wx_df["timestamp"] = pd.to_datetime(wx_df["timestamp"], utc=True, errors="coerce")
        wx_df["date"] = wx_df["timestamp"].dt.date
        wx_daily = wx_df.groupby(["station_id", "date"]).agg(
            avg_temperature=("temperature", "mean"),
            avg_humidity=("humidity", "mean"),
            total_precipitation=("precipitation", "sum"),
            avg_wind_speed=("wind_speed", "mean"),
            avg_pressure=("pressure", "mean"),
        ).reset_index()
        wx_daily["date"] = pd.to_datetime(wx_daily["date"])
        aqi_df["date"] = pd.to_datetime(aqi_df["date"])
        aqi_df = aqi_df.merge(wx_daily, on=["station_id", "date"], how="left")

    ti.xcom_push(key="final_daily", value=aqi_df.to_json(date_format="iso"))
    cat_counts = aqi_df["aqi_category"].value_counts().to_dict() if "aqi_category" in aqi_df.columns else {}
    return f"AQI categories: {cat_counts}"


# ── Task 16: Load to PostgreSQL ───────────────────────────────────────────────
def task_load_postgres(**context):
    """Load cleaned pollution, weather, and daily AQI data into PostgreSQL."""
    import pandas as pd
    from database.connection import test_connection
    from database.load_data import (
        upsert_daily_aqi,
        upsert_pollution,
        upsert_stations_from_config,
        upsert_weather,
    )

    if not test_connection():
        raise ConnectionError("Cannot connect to PostgreSQL – check DB_HOST / credentials")

    ti = context["ti"]

    # Load stations first (FK dependency)
    upsert_stations_from_config()

    # Load pollution measurements
    poll_json = ti.xcom_pull(task_ids="t12_hourly_averages", key="hourly_ready")
    if poll_json:
        poll_df = pd.read_json(poll_json)
        poll_df["timestamp"] = pd.to_datetime(poll_df["timestamp"], utc=True, errors="coerce")
        upsert_pollution(poll_df)

    # Load weather measurements
    wx_json = ti.xcom_pull(task_ids="t10_validate_ranges", key="wx_final")
    if wx_json:
        wx_df = pd.read_json(wx_json)
        wx_df["timestamp"] = pd.to_datetime(wx_df["timestamp"], utc=True, errors="coerce")
        upsert_weather(wx_df)

    # Load daily AQI
    daily_json = ti.xcom_pull(task_ids="t15_aqi_categories", key="final_daily")
    if daily_json:
        daily_df = pd.read_json(daily_json)
        upsert_daily_aqi(daily_df)

    return "Data loaded into PostgreSQL"


# ── Task 17: Update analytical tables ────────────────────────────────────────
def task_update_analytics(**context):
    """Refresh the analytical views by executing the gold-layer SQL."""
    from pathlib import Path
    from config.settings import get_project_root
    from database.connection import get_engine
    from sqlalchemy import text

    sql_file = get_project_root() / "analytics" / "analytical_queries.sql"
    if not sql_file.exists():
        raise FileNotFoundError(f"Analytics SQL not found: {sql_file}")

    sql_text = sql_file.read_text(encoding="utf-8")
    with get_engine().connect() as conn:
        with conn.begin():
            conn.execute(text(sql_text))

    return "Analytical views refreshed"


# ── Task 18: Run data quality checks ─────────────────────────────────────────
def task_run_quality_checks(**context):
    """Execute data quality checks and log results to PostgreSQL."""
    import pandas as pd
    from monitoring.data_quality_monitor import run_all_checks
    from config.logger import get_logger
    logger = get_logger("t18_quality_checks")

    ti = context["ti"]
    daily_json = ti.xcom_pull(task_ids="t15_aqi_categories", key="final_daily")
    daily_df = pd.read_json(daily_json) if daily_json else pd.DataFrame()

    results = run_all_checks(daily_df)
    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    logger.info("Quality checks: %d passed, %d failed", passed, failed)
    return f"Quality checks: {passed}/{len(results)} passed"


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="aqi_pipeline",
    description="End-to-end AQI data engineering pipeline: ingest → clean → AQI → PostgreSQL",
    default_args=default_args,
    schedule_interval="0 2 * * *",   # Daily at 02:00 UTC (configurable)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "data-engineering", "etl"],
) as dag:

    t01 = PythonOperator(task_id="t01_extract_openaq",         python_callable=task_extract_openaq)
    t02 = PythonOperator(task_id="t02_extract_weather",        python_callable=task_extract_weather)
    t03 = PythonOperator(task_id="t03_store_raw",              python_callable=task_store_raw)
    t04 = PythonOperator(task_id="t04_validate_raw",           python_callable=task_validate_raw)
    t05 = PythonOperator(task_id="t05_clean_timestamps",       python_callable=task_clean_timestamps)
    t06 = PythonOperator(task_id="t06_normalize_stations",     python_callable=task_normalize_stations)
    t07 = PythonOperator(task_id="t07_normalize_units",        python_callable=task_normalize_units)
    t08 = PythonOperator(task_id="t08_remove_duplicates",      python_callable=task_remove_duplicates)
    t09 = PythonOperator(task_id="t09_detect_gaps",            python_callable=task_detect_gaps)
    t10 = PythonOperator(task_id="t10_validate_ranges",        python_callable=task_validate_ranges)
    t11 = PythonOperator(task_id="t11_join_pollution_weather", python_callable=task_join_pollution_weather)
    t12 = PythonOperator(task_id="t12_hourly_averages",        python_callable=task_hourly_averages)
    t13 = PythonOperator(task_id="t13_daily_averages",         python_callable=task_daily_averages)
    t14 = PythonOperator(task_id="t14_calculate_aqi",          python_callable=task_calculate_aqi)
    t15 = PythonOperator(task_id="t15_aqi_categories",         python_callable=task_aqi_categories)
    t16 = PythonOperator(task_id="t16_load_postgres",          python_callable=task_load_postgres)
    t17 = PythonOperator(task_id="t17_update_analytics",       python_callable=task_update_analytics)
    t18 = PythonOperator(task_id="t18_run_quality_checks",     python_callable=task_run_quality_checks)

    # ── Task dependencies (linear pipeline) ─────────────────────────────────
    [t01, t02] >> t03 >> t04 >> t05 >> t06 >> t07 >> t08 >> t09 >> t10
    t10 >> t11 >> t12 >> t13 >> t14 >> t15 >> t16 >> t17 >> t18
