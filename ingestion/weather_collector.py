"""
ingestion/weather_collector.py
================================
Fetches hourly weather data from the Open-Meteo API (free, no API key needed).

API docs: https://open-meteo.com/en/docs

What it does:
  1. For each configured station (lat/lon from config.yaml), calls the
     Open-Meteo forecast/history endpoint.
  2. Saves the raw JSON response to data/raw/openmeteo/<date>/<station>_<time>.json
  3. Logs the ingestion run to ingestion_log.jsonl
  4. Returns a normalised pandas DataFrame ready for the ETL pipeline.

Weather variables collected per config.yaml:
  temperature_2m, relative_humidity_2m, precipitation,
  wind_speed_10m, wind_direction_10m, surface_pressure

NOTE: Open-Meteo historical data endpoint requires dates in the past.
      The module automatically chooses the correct endpoint based on date range.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from config.logger import get_logger
from config.settings import get_project_root, settings
from ingestion.ingestion_logger import log_ingestion_run
from ingestion.raw_storage import save_raw_response

logger = get_logger(__name__, log_file="ingestion.log")

# ── Config shortcuts ─────────────────────────────────────────────────────────
_OM_CFG = settings["ingestion"]["openmeteo"]
_BASE_URL = _OM_CFG["base_url"]
_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY_VARS = _OM_CFG.get(
    "hourly_variables",
    [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
    ],
)
_LOOKBACK_DAYS = _OM_CFG.get("lookback_days", 7)
_RETRY_ATTEMPTS = _OM_CFG.get("retry_attempts", 3)
_RETRY_DELAY = _OM_CFG.get("retry_delay_seconds", 5)
_STATIONS = settings["ingestion"]["stations"]


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _request_with_retry(url: str, params: dict) -> requests.Response:
    """GET request with exponential back-off retry."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code < 500:
                return resp
            logger.warning(
                "Open-Meteo returned %d on attempt %d/%d – retrying in %ds",
                resp.status_code, attempt, _RETRY_ATTEMPTS, _RETRY_DELAY,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            logger.warning(
                "Connection error on attempt %d/%d: %s – retrying in %ds",
                attempt, _RETRY_ATTEMPTS, exc, _RETRY_DELAY,
            )
        time.sleep(_RETRY_DELAY)
    raise requests.ConnectionError(
        f"Open-Meteo request failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
    )


# ── Core fetcher ─────────────────────────────────────────────────────────────

def fetch_weather_for_station(
    station_meta: dict,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> tuple[dict, int]:
    """
    Fetch hourly weather data for one station from Open-Meteo.

    Open-Meteo has two endpoints:
      - forecast  : up to 7 days in the future (also covers recent past)
      - archive   : historical data beyond 5 days ago

    This function uses the archive endpoint for historical lookback.

    Parameters
    ----------
    station_meta : dict with keys: id, name, city, latitude, longitude
    date_from    : UTC datetime. Defaults to lookback_days ago.
    date_to      : UTC datetime. Defaults to yesterday (archive has 1-day lag).

    Returns
    -------
    (raw_json_dict, http_status_code)
    """
    now_utc = datetime.now(timezone.utc)
    if date_to is None:
        # Archive has a ~1-day lag; go to yesterday to be safe
        date_to = now_utc - timedelta(days=1)
    if date_from is None:
        date_from = now_utc - timedelta(days=_LOOKBACK_DAYS)

    # For data older than 5 days, use the archive endpoint
    cutoff = now_utc - timedelta(days=5)
    url = _HISTORICAL_URL if date_from < cutoff else _BASE_URL

    params = {
        "latitude": station_meta["latitude"],
        "longitude": station_meta["longitude"],
        "hourly": ",".join(_HOURLY_VARS),
        "start_date": date_from.strftime("%Y-%m-%d"),
        "end_date": date_to.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }

    logger.info(
        "Fetching Open-Meteo | station=%s | %s → %s",
        station_meta["id"],
        params["start_date"],
        params["end_date"],
    )

    try:
        resp = _request_with_retry(url, params)
        http_status = resp.status_code
        resp.raise_for_status()
        return resp.json(), http_status
    except Exception as exc:
        logger.error(
            "Open-Meteo fetch error for %s: %s", station_meta["id"], exc
        )
        log_ingestion_run(
            source="openmeteo",
            station_id=station_meta["id"],
            request_url=url,
            api_status_code=0,
            record_count=0,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return {}, 0


def _normalise_weather(raw: dict, station_meta: dict) -> pd.DataFrame:
    """
    Convert the Open-Meteo JSON response to a flat DataFrame.

    Open-Meteo returns:
      {
        "hourly": {
          "time": ["2024-01-01T00:00", ...],
          "temperature_2m": [22.3, ...],
          ...
        }
      }

    Output columns:
        timestamp, station_id, station_name, city,
        latitude, longitude,
        temperature, humidity, precipitation,
        wind_speed, wind_direction, pressure
    """
    hourly = raw.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"timestamp": times})
    # Map Open-Meteo variable names → our standard column names
    column_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "precipitation": "precipitation",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "surface_pressure": "pressure",
    }
    for api_col, our_col in column_map.items():
        df[our_col] = hourly.get(api_col, None)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["station_id"] = station_meta["id"]
    df["station_name"] = station_meta["name"]
    df["city"] = station_meta["city"]
    df["latitude"] = station_meta["latitude"]
    df["longitude"] = station_meta["longitude"]
    df["source"] = "Open-Meteo"

    return df[[
        "timestamp", "station_id", "station_name", "city",
        "latitude", "longitude",
        "temperature", "humidity", "precipitation",
        "wind_speed", "wind_direction", "pressure",
        "source",
    ]]


# ── Mock dataset ─────────────────────────────────────────────────────────────

def _generate_mock_weather(station_meta: dict) -> pd.DataFrame:
    """
    Generate clearly-labelled synthetic weather data for pipeline testing.

    ⚠️  THIS IS NOT REAL DATA. Used only when the API is unreachable.
    """
    import numpy as np

    logger.warning(
        "⚠️  Using MOCK weather data for station=%s. NOT real measurements.",
        station_meta["id"],
    )
    now = datetime.now(timezone.utc)
    hours = 24 * _LOOKBACK_DAYS
    timestamps = [now - timedelta(hours=h) for h in range(hours, 0, -1)]
    rng = np.random.default_rng(seed=99)

    rows = []
    for ts in timestamps:
        rows.append({
            "timestamp": ts,
            "station_id": station_meta["id"],
            "station_name": station_meta["name"],
            "city": station_meta["city"],
            "latitude": station_meta["latitude"],
            "longitude": station_meta["longitude"],
            "temperature": round(float(rng.uniform(15, 42)), 1),
            "humidity": round(float(rng.uniform(20, 95)), 1),
            "precipitation": round(float(rng.uniform(0, 10)), 2),
            "wind_speed": round(float(rng.uniform(0, 30)), 1),
            "wind_direction": round(float(rng.uniform(0, 360)), 1),
            "pressure": round(float(rng.uniform(990, 1015)), 1),
            "source": "MOCK_OpenMeteo",   # clearly labelled
        })
    return pd.DataFrame(rows)


# ── Public entry point ────────────────────────────────────────────────────────

def collect_all_stations(
    use_mock_on_failure: bool = True,
) -> pd.DataFrame:
    """
    Collect weather data for all stations defined in config.yaml.

    Parameters
    ----------
    use_mock_on_failure : Fall back to mock data if the API fails.

    Returns
    -------
    pd.DataFrame with standardised weather columns
    """
    all_frames: list[pd.DataFrame] = []

    for station in _STATIONS:
        sid = station["id"]
        raw_data, http_status = fetch_weather_for_station(station)

        if raw_data and http_status == 200:
            # Save raw response
            hourly_len = len(raw_data.get("hourly", {}).get("time", []))
            raw_path = save_raw_response(
                source="openmeteo",
                station_id=sid,
                payload=raw_data,
                api_status=http_status,
                record_count=hourly_len,
                request_url=_HISTORICAL_URL,
                extra_meta={"city": station["city"]},
            )
            log_ingestion_run(
                source="openmeteo",
                station_id=sid,
                request_url=_HISTORICAL_URL,
                api_status_code=http_status,
                record_count=hourly_len,
                success=True,
                raw_file_path=str(raw_path),
            )
            df = _normalise_weather(raw_data, station)
            all_frames.append(df)

        elif use_mock_on_failure:
            logger.warning(
                "No weather data for station=%s (http=%d). Using mock.",
                sid, http_status,
            )
            all_frames.append(_generate_mock_weather(station))

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        logger.info(
            "Open-Meteo collection complete. Total records: %d", len(combined)
        )
        return combined

    logger.warning("No weather data collected at all.")
    return pd.DataFrame()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = collect_all_stations(use_mock_on_failure=True)
    print(f"\nCollected {len(df)} weather records")
    if not df.empty:
        print(df.head(10).to_string(index=False))
        print("\nStations:", df["station_id"].unique().tolist())
        print("Date range:", df["timestamp"].min(), "→", df["timestamp"].max())
