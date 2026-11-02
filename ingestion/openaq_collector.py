"""
ingestion/openaq_collector.py
==============================
Fetches air quality measurements from the OpenAQ v3 REST API.

API docs: https://docs.openaq.org/

What it does:
  1. For each configured station, calls the OpenAQ /measurements endpoint.
  2. Saves the raw JSON response to data/raw/openaq/<date>/<station>_<time>.json
  3. Logs the ingestion run (success or failure) to ingestion_log.jsonl
  4. Returns a normalised pandas DataFrame for the ETL pipeline to consume.

NOTE: If the OpenAQ API is unreachable (e.g. during local dev without internet),
      the module falls back to a clearly-labelled MOCK dataset so the rest of
      the pipeline can still be exercised. The mock path is logged as a warning.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from config.logger import get_logger
from config.settings import get_openaq_api_key, get_project_root, settings
from ingestion.ingestion_logger import log_ingestion_run
from ingestion.raw_storage import save_raw_response

logger = get_logger(__name__, log_file="ingestion.log")

# ── Config shortcuts ─────────────────────────────────────────────────────────
_OAQ_CFG = settings["ingestion"]["openaq"]
_BASE_URL = _OAQ_CFG["base_url"]
_LIMIT = _OAQ_CFG.get("limit", 1000)
_LOOKBACK_DAYS = _OAQ_CFG.get("lookback_days", 7)
_RETRY_ATTEMPTS = _OAQ_CFG.get("retry_attempts", 3)
_RETRY_DELAY = _OAQ_CFG.get("retry_delay_seconds", 5)
_STATIONS = settings["ingestion"]["stations"]


# ── HTTP helper ──────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    """Return request headers; include API key if available."""
    headers = {"Accept": "application/json"}
    api_key = get_openaq_api_key()
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _request_with_retry(url: str, params: dict) -> requests.Response:
    """
    Perform a GET request with retry logic.

    Retries on connection errors, timeouts, and 5xx server errors.
    Raises requests.HTTPError on final failure.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, headers=_get_headers(), timeout=30)
            if resp.status_code < 500:
                # 4xx is a client error – don't retry, surface immediately
                return resp
            logger.warning(
                "OpenAQ returned %d on attempt %d/%d – retrying in %ds",
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
        f"OpenAQ request failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
    )


# ── Core fetcher ─────────────────────────────────────────────────────────────

def fetch_measurements_for_location(
    location_id: int,
    station_id: str,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> tuple[list[dict], int]:
    """
    Fetch all measurements for a single OpenAQ location ID.

    Parameters
    ----------
    location_id : OpenAQ numeric location ID
    station_id  : Human-readable station identifier (used for file naming)
    date_from   : Start datetime (UTC). Defaults to lookback_days ago.
    date_to     : End datetime (UTC). Defaults to now.

    Returns
    -------
    (records, http_status_code)
      records : list of raw measurement dicts
      http_status_code : last HTTP status code received
    """
    now_utc = datetime.now(timezone.utc)
    if date_to is None:
        date_to = now_utc
    if date_from is None:
        date_from = now_utc - timedelta(days=_LOOKBACK_DAYS)

    url = f"{_BASE_URL}/locations/{location_id}/measurements"
    params = {
        "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": _LIMIT,
    }

    all_records: list[dict] = []
    http_status = 0
    page = 1

    # ── Paginate through results ─────────────────────────────────────────────
    while True:
        params["page"] = page
        logger.info(
            "Fetching OpenAQ | location_id=%d | station=%s | page=%d",
            location_id, station_id, page,
        )
        try:
            resp = _request_with_retry(url, params)
            http_status = resp.status_code
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("OpenAQ fetch error for %s: %s", station_id, exc)
            log_ingestion_run(
                source="openaq",
                station_id=station_id,
                request_url=url,
                api_status_code=http_status,
                record_count=len(all_records),
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return all_records, http_status

        results = data.get("results", [])
        all_records.extend(results)

        # Check if there are more pages
        meta = data.get("meta", {})
        found = meta.get("found", 0)
        if len(all_records) >= found or len(results) == 0:
            break
        page += 1

    return all_records, http_status


def _normalise_records(raw_records: list[dict], station_meta: dict) -> pd.DataFrame:
    """
    Convert raw OpenAQ measurement dicts to a clean, flat DataFrame.

    Output columns:
        timestamp, station_id, station_name, city,
        latitude, longitude, pollutant, value, unit, source
    """
    rows = []
    for rec in raw_records:
        # Each measurement has a parameter name and a value object
        parameter = rec.get("parameter", "")
        value_obj = rec.get("value", None)
        unit = rec.get("unit", "")

        # Timestamp – prefer "date.utc"
        date_info = rec.get("date", {})
        ts_str = date_info.get("utc") or date_info.get("local", "")

        # Coordinates from the measurement itself, fall back to station meta
        coords = rec.get("coordinates") or {}
        lat = coords.get("latitude") or station_meta.get("latitude")
        lon = coords.get("longitude") or station_meta.get("longitude")

        rows.append({
            "timestamp": ts_str,
            "station_id": station_meta["id"],
            "station_name": station_meta["name"],
            "city": station_meta["city"],
            "latitude": lat,
            "longitude": lon,
            "pollutant": parameter,
            "value": value_obj,
            "unit": unit,
            "source": "OpenAQ",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "station_id", "station_name", "city",
            "latitude", "longitude", "pollutant", "value", "unit", "source",
        ])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# ── Mock dataset (used when API is unavailable) ──────────────────────────────

def _generate_mock_data(station_meta: dict) -> pd.DataFrame:
    """
    Generate clearly-labelled synthetic data for pipeline testing.

    ⚠️  THIS IS NOT REAL DATA. Used only when the API is unreachable.
    Generates plausible-looking but random values purely for testing.
    """
    import numpy as np

    logger.warning(
        "⚠️  Using MOCK OpenAQ data for station=%s. NOT real measurements.",
        station_meta["id"],
    )
    now = datetime.now(timezone.utc)
    hours = 24 * _LOOKBACK_DAYS
    timestamps = [now - timedelta(hours=h) for h in range(hours, 0, -1)]
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    # Realistic value ranges for each pollutant
    ranges = {
        "pm25": (10, 180), "pm10": (20, 250), "no2": (10, 120),
        "so2": (2, 40),    "co": (0.2, 2.5),  "o3": (20, 100),
    }
    rows = []
    rng = np.random.default_rng(seed=42)
    for ts in timestamps:
        for pol in pollutants:
            lo, hi = ranges[pol]
            rows.append({
                "timestamp": ts,
                "station_id": station_meta["id"],
                "station_name": station_meta["name"],
                "city": station_meta["city"],
                "latitude": station_meta["latitude"],
                "longitude": station_meta["longitude"],
                "pollutant": pol,
                "value": round(float(rng.uniform(lo, hi)), 2),
                "unit": "µg/m³" if pol != "co" else "ppm",
                "source": "MOCK_OpenAQ",   # clearly labelled as mock
            })
    return pd.DataFrame(rows)


# ── Public entry point ────────────────────────────────────────────────────────

def collect_all_stations(
    use_mock_on_failure: bool = True,
) -> pd.DataFrame:
    """
    Collect measurements for all stations defined in config.yaml.

    Parameters
    ----------
    use_mock_on_failure : If True and the API fails, fall back to mock data.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, station_id, station_name, city,
        latitude, longitude, pollutant, value, unit, source
    """
    all_frames: list[pd.DataFrame] = []

    for station in _STATIONS:
        sid = station["id"]
        # OpenAQ location IDs are stored in config under openaq.location_ids
        # We map by position (first station → first location_id, etc.)
        station_index = _STATIONS.index(station)
        location_ids = _OAQ_CFG.get("location_ids", [])

        if station_index < len(location_ids):
            loc_id = location_ids[station_index]
        else:
            logger.warning(
                "No OpenAQ location_id configured for station=%s – using mock.", sid
            )
            all_frames.append(_generate_mock_data(station))
            continue

        # ── Fetch from API ───────────────────────────────────────────────────
        raw_records, http_status = fetch_measurements_for_location(
            location_id=loc_id,
            station_id=sid,
        )

        if raw_records:
            # Save raw response
            raw_path = save_raw_response(
                source="openaq",
                station_id=sid,
                payload=raw_records,
                api_status=http_status,
                record_count=len(raw_records),
                request_url=f"{_BASE_URL}/locations/{loc_id}/measurements",
                extra_meta={"location_id": loc_id, "city": station["city"]},
            )
            log_ingestion_run(
                source="openaq",
                station_id=sid,
                request_url=f"{_BASE_URL}/locations/{loc_id}/measurements",
                api_status_code=http_status,
                record_count=len(raw_records),
                success=True,
                raw_file_path=str(raw_path),
            )
            df = _normalise_records(raw_records, station)
            all_frames.append(df)

        elif use_mock_on_failure:
            # API returned no data or failed – use mock
            logger.warning(
                "No data from OpenAQ for station=%s (http=%d). Using mock data.",
                sid, http_status,
            )
            all_frames.append(_generate_mock_data(station))

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        logger.info(
            "OpenAQ collection complete. Total records: %d", len(combined)
        )
        return combined

    logger.warning("No OpenAQ data collected at all.")
    return pd.DataFrame()


# ── CLI entry point (run this file directly to test) ─────────────────────────

if __name__ == "__main__":
    df = collect_all_stations(use_mock_on_failure=True)
    print(f"\nCollected {len(df)} records")
    if not df.empty:
        print(df.head(10).to_string(index=False))
        print("\nPollutants:", df["pollutant"].unique().tolist())
        print("Stations:", df["station_id"].unique().tolist())
        print("Date range:", df["timestamp"].min(), "→", df["timestamp"].max())
