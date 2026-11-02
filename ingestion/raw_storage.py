"""
ingestion/raw_storage.py
========================
Handles saving raw API responses to the data/raw/ landing zone.

Rules enforced here:
  1. Raw files are NEVER overwritten – each run creates a timestamped file.
  2. The directory structure is: data/raw/<source>/<YYYY-MM-DD>/
  3. Every saved file has a companion .meta.json sidecar with extraction info.
  4. No transformation is done here – data is stored exactly as received.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.logger import get_logger
from config.settings import get_project_root

logger = get_logger(__name__, log_file="ingestion.log")


def _build_raw_path(source: str, station_id: str, run_ts: datetime) -> Path:
    """
    Build the file path for a raw data file.

    Pattern: data/raw/<source>/<YYYY-MM-DD>/<station_id>_<HH-MM-SS>.json

    Parameters
    ----------
    source     : Data source label, e.g. "openaq" or "openmeteo"
    station_id : Station / location identifier string
    run_ts     : datetime of the extraction run (UTC)

    Returns
    -------
    Path object (file does NOT yet exist)
    """
    date_str = run_ts.strftime("%Y-%m-%d")
    time_str = run_ts.strftime("%H-%M-%S")
    directory = get_project_root() / "data" / "raw" / source / date_str
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{station_id}_{time_str}.json"
    return directory / filename


def save_raw_response(
    source: str,
    station_id: str,
    payload: Any,
    api_status: int,
    record_count: int,
    request_url: str,
    extra_meta: dict | None = None,
) -> Path:
    """
    Save a raw API response payload to disk and write a metadata sidecar.

    Parameters
    ----------
    source        : "openaq" or "openmeteo"
    station_id    : Station / location ID string
    payload       : The raw data to persist (dict, list, or str)
    api_status    : HTTP status code returned by the API
    record_count  : Number of records received in this response
    request_url   : The exact URL that was called
    extra_meta    : Any additional key-value pairs to store in the sidecar

    Returns
    -------
    Path to the saved raw data file
    """
    run_ts = datetime.now(timezone.utc)
    raw_path = _build_raw_path(source, station_id, run_ts)

    # ── Save raw payload ─────────────────────────────────────────────────────
    with open(raw_path, "w", encoding="utf-8") as fh:
        if isinstance(payload, (dict, list)):
            json.dump(payload, fh, indent=2, default=str)
        else:
            fh.write(str(payload))

    logger.info(
        "Raw data saved | source=%s | station=%s | records=%d | file=%s",
        source,
        station_id,
        record_count,
        raw_path.name,
    )

    # ── Write metadata sidecar ───────────────────────────────────────────────
    meta = {
        "extraction_timestamp_utc": run_ts.isoformat(),
        "source": source,
        "station_id": station_id,
        "request_url": request_url,
        "api_status_code": api_status,
        "record_count": record_count,
        "raw_file": str(raw_path),
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = raw_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return raw_path


def list_raw_files(source: str, date_str: str | None = None) -> list[Path]:
    """
    Return a list of raw data file paths for a given source.

    Parameters
    ----------
    source   : "openaq" or "openmeteo"
    date_str : Optional ISO date string "YYYY-MM-DD". If omitted, lists all dates.

    Returns
    -------
    Sorted list of .json (not .meta.json) file paths
    """
    raw_dir = get_project_root() / "data" / "raw" / source
    if not raw_dir.exists():
        return []

    if date_str:
        search_root = raw_dir / date_str
        if not search_root.exists():
            return []
    else:
        search_root = raw_dir

    files = sorted(
        [p for p in search_root.rglob("*.json") if not p.name.endswith(".meta.json")]
    )
    return files
