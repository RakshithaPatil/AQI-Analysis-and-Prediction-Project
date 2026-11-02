"""
ingestion/ingestion_logger.py
==============================
Records every ingestion run as a structured log entry in:
    data/raw/ingestion_log.jsonl   (one JSON object per line)

Each entry captures:
  - run_id              unique ID for this run
  - run_timestamp_utc   when the run started
  - source              openaq | openmeteo
  - station_id          which station was collected
  - request_url         exact API URL called
  - api_status_code     HTTP response code
  - record_count        how many records were received
  - success             True / False
  - error_type          exception class name if failed
  - error_message       exception message if failed
  - raw_file_path       path to saved raw file (if success)
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.logger import get_logger
from config.settings import get_project_root

logger = get_logger(__name__, log_file="ingestion.log")

# Path to the append-only JSONL ingestion log
_LOG_FILE = get_project_root() / "data" / "raw" / "ingestion_log.jsonl"


def _ensure_log_file() -> None:
    """Make sure the parent directory and log file exist."""
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_FILE.exists():
        _LOG_FILE.touch()


def log_ingestion_run(
    source: str,
    station_id: str,
    request_url: str,
    api_status_code: int,
    record_count: int,
    success: bool,
    raw_file_path: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """
    Append one ingestion-run record to the JSONL log file.

    Parameters
    ----------
    source           : Data source label ("openaq" / "openmeteo")
    station_id       : Station or location identifier
    request_url      : The full URL that was requested
    api_status_code  : HTTP status code (0 if connection failed)
    record_count     : Number of records received
    success          : Whether the run completed without error
    raw_file_path    : Path to saved raw file (None if failed)
    error_type       : Exception class name (None if success)
    error_message    : Exception message (None if success)
    extra            : Any additional metadata to include

    Returns
    -------
    run_id : str  — unique identifier for this run
    """
    _ensure_log_file()

    run_id = str(uuid.uuid4())
    entry = {
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "station_id": station_id,
        "request_url": request_url,
        "api_status_code": api_status_code,
        "record_count": record_count,
        "success": success,
        "raw_file_path": raw_file_path,
        "error_type": error_type,
        "error_message": error_message,
    }
    if extra:
        entry.update(extra)

    # Append as a single line to the JSONL file
    with open(_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")

    if success:
        logger.info(
            "Ingestion OK | run_id=%s | source=%s | station=%s | records=%d",
            run_id, source, station_id, record_count,
        )
    else:
        logger.error(
            "Ingestion FAILED | run_id=%s | source=%s | station=%s | error=%s: %s",
            run_id, source, station_id, error_type, error_message,
        )

    return run_id


def read_ingestion_log(last_n: Optional[int] = None) -> list[dict]:
    """
    Read the ingestion log and return as a list of dicts.

    Parameters
    ----------
    last_n : If set, return only the last N entries.

    Returns
    -------
    List of ingestion run dicts
    """
    _ensure_log_file()
    records = []
    with open(_LOG_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip corrupted lines
    if last_n is not None:
        return records[-last_n:]
    return records


def get_failed_runs(source: Optional[str] = None) -> list[dict]:
    """Return all failed ingestion runs, optionally filtered by source."""
    all_runs = read_ingestion_log()
    failed = [r for r in all_runs if not r.get("success", True)]
    if source:
        failed = [r for r in failed if r.get("source") == source]
    return failed
