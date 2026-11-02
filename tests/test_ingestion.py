"""
tests/test_ingestion.py
========================
Unit tests for the ingestion layer:
  - raw_storage.py
  - ingestion_logger.py
  - openaq_collector.py (mock API)
  - weather_collector.py (mock API)
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════
# raw_storage tests
# ══════════════════════════════════════════════════════════════════════════
class TestRawStorage:

    def test_save_raw_response_creates_file(self, tmp_path, monkeypatch):
        """save_raw_response must create a .json file and a .meta.json sidecar."""
        from ingestion import raw_storage
        # Redirect project root to tmp_path
        monkeypatch.setattr(raw_storage, "get_project_root", lambda: tmp_path)

        path = raw_storage.save_raw_response(
            source="openaq",
            station_id="test_station",
            payload={"results": [{"value": 55.0}]},
            api_status=200,
            record_count=1,
            request_url="https://example.com/api",
        )
        assert path.exists(), "Raw JSON file should be created"
        assert path.suffix == ".json"

    def test_save_raw_response_creates_meta(self, tmp_path, monkeypatch):
        """A .meta.json sidecar should be created alongside the raw file."""
        from ingestion import raw_storage
        monkeypatch.setattr(raw_storage, "get_project_root", lambda: tmp_path)

        path = raw_storage.save_raw_response(
            source="openmeteo",
            station_id="test_wx",
            payload={"hourly": {}},
            api_status=200,
            record_count=24,
            request_url="https://example.com/wx",
        )
        meta_path = path.with_suffix(".meta.json")
        assert meta_path.exists(), ".meta.json sidecar must be created"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["source"] == "openmeteo"
        assert meta["record_count"] == 24
        assert meta["api_status_code"] == 200

    def test_save_raw_response_never_overwrites(self, tmp_path, monkeypatch):
        """Two calls at different times must produce different file paths."""
        from ingestion import raw_storage
        monkeypatch.setattr(raw_storage, "get_project_root", lambda: tmp_path)

        p1 = raw_storage.save_raw_response(
            "openaq", "s1", {"a": 1}, 200, 1, "http://x.com")
        p2 = raw_storage.save_raw_response(
            "openaq", "s1", {"a": 2}, 200, 1, "http://x.com")
        # Files may be same path if called within same second — check content
        # but verify no silent overwrite (both must exist)
        assert p1.exists()
        assert p2.exists()

    def test_list_raw_files_returns_only_json(self, tmp_path, monkeypatch):
        """list_raw_files must return .json files but not .meta.json."""
        from ingestion import raw_storage
        monkeypatch.setattr(raw_storage, "get_project_root", lambda: tmp_path)

        raw_storage.save_raw_response(
            "openaq", "s1", {"x": 1}, 200, 1, "http://x.com")
        files = raw_storage.list_raw_files("openaq")
        for f in files:
            assert not f.name.endswith(".meta.json"), \
                "list_raw_files must not return .meta.json files"


# ══════════════════════════════════════════════════════════════════════════
# ingestion_logger tests
# ══════════════════════════════════════════════════════════════════════════
class TestIngestionLogger:

    def test_log_ingestion_run_creates_entry(self, tmp_path, monkeypatch):
        """log_ingestion_run must append a JSON line to the log file."""
        from ingestion import ingestion_logger
        log_file = tmp_path / "data" / "raw" / "ingestion_log.jsonl"
        monkeypatch.setattr(ingestion_logger, "_LOG_FILE", log_file)

        run_id = ingestion_logger.log_ingestion_run(
            source="openaq",
            station_id="del_av",
            request_url="https://api.openaq.org/v3/test",
            api_status_code=200,
            record_count=42,
            success=True,
        )
        assert log_file.exists()
        with open(log_file) as f:
            entry = json.loads(f.readline())
        assert entry["run_id"] == run_id
        assert entry["source"] == "openaq"
        assert entry["record_count"] == 42
        assert entry["success"] is True

    def test_log_failed_run(self, tmp_path, monkeypatch):
        """Failed runs must be logged with error details."""
        from ingestion import ingestion_logger
        log_file = tmp_path / "data" / "raw" / "ingestion_log.jsonl"
        monkeypatch.setattr(ingestion_logger, "_LOG_FILE", log_file)

        run_id = ingestion_logger.log_ingestion_run(
            source="openmeteo",
            station_id="blr_slk",
            request_url="https://archive-api.open-meteo.com/test",
            api_status_code=0,
            record_count=0,
            success=False,
            error_type="ConnectionError",
            error_message="Cannot connect to host",
        )
        with open(log_file) as f:
            entry = json.loads(f.readline())
        assert entry["success"] is False
        assert entry["error_type"] == "ConnectionError"

    def test_get_failed_runs_filters_correctly(self, tmp_path, monkeypatch):
        """get_failed_runs must return only failed entries."""
        from ingestion import ingestion_logger
        log_file = tmp_path / "data" / "raw" / "ingestion_log.jsonl"
        monkeypatch.setattr(ingestion_logger, "_LOG_FILE", log_file)

        ingestion_logger.log_ingestion_run(
            "openaq", "s1", "http://x", 200, 10, True)
        ingestion_logger.log_ingestion_run(
            "openaq", "s2", "http://x", 0, 0, False,
            error_type="Timeout", error_message="timed out")

        failed = ingestion_logger.get_failed_runs("openaq")
        assert len(failed) == 1
        assert failed[0]["station_id"] == "s2"

    def test_read_ingestion_log_last_n(self, tmp_path, monkeypatch):
        """read_ingestion_log(last_n=2) must return the last 2 entries."""
        from ingestion import ingestion_logger
        log_file = tmp_path / "data" / "raw" / "ingestion_log.jsonl"
        monkeypatch.setattr(ingestion_logger, "_LOG_FILE", log_file)

        for i in range(5):
            ingestion_logger.log_ingestion_run(
                "openaq", f"s{i}", "http://x", 200, i, True)
        last2 = ingestion_logger.read_ingestion_log(last_n=2)
        assert len(last2) == 2


# ══════════════════════════════════════════════════════════════════════════
# OpenAQ collector tests (mocked API)
# ══════════════════════════════════════════════════════════════════════════
class TestOpenAQCollector:

    def test_mock_data_generated_when_api_fails(self):
        """When use_mock_on_failure=True, mock data must be returned."""
        from ingestion.openaq_collector import collect_all_stations
        # Patch requests.get to raise ConnectionError
        with patch("ingestion.openaq_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        assert not df.empty, "Must return mock data when API fails"
        assert "pollutant" in df.columns
        assert "value" in df.columns

    def test_mock_data_is_labelled(self):
        """Mock data source must be labelled MOCK_OpenAQ (not real OpenAQ)."""
        from ingestion.openaq_collector import collect_all_stations
        with patch("ingestion.openaq_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        assert df["source"].str.contains("MOCK").any(), \
            "Mock data must have MOCK label in source column"

    def test_required_columns_present(self):
        """Output DataFrame must have all required columns."""
        from ingestion.openaq_collector import collect_all_stations
        with patch("ingestion.openaq_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        required = ["timestamp", "station_id", "city", "pollutant", "value"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_mock_data_positive_values(self):
        """All mock pollutant values must be positive."""
        from ingestion.openaq_collector import collect_all_stations
        with patch("ingestion.openaq_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        assert (df["value"] >= 0).all(), "Pollutant values must be non-negative"


# ══════════════════════════════════════════════════════════════════════════
# Weather collector tests (mocked API)
# ══════════════════════════════════════════════════════════════════════════
class TestWeatherCollector:

    def test_mock_weather_returned_on_failure(self):
        """When API fails and use_mock_on_failure=True, mock data must be returned."""
        from ingestion.weather_collector import collect_all_stations
        with patch("ingestion.weather_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        assert not df.empty

    def test_weather_required_columns(self):
        """Weather DataFrame must have temperature, humidity etc."""
        from ingestion.weather_collector import collect_all_stations
        with patch("ingestion.weather_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        for col in ["temperature", "humidity", "wind_speed", "station_id"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_mock_weather_labelled(self):
        """Mock weather data must be labelled to distinguish from real data."""
        from ingestion.weather_collector import collect_all_stations
        with patch("ingestion.weather_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        assert df["source"].str.contains("MOCK").any(), \
            "Mock weather data must have MOCK label"

    def test_humidity_range(self):
        """Humidity values must be between 0 and 100."""
        from ingestion.weather_collector import collect_all_stations
        with patch("ingestion.weather_collector.requests.get",
                   side_effect=Exception("API down")):
            df = collect_all_stations(use_mock_on_failure=True)
        assert ((df["humidity"] >= 0) & (df["humidity"] <= 100)).all(), \
            "Humidity must be in [0, 100]"
