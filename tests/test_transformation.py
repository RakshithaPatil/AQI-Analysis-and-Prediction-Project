"""tests/test_transformation.py — Unit tests for cleaning, validation, AQI."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformation.aqi_calculation import (
    calculate_sub_aqi, calculate_overall_aqi, get_aqi_category,
)
from transformation.cleaning import (
    normalise_timestamps, normalise_pollutant_names,
    normalise_units, deduplicate, calculate_daily_averages,
)


# ── AQI Calculation ───────────────────────────────────────────────────────
class TestAQICalculation:
    def test_good_pm25(self):
        assert calculate_sub_aqi("pm25", 10.0) <= 50

    def test_moderate_pm25(self):
        sub = calculate_sub_aqi("pm25", 20.0)
        assert 51 <= sub <= 100

    def test_unhealthy_pm25(self):
        sub = calculate_sub_aqi("pm25", 100.0)
        assert 151 <= sub <= 200

    def test_nan_returns_none(self):
        assert calculate_sub_aqi("pm25", np.nan) is None

    def test_negative_returns_none(self):
        assert calculate_sub_aqi("pm25", -1.0) is None

    def test_unknown_pollutant_returns_none(self):
        assert calculate_sub_aqi("xyz", 50.0) is None

    def test_overall_aqi_dominant(self):
        aqi, cat, dom = calculate_overall_aqi(pm25=200, pm10=50)
        assert dom == "pm25"
        assert aqi > 150

    def test_overall_aqi_all_none(self):
        aqi, cat, dom = calculate_overall_aqi()
        assert aqi is None

    def test_aqi_category_good(self):
        assert get_aqi_category(25) == "Good"

    def test_aqi_category_hazardous(self):
        assert get_aqi_category(400) == "Hazardous"

    def test_aqi_category_nan(self):
        assert get_aqi_category(np.nan) == "Unknown"


# ── Cleaning ──────────────────────────────────────────────────────────────
class TestCleaning:
    def _sample_df(self):
        return pd.DataFrame({
            "timestamp": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z"],
            "station_id": ["DEL_AV", "DEL_AV"],
            "station_name": [" anand vihar ", "ANAND VIHAR"],
            "city": ["delhi", "DELHI"],
            "pollutant": ["PM2.5", "pm25"],
            "value": [85.0, 90.0],
            "unit": ["ug/m3", "UG/M3"],
        })

    def test_normalise_timestamps(self):
        df = self._sample_df()
        out = normalise_timestamps(df)
        assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])

    def test_normalise_pollutant_names(self):
        df = self._sample_df()
        out = normalise_pollutant_names(df)
        assert (out["pollutant"] == "pm25").all()

    def test_normalise_units(self):
        df = self._sample_df()
        out = normalise_units(df)
        assert (out["unit"] == "µg/m³").all()

    def test_deduplicate(self):
        df = pd.DataFrame({
            "station_id": ["A","A","A"],
            "timestamp": ["2023-01-01","2023-01-01","2023-01-02"],
            "pollutant": ["pm25","pm25","pm25"],
            "value": [1,2,3],
        })
        out = deduplicate(df, subset=["station_id","timestamp","pollutant"])
        assert len(out) == 2

    def test_daily_averages_shape(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2023-01-01T00:00:00Z","2023-01-01T12:00:00Z",
                 "2023-01-02T06:00:00Z"], utc=True),
            "station_id": ["A","A","A"],
            "pollutant":  ["pm25","pm25","pm25"],
            "value": [80.0, 100.0, 90.0],
        })
        daily = calculate_daily_averages(df)
        assert len(daily) == 2   # 2 unique dates
        assert daily["value"].iloc[0] == pytest.approx(90.0)  # avg of 80+100


# ── Integration smoke test ────────────────────────────────────────────────
class TestIntegration:
    def test_pipeline_output_exists(self):
        """Verify the pipeline has produced the processed output file."""
        csv = ROOT / "data" / "processed" / "aqi_processed.csv"
        assert csv.exists(), (
            "aqi_processed.csv not found. Run: python run_pipeline.py"
        )

    def test_pipeline_output_has_expected_columns(self):
        csv = ROOT / "data" / "processed" / "aqi_processed.csv"
        if not csv.exists():
            pytest.skip("Pipeline not run yet")
        df = pd.read_csv(csv, nrows=10)
        for col in ["aqi","aqi_category","pm25","pm10","station_id","date"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_models_exist(self):
        """Verify trained models are present."""
        for fname in ["best_regressor.joblib","best_classifier.joblib",
                      "model_info.json"]:
            path = ROOT / "models" / fname
            assert path.exists(), f"Model file missing: {fname}"
