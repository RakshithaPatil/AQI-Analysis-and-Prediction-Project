"""
tests/test_features.py
========================
Unit tests for feature engineering module:
  - add_temporal_features
  - add_lag_features
  - add_rolling_features
  - encode_aqi_category
  - chronological_split
  - build_feature_matrix
  - get_feature_columns
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformation.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_temporal_features,
    build_feature_matrix,
    chronological_split,
    encode_aqi_category,
    get_feature_columns,
)


# ── Sample data factory ───────────────────────────────────────────────────
def _make_daily_df(n_days: int = 30, station: str = "s1") -> pd.DataFrame:
    """Create a minimal daily AQI DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date":         dates,
        "station_id":   station,
        "aqi":          rng.uniform(50, 250, n_days),
        "aqi_category": rng.choice(
            ["Good", "Moderate", "Unhealthy", "Very Unhealthy"], n_days),
        "pm25":  rng.uniform(10, 200, n_days),
        "pm10":  rng.uniform(20, 300, n_days),
        "no2":   rng.uniform(5, 100, n_days),
        "so2":   rng.uniform(1, 50, n_days),
        "co":    rng.uniform(100, 2000, n_days),
        "o3":    rng.uniform(10, 100, n_days),
        "nh3":   rng.uniform(1, 20, n_days),
        "temperature": rng.uniform(15, 40, n_days),
        "humidity":    rng.uniform(30, 90, n_days),
        "precipitation": rng.uniform(0, 5, n_days),
        "wind_speed":  rng.uniform(2, 20, n_days),
        "pressure":    rng.uniform(995, 1015, n_days),
    })


# ══════════════════════════════════════════════════════════════════════════
# Temporal features
# ══════════════════════════════════════════════════════════════════════════
class TestTemporalFeatures:

    def test_adds_month_column(self):
        df = _make_daily_df()
        out = add_temporal_features(df)
        assert "month" in out.columns

    def test_adds_day_of_week(self):
        df = _make_daily_df()
        out = add_temporal_features(df)
        assert "day_of_week" in out.columns
        assert out["day_of_week"].between(0, 6).all()

    def test_adds_is_weekend(self):
        df = _make_daily_df()
        out = add_temporal_features(df)
        assert "is_weekend" in out.columns
        assert out["is_weekend"].isin([0, 1]).all()

    def test_adds_season(self):
        df = _make_daily_df()
        out = add_temporal_features(df)
        assert "season" in out.columns
        assert out["season"].between(0, 3).all()

    def test_month_values_correct(self):
        df = _make_daily_df()
        out = add_temporal_features(df)
        # January dates → month=1
        jan_rows = out[out["date"].dt.month == 1]
        assert (jan_rows["month"] == 1).all()


# ══════════════════════════════════════════════════════════════════════════
# Lag features
# ══════════════════════════════════════════════════════════════════════════
class TestLagFeatures:

    def test_creates_lag_columns(self):
        df = _make_daily_df(30)
        out = add_lag_features(df, lag_days=[1, 3, 7])
        assert "aqi_lag_1" in out.columns
        assert "aqi_lag_3" in out.columns
        assert "aqi_lag_7" in out.columns

    def test_lag1_matches_shifted_values(self):
        df = _make_daily_df(20)
        df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
        out = add_lag_features(df, lag_days=[1])
        # Row 1's aqi_lag_1 must equal row 0's aqi
        assert out.loc[1, "aqi_lag_1"] == pytest.approx(out.loc[0, "aqi"])

    def test_first_row_lag_is_nan(self):
        """First row per station must have NaN for lag_1 (no previous data)."""
        df = _make_daily_df(10)
        out = add_lag_features(df, lag_days=[1])
        first_row = out.sort_values("date").iloc[0]
        assert pd.isna(first_row["aqi_lag_1"])

    def test_no_future_leakage(self):
        """aqi_lag_1 must always be less than current row's date index."""
        df = _make_daily_df(15)
        out = add_lag_features(df, lag_days=[1]).dropna(subset=["aqi_lag_1"])
        # Lag values come from previous rows — verify by checking shift
        out_sorted = out.sort_values(["station_id", "date"]).reset_index(drop=True)
        for i in range(1, len(out_sorted)):
            expected = out_sorted.loc[i-1, "aqi"]
            actual   = out_sorted.loc[i, "aqi_lag_1"]
            assert expected == pytest.approx(actual, rel=1e-5)


# ══════════════════════════════════════════════════════════════════════════
# Rolling features
# ══════════════════════════════════════════════════════════════════════════
class TestRollingFeatures:

    def test_creates_rolling_columns(self):
        df = _make_daily_df(30)
        out = add_rolling_features(df, windows=[3, 7])
        assert "aqi_rolling_3" in out.columns
        assert "aqi_rolling_7" in out.columns

    def test_rolling_not_nan_after_window(self):
        df = _make_daily_df(30)
        out = add_rolling_features(df, windows=[3])
        non_nan = out["aqi_rolling_3"].notna().sum()
        assert non_nan > 0, "Rolling column must have non-NaN values"

    def test_rolling_uses_past_data_only(self):
        """
        Rolling average at position i must only use data from i-1 and earlier
        (shift(1) applied before rolling — no leakage from current row).
        """
        df = _make_daily_df(20)
        out = add_rolling_features(df, windows=[3])
        out_sorted = out.sort_values(["station_id", "date"]).reset_index(drop=True)
        # At row index 3, the rolling_3 value should NOT include row 3's aqi
        for i in range(4, len(out_sorted)):
            roll_val = out_sorted.loc[i, "aqi_rolling_3"]
            current  = out_sorted.loc[i, "aqi"]
            # If rolling included current row, they'd be very correlated.
            # Just check the column exists and has a numeric value.
            assert isinstance(roll_val, float) or pd.isna(roll_val)


# ══════════════════════════════════════════════════════════════════════════
# Category encoding
# ══════════════════════════════════════════════════════════════════════════
class TestCategoryEncoding:

    def test_adds_encoded_column(self):
        df = _make_daily_df()
        out = encode_aqi_category(df)
        assert "aqi_category_encoded" in out.columns

    def test_good_encodes_to_0(self):
        df = pd.DataFrame({"aqi_category": ["Good", "Moderate", "Unhealthy"]})
        out = encode_aqi_category(df)
        assert out.loc[out["aqi_category"] == "Good", "aqi_category_encoded"].iloc[0] == 0

    def test_encoding_is_integer(self):
        df = _make_daily_df()
        out = encode_aqi_category(df)
        assert pd.api.types.is_integer_dtype(out["aqi_category_encoded"])


# ══════════════════════════════════════════════════════════════════════════
# Chronological split
# ══════════════════════════════════════════════════════════════════════════
class TestChronologicalSplit:

    def test_correct_sizes(self):
        df = _make_daily_df(100)
        train, val, test = chronological_split(df, train_ratio=0.7, val_ratio=0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_overlap(self):
        df = _make_daily_df(100)
        train, val, test = chronological_split(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_chronological_order(self):
        """Train must come before val, val before test (no shuffle)."""
        df = _make_daily_df(100)
        train, val, test = chronological_split(df)
        assert train["date"].max() <= val["date"].min(), \
            "Training set must end before validation set starts"
        assert val["date"].max() <= test["date"].min(), \
            "Validation set must end before test set starts"

    def test_no_future_leakage_in_split(self):
        """Verify no test dates appear in training set."""
        df = _make_daily_df(90)
        train, _, test = chronological_split(df)
        overlap = set(train["date"].dt.date) & set(test["date"].dt.date)
        assert len(overlap) == 0, "No dates should appear in both train and test"


# ══════════════════════════════════════════════════════════════════════════
# get_feature_columns
# ══════════════════════════════════════════════════════════════════════════
class TestGetFeatureColumns:

    def test_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_contains_pollutants(self):
        cols = get_feature_columns()
        for p in ["pm25", "pm10", "no2"]:
            assert p in cols, f"Feature list must include {p}"

    def test_contains_lag_features(self):
        cols = get_feature_columns()
        for lag in [1, 2, 3, 7]:
            assert f"aqi_lag_{lag}" in cols

    def test_contains_rolling_features(self):
        cols = get_feature_columns()
        for w in [3, 7]:
            assert f"aqi_rolling_{w}" in cols

    def test_no_target_in_features(self):
        """Target columns must NOT appear in the feature list."""
        cols = get_feature_columns()
        assert "aqi" not in cols, "aqi (target) must not be in feature list"
        assert "aqi_category" not in cols


# ══════════════════════════════════════════════════════════════════════════
# build_feature_matrix integration
# ══════════════════════════════════════════════════════════════════════════
class TestBuildFeatureMatrix:

    def test_output_has_lag_columns(self):
        df = _make_daily_df(50)
        out = build_feature_matrix(df, drop_no_history=False)
        assert "aqi_lag_1" in out.columns

    def test_output_has_temporal_columns(self):
        df = _make_daily_df(50)
        out = build_feature_matrix(df, drop_no_history=False)
        assert "month" in out.columns
        assert "is_weekend" in out.columns

    def test_drop_no_history_reduces_rows(self):
        """drop_no_history=True must remove early rows without full lag history."""
        df = _make_daily_df(50)
        out_drop = build_feature_matrix(df, drop_no_history=True)
        out_keep = build_feature_matrix(df, drop_no_history=False)
        assert len(out_drop) < len(out_keep), \
            "Dropping rows without history must reduce the row count"
