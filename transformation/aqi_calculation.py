"""
transformation/aqi_calculation.py
===================================
Calculates the Air Quality Index (AQI) using the US EPA standard.

Standard:
  United States Environmental Protection Agency (US EPA)
  "Technical Assistance Document for the Reporting of Daily Air Quality –
   the Air Quality Index (AQI)"
  Reference: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf

Formula:
  AQI_p = ((I_high - I_low) / (C_high - C_low)) × (C_p - C_low) + I_low

  Where:
    C_p     = measured concentration of pollutant p
    C_low   = breakpoint concentration ≤ C_p
    C_high  = breakpoint concentration ≥ C_p
    I_low   = AQI index corresponding to C_low
    I_high  = AQI index corresponding to C_high

  The overall AQI is the maximum AQI across all pollutants calculated.
  The pollutant with the highest sub-index is the "dominant pollutant".

Breakpoints:
  All breakpoints are taken directly from the EPA document cited above.
  Units: PM2.5 and PM10 in µg/m³ (24-hr average), O3 in ppb (8-hr avg),
         CO in ppm (8-hr avg), SO2 in ppb (1-hr avg), NO2 in ppb (1-hr avg).

NOTE ON UNIT CONVERSIONS:
  This implementation works with µg/m³ values for all pollutants as that
  is what OpenAQ typically provides. Where the EPA standard uses ppb,
  the breakpoints below have been converted to µg/m³ at standard conditions
  (25°C, 1 atm) using accepted molecular-weight conversion factors.
  If your data is already in ppb, set use_ppb=True in calculate_sub_aqi().

AQI Categories:
  0–50     : Good
  51–100   : Moderate
  101–150  : Unhealthy for Sensitive Groups
  151–200  : Unhealthy
  201–300  : Very Unhealthy
  301–500  : Hazardous
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config.logger import get_logger

logger = get_logger(__name__, log_file="aqi.log")


# ══════════════════════════════════════════════════════════════════════════════
# US-EPA Breakpoint Tables
# All concentration values are in µg/m³ (converted from ppb where needed).
# Source: EPA AQI Technical Assistance Document (Sept 2018), Table 5.
# ══════════════════════════════════════════════════════════════════════════════

# Format: list of (C_low, C_high, I_low, I_high)
# The last row intentionally sets C_high = 9999 as an open-ended upper bound.

_BREAKPOINTS: dict[str, list[tuple]] = {
    # PM2.5 – 24-hr average, µg/m³
    "pm25": [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
        (500.5, 9999,  501, 999),  # Beyond index
    ],
    # PM10 – 24-hr average, µg/m³
    "pm10": [
        (0,    54,    0,   50),
        (55,   154,  51,  100),
        (155,  254, 101,  150),
        (255,  354, 151,  200),
        (355,  424, 201,  300),
        (425,  504, 301,  400),
        (505,  604, 401,  500),
        (605, 9999, 501,  999),
    ],
    # O3 – 8-hr average, µg/m³ (converted from ppb × 2.00 at 25°C)
    "o3": [
        (0,    107,    0,   50),
        (108,  140,   51,  100),
        (141,  170,  101,  150),
        (171,  210,  151,  200),
        (211,  400,  201,  300),
        (401, 9999,  301,  500),
    ],
    # CO – 8-hr average, µg/m³ (1 ppm = 1145 µg/m³ at 25°C)
    "co": [
        (0,      5175,    0,   50),
        (5176,  11495,   51,  100),
        (11496, 13760,  101,  150),
        (13761, 17225,  151,  200),
        (17226, 34450,  201,  300),
        (34451, 57500,  301,  400),
        (57501, 9999999,401,  500),
    ],
    # SO2 – 1-hr average, µg/m³ (1 ppb = 2.62 µg/m³ at 25°C)
    "so2": [
        (0,     91,    0,   50),
        (92,    196,  51,  100),
        (197,   484,  101, 150),
        (485,  793,   151, 200),
        (794, 1584,   201, 300),
        (1585, 2629,  301, 400),
        (2630, 9999,  401, 500),
    ],
    # NO2 – 1-hr average, µg/m³ (1 ppb = 1.88 µg/m³ at 25°C)
    "no2": [
        (0,     100,    0,   50),
        (101,   188,   51,  100),
        (189,   677,  101,  150),
        (678,  1222,  151,  200),
        (1223, 2349,  201,  300),
        (2350, 3100,  301,  400),
        (3101, 9999,  401,  500),
    ],
}

# AQI category thresholds
_AQI_CATEGORIES: list[tuple[int, int, str]] = [
    (0,   50,  "Good"),
    (51,  100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
    (501, 999, "Beyond Index"),
]


# ── Core formula ──────────────────────────────────────────────────────────────

def calculate_sub_aqi(pollutant: str, concentration: float) -> Optional[float]:
    """
    Calculate the AQI sub-index for a single pollutant at a given concentration.

    Parameters
    ----------
    pollutant     : Canonical pollutant name ('pm25', 'pm10', 'o3', 'co', 'so2', 'no2')
    concentration : Measured concentration in µg/m³ (or ppm for CO)

    Returns
    -------
    Sub-AQI value as float, or None if the pollutant is not in the breakpoint table
    or the concentration is NaN / negative.
    """
    if pollutant not in _BREAKPOINTS:
        return None
    if concentration is None or np.isnan(concentration) or concentration < 0:
        return None

    breakpoints = _BREAKPOINTS[pollutant]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            # Linear interpolation formula (EPA Equation 1)
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi, 2)

    # Concentration exceeds all breakpoints – cap at 500
    logger.warning(
        "Concentration %.2f for %s exceeds all breakpoints. Capping AQI at 500.",
        concentration, pollutant,
    )
    return 500.0


def get_aqi_category(aqi: float) -> str:
    """
    Return the AQI category label for a given AQI value.

    Parameters
    ----------
    aqi : Numeric AQI value

    Returns
    -------
    Category string (e.g. "Good", "Moderate", "Unhealthy")
    """
    if aqi is None or np.isnan(aqi):
        return "Unknown"
    for lo, hi, label in _AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Beyond Index"


# ── Row-level AQI calculation ─────────────────────────────────────────────────

def calculate_overall_aqi(
    pm25: Optional[float] = None,
    pm10: Optional[float] = None,
    o3: Optional[float] = None,
    co: Optional[float] = None,
    so2: Optional[float] = None,
    no2: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Calculate the overall AQI for a set of pollutant concentrations.

    The overall AQI equals the maximum sub-AQI across all provided pollutants.
    The pollutant with the highest sub-AQI is the "dominant pollutant".

    Parameters
    ----------
    pm25, pm10, o3, co, so2, no2 : Concentrations in µg/m³

    Returns
    -------
    (aqi, category, dominant_pollutant)
      aqi                : Overall AQI value (float) or None
      category           : AQI category string or None
      dominant_pollutant : Pollutant name driving the AQI or None
    """
    pollutant_values = {
        "pm25": pm25,
        "pm10": pm10,
        "o3": o3,
        "co": co,
        "so2": so2,
        "no2": no2,
    }

    sub_aqis: dict[str, float] = {}
    for pol, conc in pollutant_values.items():
        if conc is not None and not np.isnan(conc):
            sub = calculate_sub_aqi(pol, conc)
            if sub is not None:
                sub_aqis[pol] = sub

    if not sub_aqis:
        return None, None, None

    dominant = max(sub_aqis, key=sub_aqis.get)
    overall_aqi = sub_aqis[dominant]
    category = get_aqi_category(overall_aqi)
    return round(overall_aqi, 2), category, dominant


# ── DataFrame-level AQI calculation ──────────────────────────────────────────

def calculate_aqi_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AQI columns to a daily-level wide-format DataFrame.

    Expected input columns (from daily pivot):
        date, station_id, pm25, pm10, o3, co, so2, no2

    Added output columns:
        aqi, aqi_category, dominant_pollutant

    Parameters
    ----------
    df : Wide-format daily DataFrame with one row per (date, station_id)

    Returns
    -------
    DataFrame with aqi, aqi_category, dominant_pollutant columns added.
    """
    df = df.copy()

    def _row_aqi(row: pd.Series) -> pd.Series:
        aqi, cat, dom = calculate_overall_aqi(
            pm25=row.get("pm25"),
            pm10=row.get("pm10"),
            o3=row.get("o3"),
            co=row.get("co"),
            so2=row.get("so2"),
            no2=row.get("no2"),
        )
        return pd.Series({
            "aqi": aqi,
            "aqi_category": cat,
            "dominant_pollutant": dom,
        })

    aqi_cols = df.apply(_row_aqi, axis=1)
    df["aqi"] = aqi_cols["aqi"]
    df["aqi_category"] = aqi_cols["aqi_category"]
    df["dominant_pollutant"] = aqi_cols["dominant_pollutant"]

    n_calculated = df["aqi"].notna().sum()
    n_failed = df["aqi"].isna().sum()
    logger.info(
        "AQI calculated | rows=%d | successful=%d | failed=%d",
        len(df), n_calculated, n_failed,
    )
    return df


def pivot_daily_pollutants(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long-format daily DataFrame (date, station_id, pollutant, value)
    to wide format (one column per pollutant) needed for AQI calculation.

    Parameters
    ----------
    daily_df : Long-format DataFrame with columns:
               date, station_id, pollutant, value

    Returns
    -------
    Wide-format DataFrame:
        date, station_id, pm25, pm10, no2, so2, co, o3
    """
    wide = daily_df.pivot_table(
        index=["date", "station_id"],
        columns="pollutant",
        values="value",
        aggfunc="mean",
    ).reset_index()

    wide.columns.name = None  # Remove the 'pollutant' axis name
    # Ensure all expected pollutant columns exist (fill with NaN if missing)
    for pol in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if pol not in wide.columns:
            wide[pol] = np.nan

    logger.info("pivot_daily_pollutants: %d rows", len(wide))
    return wide


# ── CLI / quick-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AQI Sub-index Tests ===")
    test_cases = [
        ("pm25",  12.0),   # Good boundary
        ("pm25",  35.4),   # Moderate boundary
        ("pm25",  55.4),   # USG boundary
        ("pm25", 150.4),   # Unhealthy boundary
        ("pm10",  54.0),   # Good boundary
        ("o3",   107.0),   # Good
        ("co",   5175.0),  # Good
        ("no2",  100.0),   # Good boundary
    ]
    for pol, conc in test_cases:
        sub = calculate_sub_aqi(pol, conc)
        cat = get_aqi_category(sub) if sub is not None else "N/A"
        print(f"  {pol:6s} @ {conc:8.1f} µg/m³  →  AQI={sub:6.1f}  ({cat})")

    print("\n=== Overall AQI Test ===")
    aqi, cat, dom = calculate_overall_aqi(pm25=85, pm10=120, no2=35, so2=10, co=0.8, o3=40)
    print(f"  AQI={aqi}  Category='{cat}'  Dominant='{dom}'")
