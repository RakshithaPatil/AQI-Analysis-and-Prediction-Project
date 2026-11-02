"""
transformation package
=======================
Modules that clean, validate, transform, and enrich raw ingested data.

  cleaning.py          – null handling, dedup, timestamp normalisation, unit standardisation
  validation.py        – range checks, coordinate checks, completeness checks
  feature_engineering  – lag/rolling features for ML
  aqi_calculation.py   – US-EPA AQI formula and category assignment
"""
