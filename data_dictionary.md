# 📖 Data Dictionary
## Air Quality Index Analysis and Prediction

This document describes every important column in the project's datasets,
PostgreSQL tables, and ML feature matrix.

---

## 1. Raw Dataset — `india_aqi_dataset.csv`

| Column | Data Type | Description | Source | Allowed Range | Nullable |
|--------|-----------|-------------|--------|---------------|----------|
| `date` | DATE (YYYY-MM-DD) | Date of observation | Generated / OpenAQ | 2022-01-01 → present | NOT NULL |
| `station_id` | VARCHAR | Unique station identifier (e.g. DEL_AV) | Config | — | NOT NULL |
| `station_name` | VARCHAR | Human-readable station name | Config | — | NOT NULL |
| `city` | VARCHAR | City where station is located | Config | — | NOT NULL |
| `state` | VARCHAR | State/province | Config | — | NOT NULL |
| `latitude` | FLOAT | Station latitude (WGS84) | Config | -90 to 90 | NOT NULL |
| `longitude` | FLOAT | Station longitude (WGS84) | Config | -180 to 180 | NOT NULL |
| `pm25` | FLOAT | PM2.5 concentration (µg/m³) | OpenAQ | 0 – 1000 | NULLABLE |
| `pm10` | FLOAT | PM10 concentration (µg/m³) | OpenAQ | 0 – 1500 | NULLABLE |
| `no2` | FLOAT | Nitrogen dioxide (µg/m³) | OpenAQ | 0 – 2000 | NULLABLE |
| `so2` | FLOAT | Sulfur dioxide (µg/m³) | OpenAQ | 0 – 2620 | NULLABLE |
| `co` | FLOAT | Carbon monoxide (µg/m³) | OpenAQ | 0 – 50000 | NULLABLE |
| `o3` | FLOAT | Ozone (µg/m³) | OpenAQ | 0 – 600 | NULLABLE |
| `nh3` | FLOAT | Ammonia (µg/m³) | OpenAQ | 0 – 400 | NULLABLE |
| `aqi` | FLOAT | Calculated Air Quality Index | Calculated | 0 – 500 | NULLABLE |
| `aqi_category` | VARCHAR | AQI category label | Calculated | See §5 | NULLABLE |
| `dominant_pollutant` | VARCHAR | Pollutant driving the AQI | Calculated | pm25/pm10/no2/so2/co/o3 | NULLABLE |
| `temperature` | FLOAT | Air temperature (°C) | Open-Meteo | -60 to 60 | NULLABLE |
| `humidity` | FLOAT | Relative humidity (%) | Open-Meteo | 0 to 100 | NULLABLE |
| `precipitation` | FLOAT | Precipitation (mm) | Open-Meteo | 0 to 500 | NULLABLE |
| `wind_speed` | FLOAT | Wind speed (km/h) | Open-Meteo | 0 to 120 | NULLABLE |
| `wind_direction` | FLOAT | Wind direction (degrees 0–360) | Open-Meteo | 0 to 360 | NULLABLE |
| `pressure` | FLOAT | Atmospheric pressure (hPa) | Open-Meteo | 870 to 1085 | NULLABLE |

---

## 2. PostgreSQL Table: `stations`

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `station_id` | VARCHAR(100) | Unique station identifier | PRIMARY KEY |
| `station_name` | VARCHAR(255) | Display name of the station | NOT NULL |
| `city` | VARCHAR(100) | City where station is located | — |
| `latitude` | NUMERIC(9,6) | WGS84 latitude | CHECK (-90 to 90) |
| `longitude` | NUMERIC(9,6) | WGS84 longitude | CHECK (-180 to 180) |
| `created_at` | TIMESTAMPTZ | Row creation timestamp | DEFAULT NOW() |
| `updated_at` | TIMESTAMPTZ | Last update timestamp | DEFAULT NOW() |

---

## 3. PostgreSQL Table: `pollution_measurements`

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `id` | BIGSERIAL | Auto-increment primary key | PRIMARY KEY |
| `station_id` | VARCHAR(100) | Reference to stations table | FK → stations |
| `timestamp` | TIMESTAMPTZ | Observation timestamp (UTC) | NOT NULL |
| `pollutant` | VARCHAR(20) | Canonical pollutant name | CHECK IN (pm25,pm10,no2,so2,co,o3) |
| `value` | NUMERIC(12,4) | Measured concentration (µg/m³) | CHECK ≥ 0 |
| `unit` | VARCHAR(20) | Measurement unit | — |
| `source` | VARCHAR(50) | Data source label | — |
| `created_at` | TIMESTAMPTZ | Row creation timestamp | DEFAULT NOW() |

**Unique constraint:** `(station_id, timestamp, pollutant)`

---

## 4. PostgreSQL Table: `weather_measurements`

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `id` | BIGSERIAL | Auto-increment primary key | PRIMARY KEY |
| `station_id` | VARCHAR(100) | Reference to stations table | FK → stations |
| `timestamp` | TIMESTAMPTZ | Observation timestamp (UTC) | NOT NULL |
| `temperature` | NUMERIC(6,2) | Air temperature (°C) | — |
| `humidity` | NUMERIC(6,2) | Relative humidity (%) | — |
| `precipitation` | NUMERIC(8,4) | Precipitation (mm) | — |
| `wind_speed` | NUMERIC(7,3) | Wind speed (km/h) | — |
| `wind_direction` | NUMERIC(6,2) | Wind direction (degrees) | — |
| `pressure` | NUMERIC(8,2) | Atmospheric pressure (hPa) | — |
| `source` | VARCHAR(50) | Data source (Open-Meteo) | — |

**Unique constraint:** `(station_id, timestamp)`

---

## 5. PostgreSQL Table: `daily_aqi`

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `id` | BIGSERIAL | Auto-increment primary key | PRIMARY KEY |
| `station_id` | VARCHAR(100) | Reference to stations table | FK → stations |
| `date` | DATE | Date of the AQI record | NOT NULL |
| `pm25` | NUMERIC(10,4) | Daily avg PM2.5 (µg/m³) | — |
| `pm10` | NUMERIC(10,4) | Daily avg PM10 (µg/m³) | — |
| `no2` | NUMERIC(10,4) | Daily avg NO2 (µg/m³) | — |
| `so2` | NUMERIC(10,4) | Daily avg SO2 (µg/m³) | — |
| `co` | NUMERIC(10,6) | Daily avg CO (µg/m³) | — |
| `o3` | NUMERIC(10,4) | Daily avg O3 (µg/m³) | — |
| `aqi` | NUMERIC(6,2) | Calculated AQI (US-EPA) | 0 – 500 |
| `aqi_category` | VARCHAR(50) | AQI category label | See below |
| `dominant_pollutant` | VARCHAR(20) | Pollutant with highest sub-AQI | — |
| `avg_temperature` | NUMERIC(6,2) | Daily avg temperature (°C) | — |
| `avg_humidity` | NUMERIC(6,2) | Daily avg humidity (%) | — |
| `total_precipitation` | NUMERIC(8,4) | Daily total precipitation (mm) | — |
| `avg_wind_speed` | NUMERIC(7,3) | Daily avg wind speed (km/h) | — |
| `avg_pressure` | NUMERIC(8,2) | Daily avg pressure (hPa) | — |

**Unique constraint:** `(station_id, date)`

### AQI Category Values (US-EPA Standard)

| AQI Range | Category Label |
|-----------|----------------|
| 0 – 50 | Good |
| 51 – 100 | Moderate |
| 101 – 150 | Unhealthy for Sensitive Groups |
| 151 – 200 | Unhealthy |
| 201 – 300 | Very Unhealthy |
| 301 – 500 | Hazardous |

---

## 6. PostgreSQL Table: `model_predictions`

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `id` | BIGSERIAL | Auto-increment primary key | PRIMARY KEY |
| `prediction_ts` | TIMESTAMPTZ | When prediction was made | DEFAULT NOW() |
| `station_id` | VARCHAR(100) | Station for which prediction was made | — |
| `predicted_aqi` | NUMERIC(6,2) | Model's predicted AQI value | — |
| `predicted_category` | VARCHAR(50) | Model's predicted AQI category | — |
| `model_name` | VARCHAR(100) | Name of model used | — |
| `model_version` | VARCHAR(20) | Version of model used (v1, v2…) | — |
| `input_features` | JSONB | Snapshot of input features used | — |
| `actual_aqi` | NUMERIC(6,2) | Actual AQI when available (for monitoring) | NULLABLE |
| `actual_category` | VARCHAR(50) | Actual category when available | NULLABLE |
| `error_abs` | NUMERIC(8,4) | Absolute error (GENERATED COLUMN) | GENERATED |

---

## 7. ML Feature Matrix

| Feature | Data Type | Description | Source | Leakage Risk |
|---------|-----------|-------------|--------|--------------|
| `pm25` | FLOAT | PM2.5 concentration (µg/m³) | pollution_measurements | None |
| `pm10` | FLOAT | PM10 concentration (µg/m³) | pollution_measurements | None |
| `no2` | FLOAT | NO2 concentration (µg/m³) | pollution_measurements | None |
| `so2` | FLOAT | SO2 concentration (µg/m³) | pollution_measurements | None |
| `co` | FLOAT | CO concentration (µg/m³) | pollution_measurements | None |
| `o3` | FLOAT | O3 concentration (µg/m³) | pollution_measurements | None |
| `nh3` | FLOAT | NH3 concentration (µg/m³) | pollution_measurements | None |
| `temperature` | FLOAT | Air temperature (°C) | weather_measurements | None |
| `humidity` | FLOAT | Relative humidity (%) | weather_measurements | None |
| `precipitation` | FLOAT | Precipitation (mm) | weather_measurements | None |
| `wind_speed` | FLOAT | Wind speed (km/h) | weather_measurements | None |
| `pressure` | FLOAT | Atmospheric pressure (hPa) | weather_measurements | None |
| `year` | INT | Calendar year | date | None |
| `month` | INT (1–12) | Calendar month | date | None |
| `day_of_week` | INT (0–6) | Day of week (0=Monday) | date | None |
| `is_weekend` | INT (0/1) | 1 if Saturday or Sunday | date | None |
| `season` | INT (0–3) | 0=Winter, 1=Spring, 2=Summer, 3=Autumn | date | None |
| `aqi_lag_1` | FLOAT | AQI 1 day ago | daily_aqi | **shift(1) applied** |
| `aqi_lag_2` | FLOAT | AQI 2 days ago | daily_aqi | **shift(2) applied** |
| `aqi_lag_3` | FLOAT | AQI 3 days ago | daily_aqi | **shift(3) applied** |
| `aqi_lag_7` | FLOAT | AQI 7 days ago | daily_aqi | **shift(7) applied** |
| `aqi_rolling_3` | FLOAT | 3-day rolling mean AQI (past only) | daily_aqi | **shift(1) before rolling** |
| `aqi_rolling_7` | FLOAT | 7-day rolling mean AQI (past only) | daily_aqi | **shift(1) before rolling** |

### Target Columns

| Target | Task | Description |
|--------|------|-------------|
| `aqi` | Regression | Numerical AQI value (0–500) |
| `aqi_category` | Classification | Category label (6 classes) |
| `aqi_category_encoded` | Classification | Ordinal-encoded category (0–5) |

---

## 8. Ingestion Log — `data/raw/ingestion_log.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | UUID | Unique run identifier |
| `run_timestamp_utc` | ISO 8601 | When the ingestion run started |
| `source` | STRING | openaq or openmeteo |
| `station_id` | STRING | Station identifier |
| `request_url` | STRING | Exact API URL called |
| `api_status_code` | INT | HTTP status (0 if connection failed) |
| `record_count` | INT | Number of records received |
| `success` | BOOL | Whether the run completed without error |
| `raw_file_path` | STRING | Path to saved raw JSON file |
| `error_type` | STRING | Exception class name (null if success) |
| `error_message` | STRING | Exception message (null if success) |

---

## 9. Pipeline Run Log — `logs/pipeline_summary_<run_id>.json`

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | STRING | Timestamp-based run identifier |
| `total_rows` | INT | Records in final output |
| `stations` | INT | Number of unique stations |
| `cities` | INT | Number of unique cities |
| `date_range` | ARRAY[2] | [start_date, end_date] |
| `total_time_s` | FLOAT | Total pipeline execution time in seconds |
| `category_distribution` | OBJECT | Count per AQI category |
| `steps` | ARRAY | Per-step status, row counts, elapsed time |

---

## 10. Metadata Sidecar — `data/raw/<source>/<date>/<file>.meta.json`

| Field | Type | Description |
|-------|------|-------------|
| `extraction_timestamp_utc` | ISO 8601 | When the file was saved |
| `source` | STRING | openaq or openmeteo |
| `station_id` | STRING | Station identifier |
| `request_url` | STRING | API URL called |
| `api_status_code` | INT | HTTP response code |
| `record_count` | INT | Records in the raw file |
| `raw_file` | STRING | Absolute path to the .json file |

---

*Document version: 1.0 — matches pipeline run ID 20260922_094210*
*AQI standard: US-EPA Technical Assistance Document (Sept 2018)*
