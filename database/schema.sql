-- ============================================================
-- schema.sql
-- PostgreSQL Database Schema for AQI Analysis and Prediction
-- ============================================================
-- Run this file once to create all tables, indexes, and
-- constraints. It is idempotent (IF NOT EXISTS everywhere).
-- ============================================================

-- ── Extensions ──────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TABLE: stations
-- One row per monitoring station / location.
-- ============================================================
CREATE TABLE IF NOT EXISTS stations (
    station_id      VARCHAR(100)   PRIMARY KEY,
    station_name    VARCHAR(255)   NOT NULL,
    city            VARCHAR(100),
    latitude        NUMERIC(9, 6),
    longitude       NUMERIC(9, 6),
    created_at      TIMESTAMPTZ    DEFAULT NOW(),
    updated_at      TIMESTAMPTZ    DEFAULT NOW(),
    CONSTRAINT chk_latitude  CHECK (latitude  BETWEEN -90  AND 90),
    CONSTRAINT chk_longitude CHECK (longitude BETWEEN -180 AND 180)
);

CREATE INDEX IF NOT EXISTS idx_stations_city ON stations(city);

-- ============================================================
-- TABLE: pollution_measurements
-- Raw (cleaned) hourly pollution readings.
-- ============================================================
CREATE TABLE IF NOT EXISTS pollution_measurements (
    id              BIGSERIAL      PRIMARY KEY,
    station_id      VARCHAR(100)   NOT NULL REFERENCES stations(station_id),
    timestamp       TIMESTAMPTZ    NOT NULL,
    pollutant       VARCHAR(20)    NOT NULL,
    value           NUMERIC(12, 4) NOT NULL,
    unit            VARCHAR(20),
    source          VARCHAR(50),
    created_at      TIMESTAMPTZ    DEFAULT NOW(),
    CONSTRAINT chk_pollutant CHECK (
        pollutant IN ('pm25','pm10','no2','so2','co','o3')
    ),
    CONSTRAINT chk_value_positive CHECK (value >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pollution
    ON pollution_measurements(station_id, timestamp, pollutant);
CREATE INDEX IF NOT EXISTS idx_pollution_station   ON pollution_measurements(station_id);
CREATE INDEX IF NOT EXISTS idx_pollution_timestamp ON pollution_measurements(timestamp);
CREATE INDEX IF NOT EXISTS idx_pollution_pollutant ON pollution_measurements(pollutant);

-- ============================================================
-- TABLE: weather_measurements
-- Hourly weather observations joined to stations.
-- ============================================================
CREATE TABLE IF NOT EXISTS weather_measurements (
    id              BIGSERIAL      PRIMARY KEY,
    station_id      VARCHAR(100)   NOT NULL REFERENCES stations(station_id),
    timestamp       TIMESTAMPTZ    NOT NULL,
    temperature     NUMERIC(6, 2),      -- °C
    humidity        NUMERIC(6, 2),      -- %
    precipitation   NUMERIC(8, 4),      -- mm
    wind_speed      NUMERIC(7, 3),      -- km/h
    wind_direction  NUMERIC(6, 2),      -- degrees 0–360
    pressure        NUMERIC(8, 2),      -- hPa
    source          VARCHAR(50),
    created_at      TIMESTAMPTZ    DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_weather
    ON weather_measurements(station_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_weather_station   ON weather_measurements(station_id);
CREATE INDEX IF NOT EXISTS idx_weather_timestamp ON weather_measurements(timestamp);

-- ============================================================
-- TABLE: daily_aqi
-- One row per (date, station). Stores daily averages and AQI.
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_aqi (
    id                   BIGSERIAL      PRIMARY KEY,
    station_id           VARCHAR(100)   NOT NULL REFERENCES stations(station_id),
    date                 DATE           NOT NULL,
    pm25                 NUMERIC(10, 4),
    pm10                 NUMERIC(10, 4),
    no2                  NUMERIC(10, 4),
    so2                  NUMERIC(10, 4),
    co                   NUMERIC(10, 6),
    o3                   NUMERIC(10, 4),
    aqi                  NUMERIC(6, 2),
    aqi_category         VARCHAR(50),
    dominant_pollutant   VARCHAR(20),
    -- Weather averages for the day
    avg_temperature      NUMERIC(6, 2),
    avg_humidity         NUMERIC(6, 2),
    total_precipitation  NUMERIC(8, 4),
    avg_wind_speed       NUMERIC(7, 3),
    avg_pressure         NUMERIC(8, 2),
    created_at           TIMESTAMPTZ    DEFAULT NOW(),
    updated_at           TIMESTAMPTZ    DEFAULT NOW(),
    CONSTRAINT uq_daily_aqi UNIQUE (station_id, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_aqi_station  ON daily_aqi(station_id);
CREATE INDEX IF NOT EXISTS idx_daily_aqi_date     ON daily_aqi(date);
CREATE INDEX IF NOT EXISTS idx_daily_aqi_category ON daily_aqi(aqi_category);

-- ============================================================
-- TABLE: model_predictions
-- Stores every prediction made by the FastAPI service.
-- ============================================================
CREATE TABLE IF NOT EXISTS model_predictions (
    id                 BIGSERIAL      PRIMARY KEY,
    prediction_ts      TIMESTAMPTZ    DEFAULT NOW(),
    station_id         VARCHAR(100),
    predicted_aqi      NUMERIC(6, 2),
    predicted_category VARCHAR(50),
    model_name         VARCHAR(100),
    model_version      VARCHAR(20),
    -- Input features snapshot (stored as JSONB for flexibility)
    input_features     JSONB,
    -- Actual AQI if/when it becomes available (for monitoring)
    actual_aqi         NUMERIC(6, 2),
    actual_category    VARCHAR(50),
    error_abs          NUMERIC(8, 4)  GENERATED ALWAYS AS (
                           ABS(predicted_aqi - actual_aqi)
                       ) STORED,
    created_at         TIMESTAMPTZ    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_ts      ON model_predictions(prediction_ts);
CREATE INDEX IF NOT EXISTS idx_predictions_station ON model_predictions(station_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model   ON model_predictions(model_name, model_version);

-- ============================================================
-- TABLE: ingestion_runs  (mirrors the JSONL log in PostgreSQL)
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id           UUID           PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_timestamp    TIMESTAMPTZ    DEFAULT NOW(),
    source           VARCHAR(50),
    station_id       VARCHAR(100),
    request_url      TEXT,
    api_status_code  INTEGER,
    record_count     INTEGER,
    success          BOOLEAN,
    raw_file_path    TEXT,
    error_type       VARCHAR(100),
    error_message    TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingestion_source  ON ingestion_runs(source);
CREATE INDEX IF NOT EXISTS idx_ingestion_success ON ingestion_runs(success);
CREATE INDEX IF NOT EXISTS idx_ingestion_ts      ON ingestion_runs(run_timestamp);

-- ============================================================
-- TABLE: data_quality_log
-- Stores results of every data-quality check run.
-- ============================================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    id               BIGSERIAL      PRIMARY KEY,
    check_ts         TIMESTAMPTZ    DEFAULT NOW(),
    check_name       VARCHAR(100),
    source           VARCHAR(50),
    passed           BOOLEAN,
    total_records    INTEGER,
    failed_records   INTEGER,
    failure_pct      NUMERIC(6, 2),
    details          JSONB
);
