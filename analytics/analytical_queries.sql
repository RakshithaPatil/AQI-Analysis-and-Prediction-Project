-- ============================================================
-- analytics/analytical_queries.sql
-- Gold Layer – Analytical Views for Dashboard and Reporting
-- ============================================================
-- These views aggregate data from the daily_aqi and related tables
-- and are optimised for Streamlit dashboard queries.
-- Run this AFTER schema.sql.
-- ============================================================

-- ── 1. City-level AQI analysis ────────────────────────────────
-- Average AQI per city per day (across all stations in that city)
CREATE OR REPLACE VIEW vw_city_daily_aqi AS
SELECT
    d.date,
    s.city,
    ROUND(AVG(d.aqi)::numeric, 2)          AS avg_aqi,
    ROUND(MIN(d.aqi)::numeric, 2)          AS min_aqi,
    ROUND(MAX(d.aqi)::numeric, 2)          AS max_aqi,
    COUNT(DISTINCT d.station_id)           AS station_count,
    MODE() WITHIN GROUP (ORDER BY d.aqi_category) AS dominant_category
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
GROUP BY d.date, s.city
ORDER BY d.date DESC, s.city;

-- ── 2. Station-level AQI analysis ────────────────────────────
-- Latest AQI reading per station
CREATE OR REPLACE VIEW vw_station_latest_aqi AS
SELECT DISTINCT ON (d.station_id)
    d.station_id,
    s.station_name,
    s.city,
    s.latitude,
    s.longitude,
    d.date,
    d.aqi,
    d.aqi_category,
    d.dominant_pollutant,
    d.pm25,
    d.pm10,
    d.no2,
    d.so2,
    d.co,
    d.o3
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
ORDER BY d.station_id, d.date DESC;

-- ── 3. Daily AQI trends (last 30 days) ───────────────────────
CREATE OR REPLACE VIEW vw_daily_aqi_trend AS
SELECT
    d.date,
    d.station_id,
    s.station_name,
    s.city,
    d.aqi,
    d.aqi_category,
    d.dominant_pollutant,
    d.avg_temperature,
    d.avg_humidity
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
WHERE d.date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY d.station_id, d.date;

-- ── 4. Monthly AQI patterns ───────────────────────────────────
CREATE OR REPLACE VIEW vw_monthly_aqi AS
SELECT
    DATE_TRUNC('month', d.date)            AS month,
    s.city,
    d.station_id,
    s.station_name,
    ROUND(AVG(d.aqi)::numeric, 2)          AS avg_aqi,
    ROUND(AVG(d.pm25)::numeric, 4)         AS avg_pm25,
    ROUND(AVG(d.pm10)::numeric, 4)         AS avg_pm10,
    ROUND(AVG(d.no2)::numeric, 4)          AS avg_no2,
    ROUND(AVG(d.so2)::numeric, 4)          AS avg_so2,
    ROUND(AVG(d.co)::numeric, 6)           AS avg_co,
    ROUND(AVG(d.o3)::numeric, 4)           AS avg_o3,
    COUNT(*)                               AS record_count
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
GROUP BY DATE_TRUNC('month', d.date), s.city, d.station_id, s.station_name
ORDER BY month DESC, s.city;

-- ── 5. Pollutant contribution analysis ───────────────────────
-- What fraction of days does each pollutant dominate the AQI?
CREATE OR REPLACE VIEW vw_pollutant_contribution AS
SELECT
    s.city,
    d.station_id,
    d.dominant_pollutant,
    COUNT(*)                                              AS days_dominant,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY d.station_id),
        2
    )                                                     AS pct_of_days
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
WHERE d.dominant_pollutant IS NOT NULL
GROUP BY s.city, d.station_id, d.dominant_pollutant
ORDER BY d.station_id, days_dominant DESC;

-- ── 6. Weather impact on AQI ─────────────────────────────────
-- Correlation proxy: AQI alongside weather variables per station per day
CREATE OR REPLACE VIEW vw_weather_aqi_correlation AS
SELECT
    d.date,
    d.station_id,
    s.station_name,
    s.city,
    d.aqi,
    d.aqi_category,
    d.avg_temperature      AS temperature,
    d.avg_humidity         AS humidity,
    d.total_precipitation  AS precipitation,
    d.avg_wind_speed       AS wind_speed,
    d.avg_pressure         AS pressure
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
WHERE d.avg_temperature IS NOT NULL
ORDER BY d.station_id, d.date;

-- ── 7. AQI category distribution ─────────────────────────────
CREATE OR REPLACE VIEW vw_aqi_category_distribution AS
SELECT
    s.city,
    d.station_id,
    s.station_name,
    d.aqi_category,
    COUNT(*)                                              AS day_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY d.station_id),
        2
    )                                                     AS pct_of_days
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
WHERE d.aqi_category IS NOT NULL
GROUP BY s.city, d.station_id, s.station_name, d.aqi_category
ORDER BY d.station_id,
    CASE d.aqi_category
        WHEN 'Good'                            THEN 1
        WHEN 'Moderate'                        THEN 2
        WHEN 'Unhealthy for Sensitive Groups'  THEN 3
        WHEN 'Unhealthy'                       THEN 4
        WHEN 'Very Unhealthy'                  THEN 5
        WHEN 'Hazardous'                       THEN 6
        ELSE 7
    END;

-- ── 8. Hourly pollutant pattern (avg by hour-of-day) ─────────
CREATE OR REPLACE VIEW vw_hourly_aqi_pattern AS
SELECT
    s.city,
    pm.station_id,
    s.station_name,
    EXTRACT(HOUR FROM pm.timestamp)::int  AS hour_of_day,
    pm.pollutant,
    ROUND(AVG(pm.value)::numeric, 4)      AS avg_value
FROM pollution_measurements pm
JOIN stations s ON pm.station_id = s.station_id
GROUP BY s.city, pm.station_id, s.station_name,
         EXTRACT(HOUR FROM pm.timestamp), pm.pollutant
ORDER BY pm.station_id, pm.pollutant, hour_of_day;

-- ── 9. City/station comparison (latest available date) ───────
CREATE OR REPLACE VIEW vw_station_comparison AS
SELECT
    s.city,
    d.station_id,
    s.station_name,
    d.date,
    d.aqi,
    d.aqi_category,
    d.pm25,
    d.pm10,
    d.no2,
    RANK() OVER (PARTITION BY d.date ORDER BY d.aqi DESC) AS aqi_rank
FROM daily_aqi d
JOIN stations s ON d.station_id = s.station_id
WHERE d.date = (SELECT MAX(date) FROM daily_aqi)
ORDER BY aqi_rank;

-- ── 10. Prediction performance view ──────────────────────────
CREATE OR REPLACE VIEW vw_prediction_performance AS
SELECT
    DATE(prediction_ts)           AS prediction_date,
    model_name,
    model_version,
    COUNT(*)                      AS total_predictions,
    COUNT(actual_aqi)             AS evaluated_predictions,
    ROUND(AVG(error_abs)::numeric, 4)     AS mae,
    ROUND(
        SQRT(AVG(POWER(error_abs, 2)))::numeric,
        4
    )                             AS rmse
FROM model_predictions
WHERE actual_aqi IS NOT NULL
GROUP BY DATE(prediction_ts), model_name, model_version
ORDER BY prediction_date DESC;
