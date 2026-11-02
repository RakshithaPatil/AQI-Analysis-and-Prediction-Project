"""
api/schemas.py
===============
Pydantic input/output models for the FastAPI prediction service.
All inputs are validated before reaching the model.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictionInput(BaseModel):
    """Single-record prediction request."""
    station_id:  str   = Field("unknown", example="DEL_AV")
    pm25:        float = Field(..., ge=0, le=1000,  description="PM2.5 µg/m³")
    pm10:        float = Field(..., ge=0, le=1500,  description="PM10 µg/m³")
    no2:         float = Field(..., ge=0, le=2000,  description="NO2 µg/m³")
    so2:         float = Field(..., ge=0, le=2620,  description="SO2 µg/m³")
    co:          float = Field(..., ge=0, le=50000, description="CO µg/m³")
    o3:          float = Field(..., ge=0, le=600,   description="O3 µg/m³")
    nh3:         float = Field(5.0, ge=0, le=400,   description="NH3 µg/m³")
    temperature: float = Field(..., ge=-60, le=60,  description="°C")
    humidity:    float = Field(..., ge=0,  le=100,  description="%")
    precipitation: float = Field(0.0, ge=0, le=500, description="mm")
    wind_speed:  float = Field(..., ge=0,  le=120,  description="km/h")
    pressure:    float = Field(1005.0, ge=870, le=1085, description="hPa")

    # Optional temporal — defaults to current time if not provided
    year:        Optional[int]   = None
    month:       Optional[int]   = Field(None, ge=1, le=12)
    day_of_week: Optional[int]   = Field(None, ge=0, le=6)
    is_weekend:  Optional[int]   = Field(None, ge=0, le=1)
    season:      Optional[int]   = Field(None, ge=0, le=3)

    # Optional lag/rolling — if absent, model uses 0 (imputed)
    aqi_lag_1:      Optional[float] = None
    aqi_lag_2:      Optional[float] = None
    aqi_lag_3:      Optional[float] = None
    aqi_lag_7:      Optional[float] = None
    aqi_rolling_3:  Optional[float] = None
    aqi_rolling_7:  Optional[float] = None

    model_config = {"json_schema_extra": {
        "example": {
            "station_id": "DEL_AV",
            "pm25": 85.0, "pm10": 120.0, "no2": 40.0,
            "so2": 12.0,  "co": 800.0,   "o3": 45.0,
            "nh3": 5.0,   "temperature": 28.5, "humidity": 65.0,
            "wind_speed": 8.0,
        }
    }}


class PredictionOutput(BaseModel):
    """Prediction response."""
    station_id:          str
    predicted_aqi:       float
    predicted_category:  str
    model_name:          str
    model_version:       str
    prediction_timestamp: str
    input_features:      dict


class BatchPredictionInput(BaseModel):
    """Batch prediction — list of records."""
    records: list[PredictionInput]


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    total:       int
    model_name:  str


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_name:    str
    uptime_s:      float


class ModelInfoResponse(BaseModel):
    regressor_name:        str
    classifier_name:       str
    regressor_test_mae:    float
    regressor_test_r2:     float
    classifier_test_f1:    float
    classifier_test_acc:   float
    feature_count:         int
    feature_names:         list[str]
    label_classes:         list[str]
    model_version:         str
