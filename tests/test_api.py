"""tests/test_api.py — FastAPI endpoint tests using TestClient."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient
    from api.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        assert "status" in r.json()

    def test_health_has_model_loaded(self, client):
        r = client.get("/health")
        assert "model_loaded" in r.json()


class TestModelInfo:
    def test_model_info_200_if_loaded(self, client):
        r = client.get("/model-info")
        # 200 if models loaded, 503 if not
        assert r.status_code in (200, 503)


class TestPredictEndpoint:
    VALID_PAYLOAD = {
        "station_id": "DEL_AV",
        "pm25": 85.0, "pm10": 120.0, "no2": 40.0,
        "so2": 12.0,  "co": 800.0,   "o3": 45.0,
        "nh3": 5.0,
        "temperature": 28.5, "humidity": 65.0,
        "wind_speed": 8.0,
    }

    def test_predict_valid_input(self, client):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        assert r.status_code in (200, 503)  # 503 if models not loaded

    def test_predict_returns_aqi_field(self, client):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        if r.status_code == 200:
            assert "predicted_aqi" in r.json()
            assert "predicted_category" in r.json()

    def test_predict_invalid_pm25(self, client):
        bad = {**self.VALID_PAYLOAD, "pm25": -999}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422   # Pydantic validation error

    def test_predict_missing_required_field(self, client):
        bad = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "pm25"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()
