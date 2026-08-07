"""Tests for the standalone packaged FastAPI server (`src.aerovigil_pg_bnn.api`)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src.aerovigil_pg_bnn.api import app


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_root_endpoint(api_client: TestClient):
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Aerovigil PG-BNN API"
    assert "/predict" in data["predict"]
    assert "/trend" in data["trend"]


def test_health_endpoint(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True


def test_model_info_endpoint(api_client: TestClient):
    response = api_client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "wind-turbine-pg-bnn"
    assert "vibration_rms" in data["input_features"]


def test_predict_healthy_turbine(api_client: TestClient):
    payload = {
        "vibration_rms": 12.5,
        "bearing_temp": 65.0,
        "generator_temp": 80.0,
        "power_output": 2000.0,
        "wind_speed": 9.0,
        "operating_hours": 1000.0,
    }
    response = api_client.post("/predict", json=payload, params={"n_mcmc_samples": 20})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_rul_days"] > 100.0
    assert data["risk_level"] == "LOW"
    assert data["maintenance_recommended"] is False


def test_predict_critical_turbine(api_client: TestClient):
    payload = {
        "vibration_rms": 34.0,
        "bearing_temp": 118.0,
        "generator_temp": 150.0,
        "power_output": 2400.0,
        "wind_speed": 12.0,
        "operating_hours": 78000.0,
    }
    response = api_client.post("/predict", json=payload, params={"n_mcmc_samples": 20})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_rul_days"] < 14.0
    assert data["risk_level"] == "CRITICAL"
    assert data["maintenance_recommended"] is True


def test_predict_batch(api_client: TestClient):
    payload = {
        "samples": [
            {
                "vibration_rms": 12.5,
                "bearing_temp": 65.0,
                "generator_temp": 80.0,
                "power_output": 2000.0,
                "wind_speed": 9.0,
                "operating_hours": 1000.0,
            },
            {
                "vibration_rms": 34.0,
                "bearing_temp": 118.0,
                "generator_temp": 150.0,
                "power_output": 2400.0,
                "wind_speed": 12.0,
                "operating_hours": 78000.0,
            },
        ],
        "n_mcmc_samples": 20,
    }
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["risk_level"] == "LOW"
    assert data["predictions"][1]["risk_level"] == "CRITICAL"


def test_predict_stream(api_client: TestClient):
    payload = {
        "vibration_rms": 12.5,
        "bearing_temp": 65.0,
        "generator_temp": 80.0,
        "power_output": 2000.0,
        "wind_speed": 9.0,
        "operating_hours": 1000.0,
    }
    response = api_client.post("/predict/stream", json=payload, params={"n_mcmc_samples": 3})
    assert response.status_code == 200
    text = response.text
    assert "data: {" in text
    assert "[DONE]" in text


def test_trend_endpoint(api_client: TestClient):
    payload = {
        "samples": [
            {
                "vibration_rms": 12.5,
                "bearing_temp": 65.0,
                "generator_temp": 80.0,
                "power_output": 2000.0,
                "wind_speed": 9.0,
                "operating_hours": 1000.0,
            },
            {
                "vibration_rms": 20.0,
                "bearing_temp": 88.0,
                "generator_temp": 115.0,
                "power_output": 2100.0,
                "wind_speed": 11.0,
                "operating_hours": 52000.0,
            },
            {
                "vibration_rms": 34.0,
                "bearing_temp": 118.0,
                "generator_temp": 150.0,
                "power_output": 2400.0,
                "wind_speed": 12.0,
                "operating_hours": 78000.0,
            },
        ],
        "n_mcmc_samples": 20,
    }
    response = api_client.post("/trend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["trend"]) == 3
    assert data["degradation_trend"] == "DEGRADING"
    assert data["total_rul_delta_days"] < 0.0

    # Test alias endpoint /predict/trend
    response_alias = api_client.post("/predict/trend", json=payload)
    assert response_alias.status_code == 200
    assert response_alias.json()["degradation_trend"] == "DEGRADING"


def test_predict_validation_out_of_bounds(api_client: TestClient):
    payload = {
        "vibration_rms": 999.0,  # exceeds max bound of 50.0
        "bearing_temp": 65.0,
        "generator_temp": 80.0,
        "power_output": 2000.0,
        "wind_speed": 9.0,
        "operating_hours": 1000.0,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422
