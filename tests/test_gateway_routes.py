"""Tests for the unified ``/api`` gateway routes.

These exercise the canonical inference alias, the permanent redirect from the
legacy ``/api/model-api`` path, hardware telemetry ingestion + read-back, the
offline upload endpoint, and job-type validation - all through the single
application boundary produced by :func:`src.unified_app.create_app`.
"""

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.unified_app import create_app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(create_app(include_dashboard=False)) as c:
        yield c


def test_legacy_model_api_alias_is_gone(client):
    # The old /api/model-api alias was removed in the API consolidation.
    assert client.get("/api/model-api").status_code == 404


def test_model_info_is_served_on_the_single_api(client):
    resp = client.get("/api/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["architecture"] == "Physics-Guided Bayesian Neural Network"
    assert "vibration_rms" in body["input_features"]


def test_model_batch_is_served_on_the_single_api(client):
    sample = {
        "vibration_rms": 2.1,
        "bearing_temp": 62.0,
        "generator_temp": 74.0,
        "power_output": 1350.0,
        "wind_speed": 11.0,
        "operating_hours": 21000.0,
    }
    resp = client.post("/api/model/batch", json={"samples": [sample, sample], "n_mcmc_samples": 12})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2
    assert all("predicted_rul_days" in p for p in body["predictions"])


def test_model_stream_is_served_on_the_single_api(client):
    sample = {
        "vibration_rms": 2.1,
        "bearing_temp": 62.0,
        "generator_temp": 74.0,
        "power_output": 1350.0,
        "wind_speed": 11.0,
        "operating_hours": 21000.0,
    }
    with client.stream(
        "POST", "/api/model/stream", params={"n_mcmc_samples": 8}, json=sample
    ) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert text.count('"rul"') == 8
    assert "[DONE]" in text


def test_model_trend_is_served_on_the_single_api(client):
    sample = {
        "vibration_rms": 2.1,
        "bearing_temp": 62.0,
        "generator_temp": 74.0,
        "power_output": 1350.0,
        "wind_speed": 11.0,
        "operating_hours": 21000.0,
    }
    worse = {**sample, "vibration_rms": 6.5, "bearing_temp": 88.0}
    resp = client.post("/api/model/trend", json={"samples": [sample, worse], "n_mcmc_samples": 12})
    assert resp.status_code == 200
    body = resp.json()
    assert body["degradation_trend"] in ("DEGRADING", "IMPROVING", "STABLE")
    assert len(body["trend"]) == 2


def test_canonical_model_endpoint_returns_prediction(client):
    payload = {
        "vibration_rms": 2.1,
        "bearing_temp": 62.0,
        "generator_temp": 74.0,
        "power_output": 1350.0,
        "wind_speed": 11.0,
        "operating_hours": 21000.0,
        "n_mcmc_samples": 16,
    }
    resp = client.post("/api/model", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # The PG-BNN inference response carries a mean RUL prediction.
    assert "predicted_rul_days" in body
    assert "confidence_interval_95" in body


def test_hardware_stream_ingests_and_reads_back(client):
    batch = {
        "gateway_id": "gw-alpha",
        "readings": [
            {
                "gateway_id": "gw-alpha",
                "turbine_id": "WTG-01",
                "signal": "generator_rpm",
                "value": 1512.5,
                "unit": "rpm",
                "timestamp": "2026-08-08T10:00:00Z",
            },
            {
                "gateway_id": "gw-alpha",
                "turbine_id": "WTG-01",
                "signal": "gearbox_temp",
                "value": 68.3,
                "unit": "C",
                "timestamp": "2026-08-08T10:00:00Z",
            },
        ],
    }
    resp = client.post("/api/hardware/stream", json=batch)
    assert resp.status_code == 200
    ack = resp.json()
    assert ack["ack"] is True
    assert ack["received"] == 2

    latest = client.get("/api/hardware/latest", params={"limit": 10})
    assert latest.status_code == 200
    data = latest.json()
    assert data["count"] >= 2
    signals = {r["signal"] for r in data["readings"]}
    assert {"generator_rpm", "gearbox_temp"} <= signals


def test_telemetry_upload_acknowledges_bytes(client):
    content = b"timestamp,wind_speed,power\n2026-08-08T10:00:00Z,11.0,1350\n"
    files = {"file": ("scada.csv", io.BytesIO(content), "text/csv")}
    resp = client.post("/api/telemetry/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ack"] is True
    assert body["filename"] == "scada.csv"
    assert body["bytes"] == len(content)


def test_unknown_job_type_is_rejected(client):
    resp = client.post("/api/jobs/not-a-real-job", json={})
    assert resp.status_code == 404


def test_unknown_job_id_status_is_404(client):
    resp = client.get("/api/jobs/deadbeefdeadbeef")
    assert resp.status_code == 404


def test_twin_history_has_one_operations_api_owner(client):
    """The gateway router must not shadow the operations API history route."""
    from src.api.gateway_routes import router

    assert "/twin/history" not in {route.path for route in router.routes}
    response = client.get("/api/twin/history", params={"asset_id": "WTG-001"})
    assert response.status_code == 200
    assert response.json()["asset_id"] == "WTG-001"
