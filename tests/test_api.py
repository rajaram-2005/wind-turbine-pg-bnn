"""Tests for the FastAPI advisory service (`src.api`)."""

import pytest

# These tests need the optional `api` extras. Skip gracefully if absent.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.schemas import AdvisoryResponse, FleetSummary  # noqa: E402

# Fields that must NEVER appear in any advisory payload (see src/utils/safety.py).
BLOCKED_KEYS = ("throttle_pct", "rpm_setpoint", "loto_steps", "part_sku", "torque_demand")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def _good_payload():
    return {
        "asset_id": "WTG-044",
        "telemetry": {
            "vibration_mms": 4.8,
            "temperature_c": 82.0,
            "rpm": 1780,
            "oil_viscosity_cst": 12.0,
            "load_pct": 95.0,
        },
        "bnn_state": {
            "predicted_rul_days": 14.2,
            "epistemic_uncertainty": 0.04,
            "aleatoric_uncertainty": 0.12,
        },
    }


def test_root_advertises_advisory_only(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["advisory_only"] is True
    assert "DECISION-SUPPORT ONLY" in body["disclaimer"]
    assert body["agent_team"]["agents"] == ["MIKA", "KAI"]
    assert set(body["agent_team"]["connected_surfaces"]) >= {"advisory", "fleet", "twin"}


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["advisory_only"] is True
    assert body["service"] == "wind-turbine-pg-bnn"


def test_advisory_returns_advisory_only_recommendation(client: TestClient):
    resp = client.post("/advisory", json=_good_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset_id"] == "WTG-044"
    assert body["advisory_only"] is True
    assert "Decision-support only" in body["disclaimer"]
    # Low RUL -> short inspection window.
    assert body["suggested_inspection_window_days"] <= 7
    # Two physics violations expected (vibration + temperature over limit).
    assert len(body["physics_violations"]) >= 2
    assert body["agent_team"]["team_id"] == "CYBER_PRIME_DUAL_AGENT"
    assert body["agent_team"]["agents"]["mika"]["name"] == "MIKA"
    assert body["agent_team"]["agents"]["kai"]["name"] == "KAI"
    # No forbidden direct-actuation fields leak through.
    for bad in BLOCKED_KEYS:
        assert bad not in body


def test_advisory_rejects_out_of_range_telemetry(client: TestClient):
    bad = _good_payload()
    bad["telemetry"]["vibration_mms"] = 999.0  # > 50.0 schema limit
    resp = client.post("/advisory", json=bad)
    assert resp.status_code == 422  # pydantic validation error


def test_advisory_rejects_missing_bnn_state_without_model(client: TestClient):
    bad = _good_payload()
    del bad["bnn_state"]
    resp = client.post("/advisory", json=bad)
    # No model + no bnn_state -> clean 422 (not an opaque 500, and no leak).
    assert resp.status_code == 422
    assert "bnn_state" in resp.json()["detail"]


def test_advisory_response_schema_strict():
    rec = {
        "asset_id": "X",
        "predicted_rul_days": 10.0,
        "epistemic_std": 0.1,
        "aleatoric_std": 0.2,
        "physics_violations": [],
        "suggested_inspection_window_days": 2.5,
        "rationale": "ok",
        "advisory_only": True,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "disclaimer": "Decision-support only.",
    }
    resp = AdvisoryResponse(**rec)
    assert resp.advisory_only is True
    # extra="forbid" rejects unknown (and therefore any blocked) keys.
    with pytest.raises(ValidationError):
        AdvisoryResponse(**{**rec, "throttle_pct": -10})


def test_fleet_endpoint_returns_summary_and_assets(client: TestClient):
    req = {
        "assets": [
            _good_payload(),
            {
                "asset_id": "WTG-007",
                "telemetry": {
                    "vibration_mms": 2.8,
                    "temperature_c": 60.0,
                    "rpm": 1490,
                    "oil_viscosity_cst": 34.0,
                    "load_pct": 78.0,
                },
                "bnn_state": {
                    "predicted_rul_days": 300.0,
                    "epistemic_uncertainty": 0.03,
                    "aleatoric_uncertainty": 0.09,
                },
            },
        ]
    }
    resp = client.post("/advisory/fleet", json=req)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["assets"]) == 2
    summary = FleetSummary(**body["summary"])
    assert summary.n_assets == 2
    assert summary.mean_rul_days == pytest.approx((14.2 + 300.0) / 2, rel=1e-6)
    for asset in body["assets"]:
        assert asset["advisory_only"] is True
        assert asset["agent_team"]["team_id"] == "CYBER_PRIME_DUAL_AGENT"
        for bad in BLOCKED_KEYS:
            assert bad not in asset


def test_fleet_rejects_empty(client: TestClient):
    resp = client.post("/advisory/fleet", json={"assets": []})
    assert resp.status_code == 422  # min_length=1
