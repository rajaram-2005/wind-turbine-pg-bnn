"""Tests for the FastAPI advisory service (`src.api`)."""

import json

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


# --------------------------------------------------------------------------- #
# Whole-turbine fault detection routes                                         #
# --------------------------------------------------------------------------- #
def test_faults_catalog_lists_all_subsystems(client):
    resp = client.get("/faults/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["n_subsystems"] == 12
    assert body["summary"]["total_fault_types"] >= 60
    subsystems = {f["subsystem"] for f in body["faults"]}
    assert subsystems == {
        "rotor_blades",
        "pitch",
        "hub_mainshaft",
        "gearbox",
        "hss_brake",
        "generator",
        "yaw",
        "tower_foundation",
        "nacelle_sensors",
        "cooling_hydraulics",
        "electrical",
        "scada",
    }


def test_faults_catalog_filtered_by_subsystem(client):
    resp = client.get("/faults/catalog?subsystem=gearbox")
    assert resp.status_code == 200
    faults = resp.json()["faults"]
    assert len(faults) >= 10
    assert all(f["subsystem"] == "gearbox" for f in faults)
    names = " ".join(f["name"] for f in faults).lower()
    assert "viscosity" in names and "water" in names and "filter" in names

    bad = client.get("/faults/catalog?subsystem=not-real")
    assert bad.status_code == 404


def test_faults_detect_healthy_and_faulty(client):
    healthy = {
        "asset_id": "WTG-API",
        "model_key": "NREL-5MW",
        "telemetry": {
            "vibration_mms": 2.0,
            "temperature_c": 55.0,
            "rpm": 950.0,
            "oil_viscosity_cst": 32.0,
            "load_pct": 70.0,
        },
    }
    resp = client.post("/faults/detect", json=healthy)
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset_id"] == "WTG-API"
    assert body["overall_status"] == "OK"
    assert body["summary"]["n_faults"] == 0

    faulty = {
        "asset_id": "WTG-API",
        "model_key": "NREL-5MW",
        "telemetry": {
            "vibration_mms": 8.5,
            "temperature_c": 88.0,
            "oil_temp_c": 88.0,
            "rpm": 1100.0,
            "oil_viscosity_cst": 5.0,
            "load_pct": 100.0,
            "oil_water_ppm": 1500.0,
            "oil_particles_iso4406": "20/18/16",
            "oil_tan_mgkoh_g": 2.5,
            "oil_filter_dp_bar": 2.9,
            "oil_level_pct": 5.0,
            "oil_pressure_bar": 0.6,
            "generator_temp_c": 122.0,
            "yaw_error_deg": 25.0,
        },
    }
    resp = client.post("/faults/detect", json=faulty)
    assert resp.status_code == 200
    body = resp.json()
    ids = {f["fault_id"] for f in body["faults"]}
    assert {"GB-02", "GB-04", "GB-05", "GB-06", "GB-07", "GB-08", "GB-12"} <= ids
    assert "GN-01" in ids and "YW-03" in ids and "HS-01" in ids
    assert body["oil"]["overall_status"] == "ALARM"
    assert body["overall_status"] in ("HIGH", "CRITICAL")
    # Advisory-only contract: no actuation fields leak into fault payloads.
    for f in body["faults"]:
        for key in BLOCKED_KEYS:
            assert key not in json.dumps(f)


def test_faults_detect_unknown_model(client):
    resp = client.post(
        "/faults/detect",
        json={"asset_id": "WTG-X", "model_key": "No-Such-Model", "telemetry": {}},
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Email notifications + sensor catalog routes                                  #
# --------------------------------------------------------------------------- #
def test_notifications_status_reports_configuration(client):
    resp = client.get("/notifications/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] in ("smtp", "eml")
    assert body["advisory_only"] is True
    assert "alert_recipients" in body and "cooldown_hours" in body


def test_notifications_send_faulty_snapshot_pages_alert(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AV_NOTIFY_DIR", str(tmp_path / "notes"))
    monkeypatch.delenv("AV_SMTP_HOST", raising=False)
    monkeypatch.setenv("AV_ALERT_RECIPIENTS", "ops@example.com")
    payload = {
        "asset_id": "WTG-NOTIF",
        "model_key": "NREL-5MW",
        "telemetry": {
            "vibration_mms": 9.0,
            "temperature_c": 60.0,
            "rpm": 1000.0,
            "oil_viscosity_cst": 5.0,
            "load_pct": 90.0,
            "oil_temp_c": 125.0,
            "smoke_detector_on": True,
        },
    }
    resp = client.post("/notifications/send", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["delivered"] is True
    assert body["notifications"], "expected an alert email"
    subjects = " ".join(n["subject"] for n in body["notifications"])
    assert "CRITICAL" in subjects
    eml_files = list((tmp_path / "notes").glob("*.eml"))
    assert len(eml_files) >= 1
    assert "WTG-NOTIF" in eml_files[0].read_text(encoding="utf-8")


def test_notifications_send_healthy_snapshot_force_reports(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AV_NOTIFY_DIR", str(tmp_path / "notes2"))
    monkeypatch.delenv("AV_SMTP_HOST", raising=False)
    monkeypatch.setenv("AV_REPORT_RECIPIENTS", "maint@example.com")
    payload = {
        "asset_id": "WTG-FINE",
        "model_key": "NREL-5MW",
        "telemetry": {
            "vibration_mms": 2.0,
            "temperature_c": 55.0,
            "rpm": 950.0,
            "oil_viscosity_cst": 32.0,
            "load_pct": 70.0,
        },
        "force_report": True,
        "subject": "Overnight check",
    }
    resp = client.post("/notifications/send", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["delivered"] is True
    assert len(body["notifications"]) == 1
    assert "Overnight check" in body["notifications"][0]["subject"]


def test_faults_sensors_route(client):
    resp = client.get("/faults/sensors")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["n_sensors"] >= 40
    assert body["summary"]["n_fault_types_covered"] >= 75
    fire = client.get("/faults/sensors?subsystem=gearbox")
    assert fire.status_code == 200
    assert all(s["category"] for s in fire.json()["sensors"])
    bad = client.get("/faults/sensors?subsystem=nope")
    assert bad.status_code == 404


# --------------------------------------------------------------------------- #
# Alert workflow, webhooks, digest, history/trends, work orders                 #
# --------------------------------------------------------------------------- #
def test_alerts_ack_and_resolve_workflow(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AV_NOTIFY_DIR", str(tmp_path / "alerts"))
    # Seed a tracked alert by sending a faulty snapshot.
    resp = client.post(
        "/notifications/send",
        json={
            "asset_id": "WTG-ACK",
            "model_key": "NREL-5MW",
            "telemetry": {
                "vibration_mms": 9.0,
                "temperature_c": 60.0,
                "rpm": 1000.0,
                "oil_viscosity_cst": 5.0,
                "load_pct": 90.0,
            },
        },
    )
    assert resp.status_code == 200
    alerts = client.get("/notifications/alerts").json()
    assert alerts["n_open"] >= 1
    assert any(a["asset_id"] == "WTG-ACK" for a in alerts["alerts"])

    ack = client.post(
        "/notifications/alerts/ack",
        json={"asset_id": "WTG-ACK", "fault_id": "GB-02", "operator": "ops-1"},
    )
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] is True
    state = {a["fault_id"]: a for a in client.get("/notifications/alerts").json()["alerts"]}
    assert state["GB-02"]["acknowledged"] is True

    resolved = client.post(
        "/notifications/alerts/resolve",
        json={"asset_id": "WTG-ACK", "fault_id": "GB-02", "operator": "crew"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["resolved"] is True
    open_ids = {a["fault_id"] for a in client.get("/notifications/alerts").json()["alerts"]}
    assert "GB-02" not in open_ids


def test_webhooks_status_and_test_requires_urls(client):
    status = client.get("/notifications/webhooks/status").json()
    assert "enabled" in status and "n_webhooks" in status
    test = client.post("/notifications/webhooks/test")
    assert test.status_code == 422  # no URLs configured in test env


def test_digest_requires_twins(client):
    resp = client.post("/notifications/digest")
    assert resp.status_code == 422  # no tracked twins with reports


def test_digest_after_twin_simulation(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AV_NOTIFY_DIR", str(tmp_path / "digest"))
    monkeypatch.setenv("AV_REPORT_RECIPIENTS", "maint@example.com")
    # Register + populate a twin via the simulate route.
    sim = client.post(
        "/twin/simulate",
        json={"asset_id": "WTG-DG", "model": "NREL-5MW", "profile": "overload", "hours": 3},
    )
    assert sim.status_code == 200
    resp = client.post("/notifications/digest", json={"title": "Manual digest"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["delivered"] is True
    assert "Manual digest" in body["result"]["subject"]
    assert body["n_assets"] >= 1
    assert list((tmp_path / "digest").glob("*.eml"))


def test_faults_history_and_trends(client):
    sim = client.post(
        "/twin/simulate",
        json={"asset_id": "WTG-HST", "model": "NREL-5MW", "profile": "overload", "hours": 4},
    )
    assert sim.status_code == 200
    history = client.get("/faults/history?asset_id=WTG-HST&limit=10")
    assert history.status_code == 200
    body = history.json()
    assert body["asset_id"] == "WTG-HST"
    assert body["n_records"] >= 1
    assert "timeline" in body and "health_score" in body["timeline"][0]

    trends = client.get("/faults/trends")
    assert trends.status_code == 200
    tbody = trends.json()
    assert tbody["n_assets"] >= 1
    assert "severity_rollup" in tbody and "subsystem_rollup" in tbody
    assert "top_faults" in tbody
    assert all("name" in f for f in tbody["top_faults"])


def test_maintenance_workorder_generate_and_list(client):
    resp = client.post(
        "/maintenance/workorder",
        json={
            "asset_id": "WTG-WO",
            "model_key": "NREL-5MW",
            "telemetry": {
                "vibration_mms": 9.0,
                "temperature_c": 60.0,
                "rpm": 1000.0,
                "oil_viscosity_cst": 5.0,
                "load_pct": 90.0,
                "oil_water_ppm": 1500.0,
            },
        },
    )
    assert resp.status_code == 200
    wo = resp.json()
    assert wo["wo_id"].startswith("WO-")
    assert wo["asset_id"] == "WTG-WO"
    assert wo["priority"].startswith("P1") or wo["priority"].startswith("P0")
    assert wo["advisory_only"] is True
    assert wo["sensors"], "work order lists involved sensors"
    assert any(a.startswith("[GB-02]") for a in wo["recommended_actions"])

    listing = client.get("/maintenance/workorders")
    assert listing.status_code == 200
    assert listing.json()["n_workorders"] >= 1
    assert any(o["wo_id"] == wo["wo_id"] for o in listing.json()["workorders"])


def test_notifications_email_test(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AV_NOTIFY_DIR", str(tmp_path / "etest"))
    monkeypatch.setenv("AV_ALERT_RECIPIENTS", "ops@example.com")
    resp = client.post("/notifications/email/test")
    assert resp.status_code == 200
    assert resp.json()["result"]["delivered"] is True
    assert "TEST" in resp.json()["result"]["subject"]
