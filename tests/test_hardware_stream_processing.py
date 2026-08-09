"""End-to-end tests for hardware-stream processing: durable persistence, twin
updates, advisories, fleet reports, and signed-cloud imports."""


import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.unified_app import create_app  # noqa: E402

_STREAM_BATCH = {
    "gateway_id": "gw-nacelle-01",
    "readings": [
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S1", "signal": "generator_rpm",
         "value": 1500.0, "unit": "rpm", "timestamp": "2026-08-08T10:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S1", "signal": "gearbox_temp",
         "value": 68.0, "unit": "C", "timestamp": "2026-08-08T10:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S1", "signal": "vibration_rms",
         "value": 2.5, "unit": "mm/s", "timestamp": "2026-08-08T10:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S1", "signal": "wind_speed",
         "value": 11.0, "unit": "m/s", "timestamp": "2026-08-08T10:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S1", "signal": "power_output",
         "value": 1850.0, "unit": "kW", "timestamp": "2026-08-08T10:00:00Z"},
    ],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from src.data.store import reset_store
    from src.jobs.manager import reset_job_manager

    monkeypatch.setenv("AV_STORE_DB", str(tmp_path / "store.sqlite3"))
    monkeypatch.setenv("AV_JOB_DB", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.delenv("AV_MODEL_PATH", raising=False)
    monkeypatch.setenv("AV_STREAM_HEURISTIC", "1")
    reset_store()
    reset_job_manager()
    with TestClient(create_app(include_dashboard=False)) as c:
        yield c
    reset_store()
    reset_job_manager()


def test_hardware_stream_persists_and_updates_twins(client):
    resp = client.post("/api/hardware/stream", json=_STREAM_BATCH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["received"] == 5
    assert body["turbines_updated"] == 1
    assert body["advisories_computed"] == 1
    assert body["heuristic_advisories"] is True

    # Readings persisted to the durable store.
    latest = client.get("/api/hardware/latest", params={"limit": 100}).json()
    assert latest["count"] == 5

    # Twin was updated with a computed advisory.
    twin = client.get("/api/twin/status", params={"asset_id": "WTG-S1"}).json()
    assert twin["n_state_records"] >= 2  # seed + stream update
    assert twin["advisory_source"] == "stream-heuristic"
    assert twin["cumulative_wear"] > 0

    # Fleet aggregate reflects the streamed asset.
    summary = client.get("/api/fleet/summary").json()
    assert summary["n_assets"] == 1
    assert summary["at_risk_count"] == 1
    assert summary["turbines"][0]["turbine_id"] == "WTG-S1"

    # Fleet report regenerated with the stream advisory (not the CSV fallback).
    report = client.get("/api/fleet/report")
    assert report.status_code == 200
    assert "WTG-S1" in report.text
    assert "example fleet snapshot" not in report.text

    # System stats expose durable-store row counts.
    stats = client.get("/api/system/stats").json()
    assert stats["tables"]["telemetry"] == 5
    assert stats["tables"]["twin_states"] >= 1
    assert stats["tables"]["assets"] == 1
    assert stats["tables"]["reports"] >= 1


_STREAM_BATCH_SIX_SIGNAL = {
    "gateway_id": "gw-nacelle-01",
    "readings": [
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S2", "signal": "generator_rpm",
         "value": 1500.0, "unit": "rpm", "timestamp": "2026-08-08T11:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S2", "signal": "gearbox_temp",
         "value": 62.0, "unit": "C", "timestamp": "2026-08-08T11:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S2", "signal": "generator_temp",
         "value": 74.0, "unit": "C", "timestamp": "2026-08-08T11:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S2", "signal": "vibration_rms",
         "value": 2.5, "unit": "mm/s", "timestamp": "2026-08-08T11:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S2", "signal": "wind_speed",
         "value": 10.0, "unit": "m/s", "timestamp": "2026-08-08T11:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S2", "signal": "power_output",
         "value": 1500.0, "unit": "kW", "timestamp": "2026-08-08T11:00:00Z"},
        {"gateway_id": "gw-nacelle-01", "turbine_id": "WTG-S2", "signal": "operating_hours",
         "value": 22000.0, "unit": "h", "timestamp": "2026-08-08T11:00:00Z"},
    ],
}


def test_hardware_stream_uses_six_signal_model_when_batch_is_complete(client):
    """A batch carrying all six PG-BNN inputs gets a real model advisory —
    not the demo heuristic — proving the stream is wired to the trained model
    served at /api/model."""
    resp = client.post("/api/hardware/stream", json=_STREAM_BATCH_SIX_SIGNAL)
    assert resp.status_code == 200
    body = resp.json()
    assert body["received"] == 7
    assert body["serving_model_loaded"] is False
    assert body["six_signal_model_advisories"] is True
    assert body["heuristic_advisories"] is False

    asset = body["assets"][0]
    assert asset["advisory_source"] == "stream-model-six-signal"
    assert asset["predicted_rul_days"] is not None
    assert 0.0 <= asset["predicted_rul_days"] <= 3650.0

    twin = client.get("/api/twin/status", params={"asset_id": "WTG-S2"}).json()
    assert twin["advisory_source"] == "stream-model-six-signal"


def test_hardware_stream_partial_batch_falls_back_to_heuristic(client):
    """Without all six inputs (here: no generator_temp / operating_hours) the
    documented heuristic remains the advisory source."""
    resp = client.post("/api/hardware/stream", json=_STREAM_BATCH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["six_signal_model_advisories"] is False
    assert body["heuristic_advisories"] is True
    assert body["assets"][0]["advisory_source"] == "stream-heuristic"


def test_hardware_stream_records_imports_and_cloud_import_validates(client):
    up = client.post(
        "/api/telemetry/upload",
        files={"file": ("scada.csv", b"a,b\n1,2\n", "text/csv")},
        data={"source": "usb"},
    )
    assert up.status_code == 200
    assert up.json()["source"] == "usb"
    assert client.get("/api/imports").json()["count"] == 1

    # Non-HTTPS cloud URL is rejected.
    bad = client.post("/api/telemetry/import", json={"url": "ftp://example.com/x.csv"})
    assert bad.status_code == 422

    # http://localhost is tolerated for development.
    ok = client.post("/api/telemetry/import", json={"url": "http://localhost:9999/x.csv"})
    # The fetch itself will fail (nothing listening) → 502, proving the URL
    # passed scheme validation and reached the fetch step.
    assert ok.status_code == 502


def test_jobs_endpoint_lists_and_supports_evaluate(client):
    resp = client.post("/api/jobs/evaluate", json={"args": ["--checkpoint", "nope.pt"]})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    listing = client.get("/api/jobs").json()
    assert listing["count"] >= 1
    assert any(j["job_id"] == job_id for j in listing["jobs"])
