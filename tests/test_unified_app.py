"""Tests for the single-port application boundary."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.unified_app import create_app  # noqa: E402


def test_unified_health_discovers_every_surface():
    with TestClient(create_app(include_dashboard=False)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["advisory_only"] is True
        assert body["services"]["operations_api"] == "/api"
        assert body["services"]["model_api"] == "/model-api"
        assert body["agent_mesh"]["agents"] == ["MIKA", "KAI"]
        assert body["agent_mesh"]["status"] == "connected"
        assert body["agent_mesh"]["evidence_path"][-1] == "HUMAN"


def test_operations_api_is_connected_under_one_boundary():
    with TestClient(create_app(include_dashboard=False)) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["advisory_only"] is True


def test_unified_health_reports_digital_twin_registry():
    with TestClient(create_app(include_dashboard=False)) as client:
        body = client.get("/health").json()
        assert body["product"] == "AeroVigil"
        assert body["version"] == "1.0.0"
        twin = body["digital_twin"]
        assert twin["assets_tracked"] == 0
        assert twin["max_assets"] >= 1
        # Registry stats reflect live twin traffic through the mounted API.
        client.get("/api/twin/status", params={"asset_id": "WTG-UH", "model": "GE-1.5"})
        after = client.get("/health").json()
        assert after["digital_twin"]["assets_tracked"] == 1


def test_model_api_is_connected_and_initialized():
    with TestClient(create_app(include_dashboard=False)) as client:
        response = client.get("/model-api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True


def test_legacy_dashboard_redirects_to_canonical_console():
    with TestClient(create_app()) as client:
        response = client.get("/legacy", follow_redirects=False)
    assert response.status_code == 308
    assert response.headers["location"] == "/"


def test_root_serves_complete_eight_page_agent_console():
    with TestClient(create_app(include_dashboard=False)) as client:
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

    pages = (
        "overview",
        "twin",
        "fleet",
        "hardware",
        "ingestion",
        "jobs",
        "inference",
        "system",
    )
    assert html.count('<section class="page') == len(pages)
    for page in pages:
        assert f'data-page="{page}"' in html
        assert f'id="page-{page}"' in html

    assert "CYBER PRIME DUAL AGENT" in html
    assert "MIKA" in html and "KAI" in html
    assert "SCADA" in html and "PG-BNN" in html and "ISO 281" in html
    for endpoint in (
        "/api/twin/status",
        "/api/twin/history",
        "/api/fleet/summary",
        "/api/hardware/latest",
        "/api/telemetry/upload",
        "/api/telemetry/import",
        "/api/jobs/",
        "/api/model",
        "/api/system/stats",
        "/health",
    ):
        assert endpoint in html
