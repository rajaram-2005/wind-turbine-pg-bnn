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


def test_operations_api_is_connected_under_one_boundary():
    with TestClient(create_app(include_dashboard=False)) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["advisory_only"] is True


def test_model_api_is_connected_and_initialized():
    with TestClient(create_app(include_dashboard=False)) as client:
        response = client.get("/model-api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
