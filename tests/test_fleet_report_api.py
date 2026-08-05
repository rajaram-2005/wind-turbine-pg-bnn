"""Tests for GET /fleet/report (markdown reporting via the API)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402


def test_fleet_report_from_examples_fallback(tmp_path, monkeypatch):
    """Fresh app, empty twin registry → markdown from examples/fleet.csv."""
    client = TestClient(create_app())
    resp = client.get("/fleet/report", params={"title": "Q3 review"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    text = resp.text
    assert text.lstrip().startswith("# Q3 review")
    assert "example fleet snapshot" in text
    assert "ADVISORY ONLY" in text
    assert "## Fleet summary" in text


def test_fleet_report_prefers_twin_registry():
    client = TestClient(create_app())
    # Create twins with advisories: simulate supplies bnn_state → advisories.
    client.post(
        "/twin/simulate",
        json={"asset_id": "RPT-1", "model": "GE-1.5", "profile": "nominal", "hours": 2},
    )
    client.get("/twin/status", params={"asset_id": "RPT-2", "model": "Vestas-V90"})
    resp = client.get("/fleet/report")
    assert resp.status_code == 200
    text = resp.text
    assert "RPT-1" in text  # twin-sourced report (not the example fallback)
    assert "example fleet snapshot" not in text
    assert "ADVISORY ONLY" in text
