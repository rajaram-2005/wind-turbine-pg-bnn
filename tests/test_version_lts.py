"""Tests for 3-Year Enterprise Long-Term Support (LTS) metadata and version consistency."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import main
from src.api.app import create_app
from src.unified_app import create_app as create_unified_app
from src.version import (
    APP_VERSION,
    IS_LTS,
    LTS_CYCLE_YEARS,
    LTS_END_DATE,
    LTS_RELEASE_TAG,
    LTS_START_DATE,
    LTS_STATUS,
    NEXT_MAJOR_UPDATE,
    PRODUCT,
    SAFETY_BANNER,
    WEBSITE,
    get_lts_info,
)


def test_lts_constants():
    assert APP_VERSION == "1.0.0"
    assert PRODUCT == "AeroVigil"
    assert IS_LTS is True
    assert LTS_RELEASE_TAG == "v1.0.0"
    assert LTS_START_DATE == "2026-08-17"
    assert LTS_END_DATE == "2029-08-17"
    assert LTS_CYCLE_YEARS == 3
    assert NEXT_MAJOR_UPDATE == "2029-08-17"
    assert "Active LTS" in LTS_STATUS
    assert "2029" in LTS_STATUS
    assert "DECISION-SUPPORT ONLY" in SAFETY_BANNER


def test_get_lts_info():
    info = get_lts_info()
    assert isinstance(info, dict)
    assert info["product"] == "AeroVigil"
    assert info["version"] == "1.0.0"
    assert info["is_lts"] is True
    assert info["lts_tag"] == "v1.0.0"
    assert info["lts_start"] == "2026-08-17"
    assert info["lts_end"] == "2029-08-17"
    assert info["support_duration_years"] == 3
    assert info["next_major_update"] == "2029-08-17"
    assert info["website"] == WEBSITE


def test_main_cli_info(capsys):
    rc = main.main(["info"])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["version"] == "1.0.0"
    assert data["is_lts"] is True
    assert data["support_duration_years"] == 3
    assert data["lts_end"] == "2029-08-17"


def test_operations_api_health_and_root_lts():
    app = create_app()
    client = TestClient(app)

    # /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["version"] == "1.0.0"
    assert health_data["is_lts"] is True
    assert health_data["lts_support_until"] == "2029-08-17"
    assert health_data["next_major_update"] == "2029-08-17"

    # /
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    root_data = root_resp.json()
    assert root_data["version"] == "1.0.0"
    assert root_data["lts"]["is_lts"] is True
    assert root_data["lts"]["cycle_years"] == 3
    assert root_data["lts"]["support_until"] == "2029-08-17"


def test_unified_app_health_lts():
    app = create_unified_app()
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["version"] == "1.0.0"
    assert "lts" in health_data
    assert health_data["lts"]["is_lts"] is True
    assert health_data["lts"]["cycle_years"] == 3
    assert health_data["lts"]["support_until"] == "2029-08-17"
    assert health_data["lts"]["next_major_update"] == "2029-08-17"


def test_lts_docs_present():
    repo_root = Path(__file__).resolve().parents[1]
    lts_policy = repo_root / "docs" / "LTS_POLICY.md"
    releases_doc = repo_root / "docs" / "RELEASES.md"

    assert lts_policy.is_file(), "docs/LTS_POLICY.md must exist"
    assert releases_doc.is_file(), "docs/RELEASES.md must exist"

    policy_content = lts_policy.read_text(encoding="utf-8")
    assert "2026" in policy_content and "2029" in policy_content
    assert "3-Year" in policy_content or "3-year" in policy_content
    assert "v1.0.0" in policy_content
    assert "v2.0.0" in policy_content

    releases_content = releases_doc.read_text(encoding="utf-8")
    assert "v0.1.0" in releases_content
    assert "v1.0.0" in releases_content
    assert "2029" in releases_content
