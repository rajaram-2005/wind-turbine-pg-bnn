"""Offline OpenAPI docs contract: both mounted APIs must render Swagger UI
with self-hosted assets (no CDN) that support the OpenAPI version FastAPI
emits. Guards the 3.1-spec vs swagger-ui-4.x "no valid version field" bug."""

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.unified_app import create_app  # noqa: E402

_VENDOR = Path(__file__).resolve().parents[1] / "web_console" / "dist" / "vendor" / "swagger"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(include_dashboard=False)) as c:
        yield c


def test_docs_pages_use_self_hosted_assets(client):
    for path in ("/api/docs", "/model-api/docs"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert "/vendor/swagger/swagger-ui-bundle.js" in resp.text
        assert "/vendor/swagger/swagger-ui.css" in resp.text
        assert "cdn.jsdelivr.net" not in resp.text


def test_vendor_assets_exist_and_support_openapi_31(client):
    bundle = client.get("/vendor/swagger/swagger-ui-bundle.js")
    css = client.get("/vendor/swagger/swagger-ui.css")
    assert bundle.status_code == 200
    assert css.status_code == 200
    # Swagger UI must be 5.x to parse the OpenAPI 3.1 FastAPI/Pydantic v2 spec.
    match = re.search(r'PACKAGE_VERSION:"(\d+)\.', bundle.text)
    assert match is not None, "swagger-ui bundle version marker missing"
    assert int(match.group(1)) >= 5, "swagger-ui must be >= 5.x for OpenAPI 3.1"


def test_specs_declare_a_supported_openapi_version(client):
    for path in ("/api/openapi.json", "/model-api/openapi.json"):
        spec = client.get(path).json()
        version = spec.get("openapi") or spec.get("swagger")
        assert version is not None
        assert version.startswith("3.") or version == "2.0"
