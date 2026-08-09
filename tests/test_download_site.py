"""The cross-platform download site is part of the canonical deployment.

``/download`` resolves to the download page, which advertises every supported
platform and links the GitHub repository that CI publishes release binaries to.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.unified_app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(include_dashboard=False)) as c:
        yield c


def test_download_route_redirects_to_the_site(client):
    resp = client.get("/download", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"] == "/download.html"


def test_download_site_covers_every_platform(client):
    resp = client.get("/download.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    html = resp.text
    for platform in ("Windows", "macOS", "Linux", "Android", "iOS", "Web console"):
        assert platform in html
    # Release plumbing: CI workflow asset names + repository links.
    for asset in (
        "aerovigil-windows-x64.zip",
        "aerovigil-macos-universal.zip",
        "aerovigil-linux-x64.tar.gz",
        "aerovigil-android.apk",
    ):
        assert asset in html
    assert "rajaram-2005/wind-turbine-pg-bnn" in html
    # Server install paths stay visible for self-hosting.
    assert "python -m src.unified_app" in html.replace("&amp;", "&")
    assert "docker compose up aerovigil" in html


def test_health_advertises_the_download_site(client):
    body = client.get("/health").json()
    assert body["services"]["downloads"] == "/download"


def test_console_links_to_the_download_site(client):
    html = client.get("/").text
    assert 'href="/download"' in html
