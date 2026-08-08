"""Standalone AeroVigilAI web application.

Run from the repository root::

    uvicorn app.server:app --host 0.0.0.0 --port 8080

The browser app and the advisory API share one origin.  This keeps API calls
relative (``/api/...``), so the app works locally, in Docker, and behind a
reverse proxy without exposing a localhost URL to users.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.app import create_app as create_operations_api
from src.version import APP_VERSION, PRODUCT

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


def create_app() -> FastAPI:
    """Create the separately deployable web UI plus operations API."""
    operations_api = create_operations_api()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Mounted applications do not automatically receive lifespan events.
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(operations_api.router.lifespan_context(operations_api))
            yield

    application = FastAPI(
        title="AeroVigilAI Web App",
        version=APP_VERSION,
        description="Standalone operator web application; advisory-only.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "product": PRODUCT,
            "version": APP_VERSION,
            "advisory_only": True,
            "web_app": True,
            "api": "/api",
        }

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    application.mount("/api", operations_api, name="operations-api")
    return application


app = create_app()
