"""Single-process AeroVigil application – the one canonical deployment.

This module is THE deployment boundary for the project. It connects every
previously separate surface behind one host and one port (default ``8080``)::

    uvicorn src.unified_app:app --host 0.0.0.0 --port 8080

Routes
------
``/``
    Static AeroVigilAI browser console (compiled web assets).
``/api``
    The single integrated API: advisory engine, fleet, twin, AeroZip, reports,
    async job queue, hardware-gateway ingestion, and the PG-BNN model surface:

    * ``POST /api/model``            canonical six-signal RUL inference.
    * ``GET  /api/model/info``       model metadata.
    * ``POST /api/model/batch``      batch predictions.
    * ``POST /api/model/stream``     streamed Monte Carlo samples (SSE).
    * ``POST /api/model/trend``      RUL trend across a telemetry sequence.
    * ``POST /api/jobs/{job_type}``  queue a framework job -> ``job_id``.
    * ``GET  /api/jobs/{job_id}``    job status + recent logs.
    * ``POST /api/hardware/stream``  gateway telemetry ingestion.
``/health``
    Health and route discovery for the complete application.

The legacy standalone model server and Gradio dashboard have been consolidated
into this one boundary; there is no separate ``/model-api`` or ``/legacy``
mount. The packaged ``src.aerovigil_pg_bnn`` module is still importable as a
library and remains runnable standalone, but the canonical deployment exposes
exactly one API surface.

CORS is configured so the native Flutter console (Windows/macOS/Android/iOS)
can reach the ``/api`` routes from any localhost/app origin.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.aerovigil_pg_bnn.api import app as _model_app  # lifespan loads the model
from src.api.app import create_app as create_operations_api
from src.api.gateway_routes import router as gateway_router
from src.version import APP_VERSION as VERSION
from src.version import PRODUCT

# Location of the compiled browser-console assets served at ``/``.
_CONSOLE_DIR = Path(__file__).resolve().parents[1] / "web_console" / "dist"

# Default deployment port for the unified application.
DEFAULT_PORT = 8080


class _NoCacheStaticFiles(StaticFiles):
    """Static assets served with ``Cache-Control: no-cache``.

    Browsers must revalidate on every fetch, so an upgraded asset (e.g. a new
    Swagger UI bundle) can never be masked by a stale cached copy.
    """

    def file_response(self, full_path, stat_result, scope):  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


def configured_port() -> int:
    """Resolve the canonical port from deployment environment variables."""
    raw = os.environ.get("PORT", os.environ.get("AEROVIGIL_PORT", str(DEFAULT_PORT)))
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


# Origins allowed to call the /api surface from the native Flutter clients.
_CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://10.0.2.2:8080",
    "http://10.0.2.2",
    "capacitor://localhost",
    "ionic://localhost",
    "https://aerovigil.abacusai.app",
]


def create_app(*, include_dashboard: bool = True) -> FastAPI:
    """Build the unified ASGI application.

    ``include_dashboard`` is retained as a deprecated no-op for backwards
    compatibility: the legacy Gradio dashboard is no longer mounted, so there
    is nothing left to skip. The canonical deployment exposes one API surface.
    """
    del include_dashboard  # deprecated; the legacy dashboard is gone.

    operations_api = create_operations_api()
    # Attach the canonical gateway routes onto the /api surface.
    operations_api.include_router(gateway_router)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # The packaged model app is not mounted, but its lifespan still loads
        # (and later unloads) the PG-BNN into the shared module globals that
        # the /api/model* gateway routes read. Enter it explicitly so model
        # loading is identical to running the model API on its own.
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(_model_app.router.lifespan_context(_model_app))
            await stack.enter_async_context(operations_api.router.lifespan_context(operations_api))
            yield

    application = FastAPI(
        title="AeroVigil unified application",
        version=VERSION,
        description=(
            "One deployment boundary for the browser console, advisory engine, "
            "digital twin, telemetry pipeline, reporting, async job queue, "
            "hardware gateway ingestion, and PG-BNN model API. "
            "Decision-support only."
        ),
        lifespan=lifespan,
    )

    # Cross-origin access for the native Flutter clients.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|10\.0\.2\.2)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.get("/health", tags=["system"])
    def health() -> dict:
        return {
            "status": "ok",
            "product": PRODUCT,
            "version": VERSION,
            "advisory_only": True,
            "port": configured_port(),
            "services": {
                "console": "/",
                "operations_api": "/api",
                "operations_docs": "/api/docs",
                "model_inference": "/api/model",
                "model_info": "/api/model/info",
                "model_batch": "/api/model/batch",
                "model_stream": "/api/model/stream",
                "model_trend": "/api/model/trend",
                "jobs": "/api/jobs/{job_type}",
                "hardware_stream": "/api/hardware/stream",
            },
            "digital_twin": {
                "assets_tracked": len(operations_api.state.twins),
                "max_assets": operations_api.state.twin_max_assets,
            },
            "agent_mesh": {
                "team_id": "CYBER_PRIME_DUAL_AGENT",
                "agents": ["MIKA", "KAI"],
                "status": "connected",
                "evidence_path": ["SCADA", "PG-BNN", "ISO 281", "TWIN", "FLEET", "HUMAN"],
            },
        }

    # Mount the single API before the catch-all static console.
    application.mount("/api", operations_api, name="operations-api")

    # Self-hosted docs assets (Swagger UI): mounted with no-cache so a stale
    # bundle can never linger in a browser/proxy cache after an upgrade.
    _vendor_dir = _CONSOLE_DIR / "vendor"
    if _vendor_dir.is_dir():
        application.mount(
            "/vendor", _NoCacheStaticFiles(directory=str(_vendor_dir)), name="vendor"
        )

    # Root path serves the compiled AeroVigilAI browser console.
    if _CONSOLE_DIR.is_dir():
        application.mount("/", StaticFiles(directory=str(_CONSOLE_DIR), html=True), name="console")
    else:  # pragma: no cover - console assets absent in minimal installs

        @application.get("/", tags=["system"])
        def index() -> dict:
            return {
                "product": "AeroVigil",
                "message": "Browser console assets not compiled; API mode active",
                "health": "/health",
                "operations_api": "/api",
                "model_inference": "/api/model",
                "advisory_only": True,
            }

    return application


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("src.unified_app:app", host="0.0.0.0", port=configured_port(), reload=False)
