"""Single-process AeroVigil application – the one canonical deployment.

This module is THE deployment boundary for the project. It connects every
previously separate surface behind one host and one port (default ``8080``)::

    uvicorn src.unified_app:app --host 0.0.0.0 --port 8080

Routes
------
``/``
    Static AeroVigilAI browser console (compiled web assets).
``/api``
    Integrated advisory API (fleet, twin, AeroZip, reports) plus the canonical
    gateway routes:

    * ``POST /api/model``            canonical PG-BNN inference endpoint.
    * ``ANY  /api/model-api``        308 permanent redirect to ``/api/model``.
    * ``POST /api/jobs/{job_type}``  queue a framework job -> ``job_id``.
    * ``GET  /api/jobs/{job_id}``    job status + recent logs.
    * ``POST /api/hardware/stream``  gateway telemetry ingestion.
``/model-api``
    Low-level six-signal PG-BNN prediction API (retained for compatibility).
``/legacy``
    Deprecated Gradio dashboard (redirect notice only; headless API retained).
``/health``
    Health and route discovery for the complete application.

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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.aerovigil_pg_bnn.api import app as model_api
from src.api.app import create_app as create_operations_api
from src.api.gateway_routes import router as gateway_router
from src.version import APP_VERSION as VERSION
from src.version import PRODUCT

# Location of the compiled browser-console assets served at ``/``.
_CONSOLE_DIR = Path(__file__).resolve().parents[1] / "web_console" / "dist"

# Default deployment port for the unified application.
DEFAULT_PORT = 8080

# Fallback page for the deprecated Gradio dashboard path when the Gradio
# optional dependency is not installed: a static deprecation notice that
# auto-redirects to the canonical console. Mirrors gradio_app/deprecated.py.
_LEGACY_FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url=/" />
  <title>AeroVigil – dashboard moved</title>
  <style>
    body { margin:0; min-height:100vh; display:grid; place-items:center;
      background:#05121a; color:#e6f2f2; font-family:system-ui,sans-serif; }
    .card { max-width:640px; text-align:center; padding:48px 32px;
      background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
      border-radius:16px; }
    h1 { color:#2dd4bf; }
    a { display:inline-block; margin-top:16px; padding:12px 22px; color:#fff;
      background:linear-gradient(135deg,#0d9488,#06b6d4); border-radius:10px;
      text-decoration:none; font-weight:600; }
  </style>
</head>
<body>
  <div class="card">
    <h1>⚠️ This dashboard has moved</h1>
    <p>The legacy AeroVigil Gradio dashboard is <strong>deprecated</strong>.
       All operator tooling now lives in the AeroVigilAI browser console on
       the single canonical deployment at <strong>http://localhost:8080/</strong>.</p>
    <p>You are being redirected…</p>
    <a href="/" target="_top">Open the AeroVigilAI console →</a>
  </div>
</body>
</html>
"""

# Origins allowed to call the /api surface from the native Flutter clients.
_CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "capacitor://localhost",
    "ionic://localhost",
    "https://aerovigil.abacusai.app",
]


def create_app(*, include_dashboard: bool = True) -> FastAPI:
    """Build the unified ASGI application.

    ``include_dashboard=False`` skips mounting the deprecated Gradio UI (useful
    for lightweight probes and tests on machines that installed only the
    ``api`` optional dependency).
    """

    operations_api = create_operations_api()
    # Attach the canonical gateway routes onto the /api surface.
    operations_api.include_router(gateway_router)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Starlette does not automatically run lifespan handlers for mounted
        # applications. Enter both explicitly so model loading and cleanup are
        # identical to running either API on its own.
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(model_api.router.lifespan_context(model_api))
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
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
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
            "port": DEFAULT_PORT,
            "services": {
                "console": "/",
                "operations_api": "/api",
                "operations_docs": "/api/docs",
                "model_inference": "/api/model",
                "model_api": "/model-api",
                "model_docs": "/model-api/docs",
                "jobs": "/api/jobs/{job_type}",
                "hardware_stream": "/api/hardware/stream",
                "legacy_dashboard": "/legacy" if include_dashboard else None,
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

    # Mount APIs before the catch-all static console.
    application.mount("/api", operations_api, name="operations-api")
    application.mount("/model-api", model_api, name="model-api")

    # Deprecated Gradio dashboard – visible UI is a redirect notice only. A
    # static deprecation page with an auto-redirect to the canonical console
    # is served at /legacy so the route works on every Gradio version (and
    # when Gradio is not installed at all). The headless Gradio prediction
    # API stays available for legacy scripts via gradio_app/deprecated.py.
    if include_dashboard:
        application.add_api_route("/legacy", _legacy_fallback, methods=["GET"], include_in_schema=False)

    # Root path serves the compiled AeroVigilAI browser console.
    if _CONSOLE_DIR.is_dir():
        application.mount(
            "/", StaticFiles(directory=str(_CONSOLE_DIR), html=True), name="console"
        )
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


def _legacy_fallback() -> HTMLResponse:
    """Static deprecation notice served when Gradio is not installed."""
    return HTMLResponse(content=_LEGACY_FALLBACK_HTML, status_code=200)


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    port = int(os.environ.get("AEROVIGIL_PORT", DEFAULT_PORT))
    uvicorn.run("src.unified_app:app", host="0.0.0.0", port=port, reload=False)
