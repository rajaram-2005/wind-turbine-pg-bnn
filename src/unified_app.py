"""Single-process AeroVigil application.

This is the canonical deployment boundary for the project.  It exposes the
operator dashboard, the advisory/digital-twin/telemetry API, and the low-level
PG-BNN inference API on one host and one port::

    uvicorn src.unified_app:app --host 0.0.0.0 --port 8000

Routes
------
``/``
    Gradio operator dashboard.
``/api``
    Integrated advisory API (fleet, twin, AeroZip, reports).
``/model-api``
    Low-level six-signal PG-BNN prediction API.
``/health``
    Health and route discovery for the complete application.

The sub-applications remain importable for backwards compatibility, but new
installations should run this module so operators never have to coordinate
multiple servers or ports.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from src.aerovigil_pg_bnn.api import app as model_api
from src.api.app import create_app as create_operations_api
from src.version import APP_VERSION as VERSION
from src.version import PRODUCT


def create_app(*, include_dashboard: bool = True) -> FastAPI:
    """Build the unified ASGI application.

    ``include_dashboard=False`` is useful for lightweight probes and tests on
    machines that installed only the ``api`` optional dependency.
    """

    operations_api = create_operations_api()

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
            "One deployment boundary for the dashboard, advisory engine, "
            "digital twin, telemetry pipeline, reporting, and PG-BNN model API. "
            "Decision-support only."
        ),
        lifespan=lifespan,
    )

    @application.get("/health", tags=["system"])
    def health() -> dict:
        return {
            "status": "ok",
            "product": PRODUCT,
            "version": VERSION,
            "advisory_only": True,
            "services": {
                "dashboard": "/" if include_dashboard else None,
                "operations_api": "/api",
                "operations_docs": "/api/docs",
                "model_api": "/model-api",
                "model_docs": "/model-api/docs",
            },
            "digital_twin": {
                "assets_tracked": len(operations_api.state.twins),
                "max_assets": operations_api.state.twin_max_assets,
            },
        }

    # Mount APIs before the catch-all dashboard route.
    application.mount("/api", operations_api, name="operations-api")
    application.mount("/model-api", model_api, name="model-api")

    if include_dashboard:
        try:
            import gradio as gr

            from gradio_app.app import build_interface
        except ImportError as exc:  # clear install guidance instead of a partial app
            raise RuntimeError(
                "The unified dashboard requires demo dependencies; "
                "install with `pip install -e '.[api,demo]'`."
            ) from exc

        dashboard = build_interface()
        application = gr.mount_gradio_app(
            application,
            dashboard,
            path="/",
            allowed_paths=None,
        )
    else:

        @application.get("/", tags=["system"])
        def index() -> dict:
            return {
                "product": "AeroVigil",
                "message": "Unified API mode (dashboard disabled)",
                "health": "/health",
                "operations_api": "/api",
                "model_api": "/model-api",
                "advisory_only": True,
            }

    return application


app = create_app()
