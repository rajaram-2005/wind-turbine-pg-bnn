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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from app.jobs import JobManager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.aerovigil_pg_bnn.api import app as low_level_model_api
from src.api.app import create_app as create_operations_api
from src.version import APP_VERSION, PRODUCT

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


def create_app() -> FastAPI:
    """Create the separately deployable web UI plus operations API."""
    operations_api = create_operations_api()
    jobs = JobManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Mounted applications do not automatically receive lifespan events.
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(low_level_model_api.router.lifespan_context(low_level_model_api))
            await stack.enter_async_context(operations_api.router.lifespan_context(operations_api))
            yield

    application = FastAPI(
        title="AeroVigilAI Web App",
        version=APP_VERSION,
        description="Standalone operator web application; advisory-only.",
        lifespan=lifespan,
    )

    application.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class JobRequest(BaseModel):
        args: list[str] = []

    @application.post("/api/jobs/{job_type}")
    def submit_job(job_type: str, request: JobRequest) -> dict:
        try: return {"job_id": jobs.submit(job_type, request.args), "status": "Pending"}
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.jobs.get(job_id)
        if not job: raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": job_id, "status": job.status, "logs": job.logs[-100:]}

    @application.get("/api/model-api/{path:path}", include_in_schema=False)
    def model_redirect(path: str):
        suffix = f"/{path}" if path else ""
        return RedirectResponse(url=f"/api/model{suffix}", status_code=308)

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "product": PRODUCT,
            "version": APP_VERSION,
            "advisory_only": True,
            "web_app": True,
            "api": "/api",
            "model_api": "/api/model",
        }

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    # Specific mount must precede `/api`, otherwise the operations mount catches it.
    application.mount("/api/model", low_level_model_api, name="model-api")
    application.mount("/api", operations_api, name="operations-api")
    return application


app = create_app()
