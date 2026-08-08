"""Canonical `/api` gateway routes for the unified AeroVigil deployment.

These routes are attached to the operations API (mounted at ``/api`` by
:mod:`src.unified_app`) so the whole platform presents a single, coherent API
surface on one host and one port:

* ``POST /api/model``            – canonical PG-BNN inference endpoint.
* ``ANY  /api/model-api``        – permanent (308) redirect to ``/api/model``.
* ``POST /api/jobs/{job_type}``  – queue a framework job, returns ``job_id``.
* ``GET  /api/jobs/{job_id}``    – job status + recent execution logs.
* ``POST /api/hardware/stream``  – ingestion endpoint for gateway telemetry.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.jobs import ALLOWED_JOB_TYPES, get_job_manager

router = APIRouter()

# In-memory ring buffer of the most recent normalized hardware readings.
_HARDWARE_BUFFER: deque[dict[str, Any]] = deque(maxlen=5000)


# ----------------------------------------------------------------- model
@router.post("/model", tags=["model"])
async def model_inference(request: Request) -> dict[str, Any]:
    """Canonical model-inference endpoint.

    Accepts the same six-signal telemetry payload as the low-level PG-BNN API
    and returns the mean prediction together with epistemic/aleatoric
    uncertainty heads.
    """
    from src.aerovigil_pg_bnn.api import TelemetryInput, predict

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    n_samples = int(payload.pop("n_mcmc_samples", 100)) if isinstance(payload, dict) else 100
    try:
        telemetry = TelemetryInput(**payload)
    except Exception as exc:  # noqa: BLE001 - surface validation errors
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = await predict(request, telemetry, n_mcmc_samples=n_samples)
    # ``predict`` returns a pydantic model; normalise to a plain dict.
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)


@router.api_route(
    "/model-api",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
async def model_api_redirect() -> RedirectResponse:
    """Permanent redirect from the legacy ``/api/model-api`` path."""
    return RedirectResponse(url="/api/model", status_code=308)


# ------------------------------------------------------------------ jobs
class JobQueuedResponse(BaseModel):
    """Response for a freshly queued job."""

    job_id: str
    job_type: str
    status: str


@router.post("/jobs/{job_type}", response_model=JobQueuedResponse, tags=["jobs"])
async def queue_job(job_type: str, request: Request) -> JobQueuedResponse:
    """Queue a framework job and return its unique identifier.

    ``job_type`` must be one of: physics, train, federated, export,
    active-learning/active-sample, shap/explain. Extra CLI flags may be passed
    as a JSON body ``{"args": ["--config", "configs/default.yaml"]}``.
    """
    manager = get_job_manager()
    if not manager.is_allowed(job_type):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job type {job_type!r}. Allowed: {sorted(ALLOWED_JOB_TYPES)}",
        )
    extra_args: list[str] = []
    try:
        body = await request.json()
        if isinstance(body, dict) and isinstance(body.get("args"), list):
            extra_args = [str(a) for a in body["args"]]
    except Exception:  # noqa: BLE001 - empty/invalid body is fine
        extra_args = []

    job_id = await manager.queue(job_type, extra_args)
    return JobQueuedResponse(job_id=job_id, job_type=job_type, status="Pending")


@router.get("/jobs/{job_id}", tags=["jobs"])
async def job_status(job_id: str) -> dict[str, Any]:
    """Return the status and recent logs for a queued job."""
    rec = get_job_manager().get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id {job_id!r}")
    return rec.to_public()


# -------------------------------------------------------------- hardware
class HardwareReading(BaseModel):
    """A single normalized reading from a field gateway."""

    gateway_id: str
    turbine_id: Optional[str] = None
    signal: str
    value: float
    unit: Optional[str] = None
    quality: Optional[str] = "good"
    timestamp: str


class HardwareStreamBatch(BaseModel):
    """A batch of normalized readings POSTed by ``hardware_agent.py``."""

    gateway_id: str
    readings: list[HardwareReading] = Field(default_factory=list)


@router.post("/hardware/stream", tags=["hardware"])
async def hardware_stream(batch: HardwareStreamBatch) -> dict[str, Any]:
    """Ingest normalized telemetry from an industrial gateway agent."""
    received = 0
    for reading in batch.readings:
        _HARDWARE_BUFFER.append(reading.model_dump())
        received += 1
    return {
        "ack": True,
        "gateway_id": batch.gateway_id,
        "received": received,
        "buffered_total": len(_HARDWARE_BUFFER),
        "server_time": time.time(),
    }


@router.post("/telemetry/upload", tags=["telemetry"])
async def telemetry_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept an offline SCADA export uploaded from the native console.

    The file bytes are read and acknowledged with lightweight metadata; the
    contents are intentionally not persisted here (advisory-only ingestion).
    """
    data = await file.read()
    return {
        "ack": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": len(data),
        "received_at": time.time(),
    }


@router.get("/hardware/latest", tags=["hardware"])
async def hardware_latest(limit: int = 100) -> dict[str, Any]:
    """Return the most recent normalized readings for dashboards."""
    limit = max(1, min(int(limit), 1000))
    items = list(_HARDWARE_BUFFER)[-limit:]
    return {"count": len(items), "readings": items}
