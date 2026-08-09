"""Canonical `/api` gateway routes for the unified AeroVigil deployment.

These routes are attached to the operations API (mounted at ``/api`` by
:mod:`src.unified_app`) so the whole platform presents a single, coherent API
surface on one host and one port:

* ``POST /api/model``            – canonical PG-BNN inference endpoint.
* ``ANY  /api/model-api``        – permanent (308) redirect to ``/api/model``.
* ``POST /api/jobs/{job_type}``  – queue a framework job, returns ``job_id``.
* ``GET  /api/jobs``             – list recent jobs.
* ``GET  /api/jobs/{job_id}``    – job status + recent execution logs.
* ``POST /api/hardware/stream``  – gateway telemetry ingestion.
* ``GET  /api/hardware/latest``  – read-back of persisted readings.
* ``GET  /api/fleet/summary``    – fleet aggregate from the durable store.
* ``GET  /api/imports``          – offline import log.
* ``GET  /api/system/stats``     – durable-store statistics.

Hardware-stream ingestion is *not* fire-and-forget: every batch is persisted
to the durable store (:mod:`src.data.store`), the affected digital twins are
updated (advisory computed via the attached serving PG-BNN when available),
and the fleet report is regenerated automatically.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.jobs import ALLOWED_JOB_TYPES, get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Path of the pre-trained demo bundle used to serve advisories for hardware
# streams when no AV_MODEL_PATH is configured.
_DEMO_BUNDLE = Path(__file__).resolve().parents[2] / "artifacts" / "pg_bnn_demo" / "bnn_demo.pt"


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

    ``job_type`` must be one of: physics, train, evaluate, federated, export,
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


@router.get("/jobs", tags=["jobs"])
async def list_jobs(limit: int = 25) -> dict[str, Any]:
    """List recently queued jobs (newest first) for console dashboards."""
    limit = max(1, min(int(limit), 200))
    records = get_job_manager().list_jobs(limit=limit)
    return {"count": len(records), "jobs": [r.to_public() for r in records]}


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
    turbine_id: str | None = None
    signal: str
    value: float
    unit: str | None = None
    quality: str | None = "good"
    timestamp: str


class HardwareStreamBatch(BaseModel):
    """A batch of normalized readings POSTed by ``hardware_agent.py``."""

    gateway_id: str
    readings: list[HardwareReading] = Field(default_factory=list)


# Canonical twin-telemetry channel aliases for gateway signals. Hardware
# connectors often emit OEM names (generator_rpm, gearbox_temp, ...); this
# map folds them onto the digital-twin Telemetry schema.
SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {
    "vibration_mms": ("vibration_mms", "vibration_rms", "vib_rms", "vibration"),
    "temperature_c": (
        "temperature_c",
        "gearbox_temp",
        "bearing_temp",
        "generator_temp",
        "oil_temp",
    ),
    "rpm": ("rpm", "generator_rpm", "rotor_rpm", "hss_rpm", "rpm_hss"),
    "oil_viscosity_cst": ("oil_viscosity_cst", "oil_viscosity", "viscosity"),
    "load_pct": ("load_pct", "load_percent"),
}
_MIN_CHANNELS_FOR_TWIN = 4  # allow one missing channel without dropping the batch


def _stream_heuristic_enabled() -> bool:
    """Whether streams fall back to the documented heuristic advisory when no
    serving model is loaded (default on for the demo; set
    ``AV_STREAM_HEURISTIC=0`` to disable)."""
    return os.environ.get("AV_STREAM_HEURISTIC", "1") != "0"


def _ensure_stream_serving(state) -> None:
    """Lazily attach a serving PG-BNN to the operations app for streams.

    ``state`` is the operations API's ``app.state`` (it owns the ``serving``
    attribute). Uses ``AV_MODEL_PATH`` when set, else the bundled demo bundle
    so that hardware streams produce real model advisories out of the box.
    The result is cached on ``state.serving`` for the life of the process.
    """
    if getattr(state, "serving", None) is not None:
        return
    path = os.environ.get("AV_MODEL_PATH") or os.environ.get("AV_STREAM_MODEL")
    if not path and _DEMO_BUNDLE.is_file():
        path = str(_DEMO_BUNDLE)
    if not path:
        logger.info(
            "[hardware] no serving model configured; streams update twins without model advisories"
        )
        return
    try:
        from src.models.serving import load_serving_model

        # The bundled demo checkpoint predates bundle metadata; the legacy
        # kwargs reconstruct the BayesianNeuralNetwork the serving layer uses.
        state.serving = load_serving_model(path, in_features=25, hidden_sizes=(64, 64))
        logger.info("[hardware] serving PG-BNN loaded from %s", path)
    except Exception as exc:  # noqa: BLE001 - degraded mode, never crash ingestion
        logger.warning("[hardware] could not load serving model %s: %s", path, exc)


def _group_signals(batch: HardwareStreamBatch) -> dict[str, list[dict[str, Any]]]:
    """Group a batch's readings by turbine_id ('' when unspecified)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for reading in batch.readings:
        turbine = reading.turbine_id or ""
        groups.setdefault(turbine, []).append(reading.model_dump())
    return groups


def _to_telemetry(signals: list[dict[str, Any]], spec=None) -> dict[str, float] | None:
    """Map gateway signal names onto the canonical twin Telemetry channels.

    Two channels are derived when absent so a standard SCADA set (RPM,
    temperature, vibration, power, wind) still drives the twin:

    * ``load_pct``      – from ``power_output`` kW and the spec's rated MW.
    * ``oil_viscosity_cst`` – spec mid-range when no viscosity signal exists.

    Returns ``None`` when fewer than :data:`_MIN_CHANNELS_FOR_TWIN` channels
    are present (the twin cannot be updated meaningfully).
    """
    by_signal: dict[str, float] = {}
    for s in signals:
        by_signal.setdefault(str(s["signal"]), float(s["value"]))

    channels: dict[str, float] = {}
    for canonical, aliases in SIGNAL_ALIASES.items():
        for alias in aliases:
            if alias in by_signal:
                channels[canonical] = by_signal[alias]
                break

    if "load_pct" not in channels and "power_output" in by_signal and spec is not None:
        rated_kw = max(float(getattr(spec, "rated_power_mw", 1.5)) * 1000.0, 1.0)
        channels["load_pct"] = round(
            max(0.0, min(100.0, by_signal["power_output"] / rated_kw * 100.0)), 2
        )
    if "oil_viscosity_cst" not in channels and spec is not None:
        channels["oil_viscosity_cst"] = (
            float(spec.viscosity_min_cst) + float(spec.viscosity_max_cst)
        ) / 2.0

    if len(channels) < _MIN_CHANNELS_FOR_TWIN:
        return None
    return channels


def _stream_heuristic_bnn_state(tel: dict[str, float], spec) -> dict[str, Any]:
    """Deterministic, documented demo heuristic for RUL/uncertainty.

    Used only when no serving PG-BNN is available so hardware streams still
    produce advisories (clearly labeled ``advisory_source="stream-heuristic"``
    in the asset row; never used for actuation). RUL scales inversely with the
    dominant stress ratio (vibration / temperature / load vs spec limits).
    """
    vib = tel.get("vibration_mms", 0.0) / max(float(spec.vibration_limit_mms), 1e-9)
    temp = tel.get("temperature_c", 0.0) / max(float(spec.temperature_limit_c), 1e-9)
    load = tel.get("load_pct", 0.0) / 100.0
    risk = max(0.0, min(1.0, vib), min(1.0, temp), min(1.0, load))
    rul_days = round(max(14.0, min(365.0, 365.0 * (1.0 - risk) ** 2)), 1)
    return {
        "predicted_rul_days": rul_days,
        "epistemic_uncertainty": round(0.25 * rul_days, 2),
        "aleatoric_uncertainty": round(0.10 * rul_days, 2),
    }


def _health_score(tel: dict[str, float], spec) -> float:
    """Deterministic 0-100 fleet health score from twin telemetry.

    Stress is the dominant ratio (vibration / temperature / load vs spec
    limits); health decays from 100 (at ≤50% of limits) to 0 (at 100%+).
    """
    vib = tel.get("vibration_mms", 0.0) / max(float(spec.vibration_limit_mms), 1e-9)
    temp = tel.get("temperature_c", 0.0) / max(float(spec.temperature_limit_c), 1e-9)
    load = tel.get("load_pct", 0.0) / 100.0
    stress = max(vib, temp, load)
    excess = max(0.0, stress - 0.5)
    return round(max(0.0, min(100.0, 100.0 * (1.0 - 0.65 * excess / 0.5))), 2)


def _fleet_status(health: float) -> str:
    if health >= 80:
        return "Healthy"
    if health >= 65:
        return "Watch"
    return "Alert"


@router.post("/hardware/stream", tags=["hardware"])
async def hardware_stream(batch: HardwareStreamBatch, request: Request) -> dict[str, Any]:
    """Ingest normalized telemetry from an industrial gateway agent.

    Beyond acknowledging the batch this now:

    1. Persists every reading to the durable store.
    2. Updates each affected digital twin (advisory via the serving PG-BNN
       when loaded, else twin physics metrics).
    3. Upserts the per-turbine fleet-health row.
    4. Regenerates and persists the fleet report.
    """
    from src.api.app import get_or_create_twin
    from src.data.store import get_store
    from src.utils.schema import BNNState
    from src.utils.schema import Telemetry as TwinTelemetry

    store = get_store()
    received = store.record_telemetry([r.model_dump() for r in batch.readings])

    # Attach a serving model once so advisories come from the trained PG-BNN.
    _ensure_stream_serving(request.app.state)
    serving_loaded = getattr(request.app.state, "serving", None) is not None

    assets_updated: list[dict[str, Any]] = []
    advisories_computed = 0
    for turbine_id, signals in _group_signals(batch).items():
        if not turbine_id:
            # Readings that did not name a turbine belong to the gateway.
            turbine_id = batch.gateway_id
        try:
            twin = get_or_create_twin(request.app.state, turbine_id, "GE-1.5")
            tel = _to_telemetry(signals, twin.spec)
            if tel is None:
                continue
            # Model path when a serving PG-BNN is attached; otherwise a
            # deterministic, clearly-labeled heuristic bnn_state so streams
            # still produce advisory records out of the box.
            bnn_state = None
            advisory_source = None
            if not serving_loaded and _stream_heuristic_enabled():
                heuristic = _stream_heuristic_bnn_state(tel, twin.spec)
                bnn_state = BNNState(**heuristic)
                advisory_source = "stream-heuristic"
            state_record = twin.update_state(TwinTelemetry(**tel), bnn_state)
            if advisory_source:
                # Label the heuristic path explicitly so twin status, the
                # asset row and reports all say where the advisory came from.
                state_record = {**state_record, "advisory_source": advisory_source}
                if twin.state_history:
                    twin.state_history[-1]["advisory_source"] = advisory_source
        except Exception as exc:  # noqa: BLE001 - one bad turbine must not fail the batch
            logger.warning("[hardware] twin update failed for %s: %s", turbine_id, exc)
            continue

        store.record_twin_state(turbine_id, state_record)
        advisory = state_record.get("advisory") or {}
        rul = advisory.get("predicted_rul_days")
        health = _health_score(tel, twin.spec)
        asset = {
            "turbine_id": turbine_id,
            "gateway_id": batch.gateway_id,
            "model_key": twin.spec.model_name,
            "status": _fleet_status(health),
            "health_score": health,
            "availability": round(98.0 - max(0.0, 40.0 * (1.0 - health / 100.0)), 2),
            "predicted_rul_days": rul,
            "epistemic_std": advisory.get("epistemic_std"),
            "aleatoric_std": advisory.get("aleatoric_std"),
            "inspection_window_days": advisory.get("suggested_inspection_window_days"),
            "advisory_source": advisory_source or state_record.get("advisory_source"),
            "last_seen": state_record.get("timestamp"),
        }
        store.upsert_asset(asset)
        assets_updated.append({"asset": asset, "advisory": advisory})

    # Regenerate + persist the fleet report whenever assets changed.
    if assets_updated:
        try:
            from src.reporting.reports import build_fleet_report

            records = [a["advisory"] for a in assets_updated if a["advisory"]]
            if records:
                body = build_fleet_report(
                    records, title="Fleet RUL advisory report (hardware stream)"
                )
                store.record_report(
                    "fleet",
                    body,
                    title="Fleet RUL advisory report (hardware stream)",
                    meta={
                        "n_assets": len(records),
                        "generated_by": "hardware_stream",
                        "serving_model_loaded": serving_loaded,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - reporting must not break ingestion
            logger.warning("[hardware] fleet report regeneration failed: %s", exc)

    assets_updated = [a["asset"] for a in assets_updated]
    advisories_computed = len([a for a in assets_updated if a["predicted_rul_days"] is not None])

    return {
        "ack": True,
        "gateway_id": batch.gateway_id,
        "received": received,
        "turbines_updated": len(assets_updated),
        "advisories_computed": advisories_computed,
        "serving_model_loaded": serving_loaded,
        "heuristic_advisories": serving_loaded is False and _stream_heuristic_enabled(),
        "assets": assets_updated,
        "server_time": time.time(),
    }


@router.get("/hardware/latest", tags=["hardware"])
async def hardware_latest(limit: int = 100) -> dict[str, Any]:
    """Return the most recent persisted readings for dashboards."""
    from src.data.store import get_store

    limit = max(1, min(int(limit), 1000))
    items = get_store().latest_telemetry(limit)
    return {"count": len(items), "readings": items}


@router.post("/telemetry/upload", tags=["telemetry"])
async def telemetry_upload(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency marker
    source: str = Form("api"),  # noqa: B008 - FastAPI dependency marker
) -> dict[str, Any]:
    """Accept an offline SCADA export uploaded from the native console.

    Bytes are acknowledged with lightweight metadata and the import is
    recorded in the durable store (USB / cloud / API provenance).
    """
    from src.data.store import get_store

    data = await file.read()
    import_id = get_store().record_import(
        filename=file.filename or "unnamed",
        content_type=file.content_type,
        size_bytes=len(data),
        source=source or "api",
    )
    return {
        "ack": True,
        "import_id": import_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": len(data),
        "source": source,
        "received_at": time.time(),
    }


@router.get("/imports", tags=["telemetry"])
async def list_imports(limit: int = 50) -> dict[str, Any]:
    """List recently recorded offline imports (USB / cloud / API)."""
    from src.data.store import get_store

    limit = max(1, min(int(limit), 500))
    items = get_store().list_imports(limit)
    return {"count": len(items), "imports": items}


class CloudImportRequest(BaseModel):
    """A signed HTTPS URL for server-side SCADA import."""

    url: str = Field(..., description="Signed HTTPS object URL (pre-signed S3/GCS/Azure SAS)")


@router.post("/telemetry/import", tags=["telemetry"])
async def telemetry_import(req: CloudImportRequest) -> dict[str, Any]:
    """Import a SCADA export from a signed HTTPS cloud URL.

    The server fetches the object server-side (never the native client),
    records the import with ``source=cloud`` and returns its metadata. Only
    ``https://`` URLs are accepted (``http://localhost`` is tolerated for
    local development). The payload itself is advisory-only: bytes are
    acknowledged, not persisted.
    """
    from urllib.parse import urlparse

    from src.data.store import get_store

    parsed = urlparse(req.url)
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"):
        raise HTTPException(
            status_code=422, detail="Only https:// URLs (or http://localhost) are allowed"
        )
    if parsed.scheme not in ("https", "http"):
        raise HTTPException(status_code=422, detail=f"Unsupported URL scheme: {parsed.scheme!r}")

    filename = os.path.basename(parsed.path) or "cloud_import.bin"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(req.url)
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:  # noqa: BLE001 - remote failure is a client error here
        raise HTTPException(status_code=502, detail=f"Cloud fetch failed: {exc}") from exc

    content_type = resp.headers.get("content-type") if "resp" in locals() else None
    import_id = get_store().record_import(
        filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        source="cloud",
    )
    return {
        "ack": True,
        "import_id": import_id,
        "filename": filename,
        "content_type": content_type,
        "bytes": len(data),
        "source": "cloud",
        "received_at": time.time(),
    }


@router.get("/fleet/summary", tags=["fleet"])
async def fleet_summary() -> dict[str, Any]:
    """Fleet aggregate computed from the durable assets table."""
    from src.data.store import get_store

    return get_store().summarize_fleet()


@router.get("/system/stats", tags=["system"])
async def system_stats() -> dict[str, Any]:
    """Durable-store statistics (row counts + database location)."""
    from src.data.store import get_store

    stats = get_store().stats()
    try:
        stats["jobs"] = len(get_job_manager().list_jobs(limit=500))
    except Exception:  # noqa: BLE001 - jobs stats are best-effort
        stats["jobs"] = None
    return stats


@router.get("/reports", tags=["system"])
async def list_reports(kind: str | None = None, limit: int = 20) -> dict[str, Any]:
    """List persisted fleet/advisory reports (durable store)."""
    from src.data.store import get_store

    limit = max(1, min(int(limit), 200))
    items = get_store().list_reports(kind=kind, limit=limit)
    return {"count": len(items), "reports": items}


@router.get("/reports/latest", tags=["system"])
async def latest_report(kind: str = "fleet") -> dict[str, Any]:
    """Return the most recent persisted report of a given kind."""
    from fastapi import HTTPException

    from src.data.store import get_store

    rec = get_store().latest_report(kind)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"No report found for kind={kind!r}")
    return rec
