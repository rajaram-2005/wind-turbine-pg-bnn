"""FastAPI advisory service for AeroVigil (wind-turbine-pg-bnn engine).

AeroVigil v1.0.0 — https://aerovigil.abacusai.app

Run locally::

    uvicorn src.api.app:app --reload

Endpoints
---------
``GET  /``                 service info + safety banner
``GET  /health``           liveness probe (always ``advisory_only=True``)
``POST /advisory``         single-turbine snapshot -> :class:`AdvisoryResponse`
``POST /advisory/fleet``   batch of snapshots -> :class:`FleetResponse`

Every response is ADVISORY-ONLY. Outputs are screened by
:func:`src.utils.safety.enforce_safety_contract` before they leave the service,
and the response schemas deliberately model no actuation, throttle, LOTO, or
part-number fields. The service consumes the pre-computed ``bnn_state`` block
supplied in each request payload (sufficient for advisory/decision-support use
without a trained feature-extraction pipeline).
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AdvisoryRequest,
    AdvisoryResponse,
    FleetRequest,
    FleetResponse,
    FleetSummary,
    HealthResponse,
    TelemetryCompressRequest,
    TelemetryCompressResponse,
    TelemetryRestoreRequest,
    TelemetryRestoreResponse,
    TwinSimulateRequest,
)
from src.eval.calibration import expected_asset_utilization
from src.models.predictor import run_advisory
from src.utils.safety import enforce_safety_contract
from src.utils.schema import TurbinePayload

VERSION = "1.0.0"
PRODUCT = "AeroVigil"
WEBSITE = "https://aerovigil.abacusai.app"
ENV_MODEL_PATH = "AV_MODEL_PATH"
SAFETY_BANNER = (
    "AeroVigil v1.0.0 (wind-turbine-pg-bnn advisory service) — "
    "DECISION-SUPPORT ONLY. https://aerovigil.abacusai.app. "
    "Outputs are not actuation commands; review by a qualified operator "
    "and an OEM documentation cross-check are required before any "
    "maintenance action. See docs/SAFETY.md."
)


def _serving_model_path() -> str | None:
    """Where to load the serving PG-BNN from, if anywhere.

    Precedence: ``AV_MODEL_PATH`` environment variable (deployment knob),
    then ``serving.model_path`` in configs/default.yaml (normally unset).
    """
    path = os.environ.get(ENV_MODEL_PATH)
    if path:
        return path
    try:
        from src.utils.config import load_config

        return load_config().serving.model_path
    except Exception:
        return None


def _advisory_or_422(payload: TurbinePayload, serving=None, window_df=None) -> dict:
    """Run the advisory, mapping the no-model/no-bnn_state precondition to a
    clean 422 instead of an opaque 500."""
    try:
        if serving is not None and window_df is not None:
            return serving.advisory(payload, window_df)
        return run_advisory(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def create_app() -> FastAPI:
    """Application factory. Creates a fresh FastAPI instance each call."""
    app = FastAPI(
        title="AeroVigil advisory API (wind-turbine-pg-bnn) v1.0.0",
        version=VERSION,
        description=SAFETY_BANNER,
    )

    # Optional model serving: load a trained PG-BNN bundle when configured.
    # Requests without a ``telemetry_window`` still take the bnn_state path
    # untouched, so payload-based clients see zero behavior change.
    app.state.serving = None
    app.state.serving_model_path = _serving_model_path()
    if app.state.serving_model_path:
        from src.models.serving import load_serving_model

        try:
            app.state.serving = load_serving_model(app.state.serving_model_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Failed to load serving model from {app.state.serving_model_path}: {exc}"
            ) from exc

    # In-memory digital-twin registry (Phase 4): per-process, advisory-only.
    app.state.twins = {}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict:
        return {
            "product": PRODUCT,
            "service": "wind-turbine-pg-bnn",
            "version": VERSION,
            "website": WEBSITE,
            "advisory_only": True,
            "serving_model_loaded": app.state.serving is not None,
            "endpoints": [
                "/health",
                "/advisory",
                "/advisory/fleet",
                "/twin/status",
                "/twin/simulate",
                "/twin/prompt",
                "/telemetry/compress",
                "/telemetry/restore",
                "/fleet/report",
                "/docs",
            ],
            "disclaimer": SAFETY_BANNER,
        }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            advisory_only=True,
            service="wind-turbine-pg-bnn",
            version=VERSION,
            product=PRODUCT,
            website=WEBSITE,
            serving_model_loaded=app.state.serving is not None,
        )

    @app.post("/advisory", response_model=AdvisoryResponse)
    def advisory(payload: AdvisoryRequest) -> AdvisoryResponse:
        # Model-serving path requires BOTH a loaded model and a raw window.
        # Anything else is the unchanged bnn_state behavior.
        window_df = None
        if payload.telemetry_window is not None:
            window_df = pd.DataFrame(payload.telemetry_window.model_dump())
        rec = _advisory_or_422(payload, serving=app.state.serving, window_df=window_df)
        enforce_safety_contract(rec)  # defense in depth
        return AdvisoryResponse(**rec)

    @app.post("/advisory/fleet", response_model=FleetResponse)
    def advisory_fleet(req: FleetRequest) -> FleetResponse:
        records = [_advisory_or_422(p) for p in req.assets]
        for r in records:
            enforce_safety_contract(r)
        util = expected_asset_utilization(
            [r["predicted_rul_days"] for r in records]
        )
        summary = FleetSummary(
            n_assets=len(records),
            mean_utilization=util["mean_utilization"],
            fraction_at_risk=util["fraction_at_risk"],
            mean_rul_days=util["mean_rul_days"],
        )
        return FleetResponse(
            assets=[AdvisoryResponse(**r) for r in records],
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # AeroZip telemetry pipeline                                          #
    # ------------------------------------------------------------------ #
    @app.post("/telemetry/compress", response_model=TelemetryCompressResponse)
    def telemetry_compress(req: TelemetryCompressRequest) -> TelemetryCompressResponse:
        """Compress one telemetry window with AeroZip (delta + deadband +
        quantize, anomaly bypass to lossless raw)."""
        from src.models.telemetry.pipeline import compress_window

        try:
            comp = compress_window(
                req.channels.model_dump(),
                baseline_mean=req.baseline_mean,
                baseline_std=req.baseline_std,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        body = comp.to_dict()
        enforce_safety_contract(body)
        return TelemetryCompressResponse(**body)

    @app.post("/telemetry/restore", response_model=TelemetryRestoreResponse)
    def telemetry_restore(req: TelemetryRestoreRequest) -> TelemetryRestoreResponse:
        """Restore a window compressed by /telemetry/compress (AeroZip lossiness
        semantics; lossless for anomaly-bypassed windows)."""
        from src.data.ingest import CHANNELS
        from src.models.telemetry.pipeline import CompressedWindow, restore_window

        channels = tuple(req.channels or CHANNELS)
        unknown = [c for c in channels if c not in CHANNELS]
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown channels: {unknown}")
        try:
            rest = restore_window(
                CompressedWindow(
                    codec="aerozip-v1",
                    payload_b64=req.payload_b64,
                    channels=channels,
                    n_samples=0,
                    anomaly_score=0.0,
                    bypass=False,
                    raw_bytes=0,
                    compressed_bytes=0,
                )
            )
        except Exception as exc:  # decode failures are client errors (bad base64/zlib/format)
            raise HTTPException(status_code=422, detail=f"could not decode payload: {exc}")
        body = {
            "channels": {c: rest.channels[c].tolist() for c in channels},
            "n_samples": rest.n_samples,
            "anomaly_score": rest.anomaly_score,
            "bypass": rest.bypass,
        }
        enforce_safety_contract(body)
        return TelemetryRestoreResponse(**body)

    # ------------------------------------------------------------------ #
    # Digital twin ↔ advisory bridge                                      #
    # ------------------------------------------------------------------ #
    def _get_twin(asset_id: str, model_key: str):
        """Fetch (or lazily create) the in-memory twin for an asset.

        When the service has a serving model loaded it is attached so every
        twin update computes its advisory from the trained PG-BNN; otherwise
        the bnn_state path applies (unchanged engine semantics).
        """
        from src.digital_twin.specs import get_spec
        from src.digital_twin.twin import WindTurbineDigitalTwin

        twin = app.state.twins.get(asset_id)
        if twin is None:
            try:
                spec = get_spec(model_key)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            twin = WindTurbineDigitalTwin(asset_id, spec, serving_model=app.state.serving)
            # Seed with the spec's nominal operating point so status/prompt
            # are meaningful from the first call.
            from src.utils.schema import Telemetry as _Tel

            vib = spec.vibration_limit_mms * 0.6
            twin.update_state(
                _Tel(
                    vibration_mms=vib,
                    temperature_c=spec.temperature_limit_c * 0.75,
                    rpm=spec.rpm_limit_hss * 0.85,
                    oil_viscosity_cst=(spec.viscosity_min_cst + spec.viscosity_max_cst) / 2.0,
                    load_pct=75.0,
                ),
                None,
            )
            app.state.twins[asset_id] = twin
        elif app.state.serving is not None and twin.serving_model is None:
            twin.attach_serving_model(app.state.serving)
        return twin

    def _twin_status_payload(twin) -> dict:
        last = twin.state_history[-1] if twin.state_history else None
        body = {
            "asset_id": twin.asset_id,
            "model_name": twin.spec.model_name,
            "manufacturer": twin.spec.manufacturer,
            "rated_power_mw": twin.spec.rated_power_mw,
            "cumulative_wear": twin.cumulative_wear,
            "last_updated": twin.last_updated.isoformat(),
            "n_state_records": len(twin.state_history),
            "last_state": last,
            "advisory_only": True,
        }
        enforce_safety_contract(body)
        return body

    @app.get("/twin/status")
    def twin_status(asset_id: str = "WTG-001", model: str = "GE-1.5") -> dict:
        """Current digital-twin state, including the advisory engine output."""
        twin = _get_twin(asset_id, model)
        return _twin_status_payload(twin)

    @app.post("/twin/simulate")
    def twin_simulate(req: TwinSimulateRequest) -> dict:
        """Replay an operating profile on the asset's digital twin (advisory only)."""
        twin = _get_twin(req.asset_id, req.model)
        try:
            records = twin.simulate_scenario(profile=req.profile, hours=req.hours)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        advisories = [r["advisory"] for r in records if r.get("advisory")]
        body = {
            "asset_id": twin.asset_id,
            "profile": req.profile,
            "hours": req.hours,
            "steps_executed": len(records),
            "advisories_computed": len(advisories),
            "cumulative_wear": twin.cumulative_wear,
            "final_bearing_l10_hours": records[-1]["bearing_l10_hours"] if records else None,
            "last_records": records[-5:],
            "last_advisory": advisories[-1] if advisories else None,
            "advisory_only": True,
        }
        enforce_safety_contract(body)
        return body

    @app.get("/twin/prompt")
    def twin_prompt(asset_id: str = "WTG-001", model: str = "GE-1.5") -> dict:
        """Generate the contextual reliability-copilot prompt for an asset twin."""
        from src.digital_twin.prompts import generate_engineering_prompt

        twin = _get_twin(asset_id, model)
        body = {
            "asset_id": asset_id,
            "prompt": generate_engineering_prompt(twin),
            "advisory_only": True,
        }
        enforce_safety_contract(body)
        return body

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #
    @app.get("/fleet/report")
    def fleet_report(title: str = "Fleet RUL advisory report"):
        """Markdown fleet report (text/markdown) via build_fleet_report.

        Sources the advisory records of every live digital twin in the
        registry (model or bnn_state path); when no twin carries an advisory
        yet, falls back to the bundled examples/fleet.csv snapshot so the
        endpoint is demonstrable out of the box.
        """
        from fastapi.responses import PlainTextResponse

        from src.reporting.reports import build_fleet_report

        records = []
        for twin in app.state.twins.values():
            last = twin.state_history[-1] if twin.state_history else {}
            if last.get("advisory"):
                records.append(last["advisory"])

        if not records:
            from pathlib import Path

            from src.reporting.reports import advisories_from_csv

            example = Path(__file__).resolve().parents[2] / "examples" / "fleet.csv"
            if not example.is_file():
                raise HTTPException(
                    status_code=404,
                    detail="No advisories in the twin registry and no example fleet CSV available.",
                )
            records = advisories_from_csv(str(example))
            title = f"{title} (example fleet snapshot)"

        for rec in records:
            enforce_safety_contract(rec)
        return PlainTextResponse(
            content=build_fleet_report(records, title=title),
            media_type="text/markdown",
        )

    return app


# Module-level instance so ``uvicorn src.api.app:app`` works out of the box.
app = create_app()
