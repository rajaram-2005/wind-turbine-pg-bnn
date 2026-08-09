"""FastAPI advisory service for AeroVigil (wind-turbine-pg-bnn engine).

AeroVigil v1.0.0 — https://aerovigil.abacusai.app

Canonical local run (this child API is mounted under ``/api``)::

    uvicorn src.unified_app:app --host 0.0.0.0 --port 8080

The module-level ``app`` remains importable for compatibility and focused tests.

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

import contextlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agents.cyber_team import build_cyber_team_brief
from src.api.schemas import (
    AdvisoryRequest,
    AdvisoryResponse,
    AgentAskRequest,
    AgentReviewRequest,
    FleetRequest,
    FleetResponse,
    FleetSummary,
    HealthResponse,
    TelemetryCompressRequest,
    TelemetryCompressResponse,
    TelemetryRestoreRequest,
    TelemetryRestoreResponse,
    TwinScenariosRequest,
    TwinSimulateRequest,
)
from src.eval.calibration import expected_asset_utilization
from src.models.predictor import run_advisory
from src.utils.safety import enforce_safety_contract
from src.utils.schema import TurbinePayload
from src.version import APP_VERSION as VERSION
from src.version import PRODUCT, SAFETY_BANNER, WEBSITE

ENV_MODEL_PATH = "AV_MODEL_PATH"
# Memory bound for the in-memory twin registry (LRU-evicted). Overridable per
# deployment; the default comfortably covers fleet-scale demos.
ENV_TWIN_MAX_ASSETS = "AV_TWIN_MAX_ASSETS"
DEFAULT_TWIN_MAX_ASSETS = 1024


def _swagger_asset_urls() -> tuple[str, str]:
    """Self-hosted Swagger UI assets when bundled, else the public CDN.

    The canonical deployment serves ``web_console/dist`` at ``/``, so the
    bundled assets resolve at ``/vendor/swagger/...`` without any external
    network access — important for air-gapped sites and sandboxed previews.
    """
    vendor = Path(__file__).resolve().parents[2] / "web_console" / "dist" / "vendor" / "swagger"
    if (vendor / "swagger-ui-bundle.js").is_file() and (vendor / "swagger-ui.css").is_file():
        return (
            "/vendor/swagger/swagger-ui-bundle.js",
            "/vendor/swagger/swagger-ui.css",
        )
    return (
        "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
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
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _connect_agent_team(payload: TurbinePayload, recommendation: dict) -> dict:
    """Attach the same MIKA + KAI brief used by twin and dashboard surfaces."""
    enriched = dict(recommendation)
    enriched["agent_team"] = build_cyber_team_brief(
        asset_id=payload.asset_id,
        predicted_rul_days=recommendation.get("predicted_rul_days"),
        epistemic_std=recommendation.get("epistemic_std", 0.0),
        physics_violations=recommendation.get("physics_violations", []),
        telemetry=payload.telemetry.model_dump(),
    )
    return enforce_safety_contract(enriched)


def get_or_create_twin(state, asset_id: str, model_key: str = "GE-1.5"):
    """Shared, LRU-bounded digital-twin registry accessor with durable hydration.

    Used both by the advisory API routes and by the gateway router
    (:mod:`src.api.gateway_routes`) so hardware-stream ingestion updates the
    same twin registry that ``/api/twin/status`` reads from.

    If the twin is not in memory but has prior snapshots in the durable SQLite
    store, its state_history is hydrated from the latest persisted record so
    fleet dashboards can resume after a restart.
    """
    from src.digital_twin.specs import get_spec
    from src.digital_twin.twin import WindTurbineDigitalTwin

    twins: OrderedDict = state.twins
    twin = twins.get(asset_id)
    if twin is None:
        if len(twins) >= state.twin_max_assets:
            # Evict the least recently used twin to honor the registry cap.
            twins.popitem(last=False)
        try:
            spec = get_spec(model_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        twin = WindTurbineDigitalTwin(asset_id, spec, serving_model=state.serving)
        # Hydrate from durable store first; if nothing persisted, seed with
        # the spec's nominal operating point so status/prompt are meaningful.
        hydrated = False
        try:
            from src.data.store import get_store

            latest = get_store().latest_twin_state(asset_id)
            if latest:
                twin.state_history.append(latest)
                if isinstance(latest.get("cumulative_wear"), (int, float)):
                    twin.cumulative_wear = float(latest["cumulative_wear"])
                hydrated = True
        except Exception:  # noqa: BLE001 - hydration is best-effort
            hydrated = False

        if not hydrated:
            # Seed with the spec's nominal operating point so status/prompt
            # are meaningful from the first call. A bad seed must not leave a
            # half-initialized twin in the registry.
            from src.utils.schema import Telemetry as _Tel

            vib = spec.vibration_limit_mms * 0.6
            try:
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
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"cannot seed twin {asset_id}: {exc}"
                ) from exc
        twins[asset_id] = twin
    else:
        twins.move_to_end(asset_id)  # LRU touch
        if state.serving is not None and twin.serving_model is None:
            twin.attach_serving_model(state.serving)
    return twin


def create_app() -> FastAPI:
    """Application factory. Creates a fresh FastAPI instance each call."""
    app = FastAPI(
        title="AeroVigil advisory API (wind-turbine-pg-bnn) v1.0.0",
        version=VERSION,
        description=SAFETY_BANNER,
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/docs", include_in_schema=False)
    def operations_docs() -> Any:
        """Swagger UI served with self-hosted assets (no CDN dependency)."""
        from fastapi.openapi.docs import get_swagger_ui_html

        js_url, css_url = _swagger_asset_urls()
        return get_swagger_ui_html(
            openapi_url="openapi.json",
            title="AeroVigil operations API — Swagger UI",
            swagger_js_url=js_url,
            swagger_css_url=css_url,
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
    # LRU-bounded so a long-running deployment cannot leak memory on unbounded
    # asset ids. `AV_TWIN_MAX_ASSETS` overrides the default cap.
    app.state.twins: OrderedDict = OrderedDict()
    try:
        app.state.twin_max_assets = max(
            1, int(os.environ.get(ENV_TWIN_MAX_ASSETS, DEFAULT_TWIN_MAX_ASSETS))
        )
    except ValueError:
        app.state.twin_max_assets = DEFAULT_TWIN_MAX_ASSETS
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
            "agent_team": {
                "team_id": "CYBER_PRIME_DUAL_AGENT",
                "agents": ["MIKA", "KAI"],
                "connected_surfaces": [
                    "advisory",
                    "fleet",
                    "twin",
                    "prompt",
                    "copilot",
                    "review",
                    "scenario",
                    "cli",
                    "dashboard",
                ],
            },
            "endpoints": [
                "/health",
                "/advisory",
                "/advisory/fleet",
                "/twin/status",
                "/twin/history",
                "/twin/simulate",
                "/twin/scenarios",
                "/twin/prompt",
                "/agent/ask",
                "/agent/review",
                "/agent/reviews",
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
        rec = _connect_agent_team(payload, rec)
        return AdvisoryResponse(**rec)

    @app.post("/advisory/fleet", response_model=FleetResponse)
    def advisory_fleet(req: FleetRequest) -> FleetResponse:
        records = [_connect_agent_team(p, _advisory_or_422(p)) for p in req.assets]
        for r in records:
            enforce_safety_contract(r)
        util = expected_asset_utilization([r["predicted_rul_days"] for r in records])
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
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
            raise HTTPException(status_code=422, detail=f"could not decode payload: {exc}") from exc
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

        The registry is LRU-bounded: when the cap is reached the least
        recently used twin is evicted before a new asset is admitted.
        """
        return get_or_create_twin(app.state, asset_id, model_key)

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
            "history_limit": twin.max_history,
            "serving_model_loaded": app.state.serving is not None,
            "advisory_source": (last or {}).get("advisory_source"),
            "agent_team": (last or {}).get("agent_team"),
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
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        advisories = [r["advisory"] for r in records if r.get("advisory")]
        # Durable: persist the final simulated twin state snapshot.
        try:
            from src.data.store import get_store

            get_store().record_twin_state(twin.asset_id, records[-1])
        except Exception:  # noqa: BLE001 - persistence must never break simulation
            pass
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
            "agent_team": records[-1].get("agent_team") if records else None,
            "advisory_only": True,
        }
        enforce_safety_contract(body)
        return body

    # ------------------------------------------------------------------ #
    # MIKA + KAI Agent Copilot (restored from the legacy dashboard)       #
    # ------------------------------------------------------------------ #
    _KAI_KEYWORDS = ("physics", "vibration", "temperature", "bearing", "why", "l10", "iso")
    _MIKA_KEYWORDS = ("when", "maintenance", "inspect", "crew", "plan", "window", "escalate")

    @app.post("/agent/ask")
    def agent_ask(req: AgentAskRequest) -> dict:
        """Ask MIKA + KAI about an asset — deterministic, evidence-grounded routing.

        Restored from the legacy Cyber Twin "Agent Copilot": physics questions
        are answered by KAI, maintenance-planning questions by MIKA, and
        everything else by the COUNCIL shared summary. Findings are always
        rebuilt from the asset's live twin evidence — never free-form text.
        """
        twin = _get_twin(req.asset_id, req.model)
        last = twin.state_history[-1] if twin.state_history else {}
        advisory = last.get("advisory") or {}
        team = build_cyber_team_brief(
            asset_id=twin.asset_id,
            predicted_rul_days=advisory.get("predicted_rul_days"),
            epistemic_std=advisory.get("epistemic_std", 0.0),
            physics_violations=last.get("physics_violations"),
            cumulative_wear=twin.cumulative_wear,
            bearing_l10_hours=last.get("bearing_l10_hours"),
            telemetry=last.get("telemetry"),
        )
        lower = req.question.lower()
        if any(word in lower for word in _KAI_KEYWORDS):
            agent, answer = "KAI", team["agents"]["kai"]["finding"]
        elif any(word in lower for word in _MIKA_KEYWORDS):
            agent = "MIKA"
            answer = (
                f"{team['agents']['mika']['finding']} The current advisory review "
                f"window is approximately {team['review_window_days']:.1f} days."
            )
        else:
            agent, answer = "COUNCIL", team["shared_summary"]
        body = {
            "asset_id": twin.asset_id,
            "question": req.question,
            "agent": agent,
            "answer": answer,
            "risk_level": team["risk_level"],
            "review_window_days": team["review_window_days"],
            "agreement_score_pct": team["agreement_score_pct"],
            "connected_sources": team["connected_sources"],
            "team": team,
            "advisory_only": True,
        }
        enforce_safety_contract(body)
        return body

    @app.post("/agent/review")
    def agent_review(req: AgentReviewRequest) -> dict:
        """Human decision gate: durably record an advisory-only operator decision.

        Restored from the legacy Cyber Twin review gate — every acknowledgement,
        engineering-review request, or escalation is kept as an auditable row in
        the durable store. It never actuates anything.
        """
        from src.data.store import get_store

        store = get_store()
        sequence = store.record_review(req.asset_id, req.decision, req.note)
        trail = store.list_reviews(asset_id=req.asset_id, limit=8)
        body = {
            "asset_id": req.asset_id,
            "decision": req.decision,
            "note": req.note,
            "sequence": sequence,
            "recorded_at": trail[-1]["ts"] if trail else None,
            "trail": [
                {"sequence": row["id"], "decision": row["decision"], "ts": row["ts"]}
                for row in reversed(trail)
            ],
            "advisory_only": True,
        }
        enforce_safety_contract(body)
        return body

    @app.get("/agent/reviews")
    def agent_reviews(asset_id: str | None = None, limit: int = 50) -> dict:
        """Durable audit trail of human review decisions."""
        from src.data.store import get_store

        rows = get_store().list_reviews(asset_id=asset_id, limit=limit)
        body = {"count": len(rows), "reviews": rows, "advisory_only": True}
        enforce_safety_contract(body)
        return body

    @app.post("/twin/scenarios")
    def twin_scenarios(req: TwinScenariosRequest) -> dict:
        """Scenario Lab: parallel futures across operating profiles.

        Runs each requested profile on a forked twin (same spec and cumulative
        wear as the canonical asset) so the live twin is never mutated, then
        returns a side-by-side decision-runway comparison.
        """
        from src.digital_twin.twin import WindTurbineDigitalTwin

        canonical = _get_twin(req.asset_id, req.model)
        rows: list[dict[str, Any]] = []
        for profile in req.profiles:
            fork = WindTurbineDigitalTwin(f"{canonical.asset_id}::scenario", canonical.spec)
            fork.cumulative_wear = canonical.cumulative_wear
            try:
                records = fork.simulate_scenario(profile=profile, hours=req.hours)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            advisories = [r["advisory"] for r in records if r.get("advisory")]
            final_advisory = advisories[-1] if advisories else None
            final_record = records[-1] if records else {}
            rows.append(
                {
                    "profile": profile,
                    "final_rul_days": (final_advisory or {}).get("predicted_rul_days"),
                    "epistemic_std": (final_advisory or {}).get("epistemic_std"),
                    "cumulative_wear": fork.cumulative_wear,
                    "wear_delta_pct": round(
                        (fork.cumulative_wear - canonical.cumulative_wear) * 100.0, 4
                    ),
                    "bearing_l10_hours": final_record.get("bearing_l10_hours"),
                    "physics_violations": final_record.get("physics_violations", []),
                    "risk_level": (final_record.get("agent_team") or {}).get("risk_level"),
                }
            )
        ranked = sorted(
            (r for r in rows if r["final_rul_days"] is not None),
            key=lambda r: r["final_rul_days"],
        )
        body = {
            "asset_id": canonical.asset_id,
            "model_name": canonical.spec.model_name,
            "hours": req.hours,
            "baseline_wear": canonical.cumulative_wear,
            "scenarios": rows,
            "best_profile": ranked[-1]["profile"] if ranked else None,
            "worst_profile": ranked[0]["profile"] if ranked else None,
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
    @app.get("/twin/history")
    def twin_history(asset_id: str = "WTG-001", limit: int = 50) -> dict:
        """Durable twin-state history from the SQLite store (persisted snapshots)."""
        from src.data.store import get_store

        limit = max(1, min(int(limit), 500))
        history = get_store().twin_history(asset_id, limit=limit)
        body = {
            "asset_id": asset_id,
            "count": len(history),
            "history": history,
            "advisory_only": True,
        }
        enforce_safety_contract(body)
        return body

    @app.get("/fleet/report")
    def fleet_report(title: str = "Fleet RUL advisory report"):
        """Markdown fleet report (text/markdown) via build_fleet_report.

        Sources the advisory records of every live digital twin in the
        registry (model or bnn_state path); when no twin carries an advisory
        yet, falls back to the bundled examples/fleet.csv snapshot so the
        endpoint is demonstrable out of the box.
        """
        from fastapi.responses import PlainTextResponse

        from src.data.store import get_store
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
        body = build_fleet_report(records, title=title)
        # Durable: keep the latest generated report in the store as well.
        with contextlib.suppress(Exception):  # persistence must never break the report
            get_store().record_report(
                "fleet",
                body,
                title=title,
                meta={
                    "n_assets": len(records),
                    "generated_by": "fleet_report",
                    "source": "twin-registry" if any(app.state.twins.values()) else "example-csv",
                },
            )
        return PlainTextResponse(content=body, media_type="text/markdown")

    return app


# Module-level instance so ``uvicorn src.api.app:app`` works out of the box.
app = create_app()
