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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AdvisoryResponse,
    FleetRequest,
    FleetResponse,
    FleetSummary,
    HealthResponse,
)
from src.eval.calibration import expected_asset_utilization
from src.models.predictor import run_advisory
from src.utils.safety import enforce_safety_contract
from src.utils.schema import TurbinePayload

VERSION = "1.0.0"
PRODUCT = "AeroVigil"
WEBSITE = "https://aerovigil.abacusai.app"
SAFETY_BANNER = (
    "AeroVigil v1.0.0 (wind-turbine-pg-bnn advisory service) — "
    "DECISION-SUPPORT ONLY. https://aerovigil.abacusai.app. "
    "Outputs are not actuation commands; review by a qualified operator "
    "and an OEM documentation cross-check are required before any "
    "maintenance action. See docs/SAFETY.md."
)


def _advisory_or_422(payload: TurbinePayload) -> dict:
    """Run the advisory, mapping the no-model/no-bnn_state precondition to a
    clean 422 instead of an opaque 500."""
    try:
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
            "endpoints": ["/health", "/advisory", "/advisory/fleet", "/docs"],
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
        )

    @app.post("/advisory", response_model=AdvisoryResponse)
    def advisory(payload: TurbinePayload) -> AdvisoryResponse:
        rec = _advisory_or_422(payload)
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

    return app


# Module-level instance so ``uvicorn src.api.app:app`` works out of the box.
app = create_app()
