"""Request/response Pydantic schemas for the FastAPI advisory service.

These are thin API-layer wrappers around the domain models in
``src.utils.schema`` (``Telemetry``, ``BNNState``, ``TurbinePayload``) and the
advisory dict produced by ``src.models.predictor.run_advisory``.

Every response is ADVISORY-ONLY. No direct-actuation fields (throttle, torque,
pitch, rpm-setpoint, breaker, LOTO, part-number, ...) are modelled anywhere in
these schemas, and every payload is screened by ``enforce_safety_contract``
(see ``src.utils.safety``) before it leaves the service.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Re-export the domain models so API consumers can import everything from
# ``src.api.schemas`` in one place.
from src.utils.schema import BNNState, Telemetry, TurbinePayload

__all__ = [
    "AdvisoryResponse",
    "BNNState",
    "FleetRequest",
    "FleetResponse",
    "FleetSummary",
    "HealthResponse",
    "Telemetry",
    "TurbinePayload",
]


class AdvisoryResponse(BaseModel):
    """Serialized :class:`~src.utils.safety.AdvisoryRecommendation`.

    Field-for-field mirror of the dict returned by ``run_advisory``. Using
    ``extra="forbid"`` means any future field that is not on this allow-list
    will fail validation at the API boundary — a second safety net behind
    ``enforce_safety_contract``.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    predicted_rul_days: float
    epistemic_std: float
    aleatoric_std: float
    physics_violations: list[str] = Field(default_factory=list)
    suggested_inspection_window_days: float
    rationale: str
    advisory_only: bool = True
    generated_at: str
    disclaimer: str


class FleetRequest(BaseModel):
    """A batch of single-turbine snapshots (one advisory per asset)."""

    assets: list[TurbinePayload] = Field(..., min_length=1)


class FleetSummary(BaseModel):
    """Aggregate fleet utilization metrics (advisory, non-actuating)."""

    n_assets: int
    mean_utilization: float
    fraction_at_risk: float
    mean_rul_days: float


class FleetResponse(BaseModel):
    """Per-asset advisories plus an aggregate fleet summary."""

    assets: list[AdvisoryResponse]
    summary: FleetSummary


class HealthResponse(BaseModel):
    """Liveness / readiness probe."""

    status: str = "ok"
    advisory_only: bool = True
    service: str = "wind-turbine-pg-bnn"
    version: str = "0.1.0"
