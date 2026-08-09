"""Request/response Pydantic schemas for the AeroVigil FastAPI advisory service.

AeroVigil v1.0.0 — https://aerovigil.abacusai.app. These are thin API-layer
wrappers around the domain models in ``src.utils.schema`` (``Telemetry``,
``BNNState``, ``TurbinePayload``) and the advisory dict produced by
``src.models.predictor.run_advisory``.

Every response is ADVISORY-ONLY. No direct-actuation fields (throttle, torque,
pitch, rpm-setpoint, breaker, LOTO, part-number, ...) are modelled anywhere in
these schemas, and every payload is screened by ``enforce_safety_contract``
(see ``src.utils.safety``) before it leaves the service.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Re-export the domain models so API consumers can import everything from
# ``src.api.schemas`` in one place.
from src.utils.schema import BNNState, Telemetry, TurbinePayload

__all__ = [
    "AdvisoryRequest",
    "AdvisoryResponse",
    "AgentAskRequest",
    "AgentReviewRequest",
    "BNNState",
    "FleetRequest",
    "FleetResponse",
    "FleetSummary",
    "HealthResponse",
    "REVIEW_DECISIONS",
    "Telemetry",
    "TelemetryCompressRequest",
    "TelemetryCompressResponse",
    "TelemetryRestoreRequest",
    "TelemetryRestoreResponse",
    "TelemetryWindow",
    "TurbinePayload",
    "TwinScenariosRequest",
    "TwinSimulateRequest",
]


class TelemetryWindow(BaseModel):
    """A raw per-channel telemetry window (list of samples per channel).

    Used with the model-serving path: when the service has a trained PG-BNN
    loaded (``AV_MODEL_PATH`` / config) and the request carries this block,
    RUL + uncertainties are computed by the model from these samples. Equal
    sample counts are enforced for all five canonical channels.
    """

    model_config = ConfigDict(extra="forbid")

    vibration_mms: list[float] = Field(..., min_length=1, max_length=10_000)
    temperature_c: list[float] = Field(..., min_length=1, max_length=10_000)
    rpm: list[float] = Field(..., min_length=1, max_length=10_000)
    oil_viscosity_cst: list[float] = Field(..., min_length=1, max_length=10_000)
    load_pct: list[float] = Field(..., min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _equal_lengths(self) -> TelemetryWindow:
        lengths = {
            len(self.vibration_mms),
            len(self.temperature_c),
            len(self.rpm),
            len(self.oil_viscosity_cst),
            len(self.load_pct),
        }
        if len(lengths) != 1:
            raise ValueError("all telemetry_window channels must have equal sample counts")
        return self


class TelemetryCompressRequest(BaseModel):
    """Wire format for POST /telemetry/compress (AeroZip)."""

    model_config = ConfigDict(extra="forbid")

    channels: TelemetryWindow
    sample_interval_s: int = Field(600, ge=1, le=86_400)
    baseline_mean: dict[str, float] | None = None
    baseline_std: dict[str, float] | None = None


class TelemetryCompressResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codec: str
    payload_b64: str
    channels: list[str]
    n_samples: int
    anomaly_score: float
    bypass: bool
    raw_bytes: int
    compressed_bytes: int
    ratio: float
    advisory_only: bool = True


class TelemetryRestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload_b64: str
    channels: list[str] | None = None  # defaults to the five canonical channels


class TelemetryRestoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channels: dict[str, list[float]]
    n_samples: int
    anomaly_score: float
    bypass: bool
    advisory_only: bool = True


class AdvisoryRequest(TurbinePayload):
    """``TurbinePayload`` + optional raw telemetry window.

    Backward compatible: without ``telemetry_window`` (or without a loaded
    model) the request is served from the ``bnn_state`` block exactly as
    before.
    """

    telemetry_window: TelemetryWindow | None = None


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
    early_warning_triggered: bool = False
    warning_horizon_days: float = 45.0
    rationale: str
    agent_team: dict[str, Any] | None = None
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


class TwinSimulateRequest(BaseModel):
    """Wire format for POST /twin/simulate (digital-twin scenario replay)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., min_length=1)
    model: str = "GE-1.5"
    profile: str = Field("nominal", pattern="^(nominal|overload|derated|viscosity_loss)$")
    hours: float = Field(24.0, gt=0.0, le=720.0)


class TwinScenariosRequest(BaseModel):
    """Wire format for POST /twin/scenarios (parallel-futures comparison).

    Runs the requested operating profiles side by side on forked twins so the
    canonical asset twin is never mutated — the Scenario Lab of the legacy
    dashboard, rebuilt on the durable twin runtime.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., min_length=1)
    model: str = "GE-1.5"
    profiles: list[str] = Field(
        default=["nominal", "overload", "derated", "viscosity_loss"],
        min_length=1,
        max_length=4,
    )
    hours: float = Field(24.0, gt=0.0, le=720.0)

    @model_validator(mode="after")
    def _check_profiles(self) -> "TwinScenariosRequest":
        allowed = {"nominal", "overload", "derated", "viscosity_loss"}
        bad = [p for p in self.profiles if p not in allowed]
        if bad:
            raise ValueError(f"unknown profile(s): {', '.join(bad)}")
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("profiles must be unique")
        return self


# Canonical human-decision choices for the advisory review gate (advisory only).
REVIEW_DECISIONS = (
    "Acknowledge evidence",
    "Request engineering review",
    "Escalate to reliability lead",
)


class AgentAskRequest(BaseModel):
    """Wire format for POST /agent/ask (Ask MIKA + KAI)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., min_length=1)
    model: str = "GE-1.5"
    question: str = Field(..., min_length=1, max_length=500)


class AgentReviewRequest(BaseModel):
    """Wire format for POST /agent/review (human decision gate)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., min_length=1)
    decision: str = Field(..., description="One of REVIEW_DECISIONS")
    note: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _check_decision(self) -> "AgentReviewRequest":
        if self.decision not in REVIEW_DECISIONS:
            raise ValueError(
                f"decision must be one of: {' | '.join(REVIEW_DECISIONS)}"
            )
        return self


class HealthResponse(BaseModel):
    """Liveness / readiness probe."""

    status: str = "ok"
    advisory_only: bool = True
    service: str = "wind-turbine-pg-bnn"
    product: str = "AeroVigil"
    version: str = "1.0.0"
    website: str = "https://aerovigil.abacusai.app"
    serving_model_loaded: bool = False
