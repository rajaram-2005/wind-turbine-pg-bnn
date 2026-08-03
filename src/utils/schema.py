"""Pydantic schemas for telemetry payloads and BNN outputs."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Telemetry(BaseModel):
    """Drivetrain telemetry snapshot. All values are one-sample means over a
    short (e.g. 10s) window."""

    model_config = ConfigDict(extra="forbid")

    vibration_mms: float = Field(..., ge=0.0, le=50.0, description="RMS vibration, mm/s")
    temperature_c: float = Field(..., ge=-40.0, le=200.0, description="Gearbox oil / bearing temp, °C")
    rpm: float = Field(..., ge=0.0, le=3000.0, description="High-speed shaft RPM")
    oil_viscosity_cst: float = Field(..., ge=1.0, le=500.0, description="cSt at operating temp")
    load_pct: float = Field(..., ge=0.0, le=120.0, description="Generator load % of rated")


class BNNState(BaseModel):
    predicted_rul_days: float = Field(..., ge=0.0, le=3650.0)
    epistemic_uncertainty: float = Field(..., ge=0.0)
    aleatoric_uncertainty: float = Field(..., ge=0.0)


class TurbinePayload(BaseModel):
    asset_id: str = Field(..., min_length=1)
    telemetry: Telemetry
    bnn_state: Optional[BNNState] = None
    operator_intent: Optional[str] = None
