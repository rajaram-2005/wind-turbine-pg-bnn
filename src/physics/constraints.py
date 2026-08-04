"""
Physics constraints for wind-turbine drivetrains.

Encodes hard engineering limits from OEM/standard practice and provides a
differentiable soft-penalty `physics_loss()` that can be added to the BNN
training objective as L_physics to regularize RUL predictions toward
physically plausible values.

Limits used here are generic research-grade defaults. A production deployment
must substitute asset-specific OEM limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Generic research defaults. Documented in docs/SAFETY.md as non-OEM values.
@dataclass(frozen=True)
class GearboxPhysicsConstraints:
    vibration_limit_mms: float = 4.5          # mm/s RMS (ISO 10816-3 zone boundary)
    temperature_limit_c: float = 80.0         # °C continuous oil
    rpm_limit_hss: float = 1800.0             # high-speed shaft RPM
    viscosity_min_cst: float = 10.0
    viscosity_max_cst: float = 50.0


@dataclass(frozen=True)
class GeneratorPhysicsConstraints:
    temperature_limit_c: float = 120.0        # °C winding, conservative
    rpm_limit: float = 1800.0


def check_violations(
    telemetry: dict[str, float],
    gb: GearboxPhysicsConstraints | None = None,
) -> list[str]:
    """Return a list of human-readable violation strings (empty if all nominal)."""
    if gb is None:
        gb = GearboxPhysicsConstraints()
    v: list[str] = []
    if telemetry["vibration_mms"] > gb.vibration_limit_mms:
        v.append(
            f"Vibration {telemetry['vibration_mms']:.2f} mm/s exceeds limit "
            f"{gb.vibration_limit_mms:.1f} mm/s"
        )
    if telemetry["temperature_c"] > gb.temperature_limit_c:
        v.append(
            f"Temperature {telemetry['temperature_c']:.1f} °C exceeds limit "
            f"{gb.temperature_limit_c:.1f} °C"
        )
    if telemetry["rpm"] > gb.rpm_limit_hss:
        v.append(
            f"RPM {telemetry['rpm']:.0f} exceeds limit {gb.rpm_limit_hss:.0f}"
        )
    if not (gb.viscosity_min_cst <= telemetry["oil_viscosity_cst"] <= gb.viscosity_max_cst):
        v.append(
            f"Oil viscosity {telemetry['oil_viscosity_cst']:.1f} cSt outside "
            f"[{gb.viscosity_min_cst:.1f}, {gb.viscosity_max_cst:.1f}] cSt"
        )
    return v


def _soft_relu_penalty(x: torch.Tensor, limit: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Soft, differentiable hinge: penalty is ~0 when x <= limit, grows as (x-limit)^2 above.
    beta controls steepness; smooths the kink for gradient flow."""
    excess = x - limit
    # Softplus-like smooth ramp to avoid hard kink at limit
    return torch.nn.functional.softplus(beta * excess) ** 2 / (beta ** 2)


def physics_loss(
    telemetry: dict[str, torch.Tensor],
    rul_pred: torch.Tensor,
    gb: GearboxPhysicsConstraints | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute L_physics: a soft penalty that encourages predicted RUL to
    decrease smoothly as operating variables exceed their physical limits.

    This is NOT a safety interlock — it is a training regularizer.
    Returns (total_loss, breakdown_dict).
    """
    if gb is None:
        gb = GearboxPhysicsConstraints()
    vib_lim = torch.tensor(gb.vibration_limit_mms, dtype=torch.float32)
    tmp_lim = torch.tensor(gb.temperature_limit_c, dtype=torch.float32)
    rpm_lim = torch.tensor(gb.rpm_limit_hss, dtype=torch.float32)
    visc_lo = torch.tensor(gb.viscosity_min_cst, dtype=torch.float32)
    visc_hi = torch.tensor(gb.viscosity_max_cst, dtype=torch.float32)

    p_vib = _soft_relu_penalty(telemetry["vibration_mms"], vib_lim)
    p_tmp = _soft_relu_penalty(telemetry["temperature_c"], tmp_lim)
    p_rpm = _soft_relu_penalty(telemetry["rpm"], rpm_lim)
    p_visc_lo = _soft_relu_penalty(visc_lo, telemetry["oil_viscosity_cst"])  # low viscosity
    p_visc_hi = _soft_relu_penalty(telemetry["oil_viscosity_cst"], visc_hi)  # high viscosity

    # Aggregate per-channel penalty to a scalar; weight equally by default.
    channel_penalty = p_vib + p_tmp + p_rpm + p_visc_lo + p_visc_hi

    # Coupling to RUL: when channels are out-of-limits, large predicted RUL
    # should be penalized. Encourage RUL → 0 as penalty grows.
    # L = channel_penalty * softplus(RUL) — differentiable and monotonic.
    rul_coupling = channel_penalty * torch.nn.functional.softplus(rul_pred)

    total = rul_coupling.mean()
    return total, {
        "vibration": p_vib.mean(),
        "temperature": p_tmp.mean(),
        "rpm": p_rpm.mean(),
        "viscosity_low": p_visc_lo.mean(),
        "viscosity_high": p_visc_hi.mean(),
        "rul_coupling": rul_coupling.mean(),
    }


def iso_281_l10_hours(C: float, P: float, p: float = 10.0 / 3.0, rpm: float = 1500.0) -> float:
    """
    ISO 281 rated bearing life L10 in hours.
    C: basic dynamic load rating [kN], P: equivalent dynamic load [kN],
    p: life exponent (10/3 for roller bearings, 3 for ball bearings),
    rpm: rotational speed.

    Returns L10h — hours at which 10% of bearings are expected to fail.
    """
    if P <= 0 or C <= 0 or rpm <= 0:
        return float("inf")
    L10_revs = (C / P) ** p * 1e6  # millions of revolutions → revs
    return L10_revs / (rpm * 60.0)
