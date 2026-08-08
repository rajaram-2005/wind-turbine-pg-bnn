"""Aerodynamic physics models for wind turbine rotors.

Implements:
    * Heier's empirical approximation of the power coefficient Cp(beta, lambda)
    * Betz-limit enforcement
    * Rotor mechanical power extraction P = 0.5 * rho * A * v^3 * Cp
    * Jensen's (Park) wake deficit model for downstream turbines
    * A differentiable aerodynamic physics-consistency loss for training
      physics-guided neural networks.

All functions operate on ``torch.Tensor`` inputs and are fully differentiable,
so they can be embedded directly inside a training loss.

References:
    Heier, S. "Grid Integration of Wind Energy Conversion Systems." Wiley, 1998.
    Jensen, N.O. "A note on wind generator interaction." Risø-M-2411, 1983.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# ── Physical constants ────────────────────────────────────────────────────
BETZ_LIMIT: float = 16.0 / 27.0  # ≈ 0.5926, theoretical max power coefficient
DEFAULT_AIR_DENSITY: float = 1.225  # kg/m^3 at sea level, 15 °C

# Heier coefficients (widely used parametrisation)
_C1, _C2, _C3, _C4, _C5, _C6 = 0.5176, 116.0, 0.4, 5.0, 21.0, 0.0068


def lambda_i_inverse(tip_speed_ratio: torch.Tensor, pitch_deg: torch.Tensor) -> torch.Tensor:
    """Compute 1/lambda_i used in Heier's Cp approximation.

    1/lambda_i = 1/(lambda + 0.08*beta) - 0.035/(beta^3 + 1)

    Args:
        tip_speed_ratio: Tip speed ratio lambda = omega * R / v  (dimensionless).
        pitch_deg: Blade pitch angle beta in degrees.

    Returns:
        Tensor with the value of 1/lambda_i (same shape as inputs, broadcast).
    """
    lam = torch.clamp(tip_speed_ratio, min=1e-6)
    beta = torch.clamp(pitch_deg, min=0.0)
    return 1.0 / (lam + 0.08 * beta) - 0.035 / (beta.pow(3) + 1.0)


def power_coefficient(
    tip_speed_ratio: torch.Tensor,
    pitch_deg: torch.Tensor,
    enforce_betz: bool = True,
) -> torch.Tensor:
    """Heier's approximation of the rotor power coefficient Cp(beta, lambda).

    Cp = c1 * (c2/lambda_i - c3*beta - c4) * exp(-c5/lambda_i) + c6*lambda

    Args:
        tip_speed_ratio: Tip speed ratio lambda (dimensionless), > 0.
        pitch_deg: Blade pitch angle beta in degrees, >= 0.
        enforce_betz: If True, clamp the result to [0, Betz limit].

    Returns:
        Power coefficient Cp, clamped to the physically admissible range
        [0, 16/27] when ``enforce_betz`` is set.
    """
    inv_li = lambda_i_inverse(tip_speed_ratio, pitch_deg)
    cp = _C1 * (_C2 * inv_li - _C3 * pitch_deg - _C4) * torch.exp(-_C5 * inv_li)
    cp = cp + _C6 * tip_speed_ratio
    if enforce_betz:
        cp = torch.clamp(cp, min=0.0, max=BETZ_LIMIT)
    return cp


def tip_speed_ratio(
    rotor_speed_rad_s: torch.Tensor,
    wind_speed_ms: torch.Tensor,
    rotor_radius_m: float,
) -> torch.Tensor:
    """Tip speed ratio lambda = omega * R / v.

    Args:
        rotor_speed_rad_s: Rotor angular speed omega (rad/s).
        wind_speed_ms: Free-stream wind speed v (m/s).
        rotor_radius_m: Rotor radius R (m).

    Returns:
        Dimensionless tip speed ratio (wind speed floored at 0.1 m/s for
        numerical stability).
    """
    v = torch.clamp(wind_speed_ms, min=0.1)
    return rotor_speed_rad_s * rotor_radius_m / v


def rotor_mechanical_power(
    wind_speed_ms: torch.Tensor,
    cp: torch.Tensor,
    rotor_radius_m: float,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> torch.Tensor:
    """Mechanical power captured by the rotor.

    P = 0.5 * rho * A * v^3 * Cp,  with swept area A = pi * R^2.

    Args:
        wind_speed_ms: Free-stream wind speed v (m/s).
        cp: Power coefficient Cp (dimensionless, <= Betz limit).
        rotor_radius_m: Rotor radius R (m).
        air_density: Air density rho (kg/m^3).

    Returns:
        Mechanical power in Watts.
    """
    swept_area = math.pi * rotor_radius_m**2
    return 0.5 * air_density * swept_area * torch.clamp(wind_speed_ms, min=0.0).pow(3) * cp


def jensen_wake_deficit(
    distance_downstream_m: torch.Tensor,
    rotor_radius_m: float,
    thrust_coefficient: torch.Tensor,
    wake_decay: float = 0.075,
) -> torch.Tensor:
    """Jensen (Park) wake velocity deficit at a downstream distance.

    deficit = (1 - sqrt(1 - Ct)) / (1 + k*x/R)^2

    The wake wind speed is then v_wake = v_free * (1 - deficit).

    Args:
        distance_downstream_m: Downstream distance x from the wake-generating
            turbine (m), > 0.
        rotor_radius_m: Rotor radius R of the upstream turbine (m).
        thrust_coefficient: Thrust coefficient Ct in [0, 1).
        wake_decay: Wake decay constant k (0.075 onshore, ~0.04 offshore).

    Returns:
        Fractional velocity deficit in [0, 1].
    """
    ct = torch.clamp(thrust_coefficient, min=0.0, max=0.999)
    x = torch.clamp(distance_downstream_m, min=1e-3)
    expansion = (1.0 + wake_decay * x / rotor_radius_m).pow(2)
    return (1.0 - torch.sqrt(1.0 - ct)) / expansion


def waked_wind_speed(
    free_stream_ms: torch.Tensor,
    distance_downstream_m: torch.Tensor,
    rotor_radius_m: float,
    thrust_coefficient: torch.Tensor,
    wake_decay: float = 0.075,
) -> torch.Tensor:
    """Effective wind speed experienced by a downstream turbine.

    Args:
        free_stream_ms: Free-stream wind speed (m/s).
        distance_downstream_m: Distance to the upstream turbine (m).
        rotor_radius_m: Upstream rotor radius (m).
        thrust_coefficient: Upstream thrust coefficient Ct.
        wake_decay: Jensen wake decay constant k.

    Returns:
        Waked wind speed (m/s), always >= 0.
    """
    deficit = jensen_wake_deficit(
        distance_downstream_m, rotor_radius_m, thrust_coefficient, wake_decay
    )
    return torch.clamp(free_stream_ms * (1.0 - deficit), min=0.0)


def aerodynamic_physics_loss(
    predicted_power_w: torch.Tensor,
    wind_speed_ms: torch.Tensor,
    rotor_speed_rad_s: torch.Tensor,
    pitch_deg: torch.Tensor,
    rotor_radius_m: float,
    air_density: float = DEFAULT_AIR_DENSITY,
    reduction: str = "mean",
    betz_weight: float = 1.0,
) -> torch.Tensor:
    """Aerodynamic physics-consistency loss for physics-guided training.

    Two differentiable penalty terms:
        1. Residual between the model's predicted power and the power implied
           by the aerodynamic model P = 0.5*rho*A*v^3*Cp(beta, lambda)
           (normalised by rated-scale power to keep gradients well scaled).
        2. Betz-limit violation: penalise any predicted power that exceeds the
           theoretical maximum extractable power 0.5*rho*A*v^3 * 16/27.

    Args:
        predicted_power_w: Neural-network power prediction (W), shape (N,).
        wind_speed_ms: Measured wind speed (m/s), shape (N,).
        rotor_speed_rad_s: Measured rotor speed (rad/s), shape (N,).
        pitch_deg: Blade pitch angle (deg), shape (N,).
        rotor_radius_m: Rotor radius (m).
        air_density: Air density (kg/m^3).
        reduction: "mean", "sum", or "none".
        betz_weight: Multiplier for the Betz-violation penalty term.

    Returns:
        Scalar loss (or per-sample tensor when reduction="none").
    """
    lam = tip_speed_ratio(rotor_speed_rad_s, wind_speed_ms, rotor_radius_m)
    cp = power_coefficient(lam, pitch_deg)
    physics_power = rotor_mechanical_power(wind_speed_ms, cp, rotor_radius_m, air_density)

    # Normalisation scale: available wind power at each sample (avoid div-by-0)
    available = rotor_mechanical_power(
        wind_speed_ms, torch.ones_like(cp), rotor_radius_m, air_density
    )
    scale = torch.clamp(available, min=1.0)

    residual = ((predicted_power_w - physics_power) / scale).pow(2)

    betz_max = available * BETZ_LIMIT
    betz_violation = (torch.relu(predicted_power_w - betz_max) / scale).pow(2)

    loss = residual + betz_weight * betz_violation
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
