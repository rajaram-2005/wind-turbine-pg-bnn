"""Drivetrain physics models: gearbox, bearings, and vibration.

Implements:
    * Gearbox torque transfer with gear ratio and mechanical efficiency
    * Bearing wear degradation via the ISO 281 L10 basic rating life
    * Vibration stress energy (proportional to RMS velocity squared)
    * A differentiable drivetrain physics-consistency loss.

All functions are ``torch``-native and differentiable.
"""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_GEAR_RATIO: float = 97.0  # typical 1.5–3 MW turbine gearbox
DEFAULT_GEARBOX_EFFICIENCY: float = 0.97
BALL_BEARING_EXPONENT: float = 3.0
ROLLER_BEARING_EXPONENT: float = 10.0 / 3.0


def gearbox_torque_transfer(
    rotor_torque_nm: torch.Tensor,
    gear_ratio: float = DEFAULT_GEAR_RATIO,
    efficiency: float = DEFAULT_GEARBOX_EFFICIENCY,
) -> torch.Tensor:
    """High-speed-shaft torque delivered through the gearbox.

    T_hss = eta * T_rotor / n_gear

    Args:
        rotor_torque_nm: Low-speed (rotor) shaft torque (N·m).
        gear_ratio: Gearbox speed-up ratio n (dimensionless, > 0).
        efficiency: Mechanical efficiency eta in (0, 1].

    Returns:
        High-speed shaft torque (N·m).
    """
    return efficiency * rotor_torque_nm / gear_ratio


def high_speed_shaft_speed(
    rotor_speed_rad_s: torch.Tensor, gear_ratio: float = DEFAULT_GEAR_RATIO
) -> torch.Tensor:
    """High-speed-shaft angular speed omega_hss = n * omega_rotor.

    Args:
        rotor_speed_rad_s: Rotor angular speed (rad/s).
        gear_ratio: Gearbox speed-up ratio.

    Returns:
        High-speed shaft angular speed (rad/s).
    """
    return rotor_speed_rad_s * gear_ratio


def bearing_l10_life_hours(
    dynamic_load_rating_n: float,
    equivalent_load_n: torch.Tensor,
    shaft_speed_rpm: torch.Tensor,
    exponent: float = ROLLER_BEARING_EXPONENT,
) -> torch.Tensor:
    """ISO 281 basic (L10) bearing rating life, in operating hours.

    L10h = (10^6 / (60 * N)) * (C / P)^p

    Args:
        dynamic_load_rating_n: Basic dynamic load rating C (N).
        equivalent_load_n: Equivalent dynamic bearing load P (N).
        shaft_speed_rpm: Shaft rotational speed N (rpm).
        exponent: Life exponent p (3 for ball, 10/3 for roller bearings).

    Returns:
        L10 life in hours (90 % survival probability).
    """
    load = torch.clamp(equivalent_load_n, min=1.0)
    speed = torch.clamp(shaft_speed_rpm, min=1.0)
    return (1.0e6 / (60.0 * speed)) * (dynamic_load_rating_n / load).pow(exponent)


def bearing_wear_fraction(
    operating_hours: torch.Tensor,
    l10_life_hours: torch.Tensor,
) -> torch.Tensor:
    """Cumulative-damage wear fraction under the Palmgren–Miner rule.

    D = t_operated / L10h. D >= 1 indicates the design life is consumed.

    Args:
        operating_hours: Accumulated operating hours at this load level.
        l10_life_hours: L10 life at this load level (hours).

    Returns:
        Wear (damage) fraction, >= 0. Not clamped above 1 so overshoot is visible.
    """
    return operating_hours / torch.clamp(l10_life_hours, min=1.0)


def vibration_stress_energy(
    vibration_rms_mms: torch.Tensor,
    effective_mass_kg: float = 1.0,
) -> torch.Tensor:
    """Kinetic stress-energy proxy from broadband vibration.

    E ∝ 0.5 * m * v_rms^2, with v_rms in m/s. Serves as a differentiable
    severity index consistent with ISO 10816 vibration-velocity zoning.

    Args:
        vibration_rms_mms: RMS vibration velocity (mm/s).
        effective_mass_kg: Effective participating mass (kg).

    Returns:
        Stress energy (J) — a relative severity measure.
    """
    v_ms = vibration_rms_mms * 1e-3
    return 0.5 * effective_mass_kg * v_ms.pow(2)


def drivetrain_physics_loss(
    predicted_hss_torque_nm: torch.Tensor,
    rotor_torque_nm: torch.Tensor,
    predicted_wear_rate: Optional[torch.Tensor] = None,
    vibration_rms_mms: Optional[torch.Tensor] = None,
    gear_ratio: float = DEFAULT_GEAR_RATIO,
    efficiency: float = DEFAULT_GEARBOX_EFFICIENCY,
    vibration_limit_mms: float = 4.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Drivetrain physics-consistency loss.

    Terms:
        1. Torque-transfer residual: predictions must respect
           T_hss = eta * T_rotor / n (energy cannot be created in the gearbox).
        2. Monotonic wear: predicted wear rate must be non-negative
           (degradation is irreversible) — penalise negative rates.
        3. Vibration severity: predicted wear rate should not be small when
           vibration exceeds the ISO 10816 alarm level (inconsistency penalty).

    Args:
        predicted_hss_torque_nm: Model-predicted high-speed-shaft torque (N·m).
        rotor_torque_nm: Measured rotor torque (N·m).
        predicted_wear_rate: Optional model-predicted wear rate (1/h).
        vibration_rms_mms: Optional measured RMS vibration (mm/s).
        gear_ratio: Gearbox ratio n.
        efficiency: Gearbox efficiency eta.
        vibration_limit_mms: ISO 10816 alarm threshold (mm/s).
        reduction: "mean", "sum", or "none".

    Returns:
        Scalar loss (or per-sample tensor when reduction="none").
    """
    physics_torque = gearbox_torque_transfer(rotor_torque_nm, gear_ratio, efficiency)
    scale = torch.clamp(physics_torque.abs(), min=1.0)
    loss = ((predicted_hss_torque_nm - physics_torque) / scale).pow(2)

    if predicted_wear_rate is not None:
        # Wear must be irreversible: penalise negative predicted wear rates.
        loss = loss + torch.relu(-predicted_wear_rate).pow(2)

        if vibration_rms_mms is not None:
            # High vibration must not co-occur with near-zero predicted wear.
            over_limit = torch.relu(vibration_rms_mms - vibration_limit_mms)
            inconsistency = over_limit * torch.exp(-torch.clamp(predicted_wear_rate, min=0.0))
            loss = loss + 1e-2 * inconsistency.pow(2)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
