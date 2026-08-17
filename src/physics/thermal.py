"""Thermal physics models for the generator and nacelle.

Implements:
    * Generator loss dissipation (copper I^2R losses + iron core losses)
    * A lumped-parameter thermal network: ambient → winding → coolant
    * First-order temperature dynamics ODE with explicit-Euler integration
    * A differentiable thermal physics-consistency loss.

All functions are ``torch``-native and differentiable.
"""

from __future__ import annotations

import torch

DEFAULT_WINDING_RESISTANCE_OHM: float = 0.02  # per-phase stator resistance
DEFAULT_IRON_LOSS_COEFF: float = 1.2e-3  # W per (rad/s)^1.6 scale factor
DEFAULT_THERMAL_RESISTANCE_K_W: float = 0.02  # winding→coolant K/W
DEFAULT_THERMAL_CAPACITANCE_J_K: float = 5.0e4  # winding lumped capacitance


def copper_losses(
    phase_current_a: torch.Tensor,
    winding_resistance_ohm: float = DEFAULT_WINDING_RESISTANCE_OHM,
    num_phases: int = 3,
) -> torch.Tensor:
    """Ohmic (copper) losses in the stator windings.

    P_cu = m * I^2 * R  for m phases.

    Args:
        phase_current_a: RMS phase current (A).
        winding_resistance_ohm: Per-phase winding resistance (Ω).
        num_phases: Number of phases m.

    Returns:
        Copper loss power (W).
    """
    return num_phases * phase_current_a.pow(2) * winding_resistance_ohm


def iron_losses(
    electrical_speed_rad_s: torch.Tensor,
    iron_loss_coeff: float = DEFAULT_IRON_LOSS_COEFF,
    steinmetz_exponent: float = 1.6,
) -> torch.Tensor:
    """Iron (core) losses via a Steinmetz-type frequency law.

    P_fe = k_fe * omega_e^alpha, alpha ≈ 1.6 combines hysteresis and eddy terms.

    Args:
        electrical_speed_rad_s: Electrical angular frequency (rad/s).
        iron_loss_coeff: Steinmetz coefficient k_fe.
        steinmetz_exponent: Combined Steinmetz exponent alpha.

    Returns:
        Iron loss power (W).
    """
    return iron_loss_coeff * torch.clamp(electrical_speed_rad_s, min=0.0).pow(steinmetz_exponent)


def generator_heat_dissipation(
    phase_current_a: torch.Tensor,
    electrical_speed_rad_s: torch.Tensor,
    winding_resistance_ohm: float = DEFAULT_WINDING_RESISTANCE_OHM,
    iron_loss_coeff: float = DEFAULT_IRON_LOSS_COEFF,
) -> torch.Tensor:
    """Total generator heat generation: copper + iron losses.

    Args:
        phase_current_a: RMS phase current (A).
        electrical_speed_rad_s: Electrical angular frequency (rad/s).
        winding_resistance_ohm: Per-phase winding resistance (Ω).
        iron_loss_coeff: Steinmetz iron-loss coefficient.

    Returns:
        Heat generation power Q (W).
    """
    return copper_losses(phase_current_a, winding_resistance_ohm) + iron_losses(
        electrical_speed_rad_s, iron_loss_coeff
    )


def steady_state_winding_temperature(
    heat_w: torch.Tensor,
    coolant_temp_c: torch.Tensor,
    thermal_resistance_k_w: float = DEFAULT_THERMAL_RESISTANCE_K_W,
) -> torch.Tensor:
    """Steady-state winding temperature of the lumped thermal network.

    T_w,ss = T_coolant + R_th * Q

    Args:
        heat_w: Heat generation Q (W).
        coolant_temp_c: Coolant (or ambient) temperature (°C).
        thermal_resistance_k_w: Winding→coolant thermal resistance (K/W).

    Returns:
        Steady-state winding temperature (°C).
    """
    return coolant_temp_c + thermal_resistance_k_w * heat_w


def winding_temperature_derivative(
    winding_temp_c: torch.Tensor,
    heat_w: torch.Tensor,
    coolant_temp_c: torch.Tensor,
    thermal_resistance_k_w: float = DEFAULT_THERMAL_RESISTANCE_K_W,
    thermal_capacitance_j_k: float = DEFAULT_THERMAL_CAPACITANCE_J_K,
) -> torch.Tensor:
    """First-order lumped thermal ODE for the winding node.

    C_th * dT_w/dt = Q - (T_w - T_coolant) / R_th

    Args:
        winding_temp_c: Current winding temperature (°C).
        heat_w: Heat generation Q (W).
        coolant_temp_c: Coolant temperature (°C).
        thermal_resistance_k_w: Thermal resistance R_th (K/W).
        thermal_capacitance_j_k: Thermal capacitance C_th (J/K).

    Returns:
        dT_w/dt in K/s.
    """
    outflow = (winding_temp_c - coolant_temp_c) / thermal_resistance_k_w
    return (heat_w - outflow) / thermal_capacitance_j_k


def simulate_winding_temperature(
    heat_w: torch.Tensor,
    coolant_temp_c: torch.Tensor,
    initial_temp_c: float,
    dt_s: float = 10.0,
    thermal_resistance_k_w: float = DEFAULT_THERMAL_RESISTANCE_K_W,
    thermal_capacitance_j_k: float = DEFAULT_THERMAL_CAPACITANCE_J_K,
) -> torch.Tensor:
    """Explicit-Euler integration of the winding-temperature ODE over time.

    Args:
        heat_w: Heat generation time series, shape (T,).
        coolant_temp_c: Coolant temperature time series, shape (T,).
        initial_temp_c: Initial winding temperature (°C).
        dt_s: Time step (s).
        thermal_resistance_k_w: Thermal resistance (K/W).
        thermal_capacitance_j_k: Thermal capacitance (J/K).

    Returns:
        Winding temperature trajectory, shape (T,).
    """
    temps = []
    t = torch.as_tensor(initial_temp_c, dtype=heat_w.dtype, device=heat_w.device)
    for k in range(heat_w.shape[0]):
        dT = winding_temperature_derivative(
            t, heat_w[k], coolant_temp_c[k], thermal_resistance_k_w, thermal_capacitance_j_k
        )
        t = t + dt_s * dT
        temps.append(t)
    return torch.stack(temps)


def thermal_physics_loss(
    predicted_winding_temp_c: torch.Tensor,
    phase_current_a: torch.Tensor,
    electrical_speed_rad_s: torch.Tensor,
    coolant_temp_c: torch.Tensor,
    temperature_limit_c: float = 120.0,
    winding_resistance_ohm: float = DEFAULT_WINDING_RESISTANCE_OHM,
    thermal_resistance_k_w: float = DEFAULT_THERMAL_RESISTANCE_K_W,
    reduction: str = "mean",
) -> torch.Tensor:
    """Thermal physics-consistency loss.

    Terms:
        1. Steady-state residual: predicted winding temperature should match
           the lumped-network steady state T_coolant + R_th * (P_cu + P_fe).
        2. Second-law consistency: the winding cannot be colder than the
           coolant while heat is being generated — penalise violations.
        3. Insulation limit awareness: soft penalty above the class-limit
           temperature (e.g. 120 °C for class B rise).

    Args:
        predicted_winding_temp_c: Model-predicted winding temperature (°C).
        phase_current_a: Measured phase current (A).
        electrical_speed_rad_s: Electrical angular frequency (rad/s).
        coolant_temp_c: Measured coolant/ambient temperature (°C).
        temperature_limit_c: Insulation temperature limit (°C).
        winding_resistance_ohm: Winding resistance (Ω).
        thermal_resistance_k_w: Thermal resistance (K/W).
        reduction: "mean", "sum", or "none".

    Returns:
        Scalar loss (or per-sample tensor when reduction="none").
    """
    heat = generator_heat_dissipation(
        phase_current_a, electrical_speed_rad_s, winding_resistance_ohm
    )
    t_ss = steady_state_winding_temperature(heat, coolant_temp_c, thermal_resistance_k_w)

    scale = torch.clamp(t_ss.abs(), min=1.0)
    residual = ((predicted_winding_temp_c - t_ss) / scale).pow(2)

    # Winding colder than coolant while generating heat is unphysical.
    second_law = torch.relu(coolant_temp_c - predicted_winding_temp_c) * (heat > 0).float()
    second_law = (second_law / scale).pow(2)

    over_limit = (torch.relu(predicted_winding_temp_c - temperature_limit_c) / scale).pow(2)

    loss = residual + second_law + 0.1 * over_limit
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
