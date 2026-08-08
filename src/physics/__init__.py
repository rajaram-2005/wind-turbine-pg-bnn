"""Physics models for physics-guided wind-turbine learning.

Modules:
    aerodynamics: Cp(beta, lambda), rotor power, Jensen wake, aero loss.
    drivetrain: gearbox torque transfer, L10 bearing life, vibration energy.
    thermal: generator losses, lumped thermal network, temperature ODE.
    constraints: legacy soft physics constraints used by the original PG-BNN.
"""

from src.physics.aerodynamics import (
    BETZ_LIMIT,
    DEFAULT_AIR_DENSITY,
    aerodynamic_physics_loss,
    jensen_wake_deficit,
    power_coefficient,
    rotor_mechanical_power,
    tip_speed_ratio,
    waked_wind_speed,
)
from src.physics.drivetrain import (
    bearing_l10_life_hours,
    bearing_wear_fraction,
    drivetrain_physics_loss,
    gearbox_torque_transfer,
    high_speed_shaft_speed,
    vibration_stress_energy,
)
from src.physics.thermal import (
    copper_losses,
    generator_heat_dissipation,
    iron_losses,
    simulate_winding_temperature,
    steady_state_winding_temperature,
    thermal_physics_loss,
    winding_temperature_derivative,
)

__all__ = [
    "BETZ_LIMIT",
    "DEFAULT_AIR_DENSITY",
    "aerodynamic_physics_loss",
    "jensen_wake_deficit",
    "power_coefficient",
    "rotor_mechanical_power",
    "tip_speed_ratio",
    "waked_wind_speed",
    "bearing_l10_life_hours",
    "bearing_wear_fraction",
    "drivetrain_physics_loss",
    "gearbox_torque_transfer",
    "high_speed_shaft_speed",
    "vibration_stress_energy",
    "copper_losses",
    "generator_heat_dissipation",
    "iron_losses",
    "simulate_winding_temperature",
    "steady_state_winding_temperature",
    "thermal_physics_loss",
    "winding_temperature_derivative",
]
