"""Scenario simulator: deterministic telemetry snapshots for demos & tests.

``simulate_telemetry(scenario, seed)`` returns a plausible telemetry dict for
one of four scenarios:

* ``healthy``  — nominal SCADA + oil-condition channels, nothing detected.
* ``faulty``   — degraded oil (viscosity/water/particles), elevated vibration
  and temperatures, yaw error — a realistic multi-fault snapshot.
* ``critical`` — fire evidence (smoke + hot oil + blade fire), overspeed,
  extreme vibration — pages every CRITICAL alert.
* ``random``   — uniformly sampled channels within plausible bounds.

The generator is deterministic per ``seed`` (numpy default_rng), so demos
and tests are reproducible.  Used by ``POST /api/simulate/snapshot``.
"""

from __future__ import annotations

import numpy as np

SCENARIOS = ("healthy", "faulty", "critical", "random")

# Plausible sampling bounds for the "random" scenario.
_BOUNDS: dict[str, tuple[float, float]] = {
    "vibration_mms": (1.0, 8.0),
    "temperature_c": (45.0, 85.0),
    "rpm": (800.0, 1750.0),
    "oil_viscosity_cst": (5.0, 45.0),
    "load_pct": (40.0, 110.0),
    "main_bearing_temp_c": (40.0, 75.0),
    "generator_temp_c": (70.0, 125.0),
    "yaw_error_deg": (0.0, 30.0),
    "oil_water_ppm": (50.0, 1500.0),
    "oil_tan_mgkoh_g": (0.2, 2.5),
    "oil_filter_dp_bar": (0.4, 2.8),
    "oil_level_pct": (5.0, 95.0),
    "oil_pressure_bar": (0.6, 4.0),
    "brake_wear_pct": (10.0, 98.0),
    "converter_temp_c": (40.0, 90.0),
    "transformer_temp_c": (60.0, 120.0),
}


def _healthy() -> dict:
    return {
        "vibration_mms": 2.1,
        "temperature_c": 56.0,
        "oil_temp_c": 56.0,
        "rpm": 1400.0,
        "oil_viscosity_cst": 32.0,
        "load_pct": 72.0,
        "main_bearing_temp_c": 48.0,
        "generator_temp_c": 86.0,
        "oil_water_ppm": 80.0,
        "oil_moisture_pct": 22.0,
        "oil_particles_iso4406": "16/14/11",
        "oil_tan_mgkoh_g": 0.3,
        "oil_filter_dp_bar": 0.7,
        "oil_level_pct": 78.0,
        "oil_pressure_bar": 3.2,
        "oil_aeration_pct": 2.0,
        "oil_iron_ppm": 40.0,
        "yaw_error_deg": 4.0,
        "blade_pitch_deviation_deg": 0.2,
        "brake_wear_pct": 35.0,
        "converter_temp_c": 52.0,
        "transformer_temp_c": 75.0,
        "wind_speed_mps": 8.0,
        "wind_speed2_mps": 8.2,
        "comms_uptime_pct": 99.8,
        "predicted_rul_days": 240.0,
    }


def _faulty() -> dict:
    return {
        **_healthy(),
        "vibration_mms": 6.8,
        "temperature_c": 74.5,
        "oil_temp_c": 74.5,
        "rpm": 1620.0,
        "oil_viscosity_cst": 6.2,
        "load_pct": 96.0,
        "main_bearing_temp_c": 68.5,
        "generator_temp_c": 118.0,
        "generator_bearing_temp_c": 82.0,
        "oil_water_ppm": 850.0,
        "oil_moisture_pct": 62.0,
        "oil_particles_iso4406": "19/17/15",
        "oil_tan_mgkoh_g": 1.9,
        "oil_filter_dp_bar": 2.1,
        "oil_level_pct": 55.0,
        "oil_pressure_bar": 1.8,
        "oil_aeration_pct": 9.0,
        "oil_iron_ppm": 340.0,
        "bpfo_amplitude_mms": 2.4,
        "blade_pitch_deviation_deg": 1.7,
        "yaw_error_deg": 18.0,
        "cable_twist_turns": 2.9,
        "converter_temp_c": 82.0,
        "transformer_temp_c": 101.0,
        "coolant_temp_c": 52.0,
        "brake_wear_pct": 88.0,
        "predicted_rul_days": 32.0,
    }


def _critical() -> dict:
    return {
        **_faulty(),
        "vibration_mms": 12.5,
        "rpm": 1850.0,
        "oil_temp_c": 126.0,
        "oil_viscosity_cst": 4.5,
        "oil_water_ppm": 1800.0,
        "oil_iron_ppm": 620.0,
        "smoke_detector_on": True,
        "blade_fire_alarm": True,
        "fire_suppression_released": True,
        "brake_temp_c": 135.0,
        "cable_twist_turns": 3.9,
        "converter_temp_c": 96.0,
        "transformer_temp_c": 132.0,
        "predicted_rul_days": 6.0,
    }


def simulate_telemetry(scenario: str = "healthy", seed: int = 0) -> dict:
    """Deterministic telemetry snapshot for the requested scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'; choose from {SCENARIOS}")
    if scenario == "healthy":
        return _healthy()
    if scenario == "faulty":
        return _faulty()
    if scenario == "critical":
        return _critical()
    rng = np.random.default_rng(seed)
    return {key: round(float(rng.uniform(lo, hi)), 2) for key, (lo, hi) in _BOUNDS.items()}


def scenario_descriptions() -> dict[str, str]:
    """Human-readable description of each scenario (for the API/docs)."""
    return {
        "healthy": "Nominal operation — no faults expected.",
        "faulty": "Degraded oil condition, elevated vibration and temperatures —"
        " multiple HIGH/MEDIUM faults.",
        "critical": "Fire evidence, overspeed and extreme wear — CRITICAL alerts.",
        "random": "Channels sampled within plausible bounds (deterministic per seed).",
    }
