"""Shared scenario engine for single and comparative Cyber Twin forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np

SCENARIO_CONFIGS: dict[str, dict[str, float | str]] = {
    "Nominal operation": {
        "vib_mult": 1.0,
        "temp_bias": 0.0,
        "power_mult": 1.0,
        "wind_mult": 1.0,
        "degradation": 1.0,
        "shock": 0.0,
        "uncertainty": 0.055,
        "accent": "#00e5a0",
        "protocol": "BASELINE",
    },
    "High wind overload": {
        "vib_mult": 1.8,
        "temp_bias": 8.0,
        "power_mult": 1.14,
        "wind_mult": 1.28,
        "degradation": 1.5,
        "shock": 24.0,
        "uncertainty": 0.11,
        "accent": "#ff6d00",
        "protocol": "OVERLOAD",
    },
    "Derated operation": {
        "vib_mult": 0.7,
        "temp_bias": -5.0,
        "power_mult": 0.68,
        "wind_mult": 1.0,
        "degradation": 0.6,
        "shock": 0.0,
        "uncertainty": 0.045,
        "accent": "#8be9ff",
        "protocol": "PRESERVE",
    },
    "Tropical heat stress": {
        "vib_mult": 1.2,
        "temp_bias": 20.0,
        "power_mult": 0.92,
        "wind_mult": 0.9,
        "degradation": 1.3,
        "shock": 17.0,
        "uncertainty": 0.095,
        "accent": "#ffd600",
        "protocol": "THERMAL",
    },
    "Grid frequency event": {
        "vib_mult": 1.55,
        "temp_bias": 6.0,
        "power_mult": 1.08,
        "wind_mult": 1.0,
        "degradation": 1.42,
        "shock": 31.0,
        "uncertainty": 0.14,
        "accent": "#ff3df2",
        "protocol": "GRID EVENT",
    },
    "Arctic icing event": {
        "vib_mult": 1.68,
        "temp_bias": -12.0,
        "power_mult": 0.76,
        "wind_mult": 0.74,
        "degradation": 1.36,
        "shock": 27.0,
        "uncertainty": 0.12,
        "accent": "#20e3ff",
        "protocol": "ICING",
    },
}


def project_scenario(
    scenario: str,
    extra_hours: float,
    vibration: float,
    bearing_temp: float,
    generator_temp: float,
    power: float,
    wind: float,
    operating_hours: float,
    *,
    points: int = 160,
) -> dict[str, Any]:
    """Return one deterministic scenario trajectory and derived health metrics."""
    if scenario not in SCENARIO_CONFIGS:
        raise ValueError(f"unknown twin scenario: {scenario}")
    if points < 2:
        raise ValueError("points must be at least 2")

    cfg = SCENARIO_CONFIGS[scenario]
    extra_hours = max(float(extra_hours), 1.0)
    operating_hours = float(operating_hours)
    total_hours = operating_hours + extra_hours
    future_hours = np.linspace(operating_hours, total_hours, points)
    progress = np.linspace(0.0, 1.0, points)

    projected_vibration = float(vibration) * float(cfg["vib_mult"])
    projected_bearing_temp = float(bearing_temp) + float(cfg["temp_bias"])
    projected_generator_temp = float(generator_temp) + float(cfg["temp_bias"]) * 0.72
    projected_power = min(3000.0, max(0.0, float(power) * float(cfg["power_mult"])))
    projected_wind = max(0.0, float(wind) * float(cfg["wind_mult"]))

    baseline_rul = (
        450.0
        - 5.5 * max(projected_vibration - 5.0, 0) ** 1.18
        - 2.4 * max(projected_bearing_temp - 52.0, 0) ** 1.12
    )
    lifetime_wear = (future_hours / 87600.0) * 220.0 * float(cfg["degradation"])
    scenario_shock = float(cfg["shock"]) * progress**1.35
    rul = np.clip(baseline_rul - lifetime_wear - scenario_shock, 0.0, 450.0)
    uncertainty = np.maximum(4.0, rul * float(cfg["uncertainty"]) * (0.55 + progress * 0.9))
    upper = np.clip(rul + uncertainty, 0.0, 450.0)
    lower = np.clip(rul - uncertainty, 0.0, 450.0)
    wear_index = np.clip((1.0 - rul / 450.0) * 100.0, 0.0, 100.0)

    load_pct = np.clip(projected_power / 2500.0 * 100.0, 0.0, 100.0)
    vibration_pct = np.clip(projected_vibration / 45.0 * 100.0, 0.0, 100.0)
    thermal_pct = np.clip((projected_bearing_temp - 20.0) / 110.0 * 100.0, 0.0, 100.0)
    generator_pct = np.clip((projected_generator_temp - 25.0) / 145.0 * 100.0, 0.0, 100.0)
    stress_pct = float(
        np.clip(
            vibration_pct * 0.42 + thermal_pct * 0.28 + generator_pct * 0.2 + load_pct * 0.1,
            0.0,
            100.0,
        )
    )
    energy_mwh = projected_power * extra_hours / 1000.0

    return {
        "scenario": scenario,
        "config": cfg,
        "hours": future_hours,
        "progress": progress,
        "rul": rul,
        "upper": upper,
        "lower": lower,
        "wear_index": wear_index,
        "final_rul": float(rul[-1]),
        "final_wear": float(wear_index[-1]),
        "total_hours": total_hours,
        "extra_hours": extra_hours,
        "projected_vibration": projected_vibration,
        "projected_bearing_temp": projected_bearing_temp,
        "projected_generator_temp": projected_generator_temp,
        "projected_power": projected_power,
        "projected_wind": projected_wind,
        "stress_pct": stress_pct,
        "energy_mwh": energy_mwh,
    }
