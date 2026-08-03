"""
Calibration diagnostics for probabilistic RUL predictions.

Provides:
  - expected_calibration_error: ECE over predictive CDF intervals.
  - expected_asset_utilization: simple availability-style metric given a
    fleet of predicted RULs and a maintenance scheduling horizon.
  - classify_failure_mode: rule-based failure-type flagging from telemetry.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.physics.constraints import GearboxPhysicsConstraints


def expected_calibration_error(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error for Gaussian predictive distributions.
    For each confidence level p in {1/n_bins, ..., (n_bins-1)/n_bins}, measure
    the empirical fraction of truths inside the central p-interval; ECE is
    the mean absolute deviation between nominal and empirical coverage.

    Lower is better (0 = perfectly calibrated).
    """
    from scipy.stats import norm

    errors = []
    ps = np.linspace(1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1)
    for p in ps:
        z = norm.ppf(0.5 + p / 2.0)
        lo = pred_mean - z * pred_std
        hi = pred_mean + z * pred_std
        covered = ((y_true >= lo) & (y_true <= hi)).mean()
        errors.append(abs(covered - p))
    return float(np.mean(errors))


def expected_asset_utilization(
    predicted_rul_days: Sequence[float],
    planning_horizon_days: float = 90.0,
    safety_buffer_days: float = 14.0,
) -> Dict[str, float]:
    """
    Heuristic fleet utilization metric.

    A turbine is considered "available" across the horizon if its predicted
    RUL exceeds (planning horizon + safety buffer). For turbines with shorter
    RUL, utilization is proportional to RUL/horizon (i.e., curtailed by
    maintenance scheduling).
    """
    ruls = np.asarray(predicted_rul_days, dtype=np.float64)
    avail = np.minimum(1.0, np.maximum(0.0, (ruls - safety_buffer_days) / planning_horizon_days))
    return {
        "mean_utilization": float(avail.mean()),
        "fraction_at_risk": float((ruls < planning_horizon_days + safety_buffer_days).mean()),
        "mean_rul_days": float(ruls.mean()),
    }


def classify_failure_mode(
    telemetry: Dict[str, float],
    gb: GearboxPhysicsConstraints = GearboxPhysicsConstraints(),
) -> List[str]:
    """Rule-based probable failure mode flags (advisory only)."""
    flags: List[str] = []
    v = telemetry["vibration_mms"]
    t = telemetry["temperature_c"]
    rpm = telemetry["rpm"]
    visc = telemetry["oil_viscosity_cst"]
    load = telemetry["load_pct"]

    if v > gb.vibration_limit_mms and t > gb.temperature_limit_c:
        flags.append("Suspected bearing or gear-mesh damage (high vibration + elevated temperature)")
    elif v > gb.vibration_limit_mms:
        flags.append("Elevated vibration: inspect main bearing / gear teeth / alignment")
    if t > gb.temperature_limit_c and visc < gb.viscosity_min_cst:
        flags.append("Thermal risk with low oil viscosity — check cooling loop and oil oxidation")
    if visc < gb.viscosity_min_cst:
        flags.append("Low oil viscosity: risk of film breakdown at HSS contacts")
    if load > 100.0 and rpm > gb.rpm_limit_hss * 0.95:
        flags.append("Operation near rated speed at >100% load — derate advised for inspection window")
    if not flags:
        flags.append("All physics limits within nominal range; continue condition monitoring")
    return flags
