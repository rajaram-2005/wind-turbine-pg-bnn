"""
Calibration diagnostics for probabilistic RUL predictions.

Provides:
  - expected_calibration_error: ECE over predictive CDF intervals.
  - expected_asset_utilization: simple availability-style metric given a
    fleet of predicted RULs and a maintenance scheduling horizon.
  - early_warning_metrics: binary early-warning classification accuracy at a
    fixed warning horizon (default 45 days) — the headline fleet metric.
  - first_warning_lead_time_days: how far ahead of failure the first warning
    fires along a turbine's degradation trajectory.
  - classify_failure_mode: rule-based failure-type flagging from telemetry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.physics.constraints import (
    GearboxPhysicsConstraints,
    default_gearbox_constraints,
)


def _default_warning_horizon() -> float:
    """45-day early-warning horizon, sourced from configs/default.yaml
    (``eval.early_warning_horizon_days``). The YAML value is identical to the
    historical hardcoded constant, so behavior is unchanged — the config file
    is now the single source of truth for the system guarantee."""
    from src.utils.config import load_config

    return float(load_config().eval.early_warning_horizon_days)


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
) -> dict[str, float]:
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


# Threaded from configs/default.yaml (eval.early_warning_horizon_days = 45.0).
EARLY_WARNING_HORIZON_DAYS = _default_warning_horizon()


def early_warning_metrics(
    y_true_rul: Sequence[float],
    y_pred_rul: Sequence[float],
    warning_horizon_days: float = EARLY_WARNING_HORIZON_DAYS,
) -> dict[str, float]:
    """
    Binary early-warning classification metrics at a fixed warning horizon.

    An asset is "at risk" (positive) when its *true* RUL is below the warning
    horizon; the system raises a warning when the *predicted* RUL is below the
    horizon. Accuracy is the fraction of assets whose warning status matches
    reality — the headline metric for the 45-day early-warning claim.

    Also reports precision/recall/F1, the false-alarm rate, and the mean lead
    time (days before failure) of true warnings.

    Returns a dict with keys: accuracy, precision, recall, f1,
    false_alarm_rate, mean_lead_time_days, n_assets, n_true_positive,
    n_true_negative, n_false_positive, n_false_negative.
    """
    yt = np.asarray(y_true_rul, dtype=np.float64)
    yp = np.asarray(y_pred_rul, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError("y_true_rul and y_pred_rul must be 1-D arrays of equal length")
    at_risk = yt < warning_horizon_days
    warned = yp < warning_horizon_days
    tp = int((at_risk & warned).sum())
    tn = int((~at_risk & ~warned).sum())
    fp = int((~at_risk & warned).sum())
    fn = int((at_risk & ~warned).sum())
    n = int(len(yt))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_lead_time = float(yt[at_risk & warned].mean()) if tp else 0.0
    return {
        "accuracy": float((tp + tn) / n),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": float(fp / n),
        "mean_lead_time_days": mean_lead_time,
        "n_assets": n,
        "n_true_positive": tp,
        "n_true_negative": tn,
        "n_false_positive": fp,
        "n_false_negative": fn,
    }


def first_warning_lead_time_days(
    rul_remaining_days: Sequence[float],
    warned: Sequence[bool],
) -> float | None:
    """
    Days before failure at the moment the first early warning fires, along a
    single turbine's degradation trajectory.

    ``rul_remaining_days[i]`` is the true remaining useful life at assessment
    step ``i`` and ``warned[i]`` is whether the system raised a warning at
    that step. The first True entry marks the earliest warning; its remaining
    life is the lead time. Returns None if the system never warned.

    Lead time >= 45 days means the problem was announced at least 45 days
    before failure — the guarantee the early-warning system is designed for.
    """
    ruls = np.asarray(rul_remaining_days, dtype=np.float64)
    warned = np.asarray(warned, dtype=bool)
    if ruls.shape != warned.shape or ruls.ndim != 1:
        raise ValueError("rul_remaining_days and warned must be 1-D arrays of equal length")
    idx = np.argmax(warned) if warned.any() else -1
    if idx < 0:
        return None
    return float(ruls[idx])


def classify_failure_mode(
    telemetry: dict[str, float],
    gb: GearboxPhysicsConstraints | None = None,
) -> list[str]:
    """Rule-based probable failure mode flags (advisory only)."""
    if gb is None:
        gb = default_gearbox_constraints()
    flags: list[str] = []
    v = telemetry["vibration_mms"]
    t = telemetry["temperature_c"]
    rpm = telemetry["rpm"]
    visc = telemetry["oil_viscosity_cst"]
    load = telemetry["load_pct"]

    if v > gb.vibration_limit_mms and t > gb.temperature_limit_c:
        flags.append(
            "Suspected bearing or gear-mesh damage (high vibration + elevated temperature)"
        )
    elif v > gb.vibration_limit_mms:
        flags.append("Elevated vibration: inspect main bearing / gear teeth / alignment")
    if t > gb.temperature_limit_c and visc < gb.viscosity_min_cst:
        flags.append("Thermal risk with low oil viscosity — check cooling loop and oil oxidation")
    if visc < gb.viscosity_min_cst:
        flags.append("Low oil viscosity: risk of film breakdown at HSS contacts")
    if load > 100.0 and rpm > gb.rpm_limit_hss * 0.95:
        flags.append(
            "Operation near rated speed at >100% load — derate advised for inspection window"
        )
    if not flags:
        flags.append("All physics limits within nominal range; continue condition monitoring")
    return flags
