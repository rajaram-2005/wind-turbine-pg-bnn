"""
High-level advisory predictor.

Combines the sliding-window feature extractor, PG-BNN, physics constraints,
calibration/eval helpers, and the safety gate into a single
`run_advisory()` function that returns an AdvisoryRecommendation.

This is the primary public API. It intentionally returns ONLY an
AdvisoryRecommendation — no setpoints, throttles, LOTO steps, or part
numbers.
"""

from __future__ import annotations

import numpy as np
import torch

from src.eval.calibration import EARLY_WARNING_HORIZON_DAYS, classify_failure_mode
from src.physics.constraints import check_violations
from src.utils.safety import AdvisoryRecommendation, enforce_safety_contract
from src.utils.schema import TurbinePayload


def _inspection_window(rul_days: float, epistemic: float) -> float:
    """Suggest next inspection window in days, tightening with RUL/uncertainty."""
    if rul_days < 30:
        return float(np.clip(rul_days / 4.0, 1.0, 7.0))
    if rul_days < 180:
        return 14.0
    # Far-from-failure: scale with epistemic (if model is unsure, inspect sooner)
    base = 30.0
    return float(max(7.0, base - 100.0 * epistemic))


def run_advisory(
    payload: TurbinePayload,
    model: torch.nn.Module | None = None,
    feature_vector: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """
    Produce an advisory recommendation for a single turbine snapshot.

    If a trained `model` is supplied and `feature_vector` is given, we run
    MCVI prediction to obtain RUL mean + uncertainties. Otherwise we fall
    back to the bnn_state block from the payload (useful for unit tests
    and offline analysis).
    """
    tel = payload.telemetry.model_dump()
    violations = check_violations(tel)

    # RUL from model OR payload
    if model is not None and feature_vector is not None:
        x = torch.tensor(feature_vector, dtype=torch.float32, device=device).unsqueeze(0)
        model.eval()
        from src.models.bnn import predict
        out = predict(model, x, mc_samples=64)
        # Clamp the raw regression head into the same engineering range the
        # BNNState schema enforces (0..3650 days): a negative remaining-life
        # estimate means "failure now", not negative time.
        rul_days = float(np.clip(out["mean_pred"].item(), 0.0, 3650.0))
        epi = float(out["epistemic_std"].item())
        ale = float(out["aleatoric_std"].item())
    else:
        if payload.bnn_state is None:
            raise ValueError("Provide either (model + feature_vector) or payload.bnn_state")
        rul_days = float(payload.bnn_state.predicted_rul_days)
        epi = float(payload.bnn_state.epistemic_uncertainty)
        ale = float(payload.bnn_state.aleatoric_uncertainty)

    flags = classify_failure_mode(tel)

    # Build rationale text (free-form engineering narrative; explicitly non-actuating)
    rationale_parts = []
    if violations:
        rationale_parts.append("Physics limits violated: " + "; ".join(violations) + ".")
    else:
        rationale_parts.append("All physics limits within nominal bounds.")
    rationale_parts.append("Probable mode(s): " + "; ".join(flags) + ".")
    if epi > 0.2:
        rationale_parts.append(
            f"High epistemic uncertainty (σ_ep={epi:.3f}) — model has not observed "
            "similar operating regimes; recommend cross-channel data review before "
            "any maintenance decision."
        )
    if ale > 0.3:
        rationale_parts.append(
            f"High aleatoric noise (σ_al={ale:.3f}) — possible sensor degradation "
            "or turbulent operation; suggest sensor health check."
        )
    if rul_days < 30:
        rationale_parts.append(
            f"Predicted RUL is {rul_days:.1f} days (<30). Flag for maintenance "
            "planning review. This advisory does not specify a curtailment or "
            "stop command; the operator must determine appropriate action using "
            "OEM procedures."
        )

    # 45-day early-warning horizon: the system's guarantee is that the problem
    # is announced (advisory-level) at least 45 days before the predicted failure.
    early_warning_triggered = rul_days < EARLY_WARNING_HORIZON_DAYS
    if early_warning_triggered:
        rationale_parts.append(
            f"EARLY WARNING: predicted failure within {EARLY_WARNING_HORIZON_DAYS:.0f} days "
            f"(RUL = {rul_days:.1f} days). Schedule a detailed condition "
            "inspection and order spares for the probable failure mode. "
            "Advisory only — no curtailment or stop command is implied."
        )

    rec = AdvisoryRecommendation(
        asset_id=payload.asset_id,
        predicted_rul_days=rul_days,
        epistemic_std=epi,
        aleatoric_std=ale,
        physics_violations=violations,
        suggested_inspection_window_days=_inspection_window(rul_days, epi),
        early_warning_triggered=early_warning_triggered,
        warning_horizon_days=EARLY_WARNING_HORIZON_DAYS,
        rationale=" ".join(rationale_parts),
    )
    return enforce_safety_contract(rec.to_dict())
