"""Physics-grounded SHAP explanations for PG-BNN anomaly predictions.

Wraps SHAP's GradientExplainer (with a gradient-based fallback when the
``shap`` package is unavailable) and maps feature attributions onto the
physical subsystems and equation residuals they influence, so an alert is
explained as e.g. "thermal_overheating driven by generator winding
temperature" rather than an opaque feature ranking.

Anomaly-cause taxonomy:
    * mechanical_wear      — vibration / torque / bearing features
    * thermal_overheating  — temperature / current features
    * sensor_drift         — attribution concentrated on a single channel
      that disagrees with the physics residuals
    * aerodynamic_anomaly  — wind speed / rotor speed / pitch features
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: Default mapping of SCADA feature names → physics category.
DEFAULT_FEATURE_PHYSICS_MAP: Dict[str, str] = {
    "wind_speed": "aerodynamic_anomaly",
    "rotor_speed": "aerodynamic_anomaly",
    "pitch_angle": "aerodynamic_anomaly",
    "tip_speed_ratio": "aerodynamic_anomaly",
    "power_output": "aerodynamic_anomaly",
    "vibration": "mechanical_wear",
    "vibration_rms": "mechanical_wear",
    "torque": "mechanical_wear",
    "bearing_temp": "mechanical_wear",
    "oil_viscosity": "mechanical_wear",
    "gearbox_temp": "thermal_overheating",
    "generator_temp": "thermal_overheating",
    "winding_temp": "thermal_overheating",
    "phase_current": "thermal_overheating",
    "coolant_temp": "thermal_overheating",
}

#: Physics-loss module associated with each category (for residual reporting).
CATEGORY_RESIDUALS: Dict[str, str] = {
    "aerodynamic_anomaly": "src.physics.aerodynamics.aerodynamic_physics_loss",
    "mechanical_wear": "src.physics.drivetrain.drivetrain_physics_loss",
    "thermal_overheating": "src.physics.thermal.thermal_physics_loss",
    "sensor_drift": "cross-check of all physics residuals",
}


class _MeanForward(nn.Module):
    """Deterministic mean-output view of the BNN for gradient attribution."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, sample=False)
        return out[0] if isinstance(out, tuple) else out


class PhysicsSHAP:
    """SHAP-style attributions mapped onto turbine physics subsystems.

    Args:
        model: Trained PG-BNN (or any model returning (mean, log_var)).
        feature_names: Ordered SCADA feature names matching model inputs.
        background: Background dataset (N_bg, F) for the explainer baseline.
        feature_physics_map: Optional override of feature → category mapping.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Sequence[str],
        background: torch.Tensor,
        feature_physics_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = _MeanForward(model).eval()
        self.feature_names = list(feature_names)
        self.background = background
        self.map = feature_physics_map or DEFAULT_FEATURE_PHYSICS_MAP
        self._shap_explainer = self._build_shap_explainer()

    def _build_shap_explainer(self):
        """Try to build a shap.GradientExplainer; fall back to None."""
        try:
            import shap  # optional dependency

            return shap.GradientExplainer(self.model, self.background)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("shap unavailable (%s); using gradient*input fallback", exc)
            return None

    def attribute(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature attribution scores for a batch.

        Uses SHAP GradientExplainer when available, otherwise the
        gradient x (input - baseline) approximation (a one-step integrated
        gradients estimate with the background mean as baseline).

        Args:
            x: Samples to explain, shape (N, F).

        Returns:
            Attribution tensor of shape (N, F).
        """
        if self._shap_explainer is not None:
            values = self._shap_explainer.shap_values(x)
            if isinstance(values, list):
                values = values[0]
            values = torch.as_tensor(values, dtype=torch.float32)
            if values.dim() == 3:  # (N, F, out) → aggregate over outputs
                values = values.sum(dim=-1)
            return values

        baseline = self.background.mean(dim=0, keepdim=True)
        x_req = x.clone().requires_grad_(True)
        out = self.model(x_req).sum()
        (grad,) = torch.autograd.grad(out, x_req)
        return grad * (x - baseline)

    def categorize(self, attributions: torch.Tensor) -> Dict[str, float]:
        """Aggregate |attribution| mass per physics category.

        Also raises a ``sensor_drift`` score when one single feature carries
        the dominant share (> 60 %) of total attribution — a classic sign of
        a drifting/faulty sensor rather than a real physical change.

        Args:
            attributions: Attribution tensor of shape (N, F) or (F,).

        Returns:
            Dict category → normalised attribution share in [0, 1].
        """
        attr = attributions.abs()
        if attr.dim() == 2:
            attr = attr.mean(dim=0)
        total = float(attr.sum()) or 1.0

        scores: Dict[str, float] = {
            "mechanical_wear": 0.0,
            "thermal_overheating": 0.0,
            "aerodynamic_anomaly": 0.0,
            "sensor_drift": 0.0,
        }
        for i, name in enumerate(self.feature_names):
            category = self.map.get(name, "aerodynamic_anomaly")
            scores[category] += float(attr[i]) / total

        top_share = float(attr.max()) / total
        if top_share > 0.6:
            scores["sensor_drift"] = top_share
        return scores

    def explain(self, x: torch.Tensor) -> Dict:
        """Full explainability report for a batch of samples.

        Args:
            x: Samples to explain, shape (N, F).

        Returns:
            Dict with per-feature attributions, physics-category scores,
            the diagnosed root cause, and the physics residual reference.
        """
        attributions = self.attribute(x)
        mean_attr = attributions.abs().mean(dim=0)
        scores = self.categorize(attributions)
        root_cause = max(scores, key=scores.get)

        feature_report = sorted(
            (
                {
                    "feature": name,
                    "attribution": float(mean_attr[i]),
                    "physics_category": self.map.get(name, "aerodynamic_anomaly"),
                }
                for i, name in enumerate(self.feature_names)
            ),
            key=lambda d: -d["attribution"],
        )
        return {
            "root_cause": root_cause,
            "category_scores": scores,
            "physics_residual": CATEGORY_RESIDUALS[root_cause],
            "feature_attributions": feature_report,
            "explainer": "shap.GradientExplainer" if self._shap_explainer else "gradient_x_input",
        }
