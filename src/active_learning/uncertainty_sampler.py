"""Uncertainty-driven active learning on SCADA data streams.

Implements MC-Dropout / MC-weight-sampling epistemic uncertainty estimation,
threshold-based flagging of anomalous (high-uncertainty) samples, and JSON
maintenance-alert log generation for downstream CMMS integration.

Query strategy: flag samples whose *epistemic* standard deviation exceeds a
configurable threshold — high epistemic uncertainty means the model has not
seen similar operating conditions, i.e. either novel degradation or sensor
anomalies worth labelling / inspecting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.bayesian_nn import PhysicsGuidedBNN


@dataclass
class MaintenanceAlert:
    """A single high-uncertainty maintenance alert.

    Attributes:
        sample_index: Index of the flagged sample within the batch/stream.
        timestamp: UNIX epoch seconds when the alert was raised.
        prediction_mean: Model predictive mean for the sample.
        epistemic_std: Epistemic standard deviation (model uncertainty).
        aleatoric_std: Aleatoric standard deviation (observation noise).
        threshold: Threshold that was exceeded.
        features: Optional raw feature vector for traceability.
    """

    sample_index: int
    timestamp: float
    prediction_mean: float
    epistemic_std: float
    aleatoric_std: float
    threshold: float
    features: list[float] | None = None

    def to_dict(self) -> dict:
        """Serialise the alert to a JSON-compatible dict."""
        return {
            "sample_index": self.sample_index,
            "timestamp": self.timestamp,
            "prediction_mean": self.prediction_mean,
            "epistemic_std": self.epistemic_std,
            "aleatoric_std": self.aleatoric_std,
            "threshold": self.threshold,
            "severity": "high" if self.epistemic_std > 2.0 * self.threshold else "medium",
            "features": self.features,
        }


class UncertaintySampler:
    """Epistemic-uncertainty query strategy over SCADA streams.

    Args:
        model: Trained ``PhysicsGuidedBNN``.
        uncertainty_threshold: Epistemic-std threshold above which a sample
            is flagged for labelling / inspection.
        num_mc_samples: MC forward passes per sample (weight sampling; the
            model's dropout layers additionally act as MC dropout when the
            module is kept in train mode).
        sample_budget: Maximum number of samples to flag per query round.
        use_mc_dropout: Keep dropout active during sampling (MC Dropout).
    """

    def __init__(
        self,
        model: PhysicsGuidedBNN,
        uncertainty_threshold: float = 0.5,
        num_mc_samples: int = 32,
        sample_budget: int = 64,
        use_mc_dropout: bool = True,
    ) -> None:
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.num_mc_samples = num_mc_samples
        self.sample_budget = sample_budget
        self.use_mc_dropout = use_mc_dropout
        self.alert_log: list[MaintenanceAlert] = []

    @torch.no_grad()
    def estimate_uncertainty(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """MC sampling of epistemic + aleatoric uncertainty.

        Args:
            x: SCADA feature batch of shape (N, in_features).

        Returns:
            Dict with "mean", "epistemic_std", "aleatoric_std", "total_std",
            each of shape (N, out_features).
        """
        was_training = self.model.training
        # MC Dropout: keep dropout stochastic during inference if requested.
        self.model.train(self.use_mc_dropout)

        means, variances = [], []
        for _ in range(self.num_mc_samples):
            m, lv = self.model(x, sample=True)
            means.append(m)
            variances.append(torch.exp(lv))
        self.model.train(was_training)

        mc_means = torch.stack(means)
        mean = mc_means.mean(dim=0)
        epistemic_var = mc_means.var(dim=0, unbiased=False)
        aleatoric_var = torch.stack(variances).mean(dim=0)
        return {
            "mean": mean,
            "epistemic_std": epistemic_var.sqrt(),
            "aleatoric_std": aleatoric_var.sqrt(),
            "total_std": (epistemic_var + aleatoric_var).sqrt(),
        }

    def query(self, x: torch.Tensor, log_features: bool = False) -> dict:
        """Flag high-epistemic-uncertainty samples for labelling/inspection.

        Args:
            x: SCADA feature batch of shape (N, in_features).
            log_features: Include raw features in the generated alerts.

        Returns:
            Dict with:
                flagged_indices: LongTensor of flagged sample indices
                    (top-uncertainty first, capped at ``sample_budget``).
                uncertainty: full uncertainty dict from estimate_uncertainty.
                num_alerts: number of new alerts appended to the log.
        """
        stats = self.estimate_uncertainty(x)
        epi = stats["epistemic_std"].max(dim=-1).values  # worst target per sample
        over = torch.nonzero(epi > self.uncertainty_threshold).flatten()

        # Rank flagged samples by uncertainty, apply the labelling budget.
        if over.numel() > 0:
            order = torch.argsort(epi[over], descending=True)
            over = over[order][: self.sample_budget]

        now = time.time()
        for idx in over.tolist():
            self.alert_log.append(
                MaintenanceAlert(
                    sample_index=idx,
                    timestamp=now,
                    prediction_mean=float(stats["mean"][idx].mean()),
                    epistemic_std=float(epi[idx]),
                    aleatoric_std=float(stats["aleatoric_std"][idx].mean()),
                    threshold=self.uncertainty_threshold,
                    features=x[idx].tolist() if log_features else None,
                )
            )
        return {
            "flagged_indices": over,
            "uncertainty": stats,
            "num_alerts": int(over.numel()),
        }

    def write_alert_log(self, path: str) -> Path:
        """Write the accumulated maintenance alerts as a JSON file.

        Args:
            path: Destination ``.json`` path.

        Returns:
            The written path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.time(),
            "uncertainty_threshold": self.uncertainty_threshold,
            "num_alerts": len(self.alert_log),
            "alerts": [a.to_dict() for a in self.alert_log],
        }
        out.write_text(json.dumps(payload, indent=2))
        return out

    def clear_alerts(self) -> None:
        """Reset the in-memory alert log."""
        self.alert_log.clear()
