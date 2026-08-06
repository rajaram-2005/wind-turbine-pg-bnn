"""
Monte Carlo Variational Inference utilities for uncertainty quantification.
"""

import torch
from typing import Tuple, List
from .model import PhysicsGuidedBNN


class MonteCarloVI:
    """
    Monte Carlo Variational Inference for Bayesian neural network predictions.

    This class provides utilities for running MCVI to obtain uncertainty-aware
    predictions from a PhysicsGuidedBNN model.

    Args:
        model: Trained PhysicsGuidedBNN model
        num_samples: Number of Monte Carlo samples for inference (default: 100)
    """

    def __init__(self, model: PhysicsGuidedBNN, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCVI inference to get mean prediction and uncertainty.

        Args:
            x: Input tensor of shape (batch, num_features)

        Returns:
            Tuple of (mean_prediction, uncertainty)
            - mean_prediction: Mean RUL prediction (batch, 1)
            - uncertainty: Standard deviation of predictions (batch, 1)
        """
        self.model.train()  # Enable dropout for MCVI

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                rul_mean, _ = self.model(x)
                predictions.append(rul_mean)

        predictions = torch.stack(predictions, dim=0)

        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_prediction, uncertainty

    def predict_with_confidence(self, x: torch.Tensor,
                                  confidence: float = 0.95) -> dict:
        """
        Run MCVI inference with confidence intervals.

        Args:
            x: Input tensor of shape (batch, num_features)
            confidence: Confidence level (default: 0.95 for 95% CI)

        Returns:
            Dictionary with prediction stats
        """
        self.model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                rul_mean, _ = self.model(x)
                predictions.append(rul_mean)

        predictions = torch.stack(predictions, dim=0)

        alpha = (1 - confidence) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100

        return {
            "mean": predictions.mean(dim=0),
            "std": predictions.std(dim=0),
            "ci_lower": torch.quantile(predictions, lower_percentile / 100, dim=0),
            "ci_upper": torch.quantile(predictions, upper_percentile / 100, dim=0),
            "samples": predictions,
        }

    def predict_single(self, telemetry: List[float]) -> dict:
        """
        Predict RUL for a single telemetry reading.

        Args:
            telemetry: List of 6 telemetry values:
                      [vibration_rms, bearing_temp, generator_temp,
                       power_output, wind_speed, operating_hours]

        Returns:
            Dictionary with prediction results
        """
        x = torch.tensor([telemetry], dtype=torch.float32)
        result = self.predict_with_confidence(x)

        mean_rul = result["mean"].item()
        std_rul = result["std"].item()
        ci_lower = result["ci_lower"].item()
        ci_upper = result["ci_upper"].item()

        # Risk assessment
        if mean_rul < 14:
            risk = "CRITICAL"
        elif mean_rul < 30:
            risk = "HIGH"
        elif mean_rul < 45:
            risk = "MODERATE"
        else:
            risk = "LOW"

        return {
            "predicted_rul_days": round(mean_rul, 1),
            "uncertainty_days": round(std_rul, 1),
            "confidence_interval_95": [round(ci_lower, 1), round(ci_upper, 1)],
            "risk_level": risk,
            "maintenance_recommended": mean_rul < 45,
        }
