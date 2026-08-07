"""
Monte Carlo Variational Inference utilities for uncertainty quantification.
"""

from typing import Optional, Union

import numpy as np
import torch

from .model import PhysicsGuidedBNN


class MonteCarloVI:
    """
    Monte Carlo Variational Inference for Bayesian neural network predictions.

    This class provides utilities for running MCVI to obtain uncertainty-aware
    predictions from a PhysicsGuidedBNN model.

    Args:
        model: Trained PhysicsGuidedBNN model
        num_samples: Number of Monte Carlo samples for inference (default: 100)
        scaler_mean: Optional mean vector for feature normalization
        scaler_std: Optional standard deviation vector for feature normalization
    """

    def __init__(
        self,
        model: PhysicsGuidedBNN,
        num_samples: int = 100,
        scaler_mean: Optional[Union[np.ndarray, torch.Tensor, list[float]]] = None,
        scaler_std: Optional[Union[np.ndarray, torch.Tensor, list[float]]] = None,
    ):
        self.model = model
        self.num_samples = num_samples
        self.scaler_mean = (
            np.array(scaler_mean, dtype=np.float32) if scaler_mean is not None else None
        )
        self.scaler_std = np.array(scaler_std, dtype=np.float32) if scaler_std is not None else None
        if self.scaler_std is not None:
            self.scaler_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        pred_list: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                rul_mean, _ = self.model(x)
                pred_list.append(rul_mean)

        predictions = torch.stack(pred_list, dim=0)

        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_prediction, uncertainty

    def predict_with_confidence(self, x: torch.Tensor, confidence: float = 0.95) -> dict:
        """
        Run MCVI inference with confidence intervals.

        Args:
            x: Input tensor of shape (batch, num_features)
            confidence: Confidence level (default: 0.95 for 95% CI)

        Returns:
            Dictionary with prediction stats
        """
        self.model.train()

        pred_list: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                rul_mean, _ = self.model(x)
                pred_list.append(rul_mean)

        predictions = torch.stack(pred_list, dim=0)

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

    def predict_single(self, telemetry: list[float]) -> dict:
        """
        Predict RUL for a single telemetry reading.

        Args:
            telemetry: List of 6 telemetry values:
                      [vibration_rms, bearing_temp, generator_temp,
                       power_output, wind_speed, operating_hours]

        Returns:
            Dictionary with prediction results
        """
        raw = np.array([telemetry], dtype=np.float32)
        if self.scaler_mean is not None and self.scaler_std is not None:
            raw = (raw - self.scaler_mean) / self.scaler_std

        x = torch.tensor(raw, dtype=torch.float32)
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
