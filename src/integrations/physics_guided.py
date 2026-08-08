"""Serve a ``main.py`` PhysicsGuidedBNN checkpoint beside AeroVigil advisories.

The framework model and the operational RUL model deliberately remain separate:
the former may be trained for power, health, or another configured target.  Its
posterior is attached as *evidence*, never silently re-labelled as RUL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.models.bayesian_nn import PhysicsGuidedBNN
from src.utils.schema import Telemetry

DEFAULT_GEARBOX_RATIO = 90.0
DEFAULT_RATED_POWER_KW = 2000.0


@dataclass
class PhysicsGuidedServingModel:
    model: PhysicsGuidedBNN
    target_name: str = "configured physics-guided target"
    device: str = "cpu"

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "PhysicsGuidedServingModel":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = dict(checkpoint.get("config", {}))
        model = PhysicsGuidedBNN(
            in_features=int(config.get("in_features", 6)),
            hidden_dims=list(config.get("hidden_dims", [128, 128, 64])),
            out_features=int(config.get("out_features", 1)),
            prior_sigma=float(config.get("prior_sigma", 1.0)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return cls(model=model, target_name=str(config.get("target_name", cls.target_name)), device=device)

    def feature_vector(self, telemetry: Telemetry, context: dict[str, float] | None = None) -> tuple[list[float], dict[str, str]]:
        """Map operational telemetry to the framework's six-feature contract.

        Measured wind/power can be supplied in ``context``.  If absent, clearly
        labelled load-based estimates keep legacy five-signal clients usable.
        """
        context = context or {}
        ratio = float(context.get("gearbox_ratio", DEFAULT_GEARBOX_RATIO))
        rated_power = float(context.get("rated_power_kw", DEFAULT_RATED_POWER_KW))
        load = max(0.0, min(100.0, float(telemetry.load_pct))) / 100.0
        wind = context.get("wind_speed_ms")
        power = context.get("power_output_kw")
        sources = {
            "wind_speed_ms": "measured" if wind is not None else "estimated_from_load_pct",
            "power_output_kw": "measured" if power is not None else "estimated_from_load_pct",
            "rotor_speed_rad_s": "derived_from_high_speed_rpm_and_gearbox_ratio",
            "generator_temp_c": "mapped_from_temperature_c",
            "vibration_rms": "mapped_from_vibration_mms",
            "oil_viscosity": "mapped_from_oil_viscosity_cst",
        }
        wind = float(wind) if wind is not None else 12.0 * load ** (1.0 / 3.0)
        power = float(power) if power is not None else rated_power * load
        rotor_speed = float(telemetry.rpm) / max(ratio, 1.0) * (2.0 * 3.141592653589793 / 60.0)
        values = [wind, rotor_speed, float(telemetry.temperature_c), float(telemetry.vibration_mms), float(telemetry.oil_viscosity_cst), power]
        if self.model.in_features != len(values):
            raise ValueError(f"physics-guided checkpoint expects {self.model.in_features} features; AeroVigil adapter provides 6")
        return values, sources

    @torch.no_grad()
    def evaluate(self, telemetry: Telemetry, context: dict[str, float] | None = None, *, mc_samples: int = 32) -> dict[str, Any]:
        values, sources = self.feature_vector(telemetry, context)
        x = torch.tensor([values], dtype=torch.float32, device=self.device)
        posterior = self.model.predict(x, num_samples=mc_samples)
        return {
            "model_type": "PhysicsGuidedBNN",
            "target_name": self.target_name,
            "target_mean": float(posterior["mean"][0, 0].cpu()),
            "epistemic_std": float(posterior["epistemic_std"][0, 0].cpu()),
            "aleatoric_std": float(posterior["aleatoric_std"][0, 0].cpu()),
            "total_std": float(posterior["total_std"][0, 0].cpu()),
            "feature_values": values,
            "feature_sources": sources,
            "advisory_only": True,
            "interpretation": "Supplementary physics-guided posterior; not a RUL estimate unless this checkpoint was explicitly trained and validated for RUL.",
        }
