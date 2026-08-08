"""Integration tests joining main.py checkpoints to operational advisory data."""

import torch

from src.integrations.physics_guided import PhysicsGuidedServingModel
from src.models.bayesian_nn import PhysicsGuidedBNN
from src.utils.schema import Telemetry


def _telemetry() -> Telemetry:
    return Telemetry(
        vibration_mms=3.2,
        temperature_c=70.0,
        rpm=1_600.0,
        oil_viscosity_cst=28.0,
        load_pct=75.0,
    )


def test_main_checkpoint_adapter_evaluates_operational_telemetry(tmp_path):
    model = PhysicsGuidedBNN(6, hidden_dims=[4], out_features=1)
    path = tmp_path / "physics.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"in_features": 6, "hidden_dims": [4], "out_features": 1},
        },
        path,
    )

    serving = PhysicsGuidedServingModel.load(path)
    result = serving.evaluate(_telemetry(), mc_samples=2)

    assert result["model_type"] == "PhysicsGuidedBNN"
    assert result["advisory_only"] is True
    assert len(result["feature_values"]) == 6
    assert result["feature_sources"]["wind_speed_ms"] == "estimated_from_load_pct"
    assert "not a RUL estimate" in result["interpretation"]


def test_measured_framework_context_overrides_estimates(tmp_path):
    model = PhysicsGuidedBNN(6, hidden_dims=[4], out_features=1)
    path = tmp_path / "physics.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"hidden_dims": [4]}}, path)

    values, sources = PhysicsGuidedServingModel.load(path).feature_vector(
        _telemetry(), {"wind_speed_ms": 9.5, "power_output_kw": 1_250.0}
    )

    assert values[0] == 9.5
    assert values[-1] == 1_250.0
    assert sources["wind_speed_ms"] == "measured"
    assert sources["power_output_kw"] == "measured"
