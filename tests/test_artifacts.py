"""Tests for the artifact registry (src.utils.artifacts)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.models.bnn import BayesianNeuralNetwork
from src.utils.artifacts import (
    CHECKPOINT_FORMAT,
    FeatureConfig,
    export_onboarding_bundle,
    load_model_bundle,
    save_json,
    save_model_bundle,
    sidecar_path,
)
from src.utils.safety import SafetyBoundaryError


def _tiny_model(in_features: int = 25, seed: int = 0) -> BayesianNeuralNetwork:
    torch.manual_seed(seed)
    return BayesianNeuralNetwork(in_features=in_features, hidden_sizes=(8,), prior_sigma=1.0)


def test_save_load_round_trip(tmp_path):
    model = _tiny_model()
    scaler = {"vibration_mms": (1.0, 4.0), "temperature_c": (55.0, 85.0)}
    bundle = save_model_bundle(
        model,
        tmp_path / "m.pt",
        scaler=scaler,
        metadata={"purpose": "round-trip", "rmse_days": 12.3},
    )

    assert bundle.checkpoint_path.is_file()
    side = sidecar_path(tmp_path / "m.pt")
    assert side.is_file()
    sidecar = json.loads(side.read_text(encoding="utf-8"))
    assert sidecar["format"] == CHECKPOINT_FORMAT
    assert sidecar["architecture"]["in_features"] == 25
    assert sidecar["architecture"]["hidden_sizes"] == [8]
    assert sidecar["scaler_present"] is True
    assert sidecar["metadata"]["advisory_only"] is True

    loaded = load_model_bundle(tmp_path / "m.pt")
    assert loaded.architecture.in_features == 25
    assert loaded.architecture.hidden_sizes == (8,)
    assert loaded.scaler == {"vibration_mms": (1.0, 4.0), "temperature_c": (55.0, 85.0)}
    assert loaded.metadata["purpose"] == "round-trip"

    # Weights survive the round trip (compare deterministic forward pass).
    x = torch.linspace(-1, 1, 25, dtype=torch.float32).unsqueeze(0)
    model.eval()
    loaded.model.eval()
    with torch.no_grad():
        m_ref, lv_ref = model(x, sample=False)
        m_new, lv_new = loaded.model(x, sample=False)
    assert torch.allclose(m_ref, m_new)
    assert torch.allclose(lv_ref, lv_new)


def test_save_rejects_unsafe_metadata(tmp_path):
    model = _tiny_model()
    with pytest.raises(SafetyBoundaryError):
        save_model_bundle(
            model,
            tmp_path / "bad.pt",
            metadata={"notes": "ok", "rpm_setpoint": 1500},  # actuation key — fail closed
        )


def test_save_rejects_feature_dim_mismatch(tmp_path):
    model = _tiny_model(in_features=30)  # 6 channels * 5 stats
    with pytest.raises(ValueError, match="features"):
        save_model_bundle(model, tmp_path / "m.pt", features=FeatureConfig())  # 25 dims


def test_load_rejects_feature_dim_mismatch(tmp_path):
    """A hand-built checkpoint whose feature config disagrees with the model."""
    model = _tiny_model(in_features=25)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "architecture": {"in_features": 25, "hidden_sizes": [8], "prior_sigma": 1.0},
        "features": {
            "window_size": 60,
            "stride": 20,
            "stats": ["mean", "std", "min", "max", "rms"],
            "channels": ["a", "b", "c", "d", "e", "f"],  # 6 * 5 = 30 != 25
        },
        "scaler": None,
        "metadata": {"advisory_only": True},
    }
    p = tmp_path / "corrupt.pt"
    torch.save(payload, p)
    with pytest.raises(ValueError, match="feature-dim mismatch"):
        load_model_bundle(p)


def test_sidecar_disagreement_rejected(tmp_path):
    model = _tiny_model()
    bundle = save_model_bundle(model, tmp_path / "m.pt")
    sidecar = json.loads(bundle.sidecar_path.read_text(encoding="utf-8"))
    sidecar["architecture"]["hidden_sizes"] = [64, 64]
    bundle.sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="[Ss]idecar"):
        load_model_bundle(tmp_path / "m.pt")


def test_legacy_bare_state_dict(tmp_path):
    model = _tiny_model()
    p = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), p)

    # Needs an explicit architecture pin.
    with pytest.raises(ValueError, match="in_features"):
        load_model_bundle(p)

    loaded = load_model_bundle(p, in_features=25, hidden_sizes=(8,))
    assert loaded.architecture.in_features == 25
    assert loaded.scaler is None


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_bundle(tmp_path / "nope.pt")


def test_save_json_safety_gate(tmp_path):
    with pytest.raises(SafetyBoundaryError):
        save_json({"loto_steps": ["1. isolate"]}, tmp_path / "x.json")


def test_export_onboarding_bundle(tmp_path):
    class _Report:
        def to_dict(self):
            return {
                "asset_id": "WTG-NEW-1",
                "status": "promoted",
                "promoted": True,
                "rounds_completed": 1,
                "rationale": "demo",
                "advisory_only": True,
            }

    model = _tiny_model()
    paths = export_onboarding_bundle(model, _Report(), tmp_path / "onboard")
    for p in paths.values():
        assert p.is_file()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["status"] == "promoted"
    loaded = load_model_bundle(paths["checkpoint"])
    assert loaded.metadata["produced_by"] == "hermes-onboarding"
    assert loaded.metadata["promoted"] is True


def test_checkpoint_loads_with_weights_only(tmp_path):
    """Sanity: our format must remain compatible with torch.load(weights_only=True)."""
    model = _tiny_model()
    save_model_bundle(model, tmp_path / "m.pt", scaler={"vibration_mms": (1.0, 4.0)})
    payload = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=True)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert isinstance(payload["state_dict"], dict)
    np.testing.assert_equal(payload["scaler"]["vibration_mms"], (1.0, 4.0))
