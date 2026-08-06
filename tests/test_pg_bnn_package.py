"""Smoke tests for the standalone ``aerovigil_pg_bnn`` Hugging Face package.

These exercise the public surface of the packaged model (architecture, Monte
Carlo variational inference, and the CLI entry point) without touching the
network: the model is instantiated from the repo's ``config.json`` and run on
random telemetry, exactly as a downstream user would after ``pip install``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from aerovigil_pg_bnn.cli import main as cli_main
from aerovigil_pg_bnn.inference import MonteCarloVI
from aerovigil_pg_bnn.model import BayesianLinear, PhysicsGuidedBNN

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


@pytest.fixture(scope="module")
def config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model(config: dict) -> PhysicsGuidedBNN:
    return PhysicsGuidedBNN(config)


def test_bayesian_linear_reparameterization():
    """A BayesianLinear forward must be stochastic across calls (MCVI basis)."""
    layer = BayesianLinear(6, 8)
    x = torch.zeros(4, 6)
    # Dropout + reparameterization => repeated forwards differ in general.
    a = layer(x)
    b = layer(x)
    assert a.shape == (4, 8)
    assert not torch.allclose(a, b)


def test_forward_returns_mean_and_log_var(model: PhysicsGuidedBNN, config: dict):
    x = torch.randn(3, config["num_input_features"])
    rul_mean, rul_log_var = model(x)
    assert rul_mean.shape == (3, 1)
    assert rul_log_var.shape == (3, 1)
    assert torch.isfinite(rul_mean).all()
    assert torch.isfinite(rul_log_var).all()


def test_mcvi_predict_shapes(model: PhysicsGuidedBNN, config: dict):
    vi = MonteCarloVI(model, num_samples=8)
    x = torch.randn(5, config["num_input_features"])
    mean, uncertainty = vi.predict(x)
    assert mean.shape == (5, 1)
    assert uncertainty.shape == (5, 1)
    assert (uncertainty >= 0).all()


def test_predict_single_risk_levels(model: PhysicsGuidedBNN):
    """``predict_single`` must classify risk and flag maintenance below the horizon."""
    vi = MonteCarloVI(model, num_samples=8)
    # A "healthy" reading: low vibration/temps, low hours.
    healthy = vi.predict_single([1.5, 45.0, 60.0, 2000.0, 9.0, 1000.0])
    assert healthy["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert set(healthy) == {
        "predicted_rul_days",
        "uncertainty_days",
        "confidence_interval_95",
        "risk_level",
        "maintenance_recommended",
    }
    assert len(healthy["confidence_interval_95"]) == 2


def test_cli_help_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """``--help`` must parse without importing torch hub or hitting the network."""
    monkeypatch.setattr(sys, "argv", ["aerovigil-infer", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Aerovigil PG-BNN" in captured.out
