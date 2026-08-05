"""Tests for the model-serving path (src.models.serving) and its
CLI / API wiring — with backward compatibility of the bnn_state path."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.ingest import CHANNELS, SlidingWindowConfig, robust_normalize, sliding_features
from src.models.bnn import BayesianNeuralNetwork, TrainConfig, elbo_loss
from src.utils.artifacts import FeatureConfig, save_model_bundle

FEATURE_DIM = len(CHANNELS) * 5  # 25


def _window_df(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame(
        {
            "vibration_mms": 2.5 + 0.3 * np.sin(t / 9.0) + rng.normal(0, 0.05, n),
            "temperature_c": 62.0 + 2.0 * np.sin(t / 15.0) + rng.normal(0, 0.2, n),
            "rpm": 1500.0 + 40.0 * np.sin(t / 11.0) + rng.normal(0, 4.0, n),
            "oil_viscosity_cst": 32.0 - 1.0 * np.sin(t / 13.0) + rng.normal(0, 0.2, n),
            "load_pct": 80.0 + 5.0 * np.sin(t / 8.0) + rng.normal(0, 0.8, n),
        }
    )


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    """A lightly trained PG-BNN saved as a serving bundle with a fitted scaler."""
    torch.manual_seed(0)
    df = _window_df()
    _, scaler = robust_normalize(df[list(CHANNELS)])

    model = BayesianNeuralNetwork(in_features=FEATURE_DIM, hidden_sizes=(16,))
    # Quick deterministic fit: targets near ~120 days so model-mode output is
    # clearly distinguishable from a sentinel bnn_state in tests.
    rng = np.random.default_rng(0)
    X = rng.normal(0.5, 0.25, size=(256, FEATURE_DIM)).astype(np.float32)
    y = (100.0 + 20.0 * X[:, 0] + rng.normal(0, 2.0, 256)).astype(np.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    tcfg = TrainConfig(num_samples=2, kl_weight=1e-3, physics_weight=0.0)
    xt, yt = torch.tensor(X), torch.tensor(y)
    model.train()
    for _ in range(40):
        opt.zero_grad()
        loss, _ = elbo_loss(model, xt, yt, telemetry=None, cfg=tcfg)
        loss.backward()
        opt.step()
    model.eval()

    ckpt = tmp_path_factory.mktemp("registry") / "tiny.pt"
    bundle = save_model_bundle(
        model,
        ckpt,
        scaler=scaler,
        features=FeatureConfig(),
        metadata={"produced_by": "tests/test_serving.py"},
    )
    return bundle


def test_serving_load_and_features(trained_bundle):
    from src.models.serving import load_serving_model

    serving = load_serving_model(trained_bundle.checkpoint_path)
    assert serving.expected_feature_dim == FEATURE_DIM

    df = _window_df()
    feats = serving.features(df)
    assert feats.shape == (4, FEATURE_DIM)  # starts 0, 20, 40, 60

    fv = serving.latest_feature_vector(df)
    assert fv.shape == (FEATURE_DIM,)


def test_features_match_training_pipeline_order(trained_bundle):
    """Serving features must be numerically identical to the training pipeline
    (robust_normalize with the stored scaler + sliding_features)."""
    from src.models.serving import apply_scaler, load_serving_model

    serving = load_serving_model(trained_bundle.checkpoint_path)
    df = _window_df()
    norm = apply_scaler(df[list(CHANNELS)], trained_bundle.scaler)
    ref = sliding_features(norm, SlidingWindowConfig(window_size=60, stride=20))
    got = serving.features(df)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_short_window_fallback(trained_bundle):
    """A window shorter than window_size still yields one advisory (stats over
    the whole frame, same stat/channel order)."""
    from src.models.serving import load_serving_model

    serving = load_serving_model(trained_bundle.checkpoint_path)
    df = _window_df(n=10)
    feats = serving.features(df)
    assert feats.shape == (1, FEATURE_DIM)


def test_serving_feature_dim_mismatch(trained_bundle):
    """Corrupt the bundle's feature config → clean feature-dim error."""
    from src.models.serving import ServingModel
    from src.utils.artifacts import load_model_bundle

    bundle = load_model_bundle(trained_bundle.checkpoint_path)
    corrupt = ArtifactBundleLike = bundle.__class__(
        model=bundle.model,
        architecture=bundle.architecture,
        features=FeatureConfig(stats=("mean", "std")),  # 5 * 2 = 10 != 25
        scaler=bundle.scaler,
        metadata=bundle.metadata,
        checkpoint_path=bundle.checkpoint_path,
        sidecar_path=bundle.sidecar_path,
    )
    serving = ServingModel(bundle=corrupt)
    del ArtifactBundleLike
    with pytest.raises(ValueError, match="feature-dim mismatch"):
        serving.features(_window_df())


def test_model_based_advisory_uses_model_not_bnn_state(trained_bundle):
    """The model-mode advisory ignores the payload's bnn_state values."""
    from src.models.serving import load_serving_model
    from src.utils.schema import BNNState, Telemetry, TurbinePayload

    serving = load_serving_model(trained_bundle.checkpoint_path)
    df = _window_df()
    snap = df.iloc[-1]
    payload = TurbinePayload(
        asset_id="WTG-TEST",
        telemetry=Telemetry(**{c: float(snap[c]) for c in CHANNELS}),
        bnn_state=BNNState(
            predicted_rul_days=3000.0, epistemic_uncertainty=0.5, aleatoric_uncertainty=0.5
        ),
    )
    rec = serving.advisory(payload, df)
    assert rec["advisory_only"] is True
    # Model was fit near ~100 days; bnn_state sentinel is 3000 days.
    assert rec["predicted_rul_days"] < 500.0
    assert rec["epistemic_std"] > 0
    assert rec["aleatoric_std"] > 0


# --------------------------------------------------------------------------- #
# CLI wiring                                                                   #
# --------------------------------------------------------------------------- #
def test_cli_advisory_with_model(trained_bundle, tmp_path, capsys):
    from src.cli import main

    df = _window_df()
    df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=len(df), freq="10min"))
    csv_path = tmp_path / "window.csv"
    df.to_csv(csv_path, index=False)

    snap = df.iloc[-1]
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "asset_id": "WTG-CLI",
                "telemetry": {c: float(snap[c]) for c in CHANNELS},
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "advisory",
            str(payload_path),
            "--model",
            str(trained_bundle.checkpoint_path),
            "--telemetry-csv",
            str(csv_path),
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["asset_id"] == "WTG-CLI"
    assert out["advisory_only"] is True
    assert out["predicted_rul_days"] < 500.0


def test_cli_advisory_bnn_state_fallback_unchanged(tmp_path, capsys):
    """Without --model the CLI behaves exactly as before (bnn_state path)."""
    from src.cli import main

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "asset_id": "WTG-OLD",
                "telemetry": {
                    "vibration_mms": 2.0,
                    "temperature_c": 60.0,
                    "rpm": 1400.0,
                    "oil_viscosity_cst": 30.0,
                    "load_pct": 75.0,
                },
                "bnn_state": {
                    "predicted_rul_days": 14.2,
                    "epistemic_uncertainty": 0.04,
                    "aleatoric_uncertainty": 0.12,
                },
            }
        ),
        encoding="utf-8",
    )
    rc = main(["advisory", str(payload_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["predicted_rul_days"] == pytest.approx(14.2)


# --------------------------------------------------------------------------- #
# API wiring                                                                   #
# --------------------------------------------------------------------------- #
def test_api_serves_model_when_configured(trained_bundle, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.api.app import ENV_MODEL_PATH, create_app

    monkeypatch.setenv(ENV_MODEL_PATH, str(trained_bundle.checkpoint_path))
    client = TestClient(create_app())

    health = client.get("/health").json()
    assert health["serving_model_loaded"] is True
    assert health["advisory_only"] is True

    df = _window_df(n=60)
    snap = df.iloc[-1]
    payload = {
        "asset_id": "WTG-API",
        "telemetry": {c: float(snap[c]) for c in CHANNELS},
        # Sentinel bnn_state: must be IGNORED on the model path.
        "bnn_state": {
            "predicted_rul_days": 3000.0,
            "epistemic_uncertainty": 0.5,
            "aleatoric_uncertainty": 0.5,
        },
        "telemetry_window": {c: df[c].tolist() for c in CHANNELS},
    }
    resp = client.post("/advisory", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset_id"] == "WTG-API"
    assert body["advisory_only"] is True
    assert body["predicted_rul_days"] < 500.0  # model output, not the 3000-day sentinel

    # Same request WITHOUT the window → unchanged bnn_state fallback.
    resp2 = client.post("/advisory", json={k: v for k, v in payload.items() if k != "telemetry_window"})
    assert resp2.status_code == 200
    assert resp2.json()["predicted_rul_days"] == pytest.approx(3000.0)
    del fastapi


def test_api_without_model_env_unchanged(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.api.app import ENV_MODEL_PATH, create_app

    monkeypatch.delenv(ENV_MODEL_PATH, raising=False)
    client = TestClient(create_app())
    health = client.get("/health").json()
    assert health["serving_model_loaded"] is False

    resp = client.post(
        "/advisory",
        json={
            "asset_id": "WTG-PLAIN",
            "telemetry": {
                "vibration_mms": 2.0,
                "temperature_c": 60.0,
                "rpm": 1400.0,
                "oil_viscosity_cst": 30.0,
                "load_pct": 75.0,
            },
            "bnn_state": {
                "predicted_rul_days": 14.2,
                "epistemic_uncertainty": 0.04,
                "aleatoric_uncertainty": 0.12,
            },
        },
    )
    assert resp.status_code == 200
    assert resp.json()["predicted_rul_days"] == pytest.approx(14.2)


def test_api_unequal_window_lengths_rejected(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.api.app import ENV_MODEL_PATH, create_app

    monkeypatch.delenv(ENV_MODEL_PATH, raising=False)
    client = TestClient(create_app())
    window = {c: [1.0] * 10 for c in CHANNELS}
    window["rpm"] = [1.0] * 11  # length mismatch
    resp = client.post(
        "/advisory",
        json={
            "asset_id": "WTG-BADWIN",
            "telemetry": {
                "vibration_mms": 2.0,
                "temperature_c": 60.0,
                "rpm": 1400.0,
                "oil_viscosity_cst": 30.0,
                "load_pct": 75.0,
            },
            "bnn_state": {
                "predicted_rul_days": 14.2,
                "epistemic_uncertainty": 0.04,
                "aleatoric_uncertainty": 0.12,
            },
            "telemetry_window": window,
        },
    )
    assert resp.status_code == 422


def test_train_demo_bundle_loads_via_serving_cli(tmp_path):
    """End-to-end: the demo trainer's own export loads through load_serving_model."""
    from src.models.serving import load_serving_model

    out = subprocess.run(
        [sys.executable, "scripts/train_demo.py", "--out", str(tmp_path / "demo.pt")],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "demo.pt").is_file()
    assert (tmp_path / "demo.json").is_file()
    serving = load_serving_model(tmp_path / "demo.pt")
    assert serving.expected_feature_dim == FEATURE_DIM
