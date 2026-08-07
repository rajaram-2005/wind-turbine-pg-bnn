"""Tests for the digital twin ↔ advisory bridge: twin advisory computation,
API twin endpoints, and the twin-status --advisory CLI flag."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.ingest import CHANNELS, robust_normalize
from src.digital_twin.prompts import generate_engineering_prompt
from src.digital_twin.specs import get_spec
from src.digital_twin.twin import WindTurbineDigitalTwin
from src.models.bnn import BayesianNeuralNetwork
from src.utils.artifacts import FeatureConfig
from src.utils.safety import enforce_safety_contract
from src.utils.schema import BNNState, Telemetry

FEATURE_DIM = 25


def _healthy_telemetry() -> Telemetry:
    return Telemetry(
        vibration_mms=2.0, temperature_c=60.0, rpm=1400.0, oil_viscosity_cst=30.0, load_pct=70.0
    )


@pytest.fixture(scope="module")
def serving_bundle(tmp_path_factory):
    from src.utils.artifacts import FeatureConfig, save_model_bundle

    torch.manual_seed(0)
    model = BayesianNeuralNetwork(in_features=FEATURE_DIM, hidden_sizes=(8,))
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(0, 1, 120) for c in CHANNELS})
    _, scaler = robust_normalize(df)
    ckpt = tmp_path_factory.mktemp("twinreg") / "twin_model.pt"
    return save_model_bundle(model, ckpt, scaler=scaler, features=FeatureConfig())


def test_update_state_stores_bnn_state_advisory():
    twin = WindTurbineDigitalTwin("WTG-T", get_spec("GE-1.5"))
    rec = twin.update_state(
        _healthy_telemetry(),
        BNNState(predicted_rul_days=44.0, epistemic_uncertainty=0.05, aleatoric_uncertainty=0.1),
    )
    assert rec["advisory_source"] == "bnn_state"
    adv = rec["advisory"]
    assert adv is not None
    assert adv["asset_id"] == "WTG-T"
    assert adv["predicted_rul_days"] == pytest.approx(44.0)
    assert adv["early_warning_triggered"] is True  # 44 < 45-day horizon
    enforce_safety_contract(rec)


def test_update_state_without_model_or_bnn_state_has_no_advisory():
    twin = WindTurbineDigitalTwin("WTG-T2", get_spec("GE-1.5"))
    rec = twin.update_state(_healthy_telemetry(), None)
    assert rec["advisory"] is None
    assert rec["advisory_source"] is None
    enforce_safety_contract(rec)


def test_update_state_uses_attached_model(serving_bundle):
    from src.models.serving import load_serving_model

    serving = load_serving_model(serving_bundle.checkpoint_path)
    twin = WindTurbineDigitalTwin("WTG-M", get_spec("GE-1.5"), serving_model=serving)
    rec = twin.update_state(
        _healthy_telemetry(),
        BNNState(predicted_rul_days=3000.0, epistemic_uncertainty=9.0, aleatoric_uncertainty=9.0),
    )
    assert rec["advisory_source"] == "model"
    adv = rec["advisory"]
    # bnn_state sentinel (3000 days) must NOT leak through the model path.
    assert 0.0 <= adv["predicted_rul_days"] < 1000.0
    enforce_safety_contract(rec)


def test_attach_serving_model_rejects_feature_mismatch(serving_bundle, tmp_path):
    from src.models.serving import ServingModel, load_serving_model
    from src.utils.artifacts import load_model_bundle

    load_serving_model(serving_bundle.checkpoint_path)
    bundle = load_model_bundle(serving_bundle.checkpoint_path)
    corrupt = bundle.__class__(
        model=bundle.model,
        architecture=bundle.architecture,
        features=FeatureConfig(stats=("mean",)),  # 5 dims != 25
        scaler=bundle.scaler,
        metadata=bundle.metadata,
        checkpoint_path=bundle.checkpoint_path,
        sidecar_path=bundle.sidecar_path,
    )
    twin = WindTurbineDigitalTwin("WTG-X", get_spec("GE-1.5"))
    with pytest.raises(ValueError, match="feature-dim mismatch"):
        twin.attach_serving_model(ServingModel(bundle=corrupt))


class _BrokenServingModel:
    """Minimal serving-model stand-in whose advisory() always raises."""

    features_config = FeatureConfig()
    expected_feature_dim = FeatureConfig().feature_dim

    def advisory(self, payload, window_df):
        raise RuntimeError("model exploded")


def test_serving_model_failure_falls_back_to_bnn_state():
    twin = WindTurbineDigitalTwin("WTG-FB", get_spec("GE-1.5"), serving_model=_BrokenServingModel())
    rec = twin.update_state(
        _healthy_telemetry(),
        BNNState(predicted_rul_days=33.0, epistemic_uncertainty=0.05, aleatoric_uncertainty=0.1),
    )
    # The update survives the model failure and serves the bnn_state path.
    assert rec["advisory_source"] == "bnn_state"
    assert rec["advisory"]["predicted_rul_days"] == pytest.approx(33.0)
    assert "serving model advisory failed" in rec["advisory_error"]
    enforce_safety_contract(rec)


def test_serving_model_failure_without_bnn_state_records_error():
    twin = WindTurbineDigitalTwin(
        "WTG-FB2", get_spec("GE-1.5"), serving_model=_BrokenServingModel()
    )
    rec = twin.update_state(_healthy_telemetry(), None)
    assert rec["advisory"] is None
    assert rec["advisory_source"] == "model"
    assert "serving model advisory failed" in rec["advisory_error"]
    enforce_safety_contract(rec)


def test_simulate_scenario_records_advisories():
    twin = WindTurbineDigitalTwin("WTG-S", get_spec("GE-1.5"))
    records = twin.simulate_scenario(profile="nominal", hours=6)
    assert len(records) == 6
    assert all(r["advisory"] is not None for r in records)  # bnn_state path
    assert all(r["advisory_source"] == "bnn_state" for r in records)


def test_prompt_includes_advisory_section():
    twin = WindTurbineDigitalTwin("WTG-P", get_spec("GE-1.5"))
    twin.update_state(
        _healthy_telemetry(),
        BNNState(predicted_rul_days=30.0, epistemic_uncertainty=0.05, aleatoric_uncertainty=0.1),
    )
    prompt = generate_engineering_prompt(twin)
    assert "ADVISORY ENGINE OUTPUT" in prompt
    assert "ADVISORY / DECISION-SUPPORT ONLY" in prompt


# --------------------------------------------------------------------------- #
# API twin endpoints                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def api_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    return TestClient(create_app())


def test_api_twin_status_creates_and_reports(api_client):
    resp = api_client.get("/twin/status", params={"asset_id": "WTG-A1", "model": "GE-1.5"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset_id"] == "WTG-A1"
    assert body["advisory_only"] is True
    assert body["n_state_records"] >= 1
    assert body["last_state"]["physics_violations"] == []
    assert body["agent_team"]["team_id"] == "CYBER_PRIME_DUAL_AGENT"
    assert body["last_state"]["agent_team"] == body["agent_team"]
    # No serving model configured in this app → no advisory source.
    assert body["last_state"]["advisory_source"] is None

    # Repeated calls reuse the same twin (registry), not a fresh one.
    again = api_client.get("/twin/status", params={"asset_id": "WTG-A1"}).json()
    assert again["n_state_records"] == body["n_state_records"]


def test_api_twin_status_unknown_model_404(api_client):
    resp = api_client.get("/twin/status", params={"asset_id": "WTG-A2", "model": "NOPE-9000"})
    assert resp.status_code == 404


def test_api_twin_status_reports_runtime_limits(api_client):
    resp = api_client.get("/twin/status", params={"asset_id": "WTG-LIM", "model": "GE-1.5"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["history_limit"] >= 1
    assert "serving_model_loaded" in body
    assert body["advisory_source"] is None  # seeded twin has no bnn_state / model
    # Record schema carries the runtime failover field (None when healthy).
    assert body["last_state"]["advisory_error"] is None


def test_api_twin_registry_is_lru_bounded(monkeypatch):
    """AV_TWIN_MAX_ASSETS caps the twin registry; evicting LRU assets."""
    monkeypatch.setenv("AV_TWIN_MAX_ASSETS", "2")

    from fastapi.testclient import TestClient

    from src.api.app import create_app

    client = TestClient(create_app())
    for i in range(3):
        resp = client.get("/twin/status", params={"asset_id": f"WTG-EV{i}", "model": "GE-1.5"})
        assert resp.status_code == 200

    # WTG-EV0 was evicted, so re-fetching it creates a fresh (seeded) twin.
    recreated = client.get("/twin/status", params={"asset_id": "WTG-EV0"}).json()
    assert recreated["n_state_records"] == 1
    # Registry never exceeds the configured cap.
    assert len(client.app.state.twins) == 2
    assert client.app.state.twin_max_assets == 2


def test_api_twin_simulate(api_client):
    resp = api_client.post(
        "/twin/simulate",
        json={"asset_id": "WTG-SIM", "model": "GE-1.5", "profile": "overload", "hours": 4},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["steps_executed"] == 4
    assert body["advisories_computed"] == 4  # simulate_scenario supplies bnn_state
    assert body["cumulative_wear"] > 0.0
    assert body["last_advisory"]["predicted_rul_days"] > 0
    assert len(body["last_records"]) == 4
    assert body["last_records"][-1]["advisory"] is not None

    # The twin registry now has history for this asset.
    status = api_client.get("/twin/status", params={"asset_id": "WTG-SIM"}).json()
    assert status["n_state_records"] == 5  # seed + 4 simulated steps


def test_api_twin_simulate_validates_profile(api_client):
    resp = api_client.post(
        "/twin/simulate",
        json={"asset_id": "WTG-BAD", "model": "GE-1.5", "profile": "hyperdrive", "hours": 2},
    )
    assert resp.status_code == 422


def test_api_twin_prompt(api_client):
    api_client.get("/twin/status", params={"asset_id": "WTG-PR"})
    resp = api_client.get("/twin/prompt", params={"asset_id": "WTG-PR"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset_id"] == "WTG-PR"
    assert "SYSTEM INSTRUCTIONS" in body["prompt"]
    assert "WTG-PR" in body["prompt"]
    assert "CYBER PRIME DUAL-AGENT ASSESSMENT" in body["prompt"]
    assert "MIKA / Maintenance Strategist" in body["prompt"]
    assert "KAI / Physics Constraint Sentinel" in body["prompt"]
    assert body["advisory_only"] is True


# --------------------------------------------------------------------------- #
# CLI twin-status --advisory                                                  #
# --------------------------------------------------------------------------- #
def test_cli_twin_status_advisory_bnn_state(tmp_path, capsys):
    from src.cli_twin import status_main

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "telemetry": {
                    "vibration_mms": 2.1,
                    "temperature_c": 61.0,
                    "rpm": 1450.0,
                    "oil_viscosity_cst": 31.0,
                    "load_pct": 80.0,
                },
                "bnn_state": {
                    "predicted_rul_days": 25.4,
                    "epistemic_uncertainty": 0.08,
                    "aleatoric_uncertainty": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )
    rc = status_main(["--asset-id", "WTG-CLI-T", "--payload", str(payload), "--advisory"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Advisory (source: bnn_state)" in out
    assert "Predicted RUL: 25.4 days" in out
    assert "Early warning (45d): TRIGGERED" in out
    assert "Cyber Prime Agents: MIKA + KAI" in out
    assert "MIKA:" in out
    assert "KAI:" in out


def test_cli_twin_status_advisory_without_inputs(capsys):
    from src.cli_twin import status_main

    rc = status_main(["--asset-id", "WTG-NONE", "--payload", ""])
    # Default healthy seed supplies a bnn_state, so an advisory exists even here.
    assert rc == 0


def test_cli_twin_status_json_includes_advisory(capsys):
    from src.cli_twin import status_main

    rc = status_main(["--asset-id", "WTG-JSON", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["health_state"]["advisory"] is not None
    assert out["health_state"]["advisory"]["advisory_only"] is True
