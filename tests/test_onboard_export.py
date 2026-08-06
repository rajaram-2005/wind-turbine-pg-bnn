"""Phase 5: Hermes/Reptile export → serving → advisory round-trip."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.agents.hermes import HermesAgent, HermesConfig
from src.data.ingest import CHANNELS
from src.data.synthetic import SyntheticConfig, generate
from src.meta.reptile import ReptileConfig, meta_train
from src.meta.tasks import task_from_telemetry
from src.models.bnn import BayesianNeuralNetwork
from src.models.serving import load_serving_model
from src.utils.artifacts import export_onboarding_bundle, load_model_bundle
from src.utils.schema import Telemetry, TurbinePayload


@pytest.fixture(scope="module")
def onboarded(tmp_path_factory):
    """A tiny end-to-end onboarding run: Reptile meta-train (small) then
    Hermes ADAPT+GATE on one new asset, exported as deployment bundles."""
    torch.manual_seed(0)
    cfg_syn = SyntheticConfig(n_turbines=4, seq_len=800, seed=5)
    seqs = generate(cfg_syn)
    meta_tasks = [
        task_from_telemetry(
            asset_id=f"meta-{i}",
            df=df,
            rul_end_days=rul,
            n_support=6,
            sample_interval_s=cfg_syn.sample_interval_s,
            seed=10 + i,
        )
        for i, (df, rul) in enumerate(seqs[:3])
    ]
    rcfg = ReptileConfig(
        inner_lr=5e-3,
        inner_steps=3,
        meta_lr=0.4,
        tasks_per_iter=2,
        meta_iterations=3,
        num_samples=2,
        eval_mc_samples=4,
        seed=1,
    )
    meta_model = BayesianNeuralNetwork(in_features=meta_tasks[0].feature_dim, hidden_sizes=(16, 8))
    meta_model, _ = meta_train(meta_model, meta_tasks, rcfg)

    new_df, new_rul = seqs[3]
    new_task = task_from_telemetry(
        asset_id="turbine-NEW",
        df=new_df,
        rul_end_days=new_rul,
        n_support=6,
        sample_interval_s=cfg_syn.sample_interval_s,
        seed=99,
    )
    hermes = HermesAgent(
        meta_model,
        HermesConfig(adaptation=rcfg, max_rounds=1, eval_mc_samples=4, min_eval_shots=4, seed=1),
    )
    adapted, report = hermes.onboard(
        asset_id="turbine-NEW",
        support_x=new_task.support_x,
        support_y=new_task.support_y,
        unlabeled_x=new_task.query_x[:8],
        eval_x=new_task.query_x[8:],
        eval_y=new_task.query_y[8:],
    )
    out_dir = tmp_path_factory.mktemp("onboard_export")
    paths = export_onboarding_bundle(
        adapted,
        report,
        out_dir,
        features=None,  # canonical 60/20, 5 stats
        extra_metadata={"suite": "test_onboard_export"},
    )
    return {"paths": paths, "report": report, "df": new_df, "meta_model": meta_model}


def test_export_writes_bundle_and_report(onboarded):
    paths = onboarded["paths"]
    assert paths["checkpoint"].is_file()
    assert paths["sidecar"].is_file()
    assert paths["report"].is_file()

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["asset_id"] == "turbine-NEW"
    assert report["status"] in ("promoted", "shadow")
    assert report["advisory_only"] is True

    sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    assert sidecar["metadata"]["produced_by"] == "hermes-onboarding"
    assert sidecar["architecture"]["in_features"] == 25
    assert sidecar["architecture"]["hidden_sizes"] == [16, 8]


def test_export_load_advisory_round_trip(onboarded):
    """The exported Hermes model loads through the Phase-2 serving path and
    produces a model-based advisory (the deployment loop is closed)."""
    serving = load_serving_model(onboarded["paths"]["checkpoint"])
    assert serving.expected_feature_dim == 25

    df = onboarded["df"].drop(columns=["timestamp"])
    snap = df.iloc[-1]
    payload = TurbinePayload(
        asset_id="turbine-NEW",
        telemetry=Telemetry(**{c: float(np.clip(snap[c], 1e-3, None)) for c in CHANNELS}),
        bnn_state=None,  # model path: no bnn_state needed
    )
    rec = serving.advisory(payload, df)
    assert rec["advisory_only"] is True
    assert rec["asset_id"] == "turbine-NEW"
    assert 0.0 <= rec["predicted_rul_days"] <= 3650.0
    assert rec["generated_at"]

    # Bundle-level load agrees (weights round-trip).
    bundle = load_model_bundle(onboarded["paths"]["checkpoint"])
    assert bundle.metadata["onboarding_status"] == onboarded["report"].status


def test_meta_checkpoint_loads_through_serving(onboarded, tmp_path):
    """The Reptile meta-initialization itself is a deployable bundle."""
    from src.utils.artifacts import FeatureConfig, save_model_bundle

    ckpt = tmp_path / "meta.pt"
    save_model_bundle(
        onboarded["meta_model"],
        ckpt,
        features=FeatureConfig(),
        metadata={"kind": "reptile-meta"},
    )
    serving = load_serving_model(ckpt)
    df = onboarded["df"].drop(columns=["timestamp"])
    feats = serving.features(df)
    assert feats.shape[1] == 25
    snap = df.iloc[-1]
    payload = TurbinePayload(
        asset_id="meta-check",
        telemetry=Telemetry(**{c: float(np.clip(snap[c], 1e-3, None)) for c in CHANNELS}),
    )
    rec = serving.advisory(payload, df)
    assert rec["advisory_only"] is True
