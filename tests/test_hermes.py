"""Tests for the Hermes self-training onboarding agent (src/agents/hermes.py)."""

import datetime as dt

import numpy as np
import pytest
import torch

from src.agents.hermes import (
    HermesAgent,
    HermesConfig,
    OnboardingReport,
    select_confident,
)
from src.meta.reptile import ReptileConfig, meta_train
from src.models.bnn import BayesianNeuralNetwork
from src.utils.safety import SafetyBoundaryError, enforce_safety_contract

DIM = 10


def _mk_model(seed=0):
    torch.manual_seed(seed)
    return BayesianNeuralNetwork(in_features=DIM, hidden_sizes=(32, 16))


def _task_data(n_support=8, n_pool=24, n_eval=12, dim=DIM, seed=0):
    """Learnable regression data: y = 50 + 20*x0 - 10*x1 (+ tiny noise)."""
    rng = np.random.default_rng(seed)
    n = n_support + n_pool + n_eval
    x = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
    y = (50.0 + 20.0 * x[:, 0] - 10.0 * x[:, 1]).astype(np.float32)
    return (
        x[:n_support],
        y[:n_support],
        x[n_support : n_support + n_pool],
        x[n_support + n_pool :],
        y[n_support + n_pool :],
    )


def _fast_cfg(**kw):
    adapt = ReptileConfig(
        inner_lr=1e-2,
        inner_steps=10,
        meta_lr=0.5,
        tasks_per_iter=2,
        meta_iterations=5,
        num_samples=2,
        eval_mc_samples=8,
        seed=0,
    )
    defaults = {
        "adaptation": adapt,
        "confidence_tau_days": 1e9,  # accept everything unless overridden
        "max_rounds": 3,
        "max_pseudo_per_round": 4,
        "promotion_max_rmse_days": 1e9,
        "promotion_min_accuracy": 0.0,
        "min_eval_shots": 4,
        "eval_mc_samples": 8,
        "seed": 0,
    }
    defaults.update(kw)
    return HermesConfig(**defaults)


def _run_agent(agent, **kw):
    sx, sy, pool, ex, ey = _task_data()
    args = {
        "asset_id": "t-001",
        "support_x": sx,
        "support_y": sy,
        "unlabeled_x": pool,
        "eval_x": ex,
        "eval_y": ey,
    }
    args.update(kw)
    return agent.onboard(**args)


# ---------------------------------------------------------------------------
# confident pseudo-label selection
# ---------------------------------------------------------------------------
def test_select_confident_filters_by_tau_and_orders_by_confidence():
    pred = {
        "epistemic_std": torch.tensor([0.5, 0.1, 5.0, 0.2]),
        "mean_pred": torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }
    idx = select_confident(pred, tau_days=0.3, k=4)
    assert list(idx) == [1, 3]  # most-confident first, 5.0 excluded
    idx = select_confident(pred, tau_days=1.0, k=2)
    assert list(idx) == [1, 3]  # capped at k
    idx = select_confident(pred, tau_days=0.05, k=4)
    assert idx.size == 0  # nothing clears tau


# ---------------------------------------------------------------------------
# onboarding flow mechanics
# ---------------------------------------------------------------------------
def test_onboard_without_unlabeled_runs_zero_rounds():
    agent = HermesAgent(_mk_model(), _fast_cfg())
    _, report = _run_agent(agent, unlabeled_x=None)
    assert report.rounds_completed == 0
    assert report.n_pseudo_labels == 0


def test_onboard_empty_unlabeled_pool_runs_zero_rounds():
    agent = HermesAgent(_mk_model(), _fast_cfg())
    _, report = _run_agent(agent, unlabeled_x=np.zeros((0, DIM), dtype=np.float32))
    assert report.rounds_completed == 0
    assert report.n_pseudo_labels == 0


def test_self_training_respects_round_and_per_round_caps():
    cfg = _fast_cfg(max_rounds=3, max_pseudo_per_round=4)
    agent = HermesAgent(_mk_model(), cfg)
    _, report = _run_agent(agent)
    assert report.rounds_completed <= 3
    assert report.n_pseudo_labels <= 3 * 4


def test_self_training_exhausts_pool_and_stops_early():
    # Pool smaller than one round's cap: after one round it is empty, so the
    # loop stops before max_rounds even though every window is "confident".
    cfg = _fast_cfg(max_rounds=5, max_pseudo_per_round=30)
    agent = HermesAgent(_mk_model(), cfg)
    sx, sy, pool, ex, ey = _task_data(n_pool=10)
    _, report = agent.onboard("t", sx, sy, unlabeled_x=pool, eval_x=ex, eval_y=ey)
    assert report.rounds_completed == 1  # pool emptied in a single round
    assert report.n_pseudo_labels == 10


def test_onboard_is_deterministic_for_fixed_seed():
    _, r1 = _run_agent(HermesAgent(_mk_model(), _fast_cfg()))
    _, r2 = _run_agent(HermesAgent(_mk_model(), _fast_cfg()))
    assert r1.n_pseudo_labels == r2.n_pseudo_labels
    assert r1.rounds_completed == r2.rounds_completed
    assert r1.eval_rmse_days == r2.eval_rmse_days


def test_onboard_never_mutates_the_fleet_meta_model():
    model = _mk_model()
    before = [p.detach().clone() for p in model.parameters()]
    _run_agent(HermesAgent(model, _fast_cfg()))
    for p, p0 in zip(model.parameters(), before):
        assert torch.equal(p, p0)


def test_onboard_validates_inputs():
    agent = HermesAgent(_mk_model(), _fast_cfg())
    x = np.zeros((5, DIM), dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError):
        agent.onboard("t", np.zeros((0, DIM), dtype=np.float32), np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError):
        agent.onboard("t", x, y[:-1])  # length mismatch
    with pytest.raises(ValueError):
        agent.onboard("t", x, y, unlabeled_x=np.zeros((3, DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        agent.onboard("t", x, y, eval_x=x[:-1], eval_y=y)  # eval shape mismatch


# ---------------------------------------------------------------------------
# promotion gate (fail-closed)
# ---------------------------------------------------------------------------
def test_gate_blocks_when_no_eval_split():
    _, report = _run_agent(HermesAgent(_mk_model(), _fast_cfg()), eval_x=None, eval_y=None)
    assert not report.promoted
    assert report.status == "shadow"
    assert report.eval_rmse_days is None
    assert "evaluation" in report.rationale


def test_gate_blocks_when_eval_too_small():
    sx, sy, pool, ex, ey = _task_data(n_eval=3)  # below min_eval_shots=4
    _, report = HermesAgent(_mk_model(), _fast_cfg()).onboard(
        "t", sx, sy, unlabeled_x=pool, eval_x=ex, eval_y=ey
    )
    assert not report.promoted
    assert report.status == "shadow"


def test_gate_promotes_when_thresholds_met():
    _, report = _run_agent(HermesAgent(_mk_model(), _fast_cfg()))
    assert report.promoted
    assert report.status == "promoted"
    assert report.eval_rmse_days is not None
    assert report.eval_early_warning_accuracy is not None
    assert "PASSED" in report.rationale


def test_gate_blocks_model_that_misses_rmse_budget():
    cfg = _fast_cfg(promotion_max_rmse_days=-1.0)  # impossible budget
    _, report = _run_agent(HermesAgent(_mk_model(), cfg))
    assert not report.promoted
    assert report.status == "shadow"
    assert "RMSE" in report.rationale
    assert "BLOCKED" in report.rationale


def test_gate_blocks_model_that_misses_accuracy_floor():
    cfg = _fast_cfg(promotion_min_accuracy=1.01)  # impossible floor
    _, report = _run_agent(HermesAgent(_mk_model(), cfg))
    assert not report.promoted
    assert report.status == "shadow"


# ---------------------------------------------------------------------------
# report + safety contract
# ---------------------------------------------------------------------------
def test_report_passes_safety_contract_and_is_advisory_only():
    _, report = _run_agent(HermesAgent(_mk_model(), _fast_cfg()))
    d = report.to_dict()  # must not raise
    assert d["advisory_only"] is True
    assert "Decision-support only" in d["disclaimer"]
    assert d["status"] in ("promoted", "shadow")
    # generated_at is an ISO-8601 timestamp
    dt.datetime.fromisoformat(d["generated_at"])
    # the same gate rejects tampered payloads
    d["throttle_command"] = 0.5
    with pytest.raises(SafetyBoundaryError):
        enforce_safety_contract(d)


def test_report_thresholds_echo_config():
    cfg = _fast_cfg(promotion_max_rmse_days=77.0, confidence_tau_days=12.0)
    _, report = _run_agent(HermesAgent(_mk_model(), cfg))
    assert report.promotion_thresholds["max_rmse_days"] == 77.0
    assert report.promotion_thresholds["confidence_tau_days"] == 12.0


def test_end_to_end_meta_trained_model_promotes_on_easy_task():
    """Integration: quick Reptile meta-train, then Hermes onboarding promotes."""
    rng = np.random.default_rng(5)
    w = np.zeros(DIM, dtype=np.float32)
    w[0], w[1] = 20.0, -10.0
    tasks = []
    from src.meta.tasks import split_task

    for t in range(6):
        b = rng.uniform(40.0, 60.0)
        xs = rng.normal(0, 1, size=(32, DIM)).astype(np.float32)
        ys = (b + xs @ w).astype(np.float32)
        tasks.append(split_task(xs, ys, 8, seed=t, asset_id=f"m{t}"))
    cfg = _fast_cfg()
    meta, _ = meta_train(_mk_model(), tasks, cfg.adaptation)
    _, report = _run_agent(HermesAgent(meta, cfg))
    assert report.status in ("promoted", "shadow")  # contract holds either way
    assert report.promoted


def test_report_frozen_dataclass_fields():
    report = OnboardingReport(
        asset_id="a",
        status="shadow",
        promoted=False,
        rounds_completed=0,
        n_labeled_shots=1,
        n_pseudo_labels=0,
        eval_rmse_days=None,
        eval_early_warning_accuracy=None,
        promotion_thresholds={},
        rationale="r",
    )
    with pytest.raises(AttributeError):  # frozen dataclass: no attribute assignment
        report.promoted = True
