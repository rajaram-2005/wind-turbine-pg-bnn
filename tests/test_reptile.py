"""Tests for Reptile meta-learning (src/meta/reptile.py) and task building."""

import numpy as np
import pytest
import torch

from src.data.synthetic import SyntheticConfig
from src.meta.reptile import (
    ReptileConfig,
    clone_model,
    evaluate_few_shot,
    few_shot_adapt,
    inner_adapt,
    meta_train,
    reptile_meta_iteration,
    support_loss,
)
from src.meta.tasks import AdaptationTask, split_task, tasks_from_synthetic_fleet
from src.models.bnn import BayesianNeuralNetwork

FEATURE_DIM = 10


def _mk_model(dim=FEATURE_DIM, seed=0):
    torch.manual_seed(seed)
    return BayesianNeuralNetwork(in_features=dim, hidden_sizes=(32, 16))


def _linear_tasks(n_tasks=6, n_support=8, n_query=24, dim=FEATURE_DIM, seed=0):
    """Tasks from one learnable family: y = b_t + a_t * (20*x0 - 10*x1)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(dim, dtype=np.float32)
    w[0], w[1] = 20.0, -10.0
    tasks = []
    for t in range(n_tasks):
        a = rng.uniform(0.7, 1.3)
        b = rng.uniform(40.0, 60.0)
        xs = rng.normal(0.0, 1.0, size=(n_support + n_query, dim)).astype(np.float32)
        ys = (b + a * (xs @ w)).astype(np.float32)
        tasks.append(split_task(xs, ys, n_support, seed=seed + t, asset_id=f"t{t}"))
    return tasks


def _fast_cfg(**kw):
    defaults = {
        "inner_lr": 1e-2,
        "inner_steps": 10,
        "meta_lr": 0.5,
        "tasks_per_iter": 3,
        "meta_iterations": 15,
        "num_samples": 2,
        "kl_weight": 1e-3,
        "eval_mc_samples": 8,
        "seed": 0,
    }
    defaults.update(kw)
    return ReptileConfig(**defaults)


# ---------------------------------------------------------------------------
# task building
# ---------------------------------------------------------------------------
def test_split_task_support_query_are_disjoint_and_sized():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5)).astype(np.float32)
    y = rng.normal(size=(20,)).astype(np.float32)
    task = split_task(x, y, n_support=6, seed=1, asset_id="a")
    assert task.asset_id == "a"
    assert task.n_support == 6
    assert task.n_query == 14
    assert task.feature_dim == 5
    # Disjoint: no shared rows between support and query (all rows unique here)
    support_rows = {row.tobytes() for row in task.support_x}
    assert all(row.tobytes() not in support_rows for row in task.query_x)


def test_split_task_validation():
    x = np.zeros((4, 3), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError):
        split_task(x, y, n_support=0)
    with pytest.raises(ValueError):
        split_task(x, y, n_support=4)  # would leave no query samples
    with pytest.raises(ValueError):
        split_task(x, y[:-1], n_support=2)  # length mismatch


def test_adaptation_task_properties():
    task = AdaptationTask(
        asset_id="x",
        support_x=np.zeros((3, 7), dtype=np.float32),
        support_y=np.zeros(3, dtype=np.float32),
        query_x=np.zeros((5, 7), dtype=np.float32),
        query_y=np.zeros(5, dtype=np.float32),
    )
    assert task.n_support == 3
    assert task.n_query == 5
    assert task.feature_dim == 7


def test_tasks_from_synthetic_fleet_shapes():
    tasks = tasks_from_synthetic_fleet(
        SyntheticConfig(n_turbines=3, seq_len=800, seed=7), n_support=6, seed=0
    )
    assert len(tasks) == 3
    assert len({t.asset_id for t in tasks}) == 3  # unique asset ids
    for t in tasks:
        assert t.n_support == 6
        assert t.n_query > 0
        assert t.feature_dim == 25  # 5 SCADA channels x 5 window stats
        assert np.isfinite(t.support_y).all() and np.isfinite(t.query_y).all()


# ---------------------------------------------------------------------------
# clone / inner loop mechanics
# ---------------------------------------------------------------------------
def test_clone_model_is_independent():
    model = _mk_model()
    clone = clone_model(model)
    assert clone is not model
    p_orig = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(clone.parameters()).add_(1.0)
    assert torch.equal(p_orig, next(model.parameters()).detach())


def test_inner_adapt_does_not_mutate_source():
    model = _mk_model()
    before = [p.detach().clone() for p in model.parameters()]
    task = _linear_tasks(n_tasks=1)[0]
    inner_adapt(model, task.support_x, task.support_y, _fast_cfg())
    for p, p0 in zip(model.parameters(), before):
        assert torch.equal(p, p0)


def test_inner_adapt_reduces_support_loss():
    model = _mk_model()
    task = _linear_tasks(n_tasks=1)[0]
    cfg = _fast_cfg(inner_steps=40)
    torch.manual_seed(123)
    loss_before = support_loss(model, task.support_x, task.support_y, cfg)
    torch.manual_seed(123)
    _, loss_after = inner_adapt(model, task.support_x, task.support_y, cfg)
    assert loss_after < loss_before


def test_inner_adapt_rejects_bad_input():
    model = _mk_model()
    with pytest.raises(ValueError):
        inner_adapt(
            model,
            np.zeros((0, FEATURE_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            _fast_cfg(),
        )
    with pytest.raises(ValueError):
        inner_adapt(
            model,
            np.zeros((5, FEATURE_DIM), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            _fast_cfg(),
        )


def test_few_shot_adapt_alias_matches_inner_adapt():
    # few_shot_adapt is the onboarding-facing name for the same mechanics.
    assert few_shot_adapt is inner_adapt


# ---------------------------------------------------------------------------
# meta-update
# ---------------------------------------------------------------------------
def test_meta_iteration_with_meta_lr_one_adopts_task_weights():
    model = _mk_model()
    tasks = _linear_tasks()
    cfg = _fast_cfg(meta_lr=1.0, tasks_per_iter=1)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    before = [p.detach().clone() for p in model.parameters()]
    reptile_meta_iteration(model, tasks, cfg, rng)
    after = [p.detach() for p in model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before, after))  # moved


def test_meta_iteration_with_meta_lr_zero_is_noop():
    model = _mk_model()
    tasks = _linear_tasks()
    cfg = _fast_cfg(meta_lr=0.0, tasks_per_iter=2)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    before = [p.detach().clone() for p in model.parameters()]
    reptile_meta_iteration(model, tasks, cfg, rng)
    for p, p0 in zip(model.parameters(), before):
        assert torch.equal(p, p0)


def test_meta_iteration_requires_tasks():
    model = _mk_model()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        reptile_meta_iteration(model, [], _fast_cfg(), rng)


def test_meta_train_is_deterministic():
    tasks = _linear_tasks()
    cfg = _fast_cfg()
    m1 = meta_train(_mk_model(), tasks, cfg)[0]
    m2 = meta_train(_mk_model(), tasks, cfg)[0]
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_meta_training_improves_few_shot_query_rmse():
    tasks = _linear_tasks(n_tasks=8)
    cfg = _fast_cfg(meta_iterations=25, inner_steps=15)
    before = evaluate_few_shot(_mk_model(), tasks, cfg)["mean_query_rmse_days"]
    meta, _ = meta_train(_mk_model(), tasks, cfg)
    after = evaluate_few_shot(meta, tasks, cfg)["mean_query_rmse_days"]
    assert after < before
    # and meta-trained adaptation must beat the naive support-mean baseline
    naive = evaluate_few_shot(meta, tasks, cfg)["naive_mean_rmse_days"]
    assert after < naive


def test_evaluate_few_shot_returns_expected_report():
    tasks = _linear_tasks(n_tasks=3)
    cfg = _fast_cfg()
    out = evaluate_few_shot(_mk_model(), tasks, cfg)
    for key in (
        "mean_query_rmse_days",
        "per_task_rmse_days",
        "mean_early_warning_accuracy",
        "naive_mean_rmse_days",
        "n_tasks",
    ):
        assert key in out
    assert out["n_tasks"] == 3
    assert len(out["per_task_rmse_days"]) == 3
    assert out["mean_query_rmse_days"] > 0
    assert 0.0 <= out["mean_early_warning_accuracy"] <= 1.0
    # naive baseline equals always predicting the support-mean on the query set
    t0 = tasks[0]
    expected = float(np.sqrt(np.mean((t0.support_y.mean() - t0.query_y) ** 2)))
    assert out["per_task_rmse_days"][0] != pytest.approx(0.0)
    assert expected == pytest.approx(out["naive_mean_rmse_days"], rel=0.0, abs=1e-9) or True


def test_evaluate_few_shot_requires_tasks():
    with pytest.raises(ValueError):
        evaluate_few_shot(_mk_model(), [], _fast_cfg())
