"""Reptile meta-learning for few-shot onboarding of new fleet assets.

Reptile (Nichol, Achiam & Schulman, 2018; arXiv:1803.02999) is a first-order
meta-learning algorithm. For each meta-iteration, sample a batch of tasks,
run a few gradient steps on each task's support set starting from the current
meta-initialization φ (the *inner loop*), then move φ toward the average of
the adapted weights::

    θ_k = SGD_k steps(φ, support_k)          # per sampled task
    φ   ← φ + ε · mean_k(θ_k − φ)            # meta-update, ε = meta_lr

The result is an initialization from which a handful of labeled windows from
a *new* turbine (few-shot onboarding) yields an accurate advisory model after
a few gradient steps — instead of weeks of data collection and a full retrain.

Everything here operates on model weights and feature tensors only. Adapted
models feed the same advisory pipeline (`src/models/predictor.run_advisory`)
as any other model; no control path is touched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from src.eval.calibration import early_warning_metrics
from src.meta.tasks import AdaptationTask
from src.models.bnn import BayesianNeuralNetwork, TrainConfig, elbo_loss, predict


@dataclass
class ReptileConfig:
    inner_lr: float = 5e-3          # Adam lr inside the task loop
    inner_steps: int = 5            # gradient steps per task support set
    meta_lr: float = 0.4            # ε: interpolation toward adapted weights (1.0 = full)
    tasks_per_iter: int = 4         # tasks sampled per meta-iteration
    meta_iterations: int = 25
    num_samples: int = 3            # MC samples per ELBO evaluation (inner loop)
    kl_weight: float = 1e-3
    eval_mc_samples: int = 16       # MC samples for query-set evaluation
    seed: int = 0


def clone_model(model: BayesianNeuralNetwork) -> BayesianNeuralNetwork:
    """Deep-copy a model so adaptation never mutates the source weights."""
    return copy.deepcopy(model)


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def _inner_train_cfg(cfg: ReptileConfig) -> TrainConfig:
    # No telemetry in the inner loop (normalized features only): physics
    # coupling stays with the prior, exactly like scripts/train_demo.py.
    return TrainConfig(
        num_samples=cfg.num_samples,
        kl_weight=cfg.kl_weight,
        physics_weight=0.0,
    )


def support_loss(model: BayesianNeuralNetwork, x: np.ndarray, y: np.ndarray, cfg: ReptileConfig) -> float:
    """Mean ELBO of `model` on (x, y) without updating it."""
    with torch.no_grad():
        loss, _ = elbo_loss(model, _to_tensor(x), _to_tensor(y), telemetry=None, cfg=_inner_train_cfg(cfg))
    return float(loss.item())


def inner_adapt(
    model: BayesianNeuralNetwork,
    support_x: np.ndarray,
    support_y: np.ndarray,
    cfg: ReptileConfig,
) -> tuple[BayesianNeuralNetwork, float]:
    """Clone `model` and run `cfg.inner_steps` Adam steps on the support set.

    Returns (adapted clone, final support loss). The source model is never
    mutated — callers may safely pass the fleet meta-model.
    """
    xt = _to_tensor(support_x)
    yt = _to_tensor(support_y)
    if xt.ndim != 2 or yt.ndim != 1 or xt.shape[0] != yt.shape[0]:
        raise ValueError("support_x must be (n, d) and support_y (n,)")
    if yt.shape[0] < 1:
        raise ValueError("cannot adapt on an empty support set")
    tcfg = _inner_train_cfg(cfg)
    adapted = clone_model(model)
    opt = torch.optim.Adam(adapted.parameters(), lr=cfg.inner_lr)
    last_loss = float("nan")
    adapted.train()
    for _ in range(cfg.inner_steps):
        opt.zero_grad()
        loss, _ = elbo_loss(adapted, xt, yt, telemetry=None, cfg=tcfg)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    return adapted, last_loss


# Public alias with onboarding semantics (identical mechanics to inner_adapt).
few_shot_adapt = inner_adapt


def reptile_meta_iteration(
    meta_model: BayesianNeuralNetwork,
    tasks: list[AdaptationTask],
    cfg: ReptileConfig,
    rng: np.random.Generator,
) -> dict:
    """One Reptile meta-iteration: adapt on a sampled task batch, then move
    the meta-weights toward the mean of the adapted weights."""
    if not tasks:
        raise ValueError("meta-training needs at least one task")
    k = min(cfg.tasks_per_iter, len(tasks))
    idx = rng.choice(len(tasks), size=k, replace=False)

    meta_params = list(meta_model.parameters())
    direction = [torch.zeros_like(p) for p in meta_params]
    losses: list[float] = []
    task_ids: list[str] = []
    for i in idx:
        task = tasks[int(i)]
        adapted, loss = inner_adapt(meta_model, task.support_x, task.support_y, cfg)
        losses.append(loss)
        task_ids.append(task.asset_id)
        with torch.no_grad():
            for acc, p_meta, p_task in zip(direction, meta_params, adapted.parameters()):
                acc += (p_task - p_meta) / k
    with torch.no_grad():
        for p_meta, d in zip(meta_params, direction):
            p_meta.add_(d, alpha=cfg.meta_lr)
    return {
        "mean_support_loss": float(np.mean(losses)),
        "task_ids": task_ids,
        "mean_param_shift": float(np.mean([float(d.abs().mean()) for d in direction])),
    }


def meta_train(
    meta_model: BayesianNeuralNetwork,
    tasks: list[AdaptationTask],
    cfg: ReptileConfig,
) -> tuple[BayesianNeuralNetwork, dict]:
    """Meta-train `meta_model` in place over the fleet's tasks. Returns the
    model and a per-iteration history log."""
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    history = {"iterations": []}
    for it in range(cfg.meta_iterations):
        log = reptile_meta_iteration(meta_model, tasks, cfg, rng)
        log["iteration"] = it
        history["iterations"].append(log)
    return meta_model, history


@torch.no_grad()
def _query_predictions(model: BayesianNeuralNetwork, query_x: np.ndarray, mc_samples: int) -> np.ndarray:
    return predict(model, _to_tensor(query_x), mc_samples=mc_samples)["mean_pred"].numpy()


def evaluate_few_shot(
    meta_model: BayesianNeuralNetwork,
    tasks: list[AdaptationTask],
    cfg: ReptileConfig,
) -> dict:
    """Adapt `meta_model` few-shot on each task's support set and score the
    adapted model on the task's query set.

    Reports query RMSE (days), 45-day early-warning classification accuracy
    (the fleet's headline metric), and a naive baseline that always predicts
    the support-set label mean — the floor any useful adaptation must beat.
    """
    if not tasks:
        raise ValueError("evaluation needs at least one task")
    torch.manual_seed(cfg.seed * 1000 + 17)  # deterministic MC noise for evals
    rmses: list[float] = []
    accs: list[float] = []
    naive: list[float] = []
    for task in tasks:
        adapted, _ = inner_adapt(meta_model, task.support_x, task.support_y, cfg)
        pred = _query_predictions(adapted, task.query_x, cfg.eval_mc_samples)
        rmses.append(float(np.sqrt(np.mean((pred - task.query_y) ** 2))))
        accs.append(early_warning_metrics(task.query_y, pred)["accuracy"])
        naive.append(float(np.sqrt(np.mean((task.support_y.mean() - task.query_y) ** 2))))
    return {
        "mean_query_rmse_days": float(np.mean(rmses)),
        "per_task_rmse_days": rmses,
        "mean_early_warning_accuracy": float(np.mean(accs)),
        "naive_mean_rmse_days": float(np.mean(naive)),
        "n_tasks": len(tasks),
    }
