"""Few-shot adaptation tasks for fleet onboarding.

A *task* is one turbine's degradation history, seen from the perspective of a
newly onboarded asset: a small *support* set of labeled feature windows (all
a new turbine realistically has after a few inspections) plus a larger *query*
set used only to score how well adaptation worked.

Tasks are the unit of Reptile meta-learning (`src/meta/reptile.py`) and of
the Hermes onboarding agent (`src/agents/hermes.py`). Every task carries
RUL-day labels compatible with the rest of the advisory pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.ingest import (
    CHANNELS,
    SlidingWindowConfig,
    robust_normalize,
    sliding_features,
)
from src.data.synthetic import SyntheticConfig, generate


@dataclass
class AdaptationTask:
    """One turbine's few-shot learning problem.

    ``support_*``: the scarce labeled shots available at onboarding time.
    ``query_*``: held-out labeled windows used ONLY to evaluate the adapted
    model — never for adaptation itself.
    """

    asset_id: str
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray

    @property
    def n_support(self) -> int:
        return int(self.support_y.shape[0])

    @property
    def n_query(self) -> int:
        return int(self.query_y.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.support_x.shape[1])


def split_task(
    x: np.ndarray,
    y: np.ndarray,
    n_support: int,
    seed: int = 0,
    asset_id: str = "task",
) -> AdaptationTask:
    """Split labeled windows into a random support/query partition."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array of feature windows")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if n_support < 1:
        raise ValueError("n_support must be >= 1 (a task needs at least one shot)")
    if n_support >= y.shape[0]:
        raise ValueError("n_support must leave at least one window for the query set")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(y.shape[0])
    s, q = idx[:n_support], idx[n_support:]
    return AdaptationTask(
        asset_id=asset_id,
        support_x=x[s],
        support_y=y[s],
        query_x=x[q],
        query_y=y[q],
    )


def task_from_telemetry(
    asset_id: str,
    df: pd.DataFrame,
    rul_end_days: float,
    n_support: int,
    sample_interval_s: int = 600,
    seed: int = 0,
    sw: SlidingWindowConfig | None = None,
) -> AdaptationTask:
    """Build a task from one turbine's telemetry frame.

    Window features and per-window RUL labels follow the same scheme as
    `src.data.synthetic.features_and_labels` (robust-normalized channels,
    sliding-window stats, RUL interpolated linearly down to end-of-sequence).
    """
    if sw is None:
        sw = SlidingWindowConfig(window_size=60, stride=20)
    norm, _ = robust_normalize(df[list(CHANNELS)])
    feats = sliding_features(norm, sw)
    if len(feats) <= n_support:
        raise ValueError(
            f"telemetry yields {len(feats)} windows, not enough for "
            f"n_support={n_support} plus a query set"
        )
    start_rul = rul_end_days + (len(df) * sample_interval_s / 86400.0)
    ruls = np.linspace(start_rul, rul_end_days, len(feats), dtype=np.float32)
    return split_task(feats, ruls, n_support, seed=seed, asset_id=asset_id)


def tasks_from_synthetic_fleet(
    cfg: SyntheticConfig | None = None,
    n_support: int = 8,
    seed: int = 0,
) -> list[AdaptationTask]:
    """Build one task per turbine of the synthetic fleet generator."""
    if cfg is None:
        cfg = SyntheticConfig()
    seqs = generate(cfg)
    tasks = []
    for i, (df, rul_end) in enumerate(seqs):
        tasks.append(
            task_from_telemetry(
                asset_id=f"fleet-{cfg.seed}-{i}",
                df=df,
                rul_end_days=rul_end,
                n_support=n_support,
                sample_interval_s=cfg.sample_interval_s,
                seed=seed * 1000 + i,
            )
        )
    return tasks
