"""Synthetic drivetrain-telemetry generator for demos and unit tests.

Simulates a population of turbines slowly degrading over time, with random
shocks near end-of-life (elevated vibration/temperature, reduced viscosity).
Labels are true RUL (days) derived from the damage integral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.ingest import CHANNELS


@dataclass
class SyntheticConfig:
    n_turbines: int = 50
    seq_len: int = 2000           # samples per turbine
    sample_interval_s: int = 600  # 10-min SCADA
    seed: int = 42


def _turbine_sequence(rng: np.random.Generator, seq_len: int) -> tuple[pd.DataFrame, float]:
    """Return (telemetry_df, rul_at_end_days). Damage integrates exponentially
    toward end-of-life; RUL at end is set by how close to threshold we got."""
    t = np.arange(seq_len)
    # Health state h ∈ [0,1], h=1 healthy, h=0 failure
    failure_rate = rng.exponential(0.0004) + 1e-5
    damage = np.cumsum(failure_rate * (1.0 + 0.5 * np.sin(t / 200.0)))
    # Random shock in late life
    shock_idx = seq_len - rng.integers(100, 400)
    shock_mag = rng.uniform(0.5, 2.0)
    damage[shock_idx:] += shock_mag * np.linspace(0, 1, seq_len - shock_idx) ** 2
    h = np.clip(1.0 - damage / damage[-1], 0.0, 1.0)
    rul_days = float(np.clip(h[-1] * 365.0, 0.0, 365.0))

    # Telemetry: start at healthy levels, drift toward fault levels as h→0
    vib = 1.5 + (4.0 - 1.5) * (1.0 - h) + rng.normal(0, 0.1, seq_len)
    temp = 55.0 + (30.0) * (1.0 - h) + rng.normal(0, 0.3, seq_len)
    rpm = 1650 + 100 * (1.0 - h) + rng.normal(0, 5, seq_len)
    visc = 30.0 - 20 * (1.0 - h) + rng.normal(0, 0.3, seq_len)
    load = 70.0 + 25 * (1.0 - h) + rng.normal(0, 1.0, seq_len)
    rpm = np.clip(rpm, 0, 2000)
    visc = np.clip(visc, 2, 80)
    load = np.clip(load, 0, 115)
    vib = np.clip(vib, 0, 20)

    df = pd.DataFrame(
        {
            "vibration_mms": vib.astype(np.float32),
            "temperature_c": temp.astype(np.float32),
            "rpm": rpm.astype(np.float32),
            "oil_viscosity_cst": visc.astype(np.float32),
            "load_pct": load.astype(np.float32),
        }
    )
    df["timestamp"] = pd.date_range("2025-01-01", periods=seq_len, freq="10min")
    return df, rul_days


def generate(cfg: SyntheticConfig | None = None) -> list[tuple[pd.DataFrame, float]]:
    if cfg is None:
        cfg = SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    return [_turbine_sequence(rng, cfg.seq_len) for _ in range(cfg.n_turbines)]


def features_and_labels(
    cfg: SyntheticConfig | None = None,
):
    """Return (X, y, (scale_lo, scale_hi)) ready for BNN training."""
    if cfg is None:
        cfg = SyntheticConfig()
    from src.data.ingest import SlidingWindowConfig, robust_normalize, sliding_features

    seqs = generate(cfg)
    sw = SlidingWindowConfig(window_size=60, stride=20)
    X_list, y_list = [], []
    for df, rul_end in seqs:
        norm, _ = robust_normalize(df[list(CHANNELS)])
        feats = sliding_features(norm, sw)
        # Assign per-window RUL: linearly interpolate from ~start rul down to rul_end
        if len(feats) == 0:
            continue
        start_rul = rul_end + (cfg.seq_len * cfg.sample_interval_s / 86400.0)
        ruls = np.linspace(start_rul, rul_end, len(feats), dtype=np.float32)
        X_list.append(feats)
        y_list.append(ruls)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y
