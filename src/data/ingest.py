"""SCADA-style telemetry ingestion and sliding-window feature extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = ("vibration_mms", "temperature_c", "rpm", "oil_viscosity_cst", "load_pct")


@dataclass
class SlidingWindowConfig:
    window_size: int = 60       # samples
    stride: int = 10            # samples
    stats: Sequence[str] = ("mean", "std", "min", "max", "rms")


def robust_normalize(df: pd.DataFrame, quantile_lo: float = 0.01, quantile_hi: float = 0.99):
    """Robust min-max scaling using 1st/99th percentiles to limit outlier pull.

    Returns (normalized_df, scale_dict) where scale_dict holds (lo, hi) per
    column so transform can be inverted on future data.
    """
    out = pd.DataFrame(index=df.index)
    scale: dict = {}
    for c in df.columns:
        lo = float(df[c].quantile(quantile_lo))
        hi = float(df[c].quantile(quantile_hi))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        out[c] = (df[c] - lo) / (hi - lo)
        scale[c] = (lo, hi)
    return out, scale


def sliding_features(
    df: pd.DataFrame,
    cfg: SlidingWindowConfig | None = None,
    channels: Sequence[str] = CHANNELS,
) -> np.ndarray:
    """
    Convert a time-indexed telemetry frame into a 2D array of windowed feature
    vectors of shape (n_windows, len(channels)*len(stats)).
    """
    if cfg is None:
        cfg = SlidingWindowConfig()
    data = df[list(channels)].to_numpy(dtype=np.float32)
    n = len(data)
    w = cfg.window_size
    s = cfg.stride
    if n < w:
        return np.zeros((0, len(channels) * len(cfg.stats)), dtype=np.float32)
    starts = list(range(0, n - w + 1, s))
    feats: list[np.ndarray] = []
    for st in starts:
        win = data[st: st + w]
        vec = []
        for stat in cfg.stats:
            if stat == "mean":
                vec.append(win.mean(axis=0))
            elif stat == "std":
                vec.append(win.std(axis=0))
            elif stat == "min":
                vec.append(win.min(axis=0))
            elif stat == "max":
                vec.append(win.max(axis=0))
            elif stat == "rms":
                vec.append(np.sqrt((win ** 2).mean(axis=0)))
            else:
                raise ValueError(f"Unknown stat: {stat}")
        feats.append(np.concatenate(vec))
    return np.stack(feats, axis=0)


def load_csv(path: str) -> pd.DataFrame:
    """Thin wrapper so we can swap in parquet/SQL/etc. Expects a `timestamp`
    column plus CHANNELS."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df.set_index("timestamp").sort_index()
