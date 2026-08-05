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


# --------------------------------------------------------------------------- #
# AeroZip-compressed telemetry CSVs                                           #
# --------------------------------------------------------------------------- #
AEROZIP_CSV_COLUMNS = ("timestamp", "sample_interval_s", "anomaly_score", "bypass", "payload_b64")


def write_aerozip_csv(
    df: pd.DataFrame,
    path: str,
    *,
    window_samples: int = 60,
    sample_interval_s: int = 600,
    baseline_mean: dict[str, float] | None = None,
    baseline_std: dict[str, float] | None = None,
) -> str:
    """Store a telemetry frame as an AeroZip-compressed CSV (one row per window).

    Each row carries the window-start timestamp, the anomaly score surfaced by
    the compressor, the bypass flag, and the base64 payload. Lossiness follows
    the AeroZip semantics (see src/models/telemetry/pipeline.py).
    """
    from src.models.telemetry.pipeline import compress_window

    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if len(df) < 1:
        raise ValueError("cannot compress an empty telemetry frame")

    time_index = pd.to_datetime(df.index) if isinstance(df.index, pd.DatetimeIndex) else None
    rows: list[dict] = []
    for start in range(0, len(df), window_samples):
        block = df.iloc[start: start + window_samples]
        comp = compress_window(
            {c: block[c].to_numpy(dtype=np.float64) for c in CHANNELS},
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
        )
        ts = (
            time_index[start].isoformat()
            if time_index is not None
            else pd.Timestamp("1970-01-01") + pd.to_timedelta(start * sample_interval_s, unit="s")
        )
        rows.append(
            {
                "timestamp": str(ts),
                "sample_interval_s": int(sample_interval_s),
                "anomaly_score": comp.anomaly_score,
                "bypass": comp.bypass,
                "payload_b64": comp.payload_b64,
            }
        )
    pd.DataFrame(rows, columns=AEROZIP_CSV_COLUMNS).to_csv(path, index=False)
    return path


def load_aerozip_csv(path: str, *, channels=CHANNELS) -> pd.DataFrame:
    """Restore a telemetry frame written by :func:`write_aerozip_csv`.

    Timestamps are reconstructed per sample from each window's start time and
    the stored sample interval. Restored values follow AeroZip lossiness
    (lossless during anomaly bypass).
    """
    from src.models.telemetry.pipeline import CompressedWindow, restore_window

    raw = pd.read_csv(path)
    missing = [c for c in AEROZIP_CSV_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Not an AeroZip CSV (missing columns: {missing})")

    frames: list[pd.DataFrame] = []
    for _, row in raw.iterrows():
        comp = CompressedWindow(
            codec="aerozip-v1",
            payload_b64=str(row["payload_b64"]),
            channels=tuple(channels),
            n_samples=0,  # decoded from the payload itself
            anomaly_score=float(row["anomaly_score"]),
            bypass=bool(row["bypass"]),
            raw_bytes=0,
            compressed_bytes=0,
        )
        restored = restore_window(comp)
        n = restored.n_samples
        start = pd.Timestamp(row["timestamp"])
        idx = start + pd.to_timedelta(np.arange(n) * float(row["sample_interval_s"]), unit="s")
        frames.append(pd.DataFrame({c: restored.channels[c] for c in channels}, index=idx))
    if not frames:
        return pd.DataFrame(columns=list(channels))
    out = pd.concat(frames).sort_index()
    out.index.name = "timestamp"
    return out
