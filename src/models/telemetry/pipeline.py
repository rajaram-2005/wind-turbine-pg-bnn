"""AeroZip telemetry pipeline: window → compressed payload → window.

Gives the previously test-only :class:`AeroZipCompressor` a real consumer:
``compress_window`` / ``restore_window`` are the serving-grade wrappers used
by the ingestion layer, the API endpoints, and the UI metrics panel.

Semantics
---------
* Per call, compression uses a FRESH compressor (per-channel running offset =
  0), so every payload is self-contained and restore is deterministic.
* Normal mode is LOSSY: per-channel reconstruction error is bounded by the
  quantization quantum plus deadband drift (suppressed sub-deadband deltas).
* Anomaly bypass mode (anomaly_score > threshold) is LOSSLESS float64 —
  diagnostic fidelity is guaranteed during fault conditions.
* The anomaly score is always surfaced to the caller.
"""

from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from src.models.telemetry.aerozip import (
    AeroZipCompressor,
    AeroZipConfig,
    _decode_varint,
    anomaly_score,
)

CODEC = "aerozip-v1"
DEFAULT_CHANNELS = tuple(AeroZipConfig().channels)


@dataclass(frozen=True)
class CompressedWindow:
    """A transport-ready compressed window (base64 for JSON/CSV)."""

    codec: str
    payload_b64: str
    channels: tuple[str, ...]
    n_samples: int
    anomaly_score: float
    bypass: bool
    raw_bytes: int
    compressed_bytes: int

    @property
    def ratio(self) -> float:
        return self.compressed_bytes / max(self.raw_bytes, 1)

    def to_dict(self) -> dict:
        return {
            "codec": self.codec,
            "payload_b64": self.payload_b64,
            "channels": list(self.channels),
            "n_samples": self.n_samples,
            "anomaly_score": self.anomaly_score,
            "bypass": self.bypass,
            "raw_bytes": self.raw_bytes,
            "compressed_bytes": self.compressed_bytes,
            "ratio": self.ratio,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompressedWindow:
        try:
            return cls(
                codec=str(d.get("codec", CODEC)),
                payload_b64=str(d["payload_b64"]),
                channels=tuple(d["channels"]),
                n_samples=int(d.get("n_samples", 0)),
                anomaly_score=float(d.get("anomaly_score", 0.0)),
                bypass=bool(d.get("bypass", False)),
                raw_bytes=int(d.get("raw_bytes", 0)),
                compressed_bytes=int(d.get("compressed_bytes", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed CompressedWindow dict: {exc}") from exc


@dataclass(frozen=True)
class RestoredWindow:
    """Result of decoding a CompressedWindow."""

    channels: dict[str, np.ndarray]
    anomaly_score: float
    bypass: bool
    n_samples: int

    def max_abs_error(self, original: dict[str, np.ndarray]) -> dict[str, float]:
        """Per-channel max |restored - original| (useful for lossiness tests)."""
        return {
            ch: float(
                np.max(np.abs(self.channels[ch] - np.asarray(original[ch], dtype=np.float64)))
            )
            for ch in self.channels
        }


def _as_channel_dict(
    window: dict[str, np.ndarray], channels: tuple[str, ...]
) -> dict[str, np.ndarray]:
    missing = [c for c in channels if c not in window]
    if missing:
        raise ValueError(f"window is missing channels: {missing}")
    out = {c: np.asarray(window[c], dtype=np.float64).ravel() for c in channels}
    n = {len(v) for v in out.values()}
    if len(n) != 1:
        raise ValueError("all channels must have equal sample counts")
    if not n.pop():
        raise ValueError("window is empty")
    return out


def compress_window(
    window: dict[str, np.ndarray],
    *,
    cfg: AeroZipConfig | None = None,
    baseline_mean: dict[str, float] | None = None,
    baseline_std: dict[str, float] | None = None,
    anomaly_score_value: float | None = None,
) -> CompressedWindow:
    """Compress one telemetry window with a fresh AeroZip compressor.

    ``baseline_mean``/``baseline_std`` enable the z-score anomaly gate
    (score > ``cfg.anomaly_bypass_threshold`` → lossless raw bypass).
    Without baselines the score is 0 and the lossy path is used unless
    ``anomaly_score_value`` is supplied explicitly.
    """
    cfg = cfg or AeroZipConfig()
    channels = tuple(cfg.channels)
    win = _as_channel_dict(window, channels)

    if anomaly_score_value is None and baseline_mean is not None and baseline_std is not None:
        anomaly_score_value = anomaly_score(win, baseline_mean, baseline_std)

    raw = b"".join(win[c].astype(np.float64).tobytes() for c in channels)
    compressor = AeroZipCompressor(cfg)  # fresh per window → self-contained payload
    blob = compressor.compress(win, anomaly_score_value=anomaly_score_value)
    header = AeroZipCompressor.inspect_header(blob)

    return CompressedWindow(
        codec=CODEC,
        payload_b64=base64.b64encode(blob).decode("ascii"),
        channels=channels,
        n_samples=len(win[channels[0]]),
        anomaly_score=header["anomaly_score"],
        bypass=header["bypass"],
        raw_bytes=len(raw),
        compressed_bytes=len(blob),
    )


def restore_window(payload: CompressedWindow | dict | str) -> RestoredWindow:
    """Decode a window compressed by :func:`compress_window`.

    Accepts a CompressedWindow, its dict form, or a bare base64 payload.
    Reconstruction starts from the same per-channel offset (0) the fresh
    compressor used, so it is deterministic.
    """
    if isinstance(payload, CompressedWindow):
        comp = payload
    elif isinstance(payload, dict):
        comp = CompressedWindow.from_dict(payload)
    else:
        raise ValueError(
            "restore_window needs a CompressedWindow or its dict form "
            "(channel list required); use the API/pipeline helpers instead of raw blobs"
        )
    if comp.codec != CODEC:
        raise ValueError(f"Unsupported codec: {comp.codec!r}")

    blob = base64.b64decode(comp.payload_b64.encode("ascii"), validate=True)
    if not blob.startswith(b"AZ"):
        raise ValueError("Not an AeroZip payload")
    raw = zlib.decompress(blob[2:])
    mode = raw[0]
    bypass = bool(mode == 1)
    score = struct.unpack("<f", raw[1:5])[0]

    pos = 5
    channels: dict[str, np.ndarray] = {}
    cfg = AeroZipConfig(channels=comp.channels)
    for ch in comp.channels:
        (n,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        if bypass:
            vals = np.frombuffer(raw, dtype="<f8", count=n, offset=pos).astype(np.float64).copy()
            pos += 8 * n
            channels[ch] = vals
        else:
            deltas = np.empty(n, dtype=np.int64)
            for i in range(n):
                dv, pos = _decode_varint(raw, pos)
                deltas[i] = dv
            q = cfg.quanta.get(ch, 0.01)
            # Fresh compressor started at offset 0: cumulative transmitted
            # deltas × quantum reconstruct the quantized signal.
            channels[ch] = np.cumsum(deltas).astype(np.float64) * q

    return RestoredWindow(
        channels=channels,
        anomaly_score=float(score),
        bypass=bypass,
        n_samples=len(channels[comp.channels[0]]),
    )


__all__ = [
    "CODEC",
    "DEFAULT_CHANNELS",
    "CompressedWindow",
    "RestoredWindow",
    "compress_window",
    "restore_window",
    "AeroZipConfig",
    "AeroZipCompressor",
    "anomaly_score",
]
