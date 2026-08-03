"""
AeroZip: lightweight telemetry compression with anomaly bypass.

Strategy per-channel:
  1. Delta coding (store differences from previous sample).
  2. Deadband (no payload if |delta| < threshold, preserving last value).
  3. Quantization to a fixed-point scale (e.g. 0.01 engineering units).
  4. Optional varint encoding of integer deltas.

If any channel's anomaly score exceeds `anomaly_bypass_threshold` (default 0.75)
the compressor bypasses compression for that window and emits raw float64
samples — this guarantees diagnostic fidelity during fault conditions.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AeroZipConfig:
    channels: Sequence[str] = ("vibration_mms", "temperature_c", "rpm", "oil_viscosity_cst", "load_pct")
    deadbands: Dict[str, float] = field(default_factory=lambda: {
        "vibration_mms": 0.02,
        "temperature_c": 0.05,
        "rpm": 1.0,
        "oil_viscosity_cst": 0.1,
        "load_pct": 0.2,
    })
    quanta: Dict[str, float] = field(default_factory=lambda: {
        "vibration_mms": 0.01,
        "temperature_c": 0.01,
        "rpm": 1.0,
        "oil_viscosity_cst": 0.01,
        "load_pct": 0.05,
    })
    anomaly_bypass_threshold: float = 0.75  # scores above this → raw payload


def _encode_varint(value: int) -> bytes:
    """Simple signed LEB128-ish varint."""
    # ZigZag encode sign
    zz = (value << 1) ^ (value >> 63)
    out = bytearray()
    while zz >= 0x80:
        out.append((zz & 0x7F) | 0x80)
        zz >>= 7
    out.append(zz & 0x7F)
    return bytes(out)


def _decode_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    # ZigZag decode
    return ((result >> 1) ^ -(result & 1)), pos


def anomaly_score(
    window: Dict[str, np.ndarray],
    baseline_mean: Dict[str, float],
    baseline_std: Dict[str, float],
) -> float:
    """Simple z-based anomaly score in [0,1] across channels. Higher = more anomalous."""
    z_max = 0.0
    for ch, vals in window.items():
        mu = baseline_mean.get(ch, 0.0)
        sd = max(baseline_std.get(ch, 1.0), 1e-6)
        z = float(np.max(np.abs(vals - mu) / sd))
        z_max = max(z_max, z)
    # Map z to [0,1] with a soft sigmoid at ~z=3
    return float(1.0 / (1.0 + np.exp(-(z_max - 3.0) / 0.5)))


class AeroZipCompressor:
    def __init__(self, cfg: Optional[AeroZipConfig] = None):
        self.cfg = cfg or AeroZipConfig()
        # Per-channel running previous quantized value
        self._prev_q: Dict[str, int] = {c: 0 for c in self.cfg.channels}

    def compress(
        self,
        window: Dict[str, np.ndarray],
        anomaly_score_value: Optional[float] = None,
        baseline_mean: Optional[Dict[str, float]] = None,
        baseline_std: Optional[Dict[str, float]] = None,
    ) -> bytes:
        # Auto-score if not provided
        if anomaly_score_value is None:
            if baseline_mean is not None and baseline_std is not None:
                anomaly_score_value = anomaly_score(window, baseline_mean, baseline_std)
            else:
                anomaly_score_value = 0.0

        if anomaly_score_value > self.cfg.anomaly_bypass_threshold:
            # Anomaly bypass: write raw little-endian float64 arrays with a flag byte
            payload = b"\x01" + struct.pack("<f", anomaly_score_value)
            for ch in self.cfg.channels:
                vals = window[ch].astype(np.float64)
                payload += struct.pack("<I", len(vals)) + vals.tobytes(order="C")
            return b"AZ" + zlib.compress(payload, level=1)

        # Normal mode: delta + quantize + varint
        parts: List[bytes] = [b"\x00", struct.pack("<f", anomaly_score_value)]
        for ch in self.cfg.channels:
            q = self.cfg.quanta[ch]
            d = self.cfg.deadbands[ch]
            vals = window[ch].astype(np.float64)
            quantized = np.round(vals / q).astype(np.int64)
            deltas = np.diff(np.concatenate(([self._prev_q[ch]], quantized)))
            # Deadband: collapse deltas whose engineering-equivalent is below threshold
            delta_eng = deltas.astype(np.float64) * q
            deltas = np.where(np.abs(delta_eng) < d, 0, deltas).astype(np.int64)
            ch_bytes = bytearray()
            ch_bytes.extend(struct.pack("<I", len(deltas)))
            for dv in deltas:
                ch_bytes.extend(_encode_varint(int(dv)))
            parts.append(bytes(ch_bytes))
            # Update running previous to last transmitted value (not just last seen),
            # so deadband doesn't cause drift.
            if len(quantized):
                # Transmitted value accumulates non-zero deltas
                tx_sum = int(self._prev_q[ch]) + int(deltas.sum())
                self._prev_q[ch] = tx_sum

        return b"AZ" + zlib.compress(b"".join(parts), level=6)

    def compress_stats(self, window: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Helper: compare compressed vs raw size for reporting."""
        raw = b"".join(window[c].astype(np.float64).tobytes() for c in self.cfg.channels)
        comp = self.compress(window, anomaly_score_value=0.0)
        return {
            "raw_bytes": len(raw),
            "compressed_bytes": len(comp),
            "ratio": len(comp) / max(len(raw), 1),
            "bypass": False,
        }

    @staticmethod
    def inspect_header(blob: bytes) -> Dict[str, float]:
        """Quickly inspect whether a payload was anomaly-bypassed and its score."""
        if not blob.startswith(b"AZ"):
            raise ValueError("Not an AeroZip payload")
        raw = zlib.decompress(blob[2:])
        mode = raw[0]
        score = struct.unpack("<f", raw[1:5])[0]
        return {"bypass": bool(mode == 1), "anomaly_score": float(score)}
