"""Model serving: turn a persisted PG-BNN checkpoint into advisories.

Bridges the artifact registry (:mod:`src.utils.artifacts`), the ingestion
feature pipeline (``robust_normalize`` + ``sliding_features``), and the
model-aware path of :func:`src.models.predictor.run_advisory` — so the
API/UI/CLI can serve a trained model instead of requiring a pre-computed
``bnn_state`` block in every request.

Everything remains advisory-only: output still flows through
``run_advisory`` → ``enforce_safety_contract``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingest import SlidingWindowConfig, robust_normalize, sliding_features
from src.models.predictor import run_advisory
from src.utils.artifacts import ArtifactBundle, FeatureConfig, load_model_bundle
from src.utils.schema import TurbinePayload


def apply_scaler(df: pd.DataFrame, scaler: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Normalize with a FITTED scaler map (never refit quantiles at serve time)."""
    out = pd.DataFrame(index=df.index)
    for col, (lo, hi) in scaler.items():
        rng = hi - lo
        if abs(rng) < 1e-9:
            rng = 1e-9
        out[col] = (df[col].astype(np.float64) - lo) / rng
    return out


def _window_stats(win: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    """One feature vector for a single window, in sliding_features order
    (stats outer, channels inner)."""
    parts = []
    for stat in stats:
        if stat == "mean":
            parts.append(win.mean(axis=0))
        elif stat == "std":
            parts.append(win.std(axis=0))
        elif stat == "min":
            parts.append(win.min(axis=0))
        elif stat == "max":
            parts.append(win.max(axis=0))
        elif stat == "rms":
            parts.append(np.sqrt((win**2).mean(axis=0)))
        else:
            raise ValueError(f"Unknown stat: {stat}")
    return np.concatenate(parts)


@dataclass
class ServingModel:
    """A loaded PG-BNN plus the exact feature pipeline it was trained with."""

    bundle: ArtifactBundle
    device: str = "cpu"

    @property
    def model(self):
        return self.bundle.model

    @property
    def scaler(self):
        return self.bundle.scaler

    @property
    def features_config(self) -> FeatureConfig:
        return self.bundle.features

    @property
    def expected_feature_dim(self) -> int:
        return int(self.bundle.architecture.in_features)

    # ------------------------------------------------------------------ #
    # Features                                                            #
    # ------------------------------------------------------------------ #
    def feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a raw telemetry frame with the TRAINING scaler."""
        missing = [c for c in self.features_config.channels if c not in df.columns]
        if missing:
            raise ValueError(f"Telemetry frame missing channels: {missing}")
        frame = df[list(self.features_config.channels)]
        if self.scaler is not None:
            return apply_scaler(frame, self.scaler)
        # No scaler persisted (e.g. legacy checkpoint): fit robust quantiles
        # on the supplied frame — degenerate but keeps the demo path alive.
        norm, _ = robust_normalize(frame)
        return norm

    def features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute the (n_windows, feature_dim) feature matrix the model expects."""
        cfg = self.features_config
        norm = self.feature_frame(df)
        sw = SlidingWindowConfig(window_size=cfg.window_size, stride=cfg.stride, stats=cfg.stats)
        feats = sliding_features(norm, sw, channels=cfg.channels)
        if feats.shape[0] == 0:
            # Short history: fall back to window stats over the whole frame so
            # a sparse snapshot still yields ONE advisory (documented behavior).
            win = norm.to_numpy(dtype=np.float32)
            if win.shape[0] == 0:
                raise ValueError("telemetry frame is empty")
            feats = _window_stats(win, cfg.stats)[None, :].astype(np.float32)
        if feats.shape[1] != self.expected_feature_dim:
            raise ValueError(
                f"feature-dim mismatch: produced {feats.shape[1]} features but the "
                f"model expects {self.expected_feature_dim}. Rebuild the window "
                "with the same channels/stats used at training time."
            )
        return feats

    def latest_feature_vector(self, df: pd.DataFrame) -> np.ndarray:
        """Feature vector of the most recent window (what an advisory uses)."""
        feats = self.features(df)
        return feats[-1]

    # ------------------------------------------------------------------ #
    # Advisory                                                            #
    # ------------------------------------------------------------------ #
    def advisory(
        self,
        payload: TurbinePayload,
        telemetry_window: pd.DataFrame,
        device: str | None = None,
    ) -> dict:
        """Run a model-based advisory for the payload's asset.

        ``telemetry_window`` is a raw channel frame (columns = the five
        channels); the snapshot in ``payload.telemetry`` still drives the
        physics-violation check, while the model computes RUL/uncertainties
        from the window features.
        """
        fv = self.latest_feature_vector(telemetry_window)
        return run_advisory(
            payload,
            model=self.model,
            feature_vector=fv,
            device=device or self.device,
        )


def load_serving_model(
    path: str | Path,
    *,
    device: str = "cpu",
    **legacy_kwargs,
) -> ServingModel:
    """Load a checkpoint bundle and return its serving wrapper.

    Extra kwargs (``in_features``, ``hidden_sizes``, ...) are forwarded to the
    bundle loader for legacy bare-state_dict checkpoints.
    """
    bundle = load_model_bundle(path, device=device, **legacy_kwargs)
    return ServingModel(bundle=bundle, device=device)
