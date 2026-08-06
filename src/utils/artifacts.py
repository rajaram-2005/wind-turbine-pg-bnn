"""Artifact registry for AeroVigil (wind-turbine-pg-bnn).

One canonical way to persist a trained PG-BNN so the API/UI/CLI can serve it
later — previously ``scripts/train_demo.py`` saved a bare state_dict to the
CWD and nothing could load it.

A *bundle* is a pair of files rooted at a checkpoint path:

``artifacts/bnn_demo.pt``
    ``torch.save`` payload with the model state_dict, the model architecture
    (in_features / hidden_sizes / prior_sigma), the fitted robust scaler, the
    feature-window settings, and free-form metadata.

``artifacts/bnn_demo.json``
    Human-readable JSON sidecar mirroring the non-tensor parts of the
    checkpoint (architecture, feature config, metadata, schema version) so an
    operator can inspect a bundle without loading torch.

The default registry directory is ``artifacts/`` (gitignored). All metadata
that crosses this boundary is plain JSON/ASCII and advisory-only; do NOT put
raw config mappings (which contain ``allow_actuation``) into metadata — the
safety contract's key scanner blocks actuation keys by design.

Checkpoint loading uses ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.utils.safety import enforce_safety_contract

CHECKPOINT_FORMAT = "aerovigil-checkpoint/v1"
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Canonical feature stat order used by every training path so far.
DEFAULT_FEATURE_STATS = ("mean", "std", "min", "max", "rms")


def default_artifacts_dir() -> Path:
    """Registry directory (created on demand, gitignored by the repo)."""
    DEFAULT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_ARTIFACTS_DIR


def sidecar_path(checkpoint_path: str | Path) -> Path:
    """The JSON sidecar paired with a checkpoint path."""
    p = Path(checkpoint_path)
    return p.with_suffix(".json")


@dataclass(frozen=True)
class ModelArchitecture:
    """Reconstruction parameters for :class:`BayesianNeuralNetwork`."""

    in_features: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    prior_sigma: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "hidden_sizes": list(self.hidden_sizes),
            "prior_sigma": self.prior_sigma,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelArchitecture:
        return cls(
            in_features=int(d["in_features"]),
            hidden_sizes=tuple(int(h) for h in d.get("hidden_sizes", (64, 64))),
            prior_sigma=float(d.get("prior_sigma", 1.0)),
        )


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-extraction settings a served model expects (window + stats)."""

    window_size: int = 60
    stride: int = 20
    stats: tuple[str, ...] = DEFAULT_FEATURE_STATS
    channels: tuple[str, ...] = field(
        default=("vibration_mms", "temperature_c", "rpm", "oil_viscosity_cst", "load_pct")
    )

    @property
    def feature_dim(self) -> int:
        return len(self.channels) * len(self.stats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "stride": self.stride,
            "stats": list(self.stats),
            "channels": list(self.channels),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FeatureConfig:
        if not d:
            return cls()
        return cls(
            window_size=int(d.get("window_size", 60)),
            stride=int(d.get("stride", 20)),
            stats=tuple(d.get("stats", DEFAULT_FEATURE_STATS)),
            channels=tuple(
                d.get(
                    "channels",
                    ("vibration_mms", "temperature_c", "rpm", "oil_viscosity_cst", "load_pct"),
                )
            ),
        )


@dataclass
class ArtifactBundle:
    """A loaded checkpoint: model + scaler + feature config + metadata."""

    model: Any
    architecture: ModelArchitecture
    features: FeatureConfig
    scaler: dict[str, tuple[float, float]] | None
    metadata: dict[str, Any]
    checkpoint_path: Path
    sidecar_path: Path


def _infer_architecture(model: torch.nn.Module) -> ModelArchitecture:
    from src.models.bnn import BayesianNeuralNetwork

    if not isinstance(model, BayesianNeuralNetwork):
        raise TypeError(
            "artifact registry currently supports BayesianNeuralNetwork checkpoints "
            f"only, got {type(model).__name__}"
        )
    first = model.linears[0]
    return ModelArchitecture(
        in_features=first.in_features,
        hidden_sizes=tuple(layer.out_features for layer in model.linears),
        prior_sigma=first.prior_sigma,
    )


def save_model_bundle(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    scaler: dict[str, tuple[float, float] | list[float]] | None = None,
    features: FeatureConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactBundle:
    """Persist a trained model + scaler + JSON sidecar.

    ``scaler`` is the robust-normalization map produced by
    ``src.data.ingest.robust_normalize`` (``{column: (lo, hi)}``) so serving
    can normalize new telemetry with the TRAINING quantiles, never refit.
    ``metadata`` is free-form advisory text/numbers (screened by the safety
    contract before it is written).
    """
    ckpt = Path(checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    arch = _infer_architecture(model)
    feats = features or FeatureConfig()
    if arch.in_features != feats.feature_dim:
        raise ValueError(
            f"Model expects {arch.in_features} features but feature config "
            f"({feats}) produces {feats.feature_dim}"
        )

    meta = dict(metadata or {})
    meta.setdefault("advisory_only", True)
    meta.setdefault("created_utc", _dt.datetime.now(_dt.timezone.utc).isoformat())
    # Defense in depth: metadata that crosses the artifact boundary must
    # itself satisfy the advisory-only contract (fail closed on bad keys).
    enforce_safety_contract(meta)

    scaler_clean = (
        {str(k): (float(v[0]), float(v[1])) for k, v in scaler.items()} if scaler else None
    )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "architecture": arch.to_dict(),
        "features": feats.to_dict(),
        "scaler": scaler_clean,
        "metadata": meta,
    }
    torch.save(payload, ckpt)

    sidecar = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint": ckpt.name,
        "architecture": arch.to_dict(),
        "features": feats.to_dict(),
        "scaler_present": scaler_clean is not None,
        "metadata": meta,
    }
    enforce_safety_contract(sidecar)
    side = sidecar_path(ckpt)
    side.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return ArtifactBundle(
        model=model,
        architecture=arch,
        features=feats,
        scaler=scaler_clean,
        metadata=meta,
        checkpoint_path=ckpt,
        sidecar_path=side,
    )


def _load_checkpoint_dict(
    ckpt: Path,
    *,
    in_features: int | None,
    hidden_sizes: tuple[int, ...],
    prior_sigma: float,
) -> dict[str, Any]:
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and raw.get("format") == CHECKPOINT_FORMAT:
        return raw
    # Legacy fallback: a bare ``state_dict`` (e.g. a v0.1.0 bnn_demo.pt). The
    # architecture cannot be recovered from the file, so the caller must pin
    # it explicitly.
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        if in_features is None:
            raise ValueError(
                f"{ckpt} is a bare state_dict without bundle metadata; pass "
                "in_features=... (and optionally hidden_sizes=...) to load it."
            )
        return {
            "format": "legacy-state-dict",
            "state_dict": raw,
            "architecture": ModelArchitecture(
                in_features=in_features,
                hidden_sizes=hidden_sizes,
                prior_sigma=prior_sigma,
            ).to_dict(),
            "features": None,
            "scaler": None,
            "metadata": {"advisory_only": True, "legacy_state_dict": True},
        }
    raise ValueError(f"Unrecognized checkpoint format in {ckpt}")


def load_model_bundle(
    checkpoint_path: str | Path,
    *,
    in_features: int | None = None,
    hidden_sizes: tuple[int, ...] = (64, 64),
    prior_sigma: float = 1.0,
    device: str = "cpu",
) -> ArtifactBundle:
    """Load a bundle saved by :func:`save_model_bundle`.

    Architecture comes from the checkpoint itself; feature-dim mismatches
    surface as a clean :class:`ValueError` instead of an opaque torch error.
    The optional JSON sidecar is cross-checked when present.
    """
    from src.models.bnn import BayesianNeuralNetwork

    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    payload = _load_checkpoint_dict(
        ckpt, in_features=in_features, hidden_sizes=hidden_sizes, prior_sigma=prior_sigma
    )
    arch = ModelArchitecture.from_dict(payload["architecture"])
    feats = FeatureConfig.from_dict(payload.get("features"))
    if payload.get("features") and feats.feature_dim != arch.in_features:
        raise ValueError(
            f"feature-dim mismatch: checkpoint declares {feats.feature_dim} features "
            f"but the model expects {arch.in_features}"
        )

    model = BayesianNeuralNetwork(
        in_features=arch.in_features,
        hidden_sizes=arch.hidden_sizes,
        prior_sigma=arch.prior_sigma,
    )
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint architecture mismatch in {ckpt}: {exc}") from exc
    model.to(device).eval()

    scaler_raw = payload.get("scaler")
    scaler = (
        {str(k): (float(v[0]), float(v[1])) for k, v in scaler_raw.items()} if scaler_raw else None
    )
    metadata = dict(payload.get("metadata") or {})

    # Cross-check the sidecar when it exists (cheap tamper/eject sanity).
    side = sidecar_path(ckpt)
    if side.is_file():
        sidecar = json.loads(side.read_text(encoding="utf-8"))
        if sidecar.get("architecture") != arch.to_dict():
            raise ValueError(f"Sidecar {side} disagrees with the checkpoint architecture in {ckpt}")

    return ArtifactBundle(
        model=model,
        architecture=arch,
        features=feats,
        scaler=scaler,
        metadata=metadata,
        checkpoint_path=ckpt,
        sidecar_path=side,
    )


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write any advisory-safe JSON document (e.g. an OnboardingReport)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    enforce_safety_contract(data)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def export_onboarding_bundle(
    adapted_model: torch.nn.Module,
    report: Any,
    out_dir: str | Path,
    *,
    scaler: dict[str, tuple[float, float] | list[float]] | None = None,
    features: FeatureConfig | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export a Hermes/Reptile-adapted model + its OnboardingReport.

    Returns the written paths so callers (scripts, tests, the API later) can
    point the Phase-2 serving path at the promoted model directly.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    asset = report_dict.get("asset_id", "asset")
    ckpt = out / f"hermes_{asset}.pt"
    meta = dict(extra_metadata or {})
    meta.update(
        {
            "produced_by": "hermes-onboarding",
            "onboarding_status": report_dict.get("status", "unknown"),
            "promoted": bool(report_dict.get("promoted", False)),
        }
    )
    bundle = save_model_bundle(
        adapted_model,
        ckpt,
        scaler=scaler,
        features=features,
        metadata=meta,
    )
    report_path = save_json(report_dict, out / f"hermes_{asset}_report.json")
    return {
        "checkpoint": bundle.checkpoint_path,
        "sidecar": bundle.sidecar_path,
        "report": report_path,
    }
