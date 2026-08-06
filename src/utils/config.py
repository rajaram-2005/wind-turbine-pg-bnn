"""Application configuration backbone for AeroVigil (wind-turbine-pg-bnn).

Loads ``configs/default.yaml`` into a typed :class:`AppConfig` model so the
physics limits, BNN hyperparameters, telemetry windowing, meta/Hermes knobs,
the 45-day early-warning horizon, and the UI defaults live in ONE place
instead of being hardcoded across modules.

The loader is part of the safety perimeter: any configuration that is not
strictly advisory-only (``safety.mode != "advisory_only"`` or
``safety.allow_actuation == true``) is REJECTED at load time. The advisory
service refuses to start from a non-advisory configuration — fail closed.

Note: the safety contract's key scanner (``src.utils.safety``) blocks any key
matching ``actuat(e|ion)`` at the *payload* boundary, so never stuff the raw
config mapping into an advisory payload or artifact sidecar — the
``allow_actuation`` key would trip the fail-closed gate by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Repo root: src/utils/config.py -> src/utils -> src -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# Physics                                                                      #
# --------------------------------------------------------------------------- #
class GearboxPhysicsConfig(BaseModel):
    """Gearbox hard limits (research-grade defaults, mirror configs/default.yaml)."""

    model_config = ConfigDict(extra="forbid")

    vibration_limit_mms: float = 4.5
    temperature_limit_c: float = 80.0
    rpm_limit_hss: float = 1800.0
    viscosity_min_cst: float = 10.0
    viscosity_max_cst: float = 50.0


class GeneratorPhysicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_limit_c: float = 120.0
    rpm_limit: float = 1800.0


class PhysicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gearbox: GearboxPhysicsConfig = Field(default_factory=GearboxPhysicsConfig)
    generator: GeneratorPhysicsConfig = Field(default_factory=GeneratorPhysicsConfig)


# --------------------------------------------------------------------------- #
# BNN                                                                          #
# --------------------------------------------------------------------------- #
class TrainParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 1.0e-3
    num_epochs: int = 300
    num_samples: int = 10
    kl_weight: float = 1.0e-3
    physics_weight: float = 0.2
    batch_size: int = 256


class BnnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden: list[int] = Field(default_factory=lambda: [64, 64])
    prior_sigma: float = 1.0
    train: TrainParamsConfig = Field(default_factory=TrainParamsConfig)
    predict_mc_samples: int = 64


# --------------------------------------------------------------------------- #
# Telemetry                                                                    #
# --------------------------------------------------------------------------- #
class TelemetryConfig(BaseModel):
    """Telemetry windowing (seconds); sample counts derived from the interval."""

    model_config = ConfigDict(extra="forbid")

    window_s: int = 600
    window_stride_s: int = 200
    sample_interval_s: int = 10

    @property
    def window_size_samples(self) -> int:
        return max(1, int(self.window_s // self.sample_interval_s))

    @property
    def stride_samples(self) -> int:
        return max(1, int(self.window_stride_s // self.sample_interval_s))


# --------------------------------------------------------------------------- #
# Meta-learning / Hermes                                                       #
# --------------------------------------------------------------------------- #
class ReptileParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inner_lr: float = 5.0e-3
    inner_steps: int = 5
    meta_lr: float = 0.4
    tasks_per_iter: int = 4
    meta_iterations: int = 25
    num_samples: int = 3
    kl_weight: float = 1.0e-3
    eval_mc_samples: int = 16
    seed: int = 0


class MetaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reptile: ReptileParamsConfig = Field(default_factory=ReptileParamsConfig)


class HermesParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_tau_days: float = 40.0
    max_rounds: int = 4
    max_pseudo_per_round: int = 32
    promotion_max_rmse_days: float = 120.0
    promotion_min_accuracy: float = 0.8
    min_eval_shots: int = 4
    eval_mc_samples: int = 16
    seed: int = 0


# --------------------------------------------------------------------------- #
# Eval / UI / Serving / Safety                                                 #
# --------------------------------------------------------------------------- #
class EvalConfig(BaseModel):
    """Evaluation defaults. The early-warning horizon is the headline system

    guarantee: advisories must fire at least this many days before the
    predicted failure (45 days by default).
    """

    model_config = ConfigDict(extra="forbid")

    early_warning_horizon_days: float = 45.0


class UiSnapshotDefaults(BaseModel):
    """Default single-asset snapshot shown by the Streamlit UI (documentation value)."""

    model_config = ConfigDict(extra="forbid")

    vibration_mms: float = 2.5
    temperature_c: float = 62.0
    rpm: float = 1500.0
    oil_viscosity_cst: float = 32.0
    load_pct: float = 80.0


class UiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_snapshot: UiSnapshotDefaults = Field(default_factory=UiSnapshotDefaults)


class ServingConfig(BaseModel):
    """Optional model-serving defaults. ``model_path`` is normally NOT set in

    the YAML (deployment-specific); the API env var ``AV_MODEL_PATH`` takes
    precedence and is the documented deployment knob.
    """

    model_config = ConfigDict(extra="forbid")

    model_path: str | None = None


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "advisory_only"
    allow_actuation: bool = False


# --------------------------------------------------------------------------- #
# Root config                                                                  #
# --------------------------------------------------------------------------- #
class AppConfig(BaseModel):
    """Typed mirror of ``configs/default.yaml`` (plus eval/ui conveniences)."""

    model_config = ConfigDict(extra="forbid")

    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    bnn: BnnConfig = Field(default_factory=BnnConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    meta: MetaConfig = Field(default_factory=MetaConfig)
    hermes: HermesParamsConfig = Field(default_factory=HermesParamsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    @model_validator(mode="after")
    def _enforce_advisory_only(self) -> AppConfig:
        """FAIL CLOSED: reject any configuration that is not advisory-only.

        This is a hard safety boundary, not a warning: a deployment config
        that asks for actuation support must never boot the service.
        """
        if self.safety.mode != "advisory_only":
            raise ValueError(
                f"safety.mode must be 'advisory_only', got {self.safety.mode!r}. "
                "AeroVigil is decision-support only; non-advisory configurations "
                "are rejected at load time."
            )
        if self.safety.allow_actuation:
            raise ValueError(
                "safety.allow_actuation must be false. AeroVigil issues no "
                "actuation commands; refusing to load this configuration."
            )
        return self


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the application configuration.

    ``path=None`` resolves the repository default (``configs/default.yaml``)
    relative to this file, so loading is independent of the caller's working
    directory. An explicit ``path`` to a missing file raises
    :class:`FileNotFoundError`; a non-advisory ``safety:`` block raises
    :class:`ValueError` (fail closed).
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return AppConfig(**data)


# --------------------------------------------------------------------------- #
# Conversion helpers (build domain structures from the config without         #
# changing any existing defaults — values are identical to the YAML).         #
# --------------------------------------------------------------------------- #
def gearbox_constraints_from_config(cfg: AppConfig):
    """Build :class:`GearboxPhysicsConstraints` from config physics limits."""
    from src.physics.constraints import GearboxPhysicsConstraints

    return GearboxPhysicsConstraints(**cfg.physics.gearbox.model_dump())


def generator_constraints_from_config(cfg: AppConfig):
    """Build :class:`GeneratorPhysicsConstraints` from config physics limits."""
    from src.physics.constraints import GeneratorPhysicsConstraints

    return GeneratorPhysicsConstraints(**cfg.physics.generator.model_dump())


def sliding_window_config_from_config(cfg: AppConfig):
    """Build :class:`SlidingWindowConfig` from the telemetry window settings."""
    from src.data.ingest import SlidingWindowConfig

    return SlidingWindowConfig(
        window_size=cfg.telemetry.window_size_samples,
        stride=cfg.telemetry.stride_samples,
    )


def reptile_config_from_config(cfg: AppConfig):
    """Build :class:`ReptileConfig` from the meta.reptile section."""
    from src.meta.reptile import ReptileConfig

    return ReptileConfig(**cfg.meta.reptile.model_dump())


def hermes_config_from_config(cfg: AppConfig):
    """Build :class:`HermesConfig` from the hermes section (reusing reptile adaptation params)."""
    from src.agents.hermes import HermesConfig

    params = cfg.hermes.model_dump()
    params["adaptation"] = reptile_config_from_config(cfg)
    return HermesConfig(**params)


def train_config_from_config(cfg: AppConfig):
    """Build :class:`TrainConfig` from the bnn.train section."""
    from src.models.bnn import TrainConfig

    return TrainConfig(**cfg.bnn.train.model_dump())
