"""Tests for the config backbone (src.utils.config) and its threading into
physics limits, the 45-day eval horizon, and UI defaults."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from src.data.ingest import SlidingWindowConfig
from src.eval.calibration import EARLY_WARNING_HORIZON_DAYS
from src.physics.constraints import (
    GearboxPhysicsConstraints,
    GeneratorPhysicsConstraints,
    default_gearbox_constraints,
    default_generator_constraints,
)
from src.ui.defaults import default_snapshot
from src.utils.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    gearbox_constraints_from_config,
    hermes_config_from_config,
    load_config,
    reptile_config_from_config,
    sliding_window_config_from_config,
    train_config_from_config,
)


def test_default_config_loads_and_matches_yaml():
    cfg = load_config()
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert (
        cfg.physics.gearbox.vibration_limit_mms == raw["physics"]["gearbox"]["vibration_limit_mms"]
    )
    assert (
        cfg.physics.generator.temperature_limit_c
        == raw["physics"]["generator"]["temperature_limit_c"]
    )
    assert cfg.bnn.hidden == raw["bnn"]["hidden"]
    assert cfg.bnn.train.num_epochs == raw["bnn"]["train"]["num_epochs"]
    assert cfg.telemetry.window_s == raw["telemetry"]["window_s"]
    assert cfg.meta.reptile.meta_lr == raw["meta"]["reptile"]["meta_lr"]
    assert cfg.hermes.confidence_tau_days == raw["hermes"]["confidence_tau_days"]
    assert cfg.safety.mode == "advisory_only"
    assert cfg.safety.allow_actuation is False


def test_config_round_trip_explicit_path(tmp_path):
    cfg = load_config()
    dumped = yaml.safe_dump(cfg.model_dump())
    p = tmp_path / "copy.yaml"
    p.write_text(dumped, encoding="utf-8")
    reloaded = load_config(p)
    assert reloaded.model_dump() == cfg.model_dump()


def test_appconfig_defaults_construct_without_file():
    """AppConfig() with all defaults is valid and advisory-only."""
    cfg = AppConfig()
    assert cfg.safety.mode == "advisory_only"
    assert cfg.eval.early_warning_horizon_days == 45.0


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


@pytest.mark.parametrize(
    "safety_block",
    [
        {"mode": "full_control", "allow_actuation": False},
        {"mode": "advisory_only", "allow_actuation": True},
        {"mode": "supervised_actuation", "allow_actuation": True},
    ],
)
def test_non_advisory_configs_rejected(tmp_path, safety_block):
    p = tmp_path / "unsafe.yaml"
    p.write_text(yaml.safe_dump({"safety": safety_block}), encoding="utf-8")
    with pytest.raises(ValueError, match="advisory"):
        load_config(p)


def test_config_file_rejection_is_fail_closed(tmp_path):
    """A real YAML file with actuation enabled must never load."""
    p = tmp_path / "actuation.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            safety:
              mode: advisory_only
              allow_actuation: true
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(p)


# --------------------------------------------------------------------------- #
# Threading into physics / eval / UI (values unchanged vs previous hardcodes) #
# --------------------------------------------------------------------------- #
def test_physics_defaults_threaded_from_config():
    gb = default_gearbox_constraints()
    assert isinstance(gb, GearboxPhysicsConstraints)
    assert gb.vibration_limit_mms == 4.5
    assert gb.temperature_limit_c == 80.0
    assert gb.rpm_limit_hss == 1800.0
    assert (gb.viscosity_min_cst, gb.viscosity_max_cst) == (10.0, 50.0)

    gen = default_generator_constraints()
    assert isinstance(gen, GeneratorPhysicsConstraints)
    assert gen.temperature_limit_c == 120.0
    assert gen.rpm_limit == 1800.0

    via_helper = gearbox_constraints_from_config(load_config())
    assert via_helper == gb


def test_check_violations_behavior_unchanged():
    """Config-threaded defaults produce the same violations as before."""
    from src.physics.constraints import check_violations

    tel = {
        "vibration_mms": 4.8,  # > 4.5
        "temperature_c": 82.0,  # > 80
        "rpm": 1780.0,
        "oil_viscosity_cst": 12.0,
        "load_pct": 95.0,
    }
    violations = check_violations(tel)
    assert len(violations) == 2
    assert any("Vibration" in v for v in violations)
    assert any("Temperature" in v for v in violations)


def test_early_warning_horizon_threaded_from_config():
    cfg = load_config()
    assert cfg.eval.early_warning_horizon_days == 45.0
    assert EARLY_WARNING_HORIZON_DAYS == 45.0


def test_ui_defaults_threaded_from_config():
    snap = default_snapshot()
    assert snap == {
        "vibration_mms": 2.5,
        "temperature_c": 62.0,
        "rpm": 1500.0,
        "oil_viscosity_cst": 32.0,
        "load_pct": 80.0,
    }


# --------------------------------------------------------------------------- #
# Conversion helpers                                                          #
# --------------------------------------------------------------------------- #
def test_conversion_helpers_build_domain_objects():
    cfg = load_config()

    sw = sliding_window_config_from_config(cfg)
    assert isinstance(sw, SlidingWindowConfig)
    # 600 s window / 10 s interval = 60 samples; 200 s stride = 20 samples.
    assert (sw.window_size, sw.stride) == (60, 20)

    tcfg = train_config_from_config(cfg)
    assert tcfg.num_epochs == 300
    assert tcfg.kl_weight == pytest.approx(1.0e-3)

    rcfg = reptile_config_from_config(cfg)
    assert rcfg.meta_iterations == 25
    assert rcfg.meta_lr == pytest.approx(0.4)

    hcfg = hermes_config_from_config(cfg)
    assert hcfg.max_rounds == 4
    assert hcfg.confidence_tau_days == pytest.approx(40.0)
    assert hcfg.adaptation.meta_iterations == 25
