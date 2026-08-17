"""Tests for per-asset detection-limit overrides (src/faults/limits)."""

import pytest

from src.digital_twin.specs import get_spec
from src.faults.detector import FaultDetector
from src.faults.limits import OVERRIDE_KEYS, validate_overrides


def test_validate_overrides_accepts_supported_keys():
    cleaned = validate_overrides({"vibration_limit_mms": 6.0, "temperature_limit_c": 90.0})
    assert cleaned == {"vibration_limit_mms": 6.0, "temperature_limit_c": 90.0}


def test_validate_overrides_rejects_unknown_and_bad_values():
    with pytest.raises(ValueError, match="unknown limit override"):
        validate_overrides({"blade_length_m": 50.0})
    with pytest.raises(ValueError, match="must be numeric"):
        validate_overrides({"vibration_limit_mms": "hot"})
    with pytest.raises(ValueError, match="must be positive"):
        validate_overrides({"vibration_limit_mms": -1.0})


def test_override_keys_cover_detector_limits():
    assert set(OVERRIDE_KEYS) == {
        "vibration_limit_mms",
        "temperature_limit_c",
        "rpm_limit_hss",
        "viscosity_min_cst",
        "viscosity_max_cst",
    }


def test_detector_applies_overrides():
    # NREL-5MW default vibration limit is 5.0; override to 6.0.
    base = FaultDetector(get_spec("NREL-5MW"))
    tuned = FaultDetector(get_spec("NREL-5MW"), overrides={"vibration_limit_mms": 6.0})
    assert base.effective_limits["vibration_limit_mms"] == 5.0
    assert tuned.effective_limits["vibration_limit_mms"] == 6.0
    # 5.5 mm/s is a fault with default limits, healthy with the override.
    assert base.detect({"vibration_mms": 5.5}).n_faults >= 1
    assert tuned.detect({"vibration_mms": 5.5}).n_faults == 0


def test_detector_overrides_combine_with_spec_viscosity():
    tuned = FaultDetector(
        get_spec("GE-1.5"), overrides={"viscosity_min_cst": 8.0, "viscosity_max_cst": 60.0}
    )
    # 9 cSt is a fault for GE-1.5 (min 10) but fine with the override.
    assert tuned.detect({"oil_viscosity_cst": 9.0}).n_faults == 0
    assert tuned.detect({"oil_viscosity_cst": 6.0}).n_faults >= 1


def test_effective_limits_reporting():
    detector = FaultDetector(get_spec("GE-1.5"))
    limits = detector.effective_limits
    assert limits["rpm_limit_hss"] == 1800.0
    assert limits["temperature_limit_c"] == 80.0
