"""Tests for the 45-day early-warning metrics and the expanded specs library."""

import numpy as np
import pandas as pd

from src.eval.calibration import (
    early_warning_metrics,
    first_warning_lead_time_days,
)
from src.digital_twin.specs import SPECS_LIBRARY, get_spec, list_specs


# --------------------------------------------------------------------------- #
# early_warning_metrics                                                       #
# --------------------------------------------------------------------------- #
def test_early_warning_metrics_perfect_classification():
    y_true = [300.0, 200.0, 100.0, 10.0, 30.0]   # last two fail within 45 d
    y_pred = [280.0, 180.0, 120.0, 20.0, 40.0]
    m = early_warning_metrics(y_true, y_pred, warning_horizon_days=45.0)
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["n_true_positive"] == 2
    assert m["n_true_negative"] == 3
    assert m["n_false_positive"] == 0
    assert m["n_false_negative"] == 0


def test_early_warning_metrics_counts_confusion():
    # TP: true<45 & pred<45 ; TN: true>=45 & pred>=45 ; FP: true>=45 & pred<45
    y_true = [10.0, 30.0, 100.0, 200.0, 300.0]
    y_pred = [20.0, 50.0, 40.0, 250.0, 320.0]  # 50 misses a risk; 40 false alarm
    m = early_warning_metrics(y_true, y_pred, warning_horizon_days=45.0)
    assert m["n_true_positive"] == 1
    assert m["n_true_negative"] == 2
    assert m["n_false_positive"] == 1
    assert m["n_false_negative"] == 1
    assert m["accuracy"] == 3 / 5
    assert m["mean_lead_time_days"] == 10.0  # only TP: warned 10 days before failure


def test_early_warning_metrics_rejects_shape_mismatch():
    import pytest

    with pytest.raises(ValueError):
        early_warning_metrics([1.0, 2.0], [1.0])


def test_early_warning_metrics_horizon_constant():
    # The 45-day horizon is a module-level constant used across the codebase.
    from src.eval.calibration import EARLY_WARNING_HORIZON_DAYS

    assert EARLY_WARNING_HORIZON_DAYS == 45.0


# --------------------------------------------------------------------------- #
# first_warning_lead_time_days                                                #
# --------------------------------------------------------------------------- #
def test_first_warning_lead_time_finds_earliest_warning():
    rul = [120.0, 90.0, 60.0, 30.0, 10.0]
    warned = [False, False, True, True, True]
    # First warning fires while 60 days of life remain.
    assert first_warning_lead_time_days(rul, warned) == 60.0


def test_first_warning_lead_time_none_if_never_warned():
    rul = [120.0, 90.0, 60.0]
    warned = [False, False, False]
    assert first_warning_lead_time_days(rul, warned) is None


# --------------------------------------------------------------------------- #
# Expanded specs library                                                      #
# --------------------------------------------------------------------------- #
def test_specs_library_has_eight_models():
    assert set(SPECS_LIBRARY) == {
        "GE-1.5",
        "Vestas-V90",
        "Siemens-SWT-2.3",
        "Suzlon-S97",
        "Gamesa-G114",
        "Nordex-N100",
        "Senvion-MM92",
        "NREL-5MW",
    }


def test_specs_library_new_models_are_valid():
    for key in ("Suzlon-S97", "Gamesa-G114", "Nordex-N100", "Senvion-MM92"):
        spec = get_spec(key)
        assert spec.rated_power_mw >= 1.0
        assert spec.rotor_diameter_m >= 50.0
        assert spec.gearbox_ratio >= 1.0
        assert spec.vibration_limit_mms > 0.0
        assert spec.bearing_dynamic_load_c_kn > 0.0


def test_list_specs_includes_new_models():
    summary = list_specs()
    assert "Suzlon-S97" in summary
    assert summary["Suzlon-S97"]["manufacturer"] == "Suzlon Energy"


# --------------------------------------------------------------------------- #
# Fleet CSV                                                                    #
# --------------------------------------------------------------------------- #
def test_fleet_csv_has_twenty_assets():
    df = pd.read_csv("examples/fleet.csv")
    assert len(df) == 20
    assert df["asset_id"].nunique() == 20
    assert (df["predicted_rul_days"] > 0).all()
    # Mix of healthy, at-risk, and critical assets is present.
    assert (df["predicted_rul_days"] >= 45).sum() >= 10
    assert (df["predicted_rul_days"] < 45).sum() >= 5
    assert (df["predicted_rul_days"] < 15).sum() >= 3


def test_fleet_csv_telemetry_consistent_with_rul():
    """High vibration/temperature/low viscosity should coincide with low RUL."""
    df = pd.read_csv("examples/fleet.csv")
    at_risk = df[df["predicted_rul_days"] < 45]
    healthy = df[df["predicted_rul_days"] >= 100]
    assert at_risk["vibration_mms"].mean() > healthy["vibration_mms"].mean()
    assert at_risk["temperature_c"].mean() > healthy["temperature_c"].mean()
    assert at_risk["oil_viscosity_cst"].mean() < healthy["oil_viscosity_cst"].mean()


def test_early_warning_metrics_numpy_inputs():
    y_true = np.array([10.0, 200.0])
    y_pred = np.array([20.0, 180.0])
    m = early_warning_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
