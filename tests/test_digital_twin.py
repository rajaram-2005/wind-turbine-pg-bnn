"""Tests for the Wind Turbine Digital Twin module."""

import pytest

from src.digital_twin.prompts import generate_engineering_prompt
from src.digital_twin.specs import get_spec, list_specs
from src.digital_twin.twin import WindTurbineDigitalTwin
from src.utils.schema import BNNState, Telemetry


def test_specs_library():
    specs = list_specs()
    assert "GE-1.5" in specs
    assert "Vestas-V90" in specs
    assert "NREL-5MW" in specs

    spec = get_spec("GE-1.5")
    assert spec.manufacturer == "GE Renewable Energy"
    assert spec.rated_power_mw == 1.5

    # Loose match
    spec_loose = get_spec("GE 1.5 SLE")
    assert spec_loose.model_name == "GE 1.5 SLE"

    with pytest.raises(KeyError):
        get_spec("Non-Existent-Turbine")


def test_digital_twin_operations():
    spec = get_spec("GE-1.5")
    twin = WindTurbineDigitalTwin(asset_id="WTG-TEST-01", spec=spec)

    assert twin.asset_id == "WTG-TEST-01"
    assert twin.cumulative_wear == 0.0

    # Ingest good telemetry
    tel = Telemetry(
        vibration_mms=1.2,
        temperature_c=50.0,
        rpm=1500.0,
        oil_viscosity_cst=32.0,
        load_pct=70.0,
    )
    bnn = BNNState(
        predicted_rul_days=120.0,
        epistemic_uncertainty=0.03,
        aleatoric_uncertainty=0.08,
    )

    rec = twin.update_state(tel, bnn)
    assert rec["cumulative_wear"] > 0.0
    assert len(rec["physics_violations"]) == 0
    assert rec["bearing_l10_hours"] > 0.0

    # Ingest bad telemetry (causing violations)
    tel_bad = Telemetry(
        vibration_mms=6.0,  # exceeds 4.5 limit
        temperature_c=95.0,  # exceeds 80.0 limit
        rpm=1900.0,  # exceeds 1800 limit
        oil_viscosity_cst=5.0,  # below 10 limit
        load_pct=110.0,
    )
    rec_bad = twin.update_state(tel_bad, bnn)
    assert len(rec_bad["physics_violations"]) >= 3
    # Wear should have increased faster
    assert rec_bad["cumulative_wear"] > rec["cumulative_wear"]


def test_digital_twin_simulation():
    spec = get_spec("Vestas-V90")
    twin = WindTurbineDigitalTwin(asset_id="WTG-TEST-02", spec=spec)

    # Simulate overload scenario for 5 hours
    records = twin.simulate_scenario(profile="overload", hours=5)
    assert len(records) == 5
    assert twin.cumulative_wear > 0.0
    assert records[-1]["telemetry"]["load_pct"] > 100.0


def test_engineering_prompt_generation():
    spec = get_spec("NREL-5MW")
    twin = WindTurbineDigitalTwin(asset_id="WTG-TEST-03", spec=spec)

    # Check prompt text with no history
    prompt_empty = generate_engineering_prompt(twin)
    assert "No telemetry ingested yet." in prompt_empty

    # Ingest state and check prompt content
    tel = Telemetry(
        vibration_mms=2.0,
        temperature_c=60.0,
        rpm=1100.0,
        oil_viscosity_cst=35.0,
        load_pct=80.0,
    )
    twin.update_state(tel)
    prompt = generate_engineering_prompt(twin)
    assert "Asset ID: WTG-TEST-03" in prompt
    assert "NREL 5MW Reference Turbine" in prompt
    assert "DECISION-SUPPORT ONLY" in prompt
