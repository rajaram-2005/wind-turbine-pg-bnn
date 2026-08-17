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
    assert rec["agent_team"]["team_id"] == "CYBER_PRIME_DUAL_AGENT"
    assert rec["agent_team"]["agents"]["mika"]["name"] == "MIKA"
    assert rec["agent_team"]["agents"]["kai"]["name"] == "KAI"

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
    assert "active constraint" in rec_bad["agent_team"]["agents"]["kai"]["finding"]
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


def test_simulation_is_deterministic():
    """The same asset + profile + duration must reproduce the same trajectory,
    regardless of the interpreter's hash seed (PYTHONHASHSEED)."""
    spec = get_spec("GE-1.5")
    first = WindTurbineDigitalTwin("WTG-DET", spec).simulate_scenario("overload", 6)
    second = WindTurbineDigitalTwin("WTG-DET", spec).simulate_scenario("overload", 6)

    def normalized(records):
        out = []
        for rec in records:
            rec = dict(rec)
            rec.pop("timestamp", None)  # wall-clock at twin construction
            if rec.get("advisory"):
                rec["advisory"] = {k: v for k, v in rec["advisory"].items() if k != "generated_at"}
            if rec.get("fault_report"):
                # The embedded fault report carries the same wall-clock stamp.
                rec["fault_report"] = {
                    k: v for k, v in rec["fault_report"].items() if k != "timestamp"
                }
            out.append(rec)
        return out

    assert normalized(first) == normalized(second)


def test_simulate_rejects_invalid_hours():
    spec = get_spec("GE-1.5")
    twin = WindTurbineDigitalTwin("WTG-HRS", spec)
    for bad in (0, -5, float("nan"), float("inf"), 1e9):
        with pytest.raises(ValueError):
            twin.simulate_scenario(hours=bad)
    # Fractional durations round up to whole hourly steps.
    assert len(twin.simulate_scenario(hours=2.5)) == 3


def test_state_history_is_bounded():
    spec = get_spec("GE-1.5")
    twin = WindTurbineDigitalTwin("WTG-HIST", spec, max_history=3)
    for i in range(10):
        twin.update_state(
            Telemetry(
                vibration_mms=1.0 + 0.01 * i,
                temperature_c=50.0,
                rpm=1500.0,
                oil_viscosity_cst=32.0,
                load_pct=70.0,
            )
        )
    assert len(twin.state_history) == 3
    # Oldest retained record is the 8th ingest (index 7), not the first.
    assert twin.state_history[0]["telemetry"]["vibration_mms"] == pytest.approx(1.07)


def test_max_history_rejects_invalid_values():
    with pytest.raises(ValueError):
        WindTurbineDigitalTwin("WTG-BADHIST", get_spec("GE-1.5"), max_history=0)


def test_update_state_rejects_non_finite_telemetry():
    twin = WindTurbineDigitalTwin("WTG-NAN", get_spec("GE-1.5"))
    # model_construct bypasses pydantic bounds to prove the runtime guard.
    nan_tel = Telemetry.model_construct(
        vibration_mms=float("nan"),
        temperature_c=50.0,
        rpm=1500.0,
        oil_viscosity_cst=32.0,
        load_pct=70.0,
    )
    with pytest.raises(ValueError, match="non-finite telemetry value"):
        twin.update_state(nan_tel)


def test_update_state_rejects_non_finite_bnn_state():
    twin = WindTurbineDigitalTwin("WTG-NANBNN", get_spec("GE-1.5"))
    nan_bnn = BNNState.model_construct(
        predicted_rul_days=float("inf"),
        epistemic_uncertainty=0.05,
        aleatoric_uncertainty=0.1,
    )
    with pytest.raises(ValueError, match="non-finite bnn_state value"):
        twin.update_state(
            Telemetry(
                vibration_mms=2.0,
                temperature_c=60.0,
                rpm=1400.0,
                oil_viscosity_cst=30.0,
                load_pct=70.0,
            ),
            nan_bnn,
        )


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
    assert "CYBER PRIME DUAL-AGENT ASSESSMENT" in prompt
    assert "MIKA / Maintenance Strategist" in prompt
    assert "KAI / Physics Constraint Sentinel" in prompt
    assert "DECISION-SUPPORT ONLY" in prompt
