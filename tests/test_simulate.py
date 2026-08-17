"""Tests for the scenario simulator (src/faults/simulate.py)."""

import pytest

from src.faults.detector import FaultDetector
from src.faults.simulate import SCENARIOS, scenario_descriptions, simulate_telemetry


def test_healthy_scenario_detects_nothing():
    telemetry = simulate_telemetry("healthy")
    assert telemetry["vibration_mms"] == 2.1
    report = FaultDetector(get_spec()).detect(telemetry, asset_id="WTG-D")
    assert report.overall_status == "OK"


def get_spec(_key="GE-1.5"):
    from src.digital_twin.specs import get_spec as _gs

    return _gs(_key)


def test_faulty_scenario_triggers_oil_and_drivetrain_faults():
    telemetry = simulate_telemetry("faulty")
    report = FaultDetector(get_spec("NREL-5MW")).detect(telemetry, asset_id="WTG-D")
    ids = {f.fault_id for f in report.faults}
    assert "GB-02" in ids  # oil viscosity too low
    assert "GB-04" in ids  # water contamination
    assert report.oil.overall_status == "ALARM"


def test_critical_scenario_pages_fire_alerts():
    telemetry = simulate_telemetry("critical")
    report = FaultDetector(get_spec("NREL-5MW")).detect(telemetry, asset_id="WTG-D")
    ids = {f.fault_id for f in report.faults}
    assert "RB-07" in ids  # blade fire
    assert "GB-15" in ids  # gearbox oil fire
    assert "BR-02" in ids  # overspeed
    assert report.overall_status == "CRITICAL"


def test_random_scenario_is_deterministic_and_bounded():
    first = simulate_telemetry("random", seed=7)
    second = simulate_telemetry("random", seed=7)
    assert first == second
    other = simulate_telemetry("random", seed=8)
    assert first != other
    for key, (lo, hi) in {
        "vibration_mms": (1.0, 8.0),
        "temperature_c": (45.0, 85.0),
        "oil_viscosity_cst": (5.0, 45.0),
    }.items():
        assert lo <= first[key] <= hi


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        simulate_telemetry("meltdown")
    assert set(SCENARIOS) == {"healthy", "faulty", "critical", "random"}
    assert len(scenario_descriptions()) == 4
