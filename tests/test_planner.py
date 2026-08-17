"""Tests for the 30-day maintenance planner (src/maintenance/planner.py)."""

from datetime import datetime, timezone

import pytest

from src.digital_twin.specs import get_spec
from src.digital_twin.twin import WindTurbineDigitalTwin
from src.maintenance import build_plan, summarize_plan
from src.utils.schema import Telemetry

HEALTHY = {
    "vibration_mms": 2.1,
    "temperature_c": 58.0,
    "rpm": 1400.0,
    "oil_viscosity_cst": 32.0,
    "load_pct": 70.0,
}
FAULTY = {
    **HEALTHY,
    "oil_viscosity_cst": 5.0,
    "oil_water_ppm": 1500.0,
    "vibration_mms": 9.0,
    "yaw_error_deg": 25.0,
}


def _twin(asset_id: str, telemetry: dict, rul: float | None = None, farm: str = ""):
    twin = WindTurbineDigitalTwin(asset_id, get_spec("NREL-5MW"))
    twin.farm = farm
    rec = twin.update_state(
        Telemetry(
            vibration_mms=telemetry["vibration_mms"],
            temperature_c=telemetry["temperature_c"],
            rpm=telemetry["rpm"],
            oil_viscosity_cst=telemetry["oil_viscosity_cst"],
            load_pct=telemetry["load_pct"],
        )
    )
    if rul is not None:
        rec["advisory"] = {"predicted_rul_days": rul}
        twin.state_history[-1] = rec
    return twin


def test_plan_schedules_workorders_into_weeks():
    twins = {
        "WTG-1": _twin("WTG-1", FAULTY, farm="Alpha"),
        "WTG-2": _twin("WTG-2", HEALTHY, farm="Beta"),
    }
    now = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    plan = build_plan(twins, days=30, now=now)
    assert plan["n_assets_planned"] == 2
    assert plan["n_tasks_planned"] >= 2
    assert plan["calendar"], "plan must contain at least one week"
    assert plan["total_hours"] > 0
    assert plan["energy_at_risk_mwh"] > 0
    first_week = plan["calendar"][0]
    assert first_week["n_tasks"] >= 1
    assert any(t["asset_id"] == "WTG-1" for t in first_week["tasks"])


def test_plan_uses_priorities_and_farm():
    twins = {"WTG-1": _twin("WTG-1", FAULTY, farm="Alpha")}
    plan = build_plan(twins, days=14)
    tasks = [t for week in plan["calendar"] for t in week["tasks"]]
    assert tasks[0]["farm"] == "Alpha"
    assert tasks[0]["priority"].startswith("P")
    assert any("GB-02" in t["faults"] for t in tasks)


def test_plan_adds_rul_inspection_tasks():
    twins = {
        "WTG-R": _twin("WTG-R", HEALTHY, rul=20.0),
        "WTG-F": _twin("WTG-F", HEALTHY, rul=400.0),
    }
    plan = build_plan(twins, days=60)
    kinds = [t["kind"] for week in plan["calendar"] for t in week["tasks"]]
    assert "rul_inspection" in kinds  # WTG-R inside the 45-day horizon
    inspection = [
        t for week in plan["calendar"] for t in week["tasks"] if t["kind"] == "rul_inspection"
    ]
    assert len(inspection) == 1
    assert "RUL 20 days" in inspection[0]["actions"][0]


def test_plan_rejects_bad_days():
    twins = {}
    with pytest.raises(ValueError):
        build_plan(twins, days=0)


def test_summarize_plan_renders():
    twins = {"WTG-1": _twin("WTG-1", FAULTY)}
    plan = build_plan(twins, days=10)
    summary = summarize_plan(plan)
    assert "Maintenance plan" in summary
    assert "task(s)" in summary
    assert "wk " in summary
