"""Tests for the maintenance work-order generator (src/maintenance)."""

from src.faults.detector import FaultDetector
from src.maintenance import WorkOrderGenerator, work_order_from_report

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


def test_work_order_from_healthy_report_is_low_priority():
    report = FaultDetector().detect(HEALTHY, asset_id="WTG-OK")
    wo = work_order_from_report(report)
    assert wo.wo_id.startswith("WO-")
    assert wo.asset_id == "WTG-OK"
    assert wo.n_faults == 0
    assert wo.priority.startswith("P3")
    assert wo.advisory_only is True


def test_work_order_priority_tracks_worst_fault():
    report = FaultDetector().detect(FAULTY, asset_id="WTG-BAD")
    wo = work_order_from_report(report)
    assert wo.n_faults >= 2
    assert wo.priority.startswith("P0") or wo.priority.startswith("P1")
    # Actions aggregated from every fault, deduped, prefixed with fault id.
    assert any(a.startswith("[GB-02]") for a in wo.recommended_actions)
    assert wo.sensors, "work order must list involved sensors"
    assert any(s.startswith("OC-") for s in wo.sensors)  # viscometer etc.


def test_work_order_target_date_and_checklist():
    from datetime import datetime, timezone

    now = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    report = FaultDetector().detect(FAULTY, asset_id="WTG-BAD")
    wo = work_order_from_report(report, now=now)
    # FAULTY carries HIGH faults (viscosity/water) -> P1, +1 day target.
    assert wo.priority.startswith("P1")
    assert wo.target_date == "2026-08-18"
    assert "Confirm LOTO / safe access before any work" in wo.inspection_checklist
    assert wo.estimated_hours > 0
    assert wo.generated_at.startswith("2026-08-17")


def test_work_order_is_deterministic():
    report = FaultDetector().detect(FAULTY, asset_id="WTG-BAD")
    now = "2026-08-17T12:00:00+00:00"
    first = work_order_from_report(report, now=now)
    second = work_order_from_report(report, now=now)
    assert first.wo_id == second.wo_id
    assert first.to_dict() == second.to_dict()


def test_work_order_generator_fleet():
    generator = WorkOrderGenerator()
    reports = [
        FaultDetector().detect(HEALTHY, asset_id="WTG-1"),
        FaultDetector().detect(FAULTY, asset_id="WTG-2"),
    ]
    orders = generator.generate_fleet(reports)
    assert [o.asset_id for o in orders] == ["WTG-1", "WTG-2"]
    assert orders[1].priority.startswith("P0") or orders[1].priority.startswith("P1")


def test_work_order_serialization():
    report = FaultDetector().detect(FAULTY, asset_id="WTG-BAD")
    wo = work_order_from_report(report)
    body = wo.to_dict()
    assert body["wo_id"] == wo.wo_id
    assert "priority_color" in body
    assert body["advisory_only"] is True
    assert all("fault_id" in f for f in body["faults"])
