"""Work-order generator: fault report -> prioritized maintenance plan.

Maps severity to maintenance priority and planning window, aggregates the
recommended actions of every detected fault, attaches the sensors that can
reveal the faults (from :mod:`src.faults.sensors`), and produces a
deterministic WO id so the same snapshot always yields the same order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.faults.detector import FaultReport
from src.faults.sensors import sensors_for_fault

PRIORITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# severity -> (priority label, planning window)
_PLANNING: dict[str, tuple[str, timedelta]] = {
    "CRITICAL": ("P0 — immediate", timedelta(hours=4)),
    "HIGH": ("P1 — urgent", timedelta(days=1)),
    "MEDIUM": ("P2 — planned", timedelta(days=7)),
    "LOW": ("P3 — scheduled", timedelta(days=30)),
}

_PRIORITY_COLORS = {
    "P0": "#ef4444",
    "P1": "#f97316",
    "P2": "#f59e0b",
    "P3": "#8a97a5",
}


@dataclass(frozen=True)
class WorkOrder:
    """A maintenance work order derived from one fault report."""

    wo_id: str
    asset_id: str
    generated_at: str
    priority: str  # P0..P3 label
    severity: str  # overall status of the source report
    target_date: str
    n_faults: int
    faults: list[dict] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    inspection_checklist: list[str] = field(default_factory=list)
    estimated_hours: float = 0.0
    advisory_only: bool = True

    def to_dict(self) -> dict:
        return {
            "wo_id": self.wo_id,
            "asset_id": self.asset_id,
            "generated_at": self.generated_at,
            "priority": self.priority,
            "priority_color": _PRIORITY_COLORS.get(self.priority.split(" ")[0], "#8a97a5"),
            "severity": self.severity,
            "target_date": self.target_date,
            "n_faults": self.n_faults,
            "faults": list(self.faults),
            "recommended_actions": list(self.recommended_actions),
            "sensors": list(self.sensors),
            "inspection_checklist": list(self.inspection_checklist),
            "estimated_hours": self.estimated_hours,
            "advisory_only": self.advisory_only,
        }


def work_order_from_report(report: FaultReport, now: datetime | str | None = None) -> WorkOrder:
    """Build a work order from one fault report.

    ``now`` may be a :class:`datetime.datetime` or an ISO-8601 string.
    """
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    now = now or datetime.now(timezone.utc)
    if report.faults:
        worst = min(report.faults, key=lambda f: PRIORITY_RANK[f.severity])
        severity = worst.severity
    else:
        severity = "LOW"
    priority, window = _PLANNING[severity]
    target = (now + window).date().isoformat()

    digest = (
        hashlib.sha1(f"{report.asset_id}|{report.timestamp}|{severity}".encode())
        .hexdigest()[:8]
        .upper()
    )
    wo_id = f"WO-{now:%Y%m%d}-{digest}"

    actions: list[str] = []
    seen: set[str] = set()
    for fault in report.faults:
        for action in fault.recommended_actions:
            key = action.lower()
            if key not in seen:
                seen.add(key)
                actions.append(f"[{fault.fault_id}] {action}")

    sensors: list[str] = []
    for fault in report.faults:
        for sensor in sensors_for_fault(fault.fault_id):
            label = f"{sensor.sensor_id} {sensor.name}"
            if label not in sensors:
                sensors.append(label)

    checklist = [
        "Confirm LOTO / safe access before any work",
        "Review fault evidence and recent trends in the console",
        "Perform the recommended actions in priority order",
        "Re-run detection after the work and confirm the fault cleared",
        "Update the work-order status in the maintenance log",
    ]

    estimated_hours = sum(
        {"CRITICAL": 8.0, "HIGH": 4.0, "MEDIUM": 2.0, "LOW": 1.0}[f.severity] for f in report.faults
    )

    return WorkOrder(
        wo_id=wo_id,
        asset_id=report.asset_id,
        generated_at=now.isoformat(),
        priority=priority,
        severity=severity,
        target_date=target,
        n_faults=report.n_faults,
        faults=[f.to_dict() for f in report.faults],
        recommended_actions=actions,
        sensors=sensors,
        inspection_checklist=checklist,
        estimated_hours=estimated_hours,
    )


class WorkOrderGenerator:
    """Work-order factory with fleet helpers."""

    def generate(self, report: FaultReport, now: datetime | None = None) -> WorkOrder:
        return work_order_from_report(report, now=now)

    def generate_fleet(
        self, reports: list[FaultReport], now: datetime | None = None
    ) -> list[WorkOrder]:
        return [self.generate(r, now=now) for r in reports]
