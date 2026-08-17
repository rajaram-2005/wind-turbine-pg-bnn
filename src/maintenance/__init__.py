"""Maintenance work-order generation from fault reports.

* :class:`WorkOrderGenerator` turns a :class:`src.faults.detector.FaultReport`
  into a prioritized, human-actionable work order: unique WO id, priority,
  target window, recommended actions, involved parts/sensors, inspection
  checklist and advisory-only banner.

Everything generated is **advisory** — it tells the maintenance crew what to
plan and check, never what to actuate.
"""

from src.maintenance.workorder import (
    PRIORITY_RANK,
    WorkOrder,
    WorkOrderGenerator,
    work_order_from_report,
)

__all__ = [
    "PRIORITY_RANK",
    "WorkOrder",
    "WorkOrderGenerator",
    "work_order_from_report",
]
