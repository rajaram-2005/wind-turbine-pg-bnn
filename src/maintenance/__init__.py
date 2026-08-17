"""Maintenance work-order generation from fault reports.

* :class:`WorkOrderGenerator` turns a :class:`src.faults.detector.FaultReport`
  into a prioritized, human-actionable work order: unique WO id, priority,
  target window, recommended actions, involved parts/sensors, inspection
  checklist and advisory-only banner.
* :func:`src.maintenance.planner.build_plan` rolls work orders + RUL
  forecasts across the fleet onto a 30-day weekly calendar.

Everything generated is **advisory** — it tells the maintenance crew what to
plan and check, never what to actuate.
"""

from src.maintenance.planner import build_plan, summarize_plan
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
    "build_plan",
    "summarize_plan",
    "work_order_from_report",
]
