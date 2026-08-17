"""30-day maintenance planner: fleet work orders onto a weekly calendar.

Takes every tracked twin (with its latest fault report and RUL advisory) and
produces a prioritized weekly schedule:

* each fault report becomes a :class:`src.maintenance.workorder.WorkOrder`
  with its target date, priority and estimated hours;
* assets whose predicted RUL falls inside the planning window get an
  additional ``RUL-based inspection`` task;
* each week bucket reports task count and total estimated hours, and the
  energy-at-risk (MWh) is estimated from the turbine's rated power times the
  task hours (a planning aid, not a production forecast).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.maintenance.workorder import work_order_from_report

RUL_INSPECTION_HORIZON_DAYS = 45


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def build_plan(twins: Any, days: int = 30, now: datetime | None = None) -> dict:
    """Weekly maintenance plan for every twin with a fault report.

    ``twins`` maps asset_id -> digital twin (must expose ``spec``,
    ``last_fault_report`` and ``farm``). Returns a serializable dict with a
    per-week calendar and fleet totals.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    now = now or datetime.now(timezone.utc)
    horizon = now.date() + timedelta(days=days)

    weeks: dict[date, dict[str, Any]] = defaultdict(
        lambda: {"tasks": [], "total_hours": 0.0, "energy_at_risk_mwh": 0.0}
    )
    tasks_planned = 0
    assets_planned: list[str] = []

    for asset_id, twin in twins.items():
        report = twin.last_fault_report
        if report is None:
            continue
        assets_planned.append(asset_id)
        order = work_order_from_report(report, now=now)

        # Split work-order actions into per-week tasks from today to target.
        target = date.fromisoformat(order.target_date)
        if target > horizon:
            target = horizon
        # Schedule the work order's aggregate action list in the week of the
        # target date (immediate priorities land in the current week).
        bucket = _week_start(target)
        hours = max(order.estimated_hours / max(order.n_faults, 1), 0.5) if order.n_faults else 0.0
        task = {
            "asset_id": asset_id,
            "farm": getattr(twin, "farm", "") or "",
            "kind": "workorder",
            "wo_id": order.wo_id,
            "priority": order.priority,
            "faults": [f["fault_id"] for f in order.faults],
            "actions": order.recommended_actions[:4],
            "target_date": order.target_date,
            "estimated_hours": round(hours, 1),
        }
        weeks[bucket]["tasks"].append(task)
        weeks[bucket]["total_hours"] += hours
        rated_mw = float(getattr(twin.spec, "rated_power_mw", 2.0))
        weeks[bucket]["energy_at_risk_mwh"] += hours * rated_mw
        tasks_planned += 1

        # RUL-based inspection when the forecast is inside the horizon.
        rul_days = _predicted_rul(twin)
        if rul_days is not None and rul_days <= RUL_INSPECTION_HORIZON_DAYS:
            bucket = _week_start(now.date() + timedelta(days=int(rul_days)))
            if bucket > _week_start(horizon):
                bucket = _week_start(horizon)
            task = {
                "asset_id": asset_id,
                "farm": getattr(twin, "farm", "") or "",
                "kind": "rul_inspection",
                "wo_id": None,
                "priority": "P1 — urgent" if rul_days <= 15 else "P2 — planned",
                "faults": [],
                "actions": [f"Predicted RUL {rul_days:.0f} days — schedule inspection"],
                "target_date": (now.date() + timedelta(days=int(rul_days))).isoformat(),
                "estimated_hours": 4.0,
            }
            weeks[bucket]["tasks"].append(task)
            weeks[bucket]["total_hours"] += 4.0
            rated_mw = float(getattr(twin.spec, "rated_power_mw", 2.0))
            weeks[bucket]["energy_at_risk_mwh"] += 4.0 * rated_mw
            tasks_planned += 1

    calendar = [
        {
            "week_start": week.isoformat(),
            "week_end": (week + timedelta(days=6)).isoformat(),
            "n_tasks": len(data["tasks"]),
            "total_hours": round(data["total_hours"], 1),
            "energy_at_risk_mwh": round(data["energy_at_risk_mwh"], 1),
            "tasks": data["tasks"],
        }
        for week, data in sorted(weeks.items())
    ]

    return {
        "generated_at": now.isoformat(),
        "horizon_days": days,
        "horizon_end": horizon.isoformat(),
        "n_assets_planned": len(assets_planned),
        "n_tasks_planned": tasks_planned,
        "total_hours": round(sum(w["total_hours"] for w in calendar), 1),
        "energy_at_risk_mwh": round(sum(w["energy_at_risk_mwh"] for w in calendar), 1),
        "calendar": calendar,
    }


def _predicted_rul(twin) -> float | None:
    """RUL from the twin's last advisory (best effort)."""
    try:
        history = twin.state_history
        if not history:
            return None
        advisory = history[-1].get("advisory") or {}
        for key in ("predicted_rul_days", "rul_days"):
            value = advisory.get(key)
            if value is not None:
                return float(value)
        bnn = history[-1].get("bnn_state") or {}
        value = bnn.get("predicted_rul_days")
        return float(value) if value is not None else None
    except (IndexError, TypeError, ValueError):
        return None


def summarize_plan(plan: dict) -> str:
    """Compact human summary of a plan (for CLI/console)."""
    lines = [
        f"Maintenance plan: {plan['n_tasks_planned']} task(s) over "
        f"{plan['horizon_days']} days for {plan['n_assets_planned']} asset(s) — "
        f"{plan['total_hours']} h, ~{plan['energy_at_risk_mwh']} MWh at risk.",
    ]
    for week in plan["calendar"]:
        lines.append(
            f"  wk {week['week_start']}: {week['n_tasks']} task(s), " f"{week['total_hours']} h"
        )
    return "\n".join(lines)
