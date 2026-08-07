"""Connected dual-agent synthesis for the Cyber Prime operator experience.

MIKA and KAI are deterministic advisory personas over the existing AeroVigil
signals. They do not introduce another predictive model and never actuate the
asset: MIKA translates health evidence into maintenance-planning language,
while KAI explains the physics and constraint evidence behind it.
"""

from __future__ import annotations

import math
from typing import Any

from src.utils.safety import enforce_safety_contract

RISK_LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _risk_from_rul(rul_days: float | None, violations: list[str], wear: float) -> str:
    if rul_days is not None:
        if rul_days < 14.0:
            return "CRITICAL"
        if rul_days < 30.0:
            return "HIGH"
        if rul_days < 45.0:
            return "MODERATE"
    if len(violations) >= 3 or wear >= 0.9:
        return "CRITICAL"
    if len(violations) >= 2 or wear >= 0.75:
        return "HIGH"
    if violations or wear >= 0.5 or (rul_days is not None and rul_days < 45.0):
        return "MODERATE"
    return "LOW"


def build_cyber_team_brief(
    *,
    asset_id: str,
    predicted_rul_days: float | None = None,
    epistemic_std: float = 0.0,
    physics_violations: list[str] | None = None,
    cumulative_wear: float | None = None,
    bearing_l10_hours: float | None = None,
    telemetry: dict[str, Any] | None = None,
    risk: str | None = None,
) -> dict[str, Any]:
    """Fuse BNN, telemetry, wear, and physics evidence into one team brief.

    The output is deliberately plain JSON-compatible data so the same result
    can flow through the digital-twin runtime, API, prompt, dashboard, and
    fleet cards without each surface inventing its own agent interpretation.
    """
    violations = [str(item) for item in (physics_violations or [])]
    has_wear = cumulative_wear is not None
    wear = min(1.0, max(0.0, _finite(cumulative_wear)))
    uncertainty = max(0.0, _finite(epistemic_std))
    rul = None if predicted_rul_days is None else max(0.0, _finite(predicted_rul_days))
    resolved_risk = str(risk or _risk_from_rul(rul, violations, wear)).upper()
    if resolved_risk not in RISK_LEVELS:
        raise ValueError(f"unknown risk level: {resolved_risk}")

    windows = {"LOW": 30.0, "MODERATE": 14.0, "HIGH": 7.0, "CRITICAL": 1.0}
    review_window = windows[resolved_risk]
    if rul is not None:
        review_window = min(review_window, max(1.0, rul * 0.35))

    mika_finding = {
        "LOW": "Health runway supports routine monitoring; preserve the normal inspection cadence.",
        "MODERATE": "The asset is entering a planning window; reserve inspection capacity and watch the trend.",
        "HIGH": "Degradation is accelerating; validate maintenance readiness, crew availability, and spares.",
        "CRITICAL": "The failure horizon is compressed; escalate the evidence for immediate human review.",
    }[resolved_risk]

    if violations:
        kai_finding = (
            f"Physics layer reports {len(violations)} active constraint signal(s): "
            + "; ".join(violations[:3])
            + "."
        )
    elif bearing_l10_hours is not None and math.isfinite(_finite(bearing_l10_hours, math.inf)):
        kai_finding = (
            "No active spec-limit violations; ISO 281 bearing life is "
            f"{_finite(bearing_l10_hours):,.0f} hours at the current load state."
        )
    else:
        kai_finding = "No active spec-limit violations are reported by the connected physics layer."

    telemetry = dict(telemetry or {})
    observed = []
    channel_labels = {
        "vibration_mms": "vibration",
        "vibration_rms": "vibration",
        "temperature_c": "temperature",
        "bearing_temp": "bearing temperature",
        "load_pct": "load",
        "power_output": "power",
    }
    for key, label in channel_labels.items():
        if key in telemetry and label not in observed:
            observed.append(label)

    sources = ["safety_contract"]
    if telemetry:
        sources.append("telemetry")
    if violations or bearing_l10_hours is not None:
        sources.extend(["physics_constraints", "iso_281"])
    if rul is not None:
        sources.append("pg_bnn")
    if has_wear:
        sources.append("digital_twin_wear")

    # Agreement is a team-coordination indicator, not a calibrated probability.
    agreement = 99.2 - min(22.0, uncertainty * 0.8) - (2.0 if violations else 0.0)
    agreement = min(99.9, max(70.0, agreement))
    evidence = f"Connected evidence: {', '.join(sources)}."
    if observed:
        evidence += f" Live channels represented: {', '.join(observed)}."

    payload = {
        "team_id": "CYBER_PRIME_DUAL_AGENT",
        "asset_id": str(asset_id),
        "risk_level": resolved_risk,
        "agreement_score_pct": round(agreement, 2),
        "review_window_days": round(review_window, 1),
        "shared_summary": f"MIKA and KAI agree on {resolved_risk} advisory state. {evidence}",
        "agents": {
            "mika": {
                "name": "MIKA",
                "role": "maintenance_strategist",
                "finding": mika_finding,
                "focus": "maintenance planning and escalation timing",
            },
            "kai": {
                "name": "KAI",
                "role": "physics_constraint_sentinel",
                "finding": kai_finding,
                "focus": "telemetry, ISO 281 bearing life, and physical limits",
            },
        },
        "connected_sources": sources,
        "advisory_only": True,
        "disclaimer": (
            "Decision-support only. MIKA and KAI issue no turbine commands; "
            "a qualified operator must review all findings against OEM guidance."
        ),
    }
    return enforce_safety_contract(payload)
