"""Shared advisory-evidence bridge for every AeroVigil surface.

Raw PG-BNN recommendations are deliberately model-focused.  This module adds
the deterministic MIKA + KAI evidence brief once, at the application boundary,
so API, CLI, CSV/fleet reports, and dashboard callers describe the same asset
with the same source trail.  It never issues commands or changes model scores.
"""

from __future__ import annotations

from typing import Any

from src.agents.cyber_team import build_cyber_team_brief
from src.utils.safety import enforce_safety_contract
from src.utils.schema import TurbinePayload


def connect_advisory_evidence(payload: TurbinePayload, recommendation: dict[str, Any]) -> dict[str, Any]:
    """Attach the canonical dual-agent evidence brief to an advisory.

    The function is idempotent: callers may safely pass an already-enriched
    recommendation without replacing its existing brief.  A copied mapping is
    returned so an input produced by a predictor can be reused on another
    surface without mutation.
    """
    enriched = dict(recommendation)
    if not enriched.get("agent_team"):
        enriched["agent_team"] = build_cyber_team_brief(
            asset_id=payload.asset_id,
            predicted_rul_days=recommendation.get("predicted_rul_days"),
            epistemic_std=recommendation.get("epistemic_std", 0.0),
            physics_violations=recommendation.get("physics_violations", []),
            telemetry=payload.telemetry.model_dump(),
        )
    return enforce_safety_contract(enriched)
