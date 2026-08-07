"""Tests for the connected MIKA + KAI advisory synthesis layer."""

import pytest

from src.agents.cyber_team import build_cyber_team_brief
from src.utils.safety import enforce_safety_contract


def test_team_connects_bnn_physics_twin_and_telemetry():
    brief = build_cyber_team_brief(
        asset_id="WTG-LINKED",
        predicted_rul_days=24.0,
        epistemic_std=6.0,
        physics_violations=["vibration above model limit"],
        cumulative_wear=0.72,
        bearing_l10_hours=8200.0,
        telemetry={"vibration_mms": 6.1, "temperature_c": 91.0, "load_pct": 108.0},
    )
    assert brief["team_id"] == "CYBER_PRIME_DUAL_AGENT"
    assert brief["risk_level"] == "HIGH"
    assert brief["agents"]["mika"]["role"] == "maintenance_strategist"
    assert brief["agents"]["kai"]["role"] == "physics_constraint_sentinel"
    assert "vibration above model limit" in brief["agents"]["kai"]["finding"]
    assert set(brief["connected_sources"]) == {
        "safety_contract",
        "telemetry",
        "physics_constraints",
        "iso_281",
        "pg_bnn",
        "digital_twin_wear",
    }
    assert 70.0 <= brief["agreement_score_pct"] <= 99.9
    assert brief["advisory_only"] is True
    enforce_safety_contract(brief)


@pytest.mark.parametrize(
    ("rul", "risk"),
    [(120.0, "LOW"), (40.0, "MODERATE"), (20.0, "HIGH"), (8.0, "CRITICAL")],
)
def test_team_uses_shared_risk_boundaries(rul, risk):
    brief = build_cyber_team_brief(asset_id="WTG-RISK", predicted_rul_days=rul)
    assert brief["risk_level"] == risk


def test_team_can_assess_physics_only_snapshot():
    brief = build_cyber_team_brief(
        asset_id="WTG-PHYSICS",
        physics_violations=["temperature exceeds limit", "viscosity below limit"],
        cumulative_wear=0.8,
    )
    assert brief["risk_level"] == "HIGH"
    assert "pg_bnn" not in brief["connected_sources"]
    assert "physics_constraints" in brief["connected_sources"]


def test_team_rejects_unknown_risk_override():
    with pytest.raises(ValueError, match="unknown risk"):
        build_cyber_team_brief(asset_id="WTG-BAD", risk="UNKNOWN")
