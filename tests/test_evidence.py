"""Tests for the shared cross-surface advisory evidence bridge."""

from src.agents.evidence import connect_advisory_evidence
from src.models.predictor import run_advisory
from src.utils.schema import BNNState, Telemetry, TurbinePayload


def _payload() -> TurbinePayload:
    return TurbinePayload(
        asset_id="WTG-LINKED",
        telemetry=Telemetry(
            vibration_mms=4.8,
            temperature_c=82.0,
            rpm=1780.0,
            oil_viscosity_cst=12.0,
            load_pct=95.0,
        ),
        bnn_state=BNNState(
            predicted_rul_days=14.2,
            epistemic_uncertainty=0.04,
            aleatoric_uncertainty=0.12,
        ),
    )


def test_evidence_bridge_preserves_advisory_and_connects_sources():
    payload = _payload()
    original = run_advisory(payload)
    connected = connect_advisory_evidence(payload, original)

    assert connected is not original
    assert connected["predicted_rul_days"] == original["predicted_rul_days"]
    assert connected["agent_team"]["asset_id"] == "WTG-LINKED"
    assert set(connected["agent_team"]["connected_sources"]) >= {
        "safety_contract",
        "telemetry",
        "pg_bnn",
    }
    assert "agent_team" not in original


def test_evidence_bridge_is_idempotent():
    payload = _payload()
    once = connect_advisory_evidence(payload, run_advisory(payload))
    twice = connect_advisory_evidence(payload, once)

    assert twice["agent_team"] == once["agent_team"]
