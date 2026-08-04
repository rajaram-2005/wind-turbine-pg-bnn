import pytest

from src.utils.safety import (
    AdvisoryRecommendation,
    SafetyBoundaryError,
    enforce_safety_contract,
)


def test_advisory_recommendation_has_disclaimer():
    rec = AdvisoryRecommendation(
        asset_id="WTG-044",
        predicted_rul_days=14.2,
        epistemic_std=0.04,
        aleatoric_std=0.12,
        physics_violations=["vibration over limit"],
        rationale="test",
    )
    d = rec.to_dict()
    assert d["advisory_only"] is True
    assert "Decision-support only" in d["disclaimer"]
    # Forbidden fields must NOT be present
    for bad in ("throttle_pct", "rpm_setpoint", "loto_steps", "part_sku"):
        assert bad not in d


def test_enforce_safety_contract_blocks_actuation():
    with pytest.raises(SafetyBoundaryError):
        enforce_safety_contract({"asset_id": "x", "throttle_pct": -18})
    with pytest.raises(SafetyBoundaryError):
        enforce_safety_contract({"asset_id": "x", "loto_steps": ["open breaker"]})
    with pytest.raises(SafetyBoundaryError):
        enforce_safety_contract({"asset_id": "x", "maintenance": {"part_sku": "BRG-123"}})


def test_enforce_safety_contract_allows_clean_payload():
    clean = {"asset_id": "x", "predicted_rul_days": 120.0, "rationale": "ok"}
    assert enforce_safety_contract(clean)["asset_id"] == "x"
