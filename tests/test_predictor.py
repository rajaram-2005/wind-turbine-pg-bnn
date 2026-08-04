from src.models.predictor import run_advisory
from src.utils.schema import BNNState, Telemetry, TurbinePayload


def test_run_advisory_low_rul_returns_inspection_window():
    payload = TurbinePayload(
        asset_id="WTG-044",
        telemetry=Telemetry(
            vibration_mms=4.8,
            temperature_c=82.0,
            rpm=1780,
            oil_viscosity_cst=12.0,
            load_pct=95.0,
        ),
        bnn_state=BNNState(
            predicted_rul_days=14.2,
            epistemic_uncertainty=0.04,
            aleatoric_uncertainty=0.12,
        ),
    )
    rec = run_advisory(payload)
    assert rec["asset_id"] == "WTG-044"
    assert rec["advisory_only"] is True
    assert len(rec["physics_violations"]) >= 2  # vibration & temperature
    assert rec["suggested_inspection_window_days"] <= 7
    assert "does not specify a curtailment" in rec["rationale"]
    # 45-day early warning fires for this near-failure asset.
    assert rec["early_warning_triggered"] is True
    assert rec["warning_horizon_days"] == 45.0
    assert "EARLY WARNING" in rec["rationale"]
    # Blocked keys never appear
    for k in ("throttle_pct", "rpm_setpoint", "loto_steps", "part_sku"):
        assert k not in rec


def test_run_advisory_healthy_asset_no_early_warning():
    payload = TurbinePayload(
        asset_id="WTG-007",
        telemetry=Telemetry(
            vibration_mms=2.0,
            temperature_c=58.0,
            rpm=1500,
            oil_viscosity_cst=32.0,
            load_pct=78.0,
        ),
        bnn_state=BNNState(
            predicted_rul_days=300.0,
            epistemic_uncertainty=0.03,
            aleatoric_uncertainty=0.09,
        ),
    )
    rec = run_advisory(payload)
    assert rec["early_warning_triggered"] is False
    assert "EARLY WARNING" not in rec["rationale"]
