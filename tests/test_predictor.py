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
    # Blocked keys never appear
    for k in ("throttle_pct", "rpm_setpoint", "loto_steps", "part_sku"):
        assert k not in rec
