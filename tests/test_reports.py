"""Tests for advisory report generation (`src.reporting.reports`)."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.predictor import run_advisory
from src.reporting.reports import (
    advisories_from_csv,
    advisories_from_dataframe,
    build_fleet_report,
    fleet_summary,
    fleet_summary_table,
    format_advisory_markdown,
    format_advisory_text,
    write_report,
)
from src.utils.schema import BNNState, Telemetry, TurbinePayload

BLOCKED_KEYS = ("throttle_pct", "rpm_setpoint", "loto_steps", "part_sku")


def _record(rul: float = 14.2, asset: str = "WTG-044", vib: float = 4.8) -> dict:
    payload = TurbinePayload(
        asset_id=asset,
        telemetry=Telemetry(
            vibration_mms=vib,
            temperature_c=82.0,
            rpm=1780,
            oil_viscosity_cst=12.0,
            load_pct=95.0,
        ),
        bnn_state=BNNState(
            predicted_rul_days=rul,
            epistemic_uncertainty=0.04,
            aleatoric_uncertainty=0.12,
        ),
    )
    return run_advisory(payload)


# --------------------------------------------------------------------------- #
# Single-asset formatters                                                     #
# --------------------------------------------------------------------------- #
def test_format_advisory_text_contains_key_fields():
    rec = _record()
    text = format_advisory_text(rec)
    assert "WTG-044" in text
    assert "14.2 days" in text
    assert "ADVISORY ONLY" in text
    assert "decision-support" in text.lower()
    # Physics violations rendered.
    assert "Vibration" in text
    for bad in BLOCKED_KEYS:
        assert bad not in text


def test_format_advisory_markdown_has_header_and_table():
    rec = _record(asset="WTG-007", rul=300.0)
    md = format_advisory_markdown(rec)
    assert md.startswith("# Advisory — `WTG-007`")
    assert "300.0 days" in md
    assert "| Predicted RUL" in md  # markdown table
    assert "ADVISORY ONLY" in md


def test_format_advisory_markdown_handles_no_violations():
    # Fully nominal telemetry across every channel.
    payload = TurbinePayload(
        asset_id="WTG-CLEAN",
        telemetry=Telemetry(
            vibration_mms=2.0,
            temperature_c=60.0,
            rpm=1500.0,
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
    assert rec["physics_violations"] == []
    md = format_advisory_markdown(rec)
    assert "None — all channels within nominal bounds" in md


# --------------------------------------------------------------------------- #
# Fleet formatters                                                            #
# --------------------------------------------------------------------------- #
def test_fleet_summary_aggregates():
    records = [_record(rul=10.0, asset="A"), _record(rul=200.0, asset="B")]
    s = fleet_summary(records)
    assert s["n_assets"] == 2
    assert s["mean_rul_days"] == pytest.approx(105.0)
    assert 0.0 <= s["mean_utilization"] <= 1.0
    assert 0.0 <= s["fraction_at_risk"] <= 1.0


def test_fleet_summary_table_has_header_and_rows():
    records = [_record(rul=10.0, asset="A"), _record(rul=200.0, asset="B")]
    table = fleet_summary_table(records)
    assert "| Asset |" in table
    assert "| --- |" in table
    assert "A" in table and "B" in table
    lines = [ln for ln in table.strip().splitlines() if ln.startswith("|")]
    assert len(lines) == 4  # header + separator + 2 data rows


def test_build_fleet_report_structure():
    records = [_record(rul=10.0, asset="A"), _record(rul=200.0, asset="B")]
    report = build_fleet_report(records, title="Q3 review")
    assert "Q3 review" in report
    assert "ADVISORY ONLY" in report
    assert "Assets assessed: **2**" in report
    assert "Advisory table" in report
    assert "Per-asset detail" in report
    for bad in BLOCKED_KEYS:
        assert bad not in report


def test_build_fleet_report_empty():
    report = build_fleet_report([])
    assert "No assets in fleet" in report


# --------------------------------------------------------------------------- #
# Pipeline loaders                                                            #
# --------------------------------------------------------------------------- #
def _fleet_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset_id": "WTG-001",
                "vibration_mms": 2.1,
                "temperature_c": 58.0,
                "rpm": 1500.0,
                "oil_viscosity_cst": 32.0,
                "load_pct": 80.0,
                "predicted_rul_days": 240.0,
                "epistemic_uncertainty": 0.05,
                "aleatoric_uncertainty": 0.10,
            },
            {
                "asset_id": "WTG-044",
                "vibration_mms": 4.8,
                "temperature_c": 82.0,
                "rpm": 1780.0,
                "oil_viscosity_cst": 12.0,
                "load_pct": 95.0,
                "predicted_rul_days": 14.2,
                "epistemic_uncertainty": 0.04,
                "aleatoric_uncertainty": 0.12,
            },
        ]
    )


def test_advisories_from_dataframe():
    records = advisories_from_dataframe(_fleet_df())
    assert len(records) == 2
    ids = {r["asset_id"] for r in records}
    assert ids == {"WTG-001", "WTG-044"}
    for rec in records:
        assert rec["advisory_only"] is True
        for bad in BLOCKED_KEYS:
            assert bad not in rec


def test_advisories_from_dataframe_rejects_missing_columns():
    df = _fleet_df().drop(columns=["predicted_rul_days"])
    with pytest.raises(ValueError, match="missing required columns"):
        advisories_from_dataframe(df)


def test_advisories_from_csv_matches_examples(tmp_path: Path):
    # Mirror the shipped examples/fleet.csv to a temp file and round-trip it.
    csv_path = tmp_path / "fleet.csv"
    csv_path.write_text(
        "asset_id,vibration_mms,temperature_c,rpm,oil_viscosity_cst,load_pct,"
        "predicted_rul_days,epistemic_uncertainty,aleatoric_uncertainty\n"
        "WTG-001,2.1,58.0,1500,32.0,80.0,240.0,0.05,0.10\n"
    )
    records = advisories_from_csv(str(csv_path))
    assert len(records) == 1
    assert records[0]["asset_id"] == "WTG-001"


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def test_write_report_to_file(tmp_path: Path):
    out = tmp_path / "report.md"
    write_report("# hello", str(out))
    assert out.read_text() == "# hello"


def test_write_report_stdout(capsys):
    write_report("# hello", None)
    captured = capsys.readouterr()
    assert captured.out == "# hello\n"
