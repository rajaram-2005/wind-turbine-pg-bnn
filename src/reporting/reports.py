"""Advisory report generation (text + markdown) and fleet summaries.

These functions turn the dict produced by
:func:`src.models.predictor.run_advisory` into human-readable reports for the
CLI, the Streamlit UI, and maintenance tickets.

All output is ADVISORY-ONLY. Nothing here emits actuation commands; every
record is screened by :func:`src.utils.safety.enforce_safety_contract` before
it is formatted.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.eval.calibration import expected_asset_utilization
from src.models.predictor import run_advisory
from src.utils.safety import enforce_safety_contract
from src.utils.schema import BNNState, Telemetry, TurbinePayload

# Column groups understood by the fleet loaders.
TELEMETRY_COLUMNS = (
    "vibration_mms",
    "temperature_c",
    "rpm",
    "oil_viscosity_cst",
    "load_pct",
)
BNN_COLUMNS = (
    "predicted_rul_days",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
)
FLEET_REQUIRED = ("asset_id",) + TELEMETRY_COLUMNS


# --------------------------------------------------------------------------- #
# Pipeline helpers (data -> advisory records)                                 #
# --------------------------------------------------------------------------- #
def advisories_from_dataframe(df: pd.DataFrame) -> list[dict]:
    """Build one advisory record per row of a fleet :class:`~pandas.DataFrame`.

    Expects ``asset_id`` + the five telemetry channels + the three
    ``bnn_state`` columns (``predicted_rul_days``, ``epistemic_uncertainty``,
    ``aleatoric_uncertainty``). Each row is validated through the
    :class:`~src.utils.schema.TurbinePayload` schema and screened by the
    safety gate before formatting.
    """
    missing = [c for c in FLEET_REQUIRED + BNN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fleet data missing required columns: {missing}")

    records: list[dict] = []
    for _, row in df.iterrows():
        payload = TurbinePayload(
            asset_id=str(row["asset_id"]),
            telemetry=Telemetry(**{c: float(row[c]) for c in TELEMETRY_COLUMNS}),
            bnn_state=BNNState(**{c: float(row[c]) for c in BNN_COLUMNS}),
        )
        rec = run_advisory(payload)
        enforce_safety_contract(rec)
        records.append(rec)
    return records


def advisories_from_csv(path: str) -> list[dict]:
    """Load a fleet CSV and return one advisory record per asset."""
    return advisories_from_dataframe(pd.read_csv(path))


# --------------------------------------------------------------------------- #
# Single-asset formatters                                                     #
# --------------------------------------------------------------------------- #
def format_advisory_text(rec: dict) -> str:
    """Plain-text report for a single advisory record."""
    violations = rec.get("physics_violations") or []
    if violations:
        viol_block = "\n".join(f"  - {v}" for v in violations)
    else:
        viol_block = "  (none — all channels within nominal bounds)"

    return (
        f"ADVISORY REPORT — {rec['asset_id']}\n"
        f"{'=' * 72}\n"
        f"Generated:       {rec.get('generated_at', 'n/a')}\n"
        f"ADVISORY ONLY — Decision-support, NOT an actuation command.\n\n"
        f"EARLY WARNING (45-day horizon): "
        f"{'⚠️ TRIGGERED — predicted failure within 45 days' if rec.get('early_warning_triggered') else 'not triggered'}\n\n"
        f"Predicted RUL:               {rec['predicted_rul_days']:.1f} days\n"
        f"Epistemic uncertainty (σ):   {rec['epistemic_std']:.3f}\n"
        f"Aleatoric uncertainty (σ):   {rec['aleatoric_std']:.3f}\n"
        f"Suggested inspection window: {rec['suggested_inspection_window_days']:.1f} days\n\n"
        f"Physics violations:\n{viol_block}\n\n"
        f"Rationale:\n  {rec.get('rationale', '').strip()}\n\n"
        f"Disclaimer:\n  {rec.get('disclaimer', '').strip()}\n"
    )


def format_advisory_markdown(rec: dict) -> str:
    """Markdown report for a single advisory record."""
    violations = rec.get("physics_violations") or []
    if violations:
        viol_block = "\n".join(f"- {v}" for v in violations)
    else:
        viol_block = "_None — all channels within nominal bounds._"

    if rec.get("early_warning_triggered"):
        early_warning_cell = "**⚠️ TRIGGERED — predicted failure within 45 days**"
    else:
        early_warning_cell = "not triggered"

    return (
        f"# Advisory — `{rec['asset_id']}`\n"
        f"\n"
        f"> ⚠️ **ADVISORY ONLY — decision-support, not an actuation command.**\n"
        f"> Generated: {rec.get('generated_at', 'n/a')}\n"
        f"\n"
        f"| Metric | Value |\n"
        f"| --- | --- |\n"
        f"| Predicted RUL | **{rec['predicted_rul_days']:.1f} days** |\n"
        f"| Epistemic uncertainty (σ) | {rec['epistemic_std']:.3f} |\n"
        f"| Aleatoric uncertainty (σ) | {rec['aleatoric_std']:.3f} |\n"
        f"| Suggested inspection window | {rec['suggested_inspection_window_days']:.1f} days |\n"
        f"| Early warning (45-day horizon) | {early_warning_cell} |\n"
        f"\n"
        f"## Physics violations\n\n{viol_block}\n\n"
        f"## Rationale\n\n{rec.get('rationale', '').strip()}\n\n"
        f"## Disclaimer\n\n{rec.get('disclaimer', '').strip()}\n"
    )


# --------------------------------------------------------------------------- #
# Fleet formatters                                                            #
# --------------------------------------------------------------------------- #
def fleet_summary(records: Sequence[dict]) -> dict:
    """Aggregate utilization metrics across a list of advisory records."""
    ruls = [float(r["predicted_rul_days"]) for r in records]
    util = expected_asset_utilization(ruls)
    return {
        "n_assets": len(records),
        "mean_utilization": util["mean_utilization"],
        "fraction_at_risk": util["fraction_at_risk"],
        "mean_rul_days": util["mean_rul_days"],
    }


def fleet_summary_table(records: Sequence[dict]) -> str:
    """Render a compact markdown table of all fleet advisories."""
    header = (
        "| Asset | RUL (days) | Early warning | Epistemic σ | Aleatoric σ "
        "| Inspect in (days) | Violations |\n"
        "| --- | ---: | --- | ---: | ---: | ---: | --- |\n"
    )
    rows = []
    for r in records:
        n_viol = len(r.get("physics_violations") or [])
        viol = f"{n_viol}" if n_viol else "0"
        ew = "⚠️ within 45d" if r.get("early_warning_triggered") else "no"
        rows.append(
            f"| {r['asset_id']} | {r['predicted_rul_days']:.1f} | {ew} "
            f"| {r['epistemic_std']:.3f} | {r['aleatoric_std']:.3f} "
            f"| {r['suggested_inspection_window_days']:.1f} | {viol} |"
        )
    return header + "\n".join(rows) + "\n"


def build_fleet_report(
    records: Sequence[dict],
    title: str = "Fleet RUL advisory report",
) -> str:
    """Assemble a full markdown fleet report (banner + summary + table + per-asset)."""
    if not records:
        return f"# {title}\n\n_No assets in fleet._\n"

    summary = fleet_summary(records)
    table = fleet_summary_table(records)
    details = "\n\n".join(format_advisory_markdown(r) for r in records)

    return (
        f"# {title}\n"
        f"\n"
        f"> ⚠️ **ADVISORY ONLY — decision-support, not actuation commands.**\n"
        f"> Generated: {records[0].get('generated_at', 'n/a')}\n"
        f"\n"
        f"## Fleet summary\n\n"
        f"- Assets assessed: **{summary['n_assets']}**\n"
        f"- Mean predicted RUL: **{summary['mean_rul_days']:.1f} days**\n"
        f"- Mean utilization: **{summary['mean_utilization']:.3f}**\n"
        f"- Fraction at risk (RUL < horizon + buffer): "
        f"**{summary['fraction_at_risk']:.3f}**\n"
        f"\n"
        f"## Advisory table\n\n{table}\n"
        f"\n"
        f"## Per-asset detail\n\n{details}\n"
    )


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def write_report(content: str, path: str | None = None) -> None:
    """Write a report to ``path``; ``None`` or ``"-"`` writes to stdout."""
    if path in (None, "-"):
        sys.stdout.write(content if content.endswith("\n") else content + "\n")
    else:
        Path(path).write_text(content)
