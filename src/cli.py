"""Command-line interface for wind-turbine-pg-bnn.

Subcommands
-----------
``advisory``   Compute an advisory for a single turbine payload (JSON).
``fleet``      Compute advisories for a fleet CSV and emit a report.
``report``     Render a markdown report from a single payload or a fleet CSV.

All output is ADVISORY-ONLY.

Examples
--------
    wind-turbine-bnn advisory examples/payload.json
    wind-turbine-bnn fleet examples/fleet.csv -o fleet_report.md
    wind-turbine-bnn report --fleet examples/fleet.csv --title "Q3 review"
    cat examples/payload.json | wind-turbine-bnn advisory -
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.models.predictor import run_advisory
from src.reporting.reports import (
    advisories_from_csv,
    build_fleet_report,
    format_advisory_markdown,
)
from src.utils.encoding import configure_utf8_stdio
from src.utils.safety import enforce_safety_contract
from src.utils.schema import TurbinePayload


def _read_text(arg: str | None) -> str:
    """Read a payload from a path, or from stdin when ``arg`` is ``-``/None."""
    if arg in (None, "-"):
        return sys.stdin.read()
    return Path(arg).read_text(encoding="utf-8")


def _emit(content: str, output: str | None) -> None:
    """Write ``content`` to ``output`` (file path) or stdout (``-``/None)."""
    if output in (None, "-"):
        sys.stdout.write(content if content.endswith("\n") else content + "\n")
    else:
        Path(output).write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Subcommands                                                                 #
# --------------------------------------------------------------------------- #
def cmd_advisory(args: argparse.Namespace) -> None:
    payload = TurbinePayload(**json.loads(_read_text(args.payload)))
    if args.model:
        # Model-serving path: RUL + uncertainties come from the trained PG-BNN.
        from src.data.ingest import load_csv
        from src.models.serving import load_serving_model

        if not args.telemetry_csv:
            raise SystemExit("--telemetry-csv is required when --model is used")
        serving = load_serving_model(args.model)
        df = load_csv(args.telemetry_csv)
        rec = serving.advisory(payload, df)
    else:
        # Backward-compatible path: pre-computed bnn_state in the payload.
        rec = run_advisory(payload)
    enforce_safety_contract(rec)
    _emit(json.dumps(rec, indent=2), args.output)


def cmd_fleet(args: argparse.Namespace) -> None:
    records = advisories_from_csv(args.input)
    if args.format == "json":
        out = json.dumps(records, indent=2)
    else:
        out = build_fleet_report(records, title=args.title)
    _emit(out, args.output)


def cmd_report(args: argparse.Namespace) -> None:
    if args.fleet:
        records = advisories_from_csv(args.fleet)
        out = build_fleet_report(records, title=args.title)
    else:
        payload = TurbinePayload(**json.loads(_read_text(args.payload)))
        rec = run_advisory(payload)
        enforce_safety_contract(rec)
        out = format_advisory_markdown(rec)
    _emit(out, args.output)


# --------------------------------------------------------------------------- #
# Parser                                                                      #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wind-turbine-bnn",
        description="wind-turbine-pg-bnn advisory CLI (decision-support only).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("advisory", help="Advisory for a single turbine payload (JSON).")
    pa.add_argument("payload", help="Path to JSON payload, or '-' for stdin.")
    pa.add_argument("-o", "--output", default=None, help="Write to file (default: stdout).")
    pa.add_argument(
        "--model",
        default=None,
        metavar="CHECKPOINT",
        help="Optional trained PG-BNN bundle (e.g. artifacts/bnn_demo.pt). "
        "When given, RUL and uncertainties are computed by the model; "
        "otherwise the payload's bnn_state block is used (unchanged behavior).",
    )
    pa.add_argument(
        "--telemetry-csv",
        default=None,
        metavar="WINDOW_CSV",
        help="Raw telemetry window CSV (timestamp + 5 channels) used to build "
        "model features when --model is set.",
    )
    pa.set_defaults(func=cmd_advisory)

    pf = sub.add_parser("fleet", help="Advisories for a fleet CSV.")
    pf.add_argument("input", help="Path to fleet CSV.")
    pf.add_argument("-o", "--output", default=None, help="Write to file (default: stdout).")
    pf.add_argument("--title", default="Fleet RUL advisory report")
    pf.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Report format (default: markdown).",
    )
    pf.set_defaults(func=cmd_fleet)

    pr = sub.add_parser("report", help="Render a markdown advisory report.")
    group = pr.add_mutually_exclusive_group(required=True)
    group.add_argument("--payload", default=None, help="Single-asset JSON payload path (or '-').")
    group.add_argument("--fleet", default=None, help="Fleet CSV path.")
    pr.add_argument("-o", "--output", default=None, help="Write to file (default: stdout).")
    pr.add_argument("--title", default="Fleet RUL advisory report")
    pr.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
