"""Advisory report generation (text/markdown) and fleet summaries."""

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

__all__ = [
    "advisories_from_csv",
    "advisories_from_dataframe",
    "build_fleet_report",
    "fleet_summary",
    "fleet_summary_table",
    "format_advisory_markdown",
    "format_advisory_text",
    "write_report",
]
