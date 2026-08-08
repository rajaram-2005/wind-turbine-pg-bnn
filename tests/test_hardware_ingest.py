"""Tests for safe USB/cloud hardware telemetry intake."""

import pytest

from src.hardware.ingest import import_hardware_telemetry, parse_telemetry_csv

CSV = """timestamp,vibration_mms,temperature_c,rpm,oil_viscosity_cst,load_pct
2026-01-01T00:00:00Z,2.1,62,1500,32,70
2026-01-01T00:10:00Z,2.4,64,1510,31,74
"""


def test_usb_csv_preview_uses_latest_snapshot():
    imported = import_hardware_telemetry(source="usb", csv_text=CSV)
    assert imported["source"] == "usb"
    assert imported["rows_imported"] == 2
    assert imported["latest_telemetry"]["load_pct"] == 74.0
    assert imported["advisory_only"] is True


def test_hardware_csv_requires_all_advisory_channels():
    with pytest.raises(ValueError, match="missing hardware telemetry columns"):
        parse_telemetry_csv("vibration_mms,temperature_c\n2.0,60\n")


def test_cloud_input_requires_url_before_any_fetch():
    with pytest.raises(ValueError, match="cloud import requires cloud_url"):
        import_hardware_telemetry(source="cloud")
