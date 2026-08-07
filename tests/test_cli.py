"""CLI tests, with a focus on UTF-8 handling.

Reports and prompts contain non-ASCII characters (``σ``, ``⚠️``, ``°C``,
em dashes), and payloads may contain non-ASCII asset ids. Without explicit
UTF-8 handling, the CLI raises ``UnicodeDecodeError``/``UnicodeEncodeError``
on platforms whose default text encoding is not UTF-8 (e.g. Windows
``cp1252`` consoles).
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from src.cli import _emit, _read_text, main
from src.cli_twin import prompt_main, simulate_main, status_main

UTF8_PAYLOAD = {
    "asset_id": "WTG-Énergie-風",
    "telemetry": {
        "vibration_mms": 2.1,
        "temperature_c": 62.0,
        "rpm": 1500.0,
        "oil_viscosity_cst": 31.0,
        "load_pct": 80.0,
    },
    "bnn_state": {
        "predicted_rul_days": 210.0,
        "epistemic_uncertainty": 0.03,
        "aleatoric_uncertainty": 0.06,
    },
}


@pytest.fixture
def utf8_payload_file(tmp_path):
    """A UTF-8-encoded JSON payload whose asset id is non-ASCII."""
    p = tmp_path / "payload_utf8.json"
    p.write_text(json.dumps(UTF8_PAYLOAD, ensure_ascii=False), encoding="utf-8")
    return p


def _ascii_stdout(monkeypatch):
    """Replace stdout with an ASCII-encoded stream (worst-case console)."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    return raw, stream


# --------------------------------------------------------------------------- #
# src.cli helpers                                                             #
# --------------------------------------------------------------------------- #
def test_read_text_uses_utf8(utf8_payload_file):
    text = _read_text(str(utf8_payload_file))
    assert "WTG-Énergie-風" in text


def test_emit_writes_utf8_file(tmp_path):
    out = tmp_path / "report.md"
    _emit("⚠️ σ — em dash", str(out))
    assert "⚠️".encode() in out.read_bytes()
    assert out.read_text(encoding="utf-8") == "⚠️ σ — em dash"


# --------------------------------------------------------------------------- #
# wind-turbine-bnn                                                            #
# --------------------------------------------------------------------------- #
def test_advisory_round_trips_utf8_asset_id(utf8_payload_file, capsys):
    assert main(["advisory", str(utf8_payload_file)]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["asset_id"] == "WTG-Énergie-風"


def test_report_stdout_handles_non_utf8_console(utf8_payload_file, monkeypatch):
    """A non-UTF-8 console must not crash the report subcommand."""
    raw, stream = _ascii_stdout(monkeypatch)
    assert main(["report", "--payload", str(utf8_payload_file)]) == 0
    stream.flush()
    text = raw.getvalue().decode("utf-8")
    assert "—" in text
    assert "⚠️" in text
    assert "WTG-Énergie-風" in text


def test_report_file_output_is_utf8(utf8_payload_file, tmp_path):
    out = tmp_path / "report.md"
    args = ["report", "--payload", str(utf8_payload_file), "-o", str(out)]
    assert main(args) == 0
    text = out.read_text(encoding="utf-8")
    assert "WTG-Énergie-風" in text
    assert "⚠️" in text


def test_fleet_report_utf8_asset_ids(tmp_path):
    csv_path = tmp_path / "fleet.csv"
    csv_path.write_text(
        "asset_id,vibration_mms,temperature_c,rpm,oil_viscosity_cst,load_pct,"
        "predicted_rul_days,epistemic_uncertainty,aleatoric_uncertainty\n"
        "WTG-Åland,2.1,58.0,1500,32.0,80.0,240.0,0.05,0.10\n",
        encoding="utf-8",
    )
    out = tmp_path / "fleet.md"
    assert main(["fleet", str(csv_path), "-o", str(out)]) == 0
    assert "WTG-Åland" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# twin-* CLIs                                                                 #
# --------------------------------------------------------------------------- #


def test_twin_status_handles_utf8_payload(utf8_payload_file, capsys):
    args = ["--asset-id", "WTG-Énergie-風", "--payload", str(utf8_payload_file)]
    assert status_main(args) == 0
    out = capsys.readouterr().out
    assert "WTG-Énergie-風" in out
    assert "°C" in out


def test_twin_status_non_utf8_console(utf8_payload_file, monkeypatch):
    raw, stream = _ascii_stdout(monkeypatch)
    assert status_main(["--payload", str(utf8_payload_file)]) == 0
    stream.flush()
    assert "°C" in raw.getvalue().decode("utf-8")


def test_twin_simulate_writes_utf8_json(tmp_path):
    out = tmp_path / "sim.json"
    assert simulate_main(["--hours", "1", "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data


def test_twin_prompt_writes_utf8_file(utf8_payload_file, tmp_path):
    out = tmp_path / "prompt.txt"
    args = ["--payload", str(utf8_payload_file), "-o", str(out)]
    assert prompt_main(args) == 0
    assert "°C" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Unified `wind-turbine-bnn twin ...` group                                   #
# --------------------------------------------------------------------------- #
def test_unified_cli_twin_status(utf8_payload_file, capsys):
    assert (
        main(
            ["twin", "status", "--asset-id", "WTG-Énergie-風", "--payload", str(utf8_payload_file)]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "WTG-Énergie-風" in out
    assert "°C" in out


def test_unified_cli_twin_status_json(capsys):
    rc = main(["twin", "status", "--asset-id", "WTG-UNI-JSON", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["asset_id"] == "WTG-UNI-JSON"
    assert out["health_state"]["advisory"]["advisory_only"] is True


def test_unified_cli_twin_simulate_writes_utf8_json(tmp_path):
    out = tmp_path / "uni_sim.json"
    assert (
        main(["twin", "simulate", "--asset-id", "WTG-UNI-SIM", "--hours", "2", "-o", str(out)]) == 0
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[-1]["advisory"] is not None


def test_unified_cli_twin_simulate_rejects_bad_hours(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["twin", "simulate", "--hours", "0"])
    assert exc.value.code == 1
    assert "Error:" in capsys.readouterr().err


def test_unified_cli_twin_prompt_writes_utf8_file(utf8_payload_file, tmp_path):
    out = tmp_path / "uni_prompt.txt"
    args = [
        "twin",
        "prompt",
        "--asset-id",
        "WTG-Énergie-風",
        "--payload",
        str(utf8_payload_file),
        "-o",
        str(out),
    ]
    assert main(args) == 0
    assert "°C" in out.read_text(encoding="utf-8")
    assert "WTG-Énergie-風" in out.read_text(encoding="utf-8")


def test_unified_cli_twin_unknown_command_fails():
    with pytest.raises(SystemExit):
        main(["twin", "teleport"])
