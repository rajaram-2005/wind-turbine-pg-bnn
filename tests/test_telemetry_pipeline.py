"""Tests for the AeroZip telemetry pipeline, the ingest compressed-CSV path,
and the /telemetry/* API endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import CHANNELS, load_aerozip_csv, write_aerozip_csv
from src.models.telemetry.aerozip import AeroZipConfig
from src.models.telemetry.pipeline import (
    CompressedWindow,
    compress_window,
    restore_window,
)


def _window(n: int = 120, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return {
        "vibration_mms": 2.0 + 0.5 * np.sin(t / 8.0) + rng.normal(0, 0.05, n),
        "temperature_c": 60.0 + 3.0 * np.sin(t / 14.0) + rng.normal(0, 0.2, n),
        "rpm": 1500.0 + 50.0 * np.sin(t / 10.0) + rng.normal(0, 5.0, n),
        "oil_viscosity_cst": 32.0 - 1.5 * np.sin(t / 12.0) + rng.normal(0, 0.2, n),
        "load_pct": 78.0 + 6.0 * np.sin(t / 9.0) + rng.normal(0, 0.5, n),
    }


def test_round_trip_quantization_bound():
    """With deadband = 0, reconstruction error is pure quantization:
    |restored - original| <= quantum per channel (fresh single window)."""
    cfg = AeroZipConfig(deadbands=dict.fromkeys(CHANNELS, 0.0))
    win = _window()
    comp = compress_window(win, cfg=cfg)
    assert comp.bypass is False
    rest = restore_window(comp)
    errs = rest.max_abs_error(win)
    for ch in CHANNELS:
        assert errs[ch] <= cfg.quanta[ch] + 1e-9, f"{ch}: {errs[ch]}"


def test_round_trip_default_deadband_smoke():
    """Default config round-trips with error bounded per channel. The bound
    covers quantization + deadband drift within one 60-sample window."""
    cfg = AeroZipConfig()
    win = _window(n=60)
    comp = compress_window(win, cfg=cfg)
    rest = restore_window(comp)
    errs = rest.max_abs_error(win)
    n = win["rpm"].shape[0]
    for ch in CHANNELS:
        # Worst case: every sample creeps one suppressed quantum step.
        worst = cfg.quanta[ch] * 0.5 + n * cfg.deadbands[ch]
        assert errs[ch] <= worst
        # Realistic bound for varied telemetry (what the demo data shows):
        assert errs[ch] <= 3 * cfg.deadbands[ch] + cfg.quanta[ch], f"{ch}: {errs[ch]}"


def test_round_trip_via_dict_form():
    win = _window(n=60)
    comp = compress_window(win)
    rest = restore_window(comp.to_dict())
    assert rest.n_samples == 60
    assert set(rest.channels) == set(CHANNELS)


def test_anomaly_bypass_is_lossless_and_surfaces_score():
    win = _window(n=60)
    # Flat baseline far below the window values → high anomaly score.
    base_mean = dict.fromkeys(CHANNELS, 0.0)
    base_std = dict.fromkeys(CHANNELS, 0.01)
    comp = compress_window(win, baseline_mean=base_mean, baseline_std=base_std)
    assert comp.anomaly_score > 0.75
    assert comp.bypass is True
    rest = restore_window(comp)
    assert rest.bypass is True
    assert rest.anomaly_score == pytest.approx(comp.anomaly_score, rel=1e-5)
    errs = rest.max_abs_error(win)
    for ch in CHANNELS:
        assert errs[ch] == 0.0  # lossless raw float64


def test_compressed_window_serialization():
    win = _window(n=30)
    comp = compress_window(win)
    d = comp.to_dict()
    clone = CompressedWindow.from_dict(d)
    assert clone.payload_b64 == comp.payload_b64
    assert clone.ratio == comp.ratio
    assert clone.n_samples == 30
    assert d["codec"] == "aerozip-v1"


def test_compress_rejects_bad_input():
    with pytest.raises(ValueError):
        compress_window({"vibration_mms": np.array([1.0])})
    with pytest.raises(ValueError, match="equal"):
        compress_window({c: np.array([1.0] * (10 if c == "rpm" else 11)) for c in CHANNELS})
    with pytest.raises(ValueError, match="empty"):
        compress_window({c: np.array([]) for c in CHANNELS})


def test_restore_rejects_garbage():
    with pytest.raises(ValueError):
        restore_window({"payload_b64": "bm90LWF6", "channels": list(CHANNELS)})


# --------------------------------------------------------------------------- #
# Ingest: compressed CSV                                                      #
# --------------------------------------------------------------------------- #
def test_aerozip_csv_round_trip(tmp_path):
    n = 200
    idx = pd.date_range("2025-03-01", periods=n, freq="10min")
    win = _window(n=n)
    df = pd.DataFrame(win, index=idx)
    path = tmp_path / "w.aerozip.csv"
    write_aerozip_csv(df, str(path), window_samples=60, sample_interval_s=600)

    raw = pd.read_csv(path)
    assert len(raw) == 4  # 200 samples / 60 per window
    assert set(raw.columns) == {
        "timestamp",
        "sample_interval_s",
        "anomaly_score",
        "bypass",
        "payload_b64",
    }

    restored = load_aerozip_csv(str(path))
    assert len(restored) == n
    assert list(restored.columns) == list(CHANNELS)
    assert restored.index[0] == idx[0]
    assert restored.index[-1] == idx[-1]
    cfg = AeroZipConfig()
    for ch in CHANNELS:
        err = float(np.max(np.abs(restored[ch].to_numpy() - win[ch])))
        assert err <= n * cfg.deadbands[ch] + cfg.quanta[ch]


def test_aerozip_csv_rejects_missing_columns(tmp_path):
    df = pd.DataFrame({"vibration_mms": [1.0], "temperature_c": [1.0]})
    with pytest.raises(ValueError, match="Missing"):
        write_aerozip_csv(df, str(tmp_path / "x.csv"))

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2025-01-01"], "payload_b64": ["x"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_aerozip_csv(str(bad))


# --------------------------------------------------------------------------- #
# API endpoints                                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def api_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    return TestClient(create_app())


def test_api_telemetry_compress_restore_round_trip(api_client):
    win = _window(n=60)
    resp = api_client.post(
        "/telemetry/compress",
        json={"channels": {c: win[c].tolist() for c in CHANNELS}, "sample_interval_s": 600},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["codec"] == "aerozip-v1"
    assert body["n_samples"] == 60
    assert 0.0 < body["ratio"] < 1.5
    assert body["advisory_only"] is True
    assert body["bypass"] is False

    resp2 = api_client.post("/telemetry/restore", json={"payload_b64": body["payload_b64"]})
    assert resp2.status_code == 200
    back = resp2.json()
    assert back["n_samples"] == 60
    assert back["advisory_only"] is True
    assert set(back["channels"]) == set(CHANNELS)
    cfg = AeroZipConfig()
    for ch in CHANNELS:
        err = float(np.max(np.abs(np.array(back["channels"][ch]) - win[ch])))
        worst = cfg.quanta[ch] * 0.5 + 60 * cfg.deadbands[ch]
        assert err <= worst


def test_api_telemetry_compress_bypass(api_client):
    win = _window(n=60)
    resp = api_client.post(
        "/telemetry/compress",
        json={
            "channels": {c: win[c].tolist() for c in CHANNELS},
            "baseline_mean": dict.fromkeys(CHANNELS, 0.0),
            "baseline_std": dict.fromkeys(CHANNELS, 0.01),
        },
    )
    assert resp.json()["bypass"] is True
    back = api_client.post(
        "/telemetry/restore", json={"payload_b64": resp.json()["payload_b64"]}
    ).json()
    for ch in CHANNELS:
        np.testing.assert_allclose(back["channels"][ch], win[ch], rtol=0, atol=0)


def test_api_telemetry_endpoints_validate_input(api_client):
    # Unequal channel lengths → 422
    win = {c: [1.0] * 10 for c in CHANNELS}
    win["rpm"] = [1.0] * 11
    resp = api_client.post("/telemetry/compress", json={"channels": win})
    assert resp.status_code == 422

    # Garbage payload → 422
    resp2 = api_client.post("/telemetry/restore", json={"payload_b64": "bm90LWF6"})
    assert resp2.status_code == 422

    # Unknown channel in restore request → 422
    win2 = _window(n=30)
    good = api_client.post(
        "/telemetry/compress", json={"channels": {c: win2[c].tolist() for c in CHANNELS}}
    ).json()
    resp3 = api_client.post(
        "/telemetry/restore",
        json={"payload_b64": good["payload_b64"], "channels": ["not_a_channel"]},
    )
    assert resp3.status_code == 422
