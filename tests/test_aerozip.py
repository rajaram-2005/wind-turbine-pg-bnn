import numpy as np

from src.models.telemetry.aerozip import AeroZipCompressor, AeroZipConfig, anomaly_score


def _window(n=100):
    return {
        "vibration_mms": np.ones(n) * 1.5,
        "temperature_c": np.ones(n) * 60.0,
        "rpm": np.ones(n) * 1700.0,
        "oil_viscosity_cst": np.ones(n) * 28.0,
        "load_pct": np.ones(n) * 75.0,
    }


def test_compress_decompress_roundtrip_stats_normal():
    az = AeroZipCompressor()
    w = _window(200)
    # Add small noise so deltas exist but no anomaly
    rng = np.random.default_rng(0)
    for k in w:
        w[k] = w[k] + rng.normal(0, 0.001, size=w[k].shape)
    stats = az.compress_stats(w)
    assert stats["ratio"] < 1.0  # actually compresses
    assert stats["bypass"] is False


def test_anomaly_bypass_sets_flag():
    az = AeroZipCompressor(cfg=AeroZipConfig(anomaly_bypass_threshold=0.5))
    w = _window(200)
    w["vibration_mms"][:] = 12.0  # massive spike
    blob = az.compress(w, anomaly_score_value=0.9)
    hdr = AeroZipCompressor.inspect_header(blob)
    assert hdr["bypass"] is True
    assert abs(hdr["anomaly_score"] - 0.9) < 1e-5


def test_anomaly_score_high_on_spike():
    w = _window(50)
    baseline_mean = {k: float(v.mean()) for k, v in w.items()}
    baseline_std = {k: 1.0 for k in w}
    s_normal = anomaly_score(w, baseline_mean, baseline_std)
    w["vibration_mms"][25] = 50.0
    s_spike = anomaly_score(w, baseline_mean, baseline_std)
    assert s_spike > s_normal
    assert s_spike > 0.75
