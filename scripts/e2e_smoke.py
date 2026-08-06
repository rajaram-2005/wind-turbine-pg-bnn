#!/usr/bin/env python3
"""AeroVigil end-to-end smoke test over the FastAPI service.

Exercises every public endpoint with the FastAPI TestClient — `/advisory` in
BOTH modes (bnn_state fallback AND model-serving via AV_MODEL_PATH), the fleet
batch, digital-twin endpoints, telemetry compression endpoints, and the fleet
report. Trains a tiny PG-BNN on the fly (seconds), exports it as a serving
bundle, and points the app at it.

Exit code 0 on success, non-zero on ANY failure. Advisory-only throughout.

Usage:
    python scripts/e2e_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def _build_tiny_bundle(path: str) -> None:
    """Train a tiny PG-BNN (deterministic, ~seconds) and export a bundle."""
    import torch

    from src.data.ingest import CHANNELS, SlidingWindowConfig, robust_normalize, sliding_features
    from src.data.synthetic import SyntheticConfig, generate
    from src.models.bnn import BayesianNeuralNetwork, TrainConfig, elbo_loss
    from src.utils.artifacts import FeatureConfig, save_model_bundle

    torch.manual_seed(0)
    seqs = generate(SyntheticConfig(n_turbines=4, seq_len=600, seed=21))
    Xs, ys = [], []
    sw = SlidingWindowConfig(window_size=40, stride=20)
    for df, rul_end in seqs:
        norm, _ = robust_normalize(df[list(CHANNELS)])
        feats = sliding_features(norm, sw)
        Xs.append(feats)
        ys.append(np.linspace(rul_end + 15.0, max(rul_end, 5.0), len(feats)))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)

    model = BayesianNeuralNetwork(in_features=X.shape[1], hidden_sizes=(16,))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    tcfg = TrainConfig(num_samples=2, kl_weight=1e-3, physics_weight=0.0)
    xt, yt = torch.tensor(X), torch.tensor(y)
    model.train()
    for _ in range(25):
        opt.zero_grad()
        loss, _ = elbo_loss(model, xt, yt, telemetry=None, cfg=tcfg)
        loss.backward()
        opt.step()
    model.eval()

    _, scaler = robust_normalize(seqs[0][0][list(CHANNELS)])
    save_model_bundle(
        model,
        path,
        scaler=scaler,
        features=FeatureConfig(window_size=40, stride=20),
        metadata={"produced_by": "scripts/e2e_smoke.py"},
    )
    return scaler


def main() -> int:
    from src.utils.encoding import configure_utf8_stdio  # noqa: E402

    configure_utf8_stdio()
    from fastapi.testclient import TestClient

    from src.data.ingest import CHANNELS

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "e2e_model.pt")
        print("[setup] training tiny PG-BNN + exporting serving bundle ...")
        _build_tiny_bundle(ckpt)

        os.environ["AV_MODEL_PATH"] = ckpt
        from src.api.app import create_app

        client = TestClient(create_app())

        rng = np.random.default_rng(0)
        n = 40
        t = np.arange(n)
        window = {
            "vibration_mms": (2.0 + 0.4 * np.sin(t / 8.0) + rng.normal(0, 0.05, n)).tolist(),
            "temperature_c": (60.0 + 2.0 * np.sin(t / 12.0) + rng.normal(0, 0.2, n)).tolist(),
            "rpm": (1500.0 + 30.0 * np.sin(t / 10.0) + rng.normal(0, 4.0, n)).tolist(),
            "oil_viscosity_cst": (32.0 - np.sin(t / 9.0) + rng.normal(0, 0.2, n)).tolist(),
            "load_pct": (75.0 + 4.0 * np.sin(t / 7.0) + rng.normal(0, 0.5, n)).tolist(),
        }
        snapshot = {c: float(window[c][-1]) for c in CHANNELS}
        base_payload = {
            "asset_id": "E2E-001",
            "telemetry": snapshot,
            "bnn_state": {
                "predicted_rul_days": 300.0,
                "epistemic_uncertainty": 0.05,
                "aleatoric_uncertainty": 0.1,
            },
        }

        print("\n[1] Core endpoints")
        r = client.get("/")
        body = r.json()
        check("GET /", r.status_code == 200 and body["advisory_only"] is True)
        check("… root advertises serving model", body.get("serving_model_loaded") is True)
        r = client.get("/health")
        body = r.json()
        check(
            "GET /health",
            r.status_code == 200
            and body["status"] == "ok"
            and body["advisory_only"] is True
            and body["serving_model_loaded"] is True,
        )

        print("\n[2] /advisory — bnn_state fallback (backward compatible)")
        r = client.post("/advisory", json=base_payload)
        body = r.json()
        check(
            "bnn_state mode returns exact RUL",
            r.status_code == 200 and body["predicted_rul_days"] == 300.0,
        )
        check("bnn_state mode is advisory-only", body["advisory_only"] is True)

        print("\n[3] /advisory — model-serving mode (telemetry_window)")
        model_payload = dict(base_payload)
        model_payload["telemetry_window"] = window
        r = client.post("/advisory", json=model_payload)
        body = r.json()
        check(
            "model mode computes RUL (≠ bnn_state sentinel)",
            r.status_code == 200 and body["predicted_rul_days"] != 300.0,
        )
        check(
            "model mode RUL in [0, 3650]",
            0.0 <= body.get("predicted_rul_days", -1) <= 3650.0,
            f"RUL={body.get('predicted_rul_days'):.1f}",
        )
        r = client.post(
            "/advisory",
            json={"asset_id": "E2E-NOTHING", "telemetry": snapshot, "telemetry_window": window},
        )
        check("window + no bnn_state + model → 200 (no 422)", r.status_code == 200)

        print("\n[4] /advisory/fleet")
        fleet = {"assets": [base_payload, {**base_payload, "asset_id": "E2E-002"}]}
        r = client.post("/advisory/fleet", json=fleet)
        body = r.json()
        check(
            "fleet batch",
            r.status_code == 200 and len(body["assets"]) == 2 and body["summary"]["n_assets"] == 2,
        )

        print("\n[5] Digital twin endpoints")
        r = client.get("/twin/status", params={"asset_id": "E2E-TWIN", "model": "GE-1.5"})
        body = r.json()
        check(
            "GET /twin/status (with serving model attached)",
            r.status_code == 200
            and body["last_state"]["advisory"] is not None
            and body["last_state"]["advisory_source"] == "model",
        )
        r = client.post(
            "/twin/simulate",
            json={"asset_id": "E2E-TWIN", "model": "GE-1.5", "profile": "nominal", "hours": 2},
        )
        body = r.json()
        check(
            "POST /twin/simulate",
            r.status_code == 200
            and body["steps_executed"] == 2
            and body["advisories_computed"] == 2,
        )
        check(
            "… simulated steps carried model advisories",
            body["last_records"][-1]["advisory_source"] == "model",
        )
        r = client.get("/twin/prompt", params={"asset_id": "E2E-TWIN"})
        body = r.json()
        check(
            "GET /twin/prompt",
            r.status_code == 200
            and "E2E-TWIN" in body["prompt"]
            and "ADVISORY / DECISION-SUPPORT ONLY" in body["prompt"],
        )

        print("\n[6] Telemetry (AeroZip) endpoints")
        r = client.post(
            "/telemetry/compress",
            json={"channels": window, "sample_interval_s": 600},
        )
        body = r.json()
        check(
            "POST /telemetry/compress",
            r.status_code == 200 and body["n_samples"] == n and body["codec"] == "aerozip-v1",
            f"ratio={body.get('ratio'):.3f} anomaly={body.get('anomaly_score'):.3f}",
        )
        r2 = client.post("/telemetry/restore", json={"payload_b64": body["payload_b64"]})
        back = r2.json()
        max_err = max(
            float(np.max(np.abs(np.array(back["channels"][c]) - np.array(window[c]))))
            for c in CHANNELS
        )
        check(
            "POST /telemetry/restore (lossy, rpm quantum bound)",
            r2.status_code == 200 and back["n_samples"] == n,
            f"max rount-trip err={max_err:.3f}",
        )

        print("\n[7] Fleet report endpoint")
        r = client.get("/fleet/report")
        check(
            "GET /fleet/report (markdown, twin-sourced)",
            r.status_code == 200 and r.text.lstrip().startswith("#") and "E2E-TWIN" in r.text,
            f"{len(r.text)} chars of markdown",
        )

        print("\n[8] Safety gate spot-checks")
        r = client.post(
            "/advisory",
            json={
                "asset_id": "E2E-BAD",
                "telemetry": {**snapshot, "vibration_mms": 999.0},
                "bnn_state": base_payload["bnn_state"],
            },
        )
        check("out-of-range telemetry → 422", r.status_code == 422)
        bad_win = dict(window)
        bad_win["rpm"] = bad_win["rpm"][:-1]
        r = client.post(
            "/advisory",
            json={**base_payload, "asset_id": "E2E-BAD2", "telemetry_window": bad_win},
        )
        check("unequal window lengths → 422", r.status_code == 422)

    failures = [(n, d) for n, ok, d in CHECKS if not ok]
    print("\n" + "=" * 70)
    if failures:
        print(f"E2E SMOKE: {len(failures)} FAILURE(S) out of {len(CHECKS)} checks")
        for n, d in failures:
            print(f"  - {n}: {d}")
        return 1
    print(f"E2E SMOKE: all {len(CHECKS)} checks passed ✔ (advisory-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
