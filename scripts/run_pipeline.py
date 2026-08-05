#!/usr/bin/env python3
"""AeroVigil end-to-end pipeline: config → synthetic fleet → train → eval →
export → model-based advisory smoke → artifacts/pipeline_report.md.

This is the "connect all the points" demo: every stage is driven by
configs/default.yaml and reuses the production modules (data.ingest,
data.synthetic, models.bnn, eval.calibration, utils.artifacts,
models.serving, models.predictor, utils.safety). Advisory-only, as always.

Usage:
    python scripts/run_pipeline.py                     # default (~1 min)
    python scripts/run_pipeline.py --turbines 10 --epochs 60
    python scripts/run_pipeline.py --config configs/default.yaml --out-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.ingest import CHANNELS, robust_normalize  # noqa: E402
from src.data.synthetic import SyntheticConfig, features_and_labels, generate  # noqa: E402
from src.eval.calibration import (  # noqa: E402
    EARLY_WARNING_HORIZON_DAYS,
    early_warning_metrics,
    expected_calibration_error,
)
from src.models.bnn import (  # noqa: E402
    BayesianNeuralNetwork,
    TrainConfig,
    elbo_loss,
    predict,
)
from src.models.serving import load_serving_model  # noqa: E402
from src.reporting.reports import format_advisory_markdown  # noqa: E402
from src.utils.artifacts import FeatureConfig, save_model_bundle  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.schema import Telemetry, TurbinePayload  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="config YAML (default: configs/default.yaml)")
    parser.add_argument("--turbines", type=int, default=8, help="synthetic fleet size")
    parser.add_argument("--seq-len", type=int, default=1200, help="samples per turbine")
    parser.add_argument("--epochs", type=int, default=40, help="training epochs")
    parser.add_argument("--out-dir", default="artifacts", help="artifact registry dir")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    started = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)

    # ------------------------------------------------------------------ #
    print("[1/6] Loading configuration ...")
    cfg = load_config(args.config)
    print(f"      safety.mode={cfg.safety.mode}  allow_actuation={cfg.safety.allow_actuation}")
    print(f"      horizon={cfg.eval.early_warning_horizon_days}d  "
          f"hidden={cfg.bnn.hidden}  window={cfg.telemetry.window_size_samples} samples")

    # ------------------------------------------------------------------ #
    print(f"[2/6] Generating synthetic fleet ({args.turbines} turbines x {args.seq_len} samples) ...")
    syn_cfg = SyntheticConfig(n_turbines=args.turbines, seq_len=args.seq_len, seed=42)
    X, y = features_and_labels(syn_cfg)
    print(f"      windows={len(X)}  feature_dim={X.shape[1]}  "
          f"RUL range=[{y.min():.0f}, {y.max():.0f}] days")

    # ------------------------------------------------------------------ #
    print(f"[3/6] Training PG-BNN ({args.epochs} epochs) ...")
    n_train = int(0.85 * len(X))
    perm = np.random.default_rng(0).permutation(len(X))
    Xtr, ytr = X[perm[:n_train]], y[perm[:n_train]]
    Xte, yte = X[perm[n_train:]], y[perm[n_train:]]
    model = BayesianNeuralNetwork(
        in_features=X.shape[1],
        hidden_sizes=tuple(cfg.bnn.hidden),
        prior_sigma=cfg.bnn.prior_sigma,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.bnn.train.lr)
    tcfg = TrainConfig(
        num_samples=5,
        kl_weight=cfg.bnn.train.kl_weight,
        physics_weight=0.0,
        batch_size=cfg.bnn.train.batch_size,
    )
    dl = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=tcfg.batch_size, shuffle=True,
    )
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss, _ = elbo_loss(model, xb, yb, telemetry=None, cfg=tcfg)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if (epoch + 1) % max(1, args.epochs // 4) == 0:
            print(f"      epoch {epoch + 1:3d}  loss={tot / len(Xtr):.3f}")

    # ------------------------------------------------------------------ #
    print("[4/6] Evaluating ...")
    model.eval()
    with torch.no_grad():
        out = predict(model, torch.tensor(Xte), mc_samples=cfg.bnn.predict_mc_samples)
    pred = out["mean_pred"].numpy()
    rmse = float(np.sqrt(((pred - yte) ** 2).mean()))
    ece = expected_calibration_error(yte, pred, out["total_std"].numpy(), n_bins=10)
    ew = early_warning_metrics(yte, pred, warning_horizon_days=EARLY_WARNING_HORIZON_DAYS)
    print(f"      RMSE={rmse:.2f} d  ECE={ece:.3f}")
    print(f"      early-warning @ {EARLY_WARNING_HORIZON_DAYS:.0f}d: "
          f"accuracy={ew['accuracy']:.3f} recall={ew['recall']:.3f} "
          f"false-alarm={ew['false_alarm_rate']:.3f}")

    # ------------------------------------------------------------------ #
    print(f"[5/6] Exporting serving bundle to {args.out_dir} ...")
    ref_df, _ = generate(SyntheticConfig(n_turbines=1, seq_len=args.seq_len, seed=0))[0]
    _, ref_scaler = robust_normalize(ref_df[list(CHANNELS)])
    ckpt_path = os.path.join(args.out_dir, "pipeline_model.pt")
    bundle = save_model_bundle(
        model,
        ckpt_path,
        scaler=ref_scaler,
        features=FeatureConfig(),
        metadata={
            "produced_by": "scripts/run_pipeline.py",
            "rmse_days": round(rmse, 3),
            "ece": round(ece, 4),
            "early_warning_accuracy": round(ew["accuracy"], 4),
            "fleet": f"synthetic n={args.turbines} seq_len={args.seq_len} seed=42",
        },
    )
    print(f"      checkpoint: {bundle.checkpoint_path}")
    print(f"      sidecar:    {bundle.sidecar_path}")

    # ------------------------------------------------------------------ #
    print("[6/6] Model-based advisory smoke (run_advisory via serving path) ...")
    serving = load_serving_model(bundle.checkpoint_path)
    snap = ref_df.iloc[-1]
    payload = TurbinePayload(
        asset_id="PIPELINE-SMOKE-1",
        telemetry=Telemetry(**{c: float(np.clip(snap[c], 1e-3, None)) for c in CHANNELS}),
    )
    advisory = serving.advisory(payload, ref_df[list(CHANNELS)])
    print(f"      advisory RUL={advisory['predicted_rul_days']:.1f} d "
          f"(epistemic σ={advisory['epistemic_std']:.3f}, aleatoric σ={advisory['aleatoric_std']:.3f})")
    print(f"      advisory_only={advisory['advisory_only']}  "
          f"early_warning={advisory['early_warning_triggered']}")

    # ------------------------------------------------------------------ #
    report_md = f"""# AeroVigil end-to-end pipeline report

> ⚠️ **ADVISORY ONLY — decision-support, not an actuation command.**
> Generated: {started.isoformat()}

## Configuration (configs/default.yaml)

| Section | Values |
| --- | --- |
| safety | mode=`{cfg.safety.mode}`, allow_actuation=`{cfg.safety.allow_actuation}` |
| physics.gearbox | vib≤{cfg.physics.gearbox.vibration_limit_mms} mm/s, temp≤{cfg.physics.gearbox.temperature_limit_c} °C, rpm≤{cfg.physics.gearbox.rpm_limit_hss}, visc∈[{cfg.physics.gearbox.viscosity_min_cst}, {cfg.physics.gearbox.viscosity_max_cst}] cSt |
| bnn | hidden={cfg.bnn.hidden}, prior σ={cfg.bnn.prior_sigma}, predict_mc_samples={cfg.bnn.predict_mc_samples} |
| telemetry | window={cfg.telemetry.window_s}s / stride={cfg.telemetry.window_stride_s}s / interval={cfg.telemetry.sample_interval_s}s |
| eval | early_warning_horizon={cfg.eval.early_warning_horizon_days} days |

## Data & training

- Synthetic fleet: {args.turbines} turbines × {args.seq_len} samples (seed 42) → {len(X)} windows (dim {X.shape[1]})
- Train/test split: {len(Xtr)}/{len(Xte)} windows, {args.epochs} epochs, Adam lr={cfg.bnn.train.lr}

## Evaluation

| Metric | Value |
| --- | ---: |
| RMSE | {rmse:.2f} days |
| ECE (10 bins) | {ece:.3f} |
| Early-warning accuracy @ {EARLY_WARNING_HORIZON_DAYS:.0f} d | {ew['accuracy']:.3f} |
| Early-warning recall | {ew['recall']:.3f} |
| False-alarm rate | {ew['false_alarm_rate']:.3f} |

## Exported artifacts

- Checkpoint: `{bundle.checkpoint_path}` (+ JSON sidecar `{bundle.sidecar_path}`)
- Loaded back via `src.models.serving.load_serving_model` ✔

## Model-based advisory smoke

{format_advisory_markdown(advisory)}
"""
    report_path = os.path.join(args.out_dir, "pipeline_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(f"\nReport written to {report_path}")
    print("Pipeline complete. Advisory-only: no actuation fields anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
