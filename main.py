#!/usr/bin/env python3
"""Unified CLI for the Physics-Guided AI wind-turbine framework.

Subcommands:
    train          Train the PG-BNN with the combined physics+data loss.
    evaluate       Evaluate a trained model (RMSE, NLL, calibration stats).
    export         Export the trained BNN to ONNX (mean + variance heads).
    active-sample  Run uncertainty sampling on a SCADA batch, emit alerts.
    explain        Generate a physics-grounded SHAP explainability report.
    federated      Fleet-wide federated-averaging simulation across farms.

Examples:
    python main.py train --config configs/default.yaml --epochs 50
    python main.py evaluate --checkpoint artifacts/pg_bnn.pt
    python main.py export --checkpoint artifacts/pg_bnn.pt --out artifacts/pg_bnn.onnx
    python main.py active-sample --checkpoint artifacts/pg_bnn.pt
    python main.py explain --checkpoint artifacts/pg_bnn.pt
    python main.py federated --rounds 3 --clients 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.bayesian_nn import PGBNNLoss, PhysicsGuidedBNN, train_step  # noqa: E402
from src.physics.aerodynamics import aerodynamic_physics_loss  # noqa: E402

logger = logging.getLogger("pg_ai")


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging for the CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_yaml_config(path: str) -> dict:
    """Load the YAML config as a plain dict (schema-validated lazily)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(name: str) -> torch.device:
    """Resolve 'auto' to cuda when available, else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg: dict) -> PhysicsGuidedBNN:
    """Instantiate the PG-BNN from the ``model:`` config section."""
    m = cfg.get("model", {})
    return PhysicsGuidedBNN(
        in_features=int(m.get("in_features", 6)),
        hidden_dims=list(m.get("hidden_dims", [128, 128, 64])),
        out_features=int(m.get("out_features", 1)),
        prior_sigma=float(m.get("prior_sigma", 1.0)),
        dropout=float(m.get("dropout", 0.1)),
    )


def make_synthetic_scada(
    n: int, in_features: int, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a physically-plausible synthetic SCADA dataset.

    Features: wind_speed, rotor_speed, generator_temp, vibration_rms,
    oil_viscosity, power_output (order matches configs/default.yaml).
    Target: normalised turbine power (a stand-in for the health target so
    the CLI runs end-to-end without proprietary data).
    """
    g = torch.Generator().manual_seed(seed)
    wind = 4.0 + 10.0 * torch.rand(n, generator=g)
    rotor = 0.8 + 1.2 * torch.rand(n, generator=g)  # rad/s
    gen_temp = 50.0 + 40.0 * torch.rand(n, generator=g)
    vib = 1.0 + 4.0 * torch.rand(n, generator=g)
    visc = 20.0 + 20.0 * torch.rand(n, generator=g)
    power = 0.4 * wind.pow(3) * (1.0 - 0.02 * vib) + 5.0 * torch.randn(n, generator=g)
    x = torch.stack([wind, rotor, gen_temp, vib, visc, power], dim=1)[:, :in_features]
    y = (power / power.std()).unsqueeze(1)
    return x, y


def cmd_train(args: argparse.Namespace) -> int:
    """Train the PG-BNN with L = NLL + beta*KL + lambda_physics*L_physics."""
    cfg = load_yaml_config(args.config)
    tr, ph = cfg.get("training", {}), cfg.get("physics", {})
    epochs = args.epochs or int(tr.get("epochs", 100))
    device = resolve_device(str(tr.get("device", "auto")))
    torch.manual_seed(int(tr.get("seed", 0)))

    model = build_model(cfg).to(device)
    x, y = make_synthetic_scada(4096, model.in_features, seed=int(tr.get("seed", 0)))
    x, y = x.to(device), y.to(device)
    batch = int(tr.get("batch_size", 256))
    num_batches = max(x.shape[0] // batch, 1)

    loss_fn = PGBNNLoss(
        beta_kl=float(tr.get("beta_kl", 1e-3)),
        lambda_physics=float(ph.get("lambda_aero", 0.1)),
        num_batches=num_batches,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr.get("lr", 1e-3)))
    rotor_radius = float(ph.get("rotor_radius", 60.0))
    air_density = float(ph.get("air_density", 1.225))
    power_scale = 0.5 * air_density * 3.14159 * rotor_radius**2 * 14.0**3

    def physics_fn(pred_mean: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Aerodynamic consistency of the (denormalised) power prediction."""
        return aerodynamic_physics_loss(
            predicted_power_w=pred_mean.squeeze(-1) * power_scale * 1e-3,
            wind_speed_ms=xb[:, 0],
            rotor_speed_rad_s=xb[:, 1],
            pitch_deg=torch.zeros_like(xb[:, 0]),
            rotor_radius_m=rotor_radius,
            air_density=air_density,
        )

    for epoch in range(epochs):
        perm = torch.randperm(x.shape[0], device=device)
        stats = {}
        for b in range(num_batches):
            idx = perm[b * batch : (b + 1) * batch]
            stats = train_step(
                model, loss_fn, optimizer, x[idx], y[idx],
                physics_fn=physics_fn,
                num_mc_samples=int(tr.get("num_mc_samples", 2)),
            )
        if epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1:
            logger.info(
                "epoch %3d/%d  total=%.4f nll=%.4f kl=%.2f physics=%.4f",
                epoch + 1, epochs, stats["total"], stats["nll"], stats["kl"], stats["physics"],
            )

    out = Path(args.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.get("model", {})}, out)
    logger.info("saved checkpoint to %s", out)
    return 0


def _load_checkpoint(cfg: dict, checkpoint: str) -> PhysicsGuidedBNN:
    """Rebuild the model and load trained weights."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.update(ckpt.get("config", {}))
    model = build_model({"model": model_cfg})
    model.load_state_dict(ckpt["state_dict"])
    return model


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a trained model on held-out synthetic SCADA data."""
    cfg = load_yaml_config(args.config)
    model = _load_checkpoint(cfg, args.checkpoint)
    x, y = make_synthetic_scada(1024, model.in_features, seed=42)

    num_samples = int(cfg.get("training", {}).get("predict_mc_samples", 64))
    pred = model.predict(x, num_samples=num_samples)
    rmse = float(torch.sqrt(torch.mean((pred["mean"] - y) ** 2)))
    inside = ((y - pred["mean"]).abs() <= 1.96 * pred["total_std"]).float().mean()
    report = {
        "rmse": rmse,
        "mean_aleatoric_std": float(pred["aleatoric_std"].mean()),
        "mean_epistemic_std": float(pred["epistemic_std"].mean()),
        "empirical_95ci_coverage": float(inside),
        "mc_samples": num_samples,
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export the trained BNN to ONNX with mean + variance heads."""
    from src.deployment.export_onnx import export_bnn_to_onnx

    cfg = load_yaml_config(args.config)
    dep = cfg.get("deployment", {})
    model = _load_checkpoint(cfg, args.checkpoint)
    out = export_bnn_to_onnx(
        model,
        args.out or str(dep.get("onnx_path", "artifacts/pg_bnn.onnx")),
        opset=int(dep.get("onnx_opset", 17)),
        validate=bool(dep.get("validate_export", True)),
    )
    logger.info("ONNX model written to %s", out)
    return 0


def cmd_active_sample(args: argparse.Namespace) -> int:
    """Flag high-epistemic-uncertainty SCADA samples and write the alert log."""
    from src.active_learning.uncertainty_sampler import UncertaintySampler

    cfg = load_yaml_config(args.config)
    al = cfg.get("active_learning", {})
    model = _load_checkpoint(cfg, args.checkpoint)
    x, _ = make_synthetic_scada(512, model.in_features, seed=7)
    # Inject out-of-distribution rows to demonstrate flagging.
    x[:16] = x[:16] * 3.0 + 10.0

    sampler = UncertaintySampler(
        model,
        uncertainty_threshold=float(al.get("uncertainty_threshold", 0.5)),
        num_mc_samples=int(al.get("num_mc_samples", 32)),
        sample_budget=int(al.get("sample_budget", 64)),
        use_mc_dropout=bool(al.get("use_mc_dropout", True)),
    )
    result = sampler.query(x, log_features=True)
    path = sampler.write_alert_log(
        args.out or str(al.get("alert_log_path", "artifacts/maintenance_alerts.json"))
    )
    logger.info("flagged %d samples; alert log: %s", result["num_alerts"], path)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Produce a physics-grounded SHAP explainability report."""
    from src.explainability.physics_shap import PhysicsSHAP

    cfg = load_yaml_config(args.config)
    ex = cfg.get("explainability", {})
    model = _load_checkpoint(cfg, args.checkpoint)
    names = list(ex.get("feature_names", []))[: model.in_features] or [
        f"feature_{i}" for i in range(model.in_features)
    ]
    background, _ = make_synthetic_scada(
        int(ex.get("background_samples", 128)), model.in_features, seed=1
    )
    x, _ = make_synthetic_scada(32, model.in_features, seed=99)

    explainer = PhysicsSHAP(model, names, background)
    report = explainer.explain(x)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        logger.info("report written to %s", args.out)
    return 0


def cmd_federated(args: argparse.Namespace) -> int:
    """Run a local fleet-wide federated-averaging simulation.

    Simulates ``--clients`` farms training locally for ``--local-epochs`` and
    aggregating with FedAvg for ``--rounds`` rounds. This runs end-to-end
    without a live Flower server so the job queue can exercise the federated
    path; when ``flwr`` and a reachable ``--server`` are available it instead
    connects a real Flower client via ``start_client``.
    """
    from src.federated.fed_client import FederatedConfig

    cfg = load_yaml_config(args.config)
    tr, ph = cfg.get("training", {}), cfg.get("physics", {})
    device = resolve_device(str(tr.get("device", "auto")))
    torch.manual_seed(int(tr.get("seed", 0)))

    fed_cfg = FederatedConfig(
        num_rounds=int(args.rounds),
        min_clients=int(args.clients),
        local_epochs=int(args.local_epochs),
        lr=float(tr.get("lr", 1e-3)),
        batch_size=int(tr.get("batch_size", 256)),
        beta_kl=float(tr.get("beta_kl", 1e-3)),
        lambda_physics=float(ph.get("lambda_aero", 0.1)),
        server_address=str(args.server),
    )

    # Try a real Flower client only when explicitly pointed at a server.
    if args.server and args.connect:
        try:
            from src.federated.fed_client import FlowerFederatedClient, start_client

            model = build_model(cfg).to(device)
            x, y = make_synthetic_scada(2048, model.in_features, seed=0)
            split = x.shape[0] // 2
            client = FlowerFederatedClient(
                model,
                (x[:split], y[:split]),
                (x[split:], y[split:]),
                config=fed_cfg,
            )
            logger.info("connecting to Flower server at %s", fed_cfg.server_address)
            start_client(client)
            return 0
        except Exception as exc:  # pragma: no cover - needs live server/flwr
            logger.warning("real federated client unavailable (%s); "
                           "falling back to local simulation", exc)

    # Local FedAvg simulation across synthetic farms (always runnable).
    n_clients = max(int(args.clients), 1)
    global_model = build_model(cfg).to(device)
    farms = []
    for c in range(n_clients):
        xc, yc = make_synthetic_scada(1024, global_model.in_features, seed=c + 1)
        farms.append((xc.to(device), yc.to(device)))

    batch = fed_cfg.batch_size
    for rnd in range(fed_cfg.num_rounds):
        local_states = []
        weights = []
        for c, (xc, yc) in enumerate(farms):
            local = build_model(cfg).to(device)
            local.load_state_dict(global_model.state_dict())
            num_batches = max(xc.shape[0] // batch, 1)
            loss_fn = PGBNNLoss(fed_cfg.beta_kl, fed_cfg.lambda_physics, num_batches)
            optimizer = torch.optim.Adam(local.parameters(), lr=fed_cfg.lr)
            stats = {}
            for _ in range(fed_cfg.local_epochs):
                perm = torch.randperm(xc.shape[0], device=device)
                for b in range(num_batches):
                    idx = perm[b * batch : (b + 1) * batch]
                    stats = train_step(local, loss_fn, optimizer, xc[idx], yc[idx])
            local_states.append({k: v.detach().clone() for k, v in local.state_dict().items()})
            weights.append(xc.shape[0])
            logger.info("round %d farm %d local nll=%.4f", rnd + 1, c, stats.get("nll", float("nan")))

        # Weighted FedAvg aggregation.
        total = float(sum(weights))
        agg = {}
        for key in global_model.state_dict().keys():
            acc = None
            for w, state in zip(weights, local_states):
                term = state[key].float() * (w / total)
                acc = term if acc is None else acc + term
            agg[key] = acc.to(global_model.state_dict()[key].dtype)
        global_model.load_state_dict(agg)
        logger.info("round %d/%d aggregated %d farms", rnd + 1, fed_cfg.num_rounds, n_clients)

    out = Path(args.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": global_model.state_dict(), "config": cfg.get("model", {})}, out)
    logger.info("saved federated global model to %s", out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argparse CLI with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Physics-Guided AI framework for wind-turbine predictive maintenance",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
        p.add_argument(
            "--checkpoint", default="artifacts/pg_bnn.pt", help="model checkpoint path"
        )

    p_train = sub.add_parser("train", help="train the PG-BNN")
    common(p_train)
    p_train.add_argument("--epochs", type=int, default=None, help="override epochs")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a trained model")
    common(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_export = sub.add_parser("export", help="export to ONNX")
    common(p_export)
    p_export.add_argument("--out", default=None, help="output .onnx path")
    p_export.set_defaults(func=cmd_export)

    p_active = sub.add_parser("active-sample", help="uncertainty sampling + alerts")
    common(p_active)
    p_active.add_argument("--out", default=None, help="alert log output path")
    p_active.set_defaults(func=cmd_active_sample)

    p_explain = sub.add_parser("explain", help="physics-SHAP explanation report")
    common(p_explain)
    p_explain.add_argument("--out", default=None, help="report output path")
    p_explain.set_defaults(func=cmd_explain)

    p_fed = sub.add_parser("federated", help="fleet-wide federated FedAvg simulation")
    common(p_fed)
    p_fed.add_argument("--rounds", type=int, default=3, help="federated rounds")
    p_fed.add_argument("--clients", type=int, default=2, help="number of farms")
    p_fed.add_argument("--local-epochs", type=int, default=1, help="local epochs/round")
    p_fed.add_argument("--server", default="", help="Flower server host:port")
    p_fed.add_argument("--connect", action="store_true",
                       help="connect to a live Flower server instead of simulating")
    p_fed.set_defaults(func=cmd_federated)
    return parser


def main(argv: Optional[list] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
