"""Demo training script: fit PG-BNN on synthetic drivetrain telemetry.

Trains on synthetic data and exports a *serving bundle* (checkpoint + fitted
robust scaler + JSON sidecar) to the artifact registry so the API/UI/CLI can
load it back — see src/utils/artifacts.py and src/models/serving.py.

Usage:
    python scripts/train_demo.py                      # writes artifacts/bnn_demo.pt
    python scripts/train_demo.py --out artifacts/bnn_demo.pt
"""

from __future__ import annotations
from src.utils.encoding import configure_utf8_stdio
configure_utf8_stdio()
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.ingest import CHANNELS, robust_normalize  # noqa: E402
from src.data.synthetic import SyntheticConfig, features_and_labels, generate  # noqa: E402
from src.eval.calibration import expected_calibration_error  # noqa: E402
from src.models.bnn import (  # noqa: E402
    BayesianNeuralNetwork,
    TrainConfig,
    elbo_loss,
    predict,
)
from src.utils.artifacts import FeatureConfig, save_model_bundle  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=os.path.join("artifacts", "bnn_demo.pt"),
        help="checkpoint path inside the artifact registry (default: artifacts/bnn_demo.pt)",
    )
    args = parser.parse_args(argv)

    device = "cpu"
    cfg = load_config()
    print("[1/5] Generating synthetic drivetrain telemetry ...")
    X, y = features_and_labels(SyntheticConfig(n_turbines=20, seq_len=1500))
    n = len(X)
    n_train = int(0.8 * n)
    perm = np.random.default_rng(0).permutation(n)
    Xtr, ytr = X[perm[:n_train]], y[perm[:n_train]]
    Xte, yte = X[perm[n_train:]], y[perm[n_train:]]
    print(f"    train={len(Xtr)}  test={len(Xte)}  feature_dim={X.shape[1]}")

    # Reference scaler for serving-time normalization. Each turbine is
    # robust-normalized individually at train time; for a single served model
    # we persist the scaler of one representative turbine so the feature
    # pipeline is reproducible end-to-end (documented approximation).
    ref_df, _ref_rul = generate(SyntheticConfig(n_turbines=1, seq_len=1500, seed=0))[0]
    _, ref_scaler = robust_normalize(ref_df[list(CHANNELS)])

    hidden = tuple(cfg.bnn.hidden)
    model = BayesianNeuralNetwork(in_features=X.shape[1], hidden_sizes=hidden,
                                  prior_sigma=cfg.bnn.prior_sigma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tcfg = TrainConfig(num_epochs=100, num_samples=5, batch_size=256)

    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True)

    print("[2/5] Training PG-BNN (MCVI + physics penalty) ...")
    for epoch in range(tcfg.num_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            # For the demo we don't pass raw telemetry into elbo_loss (we have
            # normalized features), so physics coupling acts weakly via prior.
            opt.zero_grad()
            loss, bd = elbo_loss(model, xb, yb, telemetry=None, cfg=tcfg)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}  loss={total_loss/len(ds):.3f}  "
                  f"nll={bd['nll']:.3f} kl={bd['kl']:.3f}")

    print("[3/5] Evaluating on test split ...")
    model.eval()
    with torch.no_grad():
        out = predict(model, torch.tensor(Xte, dtype=torch.float32),
                      mc_samples=cfg.bnn.predict_mc_samples)
    pred = out["mean_pred"].numpy()
    std = out["total_std"].numpy()
    rmse = float(np.sqrt(((pred - yte) ** 2).mean()))
    ece = expected_calibration_error(yte, pred, std, n_bins=10)
    print(f"    RMSE = {rmse:.2f} days")
    print(f"    ECE  = {ece:.3f} (lower is better)")
    print(f"    mean epistemic σ = {out['epistemic_std'].mean().item():.2f} days")
    print(f"    mean aleatoric σ = {out['aleatoric_std'].mean().item():.2f} days")

    print(f"[4/5] Exporting serving bundle to {args.out} ...")
    bundle = save_model_bundle(
        model,
        args.out,
        scaler=ref_scaler,
        features=FeatureConfig(),
        metadata={
            "produced_by": "scripts/train_demo.py",
            "rmse_days": round(rmse, 3),
            "ece": round(ece, 4),
            "feature_dim": int(X.shape[1]),
            "dataset": "synthetic (features_and_labels, seed=42)",
        },
    )
    print(f"    checkpoint: {bundle.checkpoint_path}")
    print(f"    sidecar:    {bundle.sidecar_path}")

    print("[5/5] Verifying the bundle loads through the serving path ...")
    from src.models.serving import load_serving_model

    serving = load_serving_model(bundle.checkpoint_path)
    fv = serving.latest_feature_vector(ref_df[list(CHANNELS)])
    print(f"    serving feature vector: dim={fv.shape[0]}  OK")
    print("Done. Remember: this model produces ADVISORY predictions only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
