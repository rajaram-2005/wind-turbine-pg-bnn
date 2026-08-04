"""Demo training script: fit PG-BNN on synthetic drivetrain telemetry.

Usage:
    python scripts/train_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.synthetic import SyntheticConfig, features_and_labels  # noqa: E402
from src.eval.calibration import expected_calibration_error  # noqa: E402
from src.models.bnn import (  # noqa: E402
    BayesianNeuralNetwork,
    TrainConfig,
    elbo_loss,
    predict,
)


def main():
    device = "cpu"
    print("[1/4] Generating synthetic drivetrain telemetry ...")
    X, y = features_and_labels(SyntheticConfig(n_turbines=20, seq_len=1500))
    n = len(X)
    n_train = int(0.8 * n)
    perm = np.random.default_rng(0).permutation(n)
    Xtr, ytr = X[perm[:n_train]], y[perm[:n_train]]
    Xte, yte = X[perm[n_train:]], y[perm[n_train:]]
    print(f"    train={len(Xtr)}  test={len(Xte)}  feature_dim={X.shape[1]}")

    model = BayesianNeuralNetwork(in_features=X.shape[1], hidden_sizes=(64, 64)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tcfg = TrainConfig(num_epochs=100, num_samples=5, batch_size=256)

    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True)

    print("[2/4] Training PG-BNN (MCVI + physics penalty) ...")
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

    print("[3/4] Evaluating on test split ...")
    model.eval()
    with torch.no_grad():
        out = predict(model, torch.tensor(Xte, dtype=torch.float32), mc_samples=64)
    pred = out["mean_pred"].numpy()
    std = out["total_std"].numpy()
    rmse = float(np.sqrt(((pred - yte) ** 2).mean()))
    ece = expected_calibration_error(yte, pred, std, n_bins=10)
    print(f"    RMSE = {rmse:.2f} days")
    print(f"    ECE  = {ece:.3f} (lower is better)")
    print(f"    mean epistemic σ = {out['epistemic_std'].mean().item():.2f} days")
    print(f"    mean aleatoric σ = {out['aleatoric_std'].mean().item():.2f} days")

    print("[4/4] Saving model checkpoint to ./bnn_demo.pt ...")
    torch.save(model.state_dict(), "bnn_demo.pt")
    print("Done. Remember: this model produces ADVISORY predictions only.")


if __name__ == "__main__":
    main()
