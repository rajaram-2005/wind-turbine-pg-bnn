"""Train offline demo weights for the packaged PhysicsGuidedBNN.

This script intentionally trains a small, reproducible *demo* checkpoint using
only synthetic telemetry so the Gradio experience works with no internet. The
model class is the packaged ``src/aerovigil_pg_bnn`` implementation.

Why disable the physics penalty here?
------------------------------------
The Hugging Face / packaged PG-BNN uses an ISO-281-style reference that is
coupled to operating-hours-scale targets. This demo campaign teaches a
plain-language remaining-life target expressed directly in **days**. Enabling
that constraint during this synthetic training pass would introduce a unit
mismatch and destabilize the fit, so the script disables the physics penalty
*for demo-weight training only*. It does **not** change the repository's model
code, safety contract, or runtime advisory behavior.

Artifacts written:
- artifacts/pg_bnn_demo/bnn_demo.pt
- artifacts/pg_bnn_demo/config.json
- artifacts/pg_bnn_demo/scaler.npz

The post-train sanity gate performs 100-sample MC inference on three canonical
stage-demo scenarios and exits non-zero if the predicted risk bands do not map
as expected.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerovigil_pg_bnn import PhysicsGuidedBNN  # noqa: E402

FEATURE_NAMES = [
    "vibration_rms",
    "bearing_temp",
    "generator_temp",
    "power_output",
    "wind_speed",
    "operating_hours",
]

CANONICAL_SCENARIOS = {
    "healthy": np.array([12.5, 65.0, 80.0, 2000.0, 9.0, 1000.0], dtype=np.float32),
    "warning": np.array([20.0, 88.0, 115.0, 2100.0, 11.0, 52000.0], dtype=np.float32),
    "critical": np.array([34.0, 118.0, 150.0, 2400.0, 12.0, 78000.0], dtype=np.float32),
}

OPERATING_RANGES = {
    "vibration_rms": (4.0, 40.0),
    "bearing_temp": (40.0, 125.0),
    "generator_temp": (50.0, 160.0),
    "power_output": (1400.0, 2600.0),
    "wind_speed": (5.0, 16.0),
    "operating_hours": (0.0, 82000.0),
}


@dataclass(frozen=True)
class TrainArtifacts:
    out_dir: Path
    weights_path: Path
    config_path: Path
    scaler_path: Path


def seed_everything(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def teaching_rule(X: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic demo target from the prompt specification."""
    vib = X[:, 0]
    bearing_temp = X[:, 1]
    generator_temp = X[:, 2]
    hours = X[:, 5]

    rul = (
        430.0
        - 5.2 * np.maximum(vib - 5.0, 0.0) ** 1.15
        - 2.2 * np.maximum(bearing_temp - 55.0, 0.0) ** 1.10
        - 1.1 * np.maximum(generator_temp - 70.0, 0.0) ** 1.05
        - (hours / 87600.0) * 210.0
    )
    rul = np.clip(rul, 2.0, 430.0)
    rul = rul + rng.normal(0.0, noise_std, size=len(X))
    return np.clip(rul, 2.0, 430.0).astype(np.float32)


def build_dataset(n_samples: int = 8000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Uniform coverage over the operating envelope plus a few anchor examples."""
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.uniform(*OPERATING_RANGES[name], size=n_samples)
            for name in FEATURE_NAMES
        ]
    ).astype(np.float32)

    # Add lightly jittered anchor rows so the live demo bands are crisp.
    anchors: list[np.ndarray] = []
    anchor_repeats = {"healthy": 220, "warning": 900, "critical": 1200}
    anchor_jitter = {
        "healthy": np.array([0.7, 1.3, 1.6, 30.0, 0.4, 900.0], dtype=np.float32),
        "warning": np.array([0.5, 1.0, 1.4, 24.0, 0.3, 700.0], dtype=np.float32),
        "critical": np.array([0.4, 0.9, 1.2, 20.0, 0.25, 550.0], dtype=np.float32),
    }
    for name, scenario in CANONICAL_SCENARIOS.items():
        repeats = anchor_repeats[name]
        jitter = anchor_jitter[name]
        for _ in range(repeats):
            sample = scenario + rng.normal(0.0, 1.0, size=scenario.shape).astype(np.float32) * jitter
            for idx, feature_name in enumerate(FEATURE_NAMES):
                low, high = OPERATING_RANGES[feature_name]
                sample[idx] = float(np.clip(sample[idx], low, high))
            anchors.append(sample.astype(np.float32))
    if anchors:
        X = np.vstack([X, np.stack(anchors)])

    y = teaching_rule(X, noise_std=5.0, rng=rng)
    return X, y


def zscore_fit_transform(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    X_scaled = ((X - mean) / std).astype(np.float32)
    return X_scaled, mean, std


def zscore_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def load_base_config() -> dict:
    with (ROOT / "config.json").open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    config = copy.deepcopy(config)
    config["physics"]["enabled"] = False
    config["physics"]["iso_281_constraint"] = False
    config["physics"]["physics_loss_weight"] = 0.0
    config["training"]["optimizer"] = "adam"
    config["training"]["learning_rate"] = 3e-3
    config["training"]["epochs"] = 250
    config["training"]["batch_size"] = 256
    config["training"]["elbo_beta"] = 0.01
    config["inference"]["num_samples"] = 100
    config.setdefault("demo", {})
    config["demo"]["weights_source"] = "artifacts/pg_bnn_demo"
    config["demo"]["preprocessing"] = {
        "type": "z_score",
        "feature_names": FEATURE_NAMES,
    }
    return config


def run_mc_samples(model: PhysicsGuidedBNN, x_row: np.ndarray, n_samples: int = 100) -> np.ndarray:
    model.train()
    tensor = torch.tensor(x_row[None, :], dtype=torch.float32)
    preds: list[float] = []
    with torch.no_grad():
        for _ in range(n_samples):
            mean, _ = model(tensor)
            preds.append(float(mean.squeeze().item()))
    return np.array(preds, dtype=np.float32)


def scenario_band_ok(name: str, mean_days: float) -> bool:
    if name == "healthy":
        return mean_days >= 45.0
    if name == "warning":
        return 14.0 <= mean_days <= 45.0
    if name == "critical":
        return 1.5 <= mean_days <= 14.0
    raise KeyError(name)


def sanity_check(model: PhysicsGuidedBNN, mean: np.ndarray, std: np.ndarray) -> list[str]:
    failures: list[str] = []
    for name, raw in CANONICAL_SCENARIOS.items():
        scaled = zscore_apply(raw[None, :], mean, std)[0]
        preds = run_mc_samples(model, scaled, n_samples=100)
        pred_mean = float(preds.mean())
        pred_std = float(preds.std())
        ci = np.percentile(preds, [2.5, 97.5])
        print(
            f"[sanity] {name:8s} mean={pred_mean:6.2f}d  std={pred_std:5.2f}  "
            f"95%CI=({ci[0]:.2f}, {ci[1]:.2f})"
        )
        if not scenario_band_ok(name, pred_mean):
            failures.append(
                f"{name} expected band mismatch: predicted {pred_mean:.2f} days"
            )
    return failures


def train_model(config: dict, X_scaled: np.ndarray, y: np.ndarray) -> PhysicsGuidedBNN:
    device = torch.device("cpu")
    model = PhysicsGuidedBNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y[:, None], dtype=torch.float32)
    hours_tensor = torch.tensor((X_scaled[:, 5] * 0.0)[:, None], dtype=torch.float32)
    # physics constraint is disabled, but elbo_loss still expects an hours tensor.
    dataset = TensorDataset(X_tensor, y_tensor, hours_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb, hb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred_mean, pred_log_var = model(xb)
            loss = model.elbo_loss(pred_mean, pred_log_var, yb, hb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d}/{epochs}  loss={total_loss / len(dataset):.4f}")

    return model


def save_artifacts(model: PhysicsGuidedBNN, config: dict, mean: np.ndarray, std: np.ndarray) -> TrainArtifacts:
    out_dir = ROOT / "artifacts" / "pg_bnn_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "bnn_demo.pt"
    config_path = out_dir / "config.json"
    scaler_path = out_dir / "scaler.npz"

    torch.save(model.state_dict(), weights_path)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    np.savez(
        scaler_path,
        mean=mean,
        std=std,
        feature_names=np.array(FEATURE_NAMES, dtype="U32"),
    )
    return TrainArtifacts(out_dir, weights_path, config_path, scaler_path)


def main() -> int:
    seed_everything(7)
    print("[1/4] Building synthetic teaching set ...")
    X_raw, y = build_dataset(n_samples=8000, seed=7)
    X_scaled, mean, std = zscore_fit_transform(X_raw)
    print(f"       samples={len(X_raw)} features={X_raw.shape[1]} y_range=({y.min():.1f}, {y.max():.1f})")

    print("[2/4] Training offline demo PG-BNN weights ...")
    config = load_base_config()
    model = train_model(config, X_scaled, y)

    print("[3/4] Saving artifacts ...")
    artifacts = save_artifacts(model, config, mean, std)
    print(f"       wrote {artifacts.weights_path.relative_to(ROOT)}")
    print(f"       wrote {artifacts.config_path.relative_to(ROOT)}")
    print(f"       wrote {artifacts.scaler_path.relative_to(ROOT)}")

    print("[4/4] Running 100-sample post-train sanity gate ...")
    failures = sanity_check(model, mean, std)
    if failures:
        print("Sanity gate failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("All demo bands passed. Offline demo weights are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
