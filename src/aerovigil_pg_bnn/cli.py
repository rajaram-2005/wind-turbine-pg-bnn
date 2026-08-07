"""Command-line interface for Aerovigil PG-BNN inference."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .model import PhysicsGuidedBNN


def main() -> int:
    parser = argparse.ArgumentParser(description="Aerovigil PG-BNN: Wind Turbine RUL Prediction")
    parser.add_argument(
        "--repo-id", default="AerovigilAI/wind-turbine-pg-bnn", help="Hugging Face model repo ID"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input telemetry JSON file")
    parser.add_argument("--samples", "-n", type=int, default=100, help="Number of MCVI samples")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.repo_id}...", file=sys.stderr)
    model = PhysicsGuidedBNN.from_pretrained(args.repo_id)
    model.train()  # Enable dropout for MCVI

    # Load input
    with open(args.input) as f:
        data = json.load(f)

    raw = np.array(
        [
            [
                data["vibration_rms"],
                data["bearing_temp"],
                data["generator_temp"],
                data["power_output"],
                data["wind_speed"],
                data["operating_hours"],
            ]
        ],
        dtype=np.float32,
    )

    # Check for scaler
    scaler_path = Path("artifacts/pg_bnn_demo/scaler.npz")
    if scaler_path.exists():
        s = np.load(scaler_path)
        mean, std = s["mean"].astype(np.float32), s["std"].astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        raw = (raw - mean) / std

    features = torch.tensor(raw, dtype=torch.float32)

    # Inference
    predictions = []
    with torch.no_grad():
        for _ in range(args.samples):
            rul_mean, _ = model(features)
            predictions.append(rul_mean)

    mean_rul = torch.stack(predictions).mean().item()
    uncertainty = torch.stack(predictions).std().item()

    result = {
        "predicted_rul_days": round(mean_rul, 2),
        "uncertainty_days": round(uncertainty, 2),
        "confidence_95": [
            round(mean_rul - 2 * uncertainty, 2),
            round(mean_rul + 2 * uncertainty, 2),
        ],
        "maintenance_recommended": mean_rul < 45,
    }

    output = json.dumps(result, indent=2)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
