"""ONNX export for the physics-guided BNN with uncertainty heads.

The Bayesian layers are stochastic; for edge deployment we export a
deterministic graph by fixing every Bayesian layer to its posterior-mean
weights (``sample=False``). The exported model still returns *two* outputs:

    * ``mean``     — the predictive mean
    * ``variance`` — the aleatoric variance exp(log_var)

so edge devices get calibrated observation-noise uncertainty for free.
Epistemic uncertainty (which requires weight sampling) stays on the cloud
side, where MC sampling is affordable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.bayesian_nn import PhysicsGuidedBNN

logger = logging.getLogger(__name__)


class _DeterministicWrapper(nn.Module):
    """Wraps the BNN into a deterministic (mean-weight) two-output module."""

    def __init__(self, model: PhysicsGuidedBNN) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic forward returning (mean, variance)."""
        mean, log_var = self.model(x, sample=False)
        return mean, torch.exp(log_var)


def export_bnn_to_onnx(
    model: PhysicsGuidedBNN,
    output_path: str,
    opset: int = 18,
    validate: bool = True,
    atol: float = 1e-4,
) -> Path:
    """Export a trained BNN to ONNX with mean and variance output heads.

    Args:
        model: Trained ``PhysicsGuidedBNN``.
        output_path: Destination ``.onnx`` file path.
        opset: ONNX opset version.
        validate: If True, compare ONNX Runtime output against PyTorch.
        atol: Absolute tolerance for the validation check.

    Returns:
        Path of the written ONNX file.

    Raises:
        RuntimeError: If validation is requested and outputs disagree.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _DeterministicWrapper(model).eval()
    dummy = torch.randn(1, model.in_features)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["scada_features"],
        output_names=["mean", "variance"],
        dynamic_axes={
            "scada_features": {0: "batch"},
            "mean": {0: "batch"},
            "variance": {0: "batch"},
        },
        opset_version=opset,
    )
    logger.info("Exported ONNX model to %s (opset %d)", out, opset)

    if validate:
        validate_onnx_export(model, out, atol=atol)
    return out


def validate_onnx_export(
    model: PhysicsGuidedBNN,
    onnx_path: Path,
    num_test_batches: int = 3,
    batch_size: int = 8,
    atol: float = 1e-4,
    seed: Optional[int] = 0,
) -> bool:
    """Validate the exported ONNX model against the PyTorch mean-weight model.

    Args:
        model: The source PyTorch model.
        onnx_path: Path to the exported ONNX file.
        num_test_batches: Number of random batches to compare.
        batch_size: Batch size of each random test batch.
        atol: Absolute tolerance.
        seed: Optional RNG seed for reproducibility.

    Returns:
        True on success.

    Raises:
        RuntimeError: If any output disagrees beyond ``atol``.
        ImportError: If onnxruntime is not installed.
    """
    import onnxruntime as ort  # local import: edge-only dependency

    if seed is not None:
        torch.manual_seed(seed)

    wrapper = _DeterministicWrapper(model).eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for i in range(num_test_batches):
        x = torch.randn(batch_size, model.in_features)
        with torch.no_grad():
            torch_mean, torch_var = wrapper(x)
        ort_mean, ort_var = session.run(None, {"scada_features": x.numpy()})

        if not np.allclose(torch_mean.numpy(), ort_mean, atol=atol) or not np.allclose(
            torch_var.numpy(), ort_var, atol=atol
        ):
            raise RuntimeError(
                f"ONNX validation failed on batch {i}: max mean err "
                f"{np.abs(torch_mean.numpy() - ort_mean).max():.2e}, max var err "
                f"{np.abs(torch_var.numpy() - ort_var).max():.2e} (atol={atol})"
            )
    logger.info("ONNX validation passed (%d batches, atol=%g)", num_test_batches, atol)
    return True
