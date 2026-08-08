"""Tests for the Bayesian NN, PINO operator, and ONNX export."""

from __future__ import annotations

import torch

from src.models.bayesian_nn import BayesianLinear, PGBNNLoss, PhysicsGuidedBNN, train_step
from src.models.pino_operator import PINO, FourierNeuralOperator, SpectralConv2d


class TestBayesianNN:
    def test_forward_shapes(self):
        model = PhysicsGuidedBNN(in_features=6, hidden_dims=[32, 32], out_features=2)
        x = torch.randn(17, 6)
        mean, log_var = model(x)
        assert mean.shape == (17, 2)
        assert log_var.shape == (17, 2)

    def test_kl_divergence_positive(self):
        layer = BayesianLinear(4, 3)
        assert float(layer.kl_divergence()) > 0

    def test_predict_uncertainty_keys_and_shapes(self):
        model = PhysicsGuidedBNN(in_features=4, hidden_dims=[16], out_features=1)
        out = model.predict(torch.randn(9, 4), num_samples=8)
        for key in ("mean", "aleatoric_std", "epistemic_std", "total_std"):
            assert out[key].shape == (9, 1)
        assert torch.all(out["total_std"] >= out["epistemic_std"] - 1e-6)

    def test_uncertainty_increases_ood(self):
        """Epistemic uncertainty should grow on far out-of-distribution inputs."""
        torch.manual_seed(0)
        model = PhysicsGuidedBNN(in_features=3, hidden_dims=[32, 32], out_features=1)
        x = torch.randn(512, 3)
        y = x.sum(dim=1, keepdim=True)
        loss_fn = PGBNNLoss(beta_kl=1e-3, lambda_physics=0.0, num_batches=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(200):
            train_step(model, loss_fn, opt, x, y, num_mc_samples=1)

        in_dist = model.predict(torch.randn(256, 3), num_samples=64)
        ood = model.predict(torch.randn(256, 3) * 10 + 25, num_samples=64)
        assert float(ood["epistemic_std"].mean()) > float(in_dist["epistemic_std"].mean())

    def test_combined_loss_components(self):
        model = PhysicsGuidedBNN(in_features=4, hidden_dims=[8], out_features=1)
        x, y = torch.randn(10, 4), torch.randn(10, 1)
        mean, log_var = model(x)
        parts = PGBNNLoss(1e-3, 0.5, 1)(model, mean, log_var, y, physics_loss=torch.tensor(2.0))
        assert set(parts) == {"total", "nll", "kl", "physics"}
        expected = parts["nll"] + 1e-3 * parts["kl"] + 0.5 * parts["physics"]
        assert torch.allclose(parts["total"], expected)


class TestPINO:
    def test_spectral_conv_shape(self):
        layer = SpectralConv2d(3, 5, modes1=8, modes2=8)
        out = layer(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 5, 32, 32)

    def test_fno_output_shape_and_resolution_invariance(self):
        fno = FourierNeuralOperator(in_channels=3, out_channels=1, width=16, num_blocks=2)
        assert fno(torch.randn(2, 3, 32, 48)).shape == (2, 1, 32, 48)
        assert fno(torch.randn(2, 3, 64, 64)).shape == (2, 1, 64, 64)  # other grid

    def test_pino_loss_finite_and_differentiable(self):
        pino = PINO(FourierNeuralOperator(in_channels=3, out_channels=1, width=8, num_blocks=1))
        x = torch.randn(2, 3, 16, 16)
        y = torch.randn(2, 1, 16, 16)
        total, data_loss, pde_loss = pino.loss(x, y)
        assert torch.isfinite(total) and float(data_loss) >= 0 and float(pde_loss) >= 0
        total.backward()
        grads = [p.grad for p in pino.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestONNXExport:
    def test_export_produces_valid_file(self, tmp_path):
        import pytest

        pytest.importorskip("onnx")
        from src.deployment.export_onnx import export_bnn_to_onnx

        try:
            import onnxruntime  # noqa: F401

            validate = True
        except ImportError:
            validate = False

        model = PhysicsGuidedBNN(in_features=6, hidden_dims=[16], out_features=1)
        out = export_bnn_to_onnx(
            model, str(tmp_path / "bnn.onnx"), validate=validate
        )
        assert out.exists() and out.stat().st_size > 0

        import onnx

        proto = onnx.load(str(out))
        onnx.checker.check_model(proto)
        assert [o.name for o in proto.graph.output] == ["mean", "variance"]
