import numpy as np
import torch

from src.models.bnn import BayesianNeuralNetwork, TrainConfig, elbo_loss, predict


def _mk_model(dim=25):
    return BayesianNeuralNetwork(in_features=dim, hidden_sizes=(32, 16))


def test_bnn_forward_shape():
    model = _mk_model()
    x = torch.randn(8, 25)
    mean, log_var = model(x, sample=True)
    assert mean.shape == (8,)
    assert log_var.shape == (8,)


def test_predict_returns_all_uncertainty_components():
    # Make the test deterministic and robust to untrained-model negative means.
    torch.manual_seed(0)
    np.random.seed(0)
    model = _mk_model()
    x = torch.randn(16, 25)
    out = predict(model, x, mc_samples=20)
    # Shape checks for all keys
    for k in ("mean_pred", "epistemic_std", "aleatoric_std", "total_std"):
        assert out[k].shape == (16,)
        assert torch.isfinite(out[k]).all()

    # Only uncertainty (std) components must be non-negative; mean_pred can be
    # negative for an untrained model and is not constrained here.
    for k in ("epistemic_std", "aleatoric_std", "total_std"):
        assert (out[k] >= 0).all()

    # total >= epistemic, total >= aleatoric (allow tiny numerical slack)
    assert (out["total_std"] + 1e-6 >= out["epistemic_std"] - 1e-5).all()
    assert (out["total_std"] + 1e-6 >= out["aleatoric_std"] - 1e-5).all()


def test_elbo_loss_runs_and_is_scalar():
    model = _mk_model()
    x = torch.randn(32, 25)
    y = torch.rand(32) * 365
    cfg = TrainConfig(num_samples=3, kl_weight=1e-3, physics_weight=0.0)
    loss, bd = elbo_loss(model, x, y, telemetry=None, cfg=cfg)
    assert loss.dim() == 0
    assert bd["nll"] > 0
