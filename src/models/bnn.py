"""
Bayesian Neural Network for RUL prediction (Monte Carlo Variational Inference).

Weights are treated as random variables parameterized by (mu, rho) where
sigma = softplus(rho). The variational posterior q(w) = N(mu, softplus(rho)^2)
is trained against a Gaussian likelihood whose observation noise
(aleatoric) is also learned.

- Epistemic uncertainty: variance of predictions across T forward samples
  (captured by sampling weights).
- Aleatoric uncertainty: learned observation noise sigma_obs.

This is a deliberately compact, pedagogical implementation — sufficient
for research experimentation on drivetrain telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.physics.constraints import physics_loss


# --------------------------------------------------------------------------
# Bayesian linear layer
# --------------------------------------------------------------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -k, k)
        nn.init.uniform_(self.bias_mu, -k, k)
        nn.init.constant_(self.weight_rho, -3.0)  # small initial sigma (~0.05)
        nn.init.constant_(self.bias_rho, -3.0)

    def _sigma(self, rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return F.linear(x, self.weight_mu, self.bias_mu)
        w_sig = self._sigma(self.weight_rho)
        b_sig = self._sigma(self.bias_rho)
        w = self.weight_mu + w_sig * torch.randn_like(self.weight_mu)
        b = self.bias_mu + b_sig * torch.randn_like(self.bias_mu)
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        """KL(q || p) for this layer, p = N(0, prior_sigma^2). Closed form for Gaussians."""
        s_w = self._sigma(self.weight_rho)
        s_b = self._sigma(self.bias_rho)
        kl_w = 0.5 * (s_w ** 2 + self.weight_mu ** 2) / (self.prior_sigma ** 2) - 0.5
        kl_b = 0.5 * (s_b ** 2 + self.bias_mu ** 2) / (self.prior_sigma ** 2) - 0.5
        return kl_w.sum() + kl_b.sum()


# --------------------------------------------------------------------------
# BNN: maps feature vector → (rul_mean, rul_log_var)
# --------------------------------------------------------------------------
class BayesianNeuralNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        dims = [in_features, *hidden_sizes]
        self.linears = nn.ModuleList(
            [BayesianLinear(dims[i], dims[i + 1], prior_sigma=prior_sigma) for i in range(len(dims) - 1)]
        )
        self.out_mean = BayesianLinear(dims[-1], 1, prior_sigma=prior_sigma)
        # Learned heteroscedastic log-var (aleatoric) as a global scalar + input-dependent head for flexibility
        self.out_log_var = nn.Linear(dims[-1], 1)
        nn.init.constant_(self.out_log_var.bias, -1.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.linears:
            h = layer(h, sample=sample)
            h = F.relu(h)
        mean = self.out_mean(h, sample=sample).squeeze(-1)
        log_var = self.out_log_var(h).squeeze(-1)
        return mean, log_var

    def kl(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.out_mean.weight_mu.device)
        for m in self.linears:
            total = total + m.kl()
        total = total + self.out_mean.kl()
        return total


# --------------------------------------------------------------------------
# Training / prediction helpers
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    num_epochs: int = 300
    num_samples: int = 10       # MC samples per batch for ELBO
    kl_weight: float = 1e-3     # KL scaling factor (to weight against NLL)
    physics_weight: float = 0.2 # L_physics weight
    batch_size: int = 256


def elbo_loss(
    model: BayesianNeuralNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    telemetry: dict[str, torch.Tensor] | None = None,
    cfg: TrainConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute negative ELBO = E_q[ NLL ] + KL_weight * KL(q||p) + physics_weight * L_physics."""
    if cfg is None:
        cfg = TrainConfig()
    nll_acc = torch.tensor(0.0, device=x.device)
    n = y.shape[0]
    for _ in range(cfg.num_samples):
        mean, log_var = model(x, sample=True)
        var = torch.exp(log_var) + 1e-6
        # Gaussian NLL: 0.5 * log(2πσ²) + 0.5 * (y - μ)² / σ²
        nll = 0.5 * torch.log(2 * math.pi * var) + 0.5 * (y - mean) ** 2 / var
        nll_acc = nll_acc + nll.mean()
    nll_acc = nll_acc / cfg.num_samples
    kl = model.kl() / max(n, 1)
    loss = nll_acc + cfg.kl_weight * kl

    breakdown = {"nll": nll_acc.item(), "kl": kl.item()}

    if telemetry is not None and cfg.physics_weight > 0:
        # We couple physics loss against the (sampled) predicted mean
        mean_det, _ = model(x, sample=False)
        l_phys, bd = physics_loss(telemetry, mean_det)
        loss = loss + cfg.physics_weight * l_phys
        breakdown["physics"] = l_phys.item()
        for k, v in bd.items():
            breakdown[f"physics.{k}"] = v.item()
    else:
        breakdown["physics"] = 0.0

    return loss, breakdown


@torch.no_grad()
def predict(
    model: BayesianNeuralNetwork,
    x: torch.Tensor,
    mc_samples: int = 50,
) -> dict[str, torch.Tensor]:
    """
    Predict RUL with uncertainty decomposition.

    Returns dict with:
      - mean_pred   : predictive mean (averaged over MC samples)
      - epistemic_std: std of per-sample means → model uncertainty
      - aleatoric_std: sqrt(mean of per-sample predicted vars) → sensor noise
      - total_std   : sqrt(epistemic^2 + aleatoric^2)
    """
    model.eval()
    means = []
    vars_ = []
    for _ in range(mc_samples):
        m, lv = model(x, sample=True)
        means.append(m)
        vars_.append(torch.exp(lv))
    means_t = torch.stack(means, dim=0)           # (T, B)
    vars_t = torch.stack(vars_, dim=0)            # (T, B)
    pred_mean = means_t.mean(dim=0)
    epistemic_var = means_t.var(dim=0)
    aleatoric_var = vars_t.mean(dim=0)
    total_var = epistemic_var + aleatoric_var
    return {
        "mean_pred": pred_mean,
        "epistemic_std": torch.sqrt(torch.clamp(epistemic_var, min=1e-8)),
        "aleatoric_std": torch.sqrt(torch.clamp(aleatoric_var, min=1e-8)),
        "total_std": torch.sqrt(torch.clamp(total_var, min=1e-8)),
    }
