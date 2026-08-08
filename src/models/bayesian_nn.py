"""Bayesian neural network with Bayes-by-Backprop weight uncertainty.

Implements:
    * ``BayesianLinear`` — a fully-connected layer whose weights and biases
      are Gaussian random variables trained with the local reparameterisation
      trick (Blundell et al., "Weight Uncertainty in Neural Networks", 2015).
    * ``PhysicsGuidedBNN`` — a configurable-depth BNN with a heteroscedastic
      (aleatoric) noise head and Monte-Carlo predictive uncertainty.
    * ``PGBNNLoss`` — the combined training objective
        L_total = NLL + beta * KL + lambda_physics * L_physics.

Uncertainty decomposition:
    * Aleatoric — learned per-sample observation noise (log-variance head).
    * Epistemic — variance of the MC predictive mean across weight samples.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Linear layer with a factorised Gaussian posterior over weights.

    Posterior: q(w) = N(mu, sigma^2) with sigma = softplus(rho) for positivity.
    Prior: p(w) = N(0, prior_sigma^2).

    The forward pass samples w = mu + sigma * eps (reparameterisation trick),
    keeping the computation differentiable w.r.t. mu and rho.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        prior_sigma: Standard deviation of the zero-mean Gaussian prior.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational parameters (initialised like Kaiming + small noise)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise variational parameters."""
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -std, std)
        nn.init.constant_(self.weight_rho, -5.0)  # softplus(-5) ≈ 0.0067
        nn.init.uniform_(self.bias_mu, -std, std)
        nn.init.constant_(self.bias_rho, -5.0)

    @property
    def weight_sigma(self) -> torch.Tensor:
        """Posterior weight standard deviation sigma = softplus(rho)."""
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self) -> torch.Tensor:
        """Posterior bias standard deviation."""
        return F.softplus(self.bias_rho)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with optionally sampled weights.

        Args:
            x: Input of shape (N, in_features).
            sample: If True, draw a weight sample; if False, use the
                posterior mean (deterministic — used for ONNX export).

        Returns:
            Output of shape (N, out_features).
        """
        if sample:
            w_eps = torch.randn_like(self.weight_mu)
            b_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + self.weight_sigma * w_eps
            bias = self.bias_mu + self.bias_sigma * b_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Closed-form KL(q(w) || p(w)) for factorised Gaussians.

        KL = sum over params of
            log(sigma_p / sigma_q) + (sigma_q^2 + mu^2) / (2 sigma_p^2) - 1/2

        Returns:
            Scalar KL divergence for this layer.
        """
        def _kl(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            sp = self.prior_sigma
            return (
                torch.log(sp / sigma) + (sigma.pow(2) + mu.pow(2)) / (2.0 * sp**2) - 0.5
            ).sum()

        return _kl(self.weight_mu, self.weight_sigma) + _kl(self.bias_mu, self.bias_sigma)


class PhysicsGuidedBNN(nn.Module):
    """Bayesian MLP with heteroscedastic noise head for turbine health targets.

    Architecture:
        input → [BayesianLinear + ReLU + Dropout] * len(hidden_dims)
              → BayesianLinear head producing (mean, log_var) per output.

    Args:
        in_features: Number of input SCADA features.
        hidden_dims: Sizes of the hidden layers, e.g. [128, 128, 64].
        out_features: Number of predicted targets (e.g. 1 for RUL or power).
        prior_sigma: Prior std for all Bayesian layers.
        dropout: Dropout probability (also enables MC-dropout at inference).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Optional[List[int]] = None,
        out_features: int = 1,
        prior_sigma: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128, 64]
        self.in_features = in_features
        self.out_features = out_features

        dims = [in_features] + hidden_dims
        self.hidden = nn.ModuleList(
            BayesianLinear(dims[i], dims[i + 1], prior_sigma) for i in range(len(hidden_dims))
        )
        self.dropout = nn.Dropout(dropout)
        # Head outputs [mean, log_var] for each target → aleatoric uncertainty.
        self.head = BayesianLinear(dims[-1], 2 * out_features, prior_sigma)

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single stochastic forward pass.

        Args:
            x: Inputs of shape (N, in_features).
            sample: Sample weights (True) or use posterior means (False).

        Returns:
            Tuple (mean, log_var), each of shape (N, out_features).
        """
        h = x
        for layer in self.hidden:
            h = self.dropout(F.relu(layer(h, sample=sample)))
        out = self.head(h, sample=sample)
        mean, log_var = out.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mean, log_var

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence of all Bayesian layers."""
        kl = torch.zeros((), device=self.head.weight_mu.device)
        for layer in self.hidden:
            kl = kl + layer.kl_divergence()
        return kl + self.head.kl_divergence()

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, num_samples: int = 64
    ) -> Dict[str, torch.Tensor]:
        """Monte-Carlo predictive distribution with uncertainty decomposition.

        Args:
            x: Inputs of shape (N, in_features).
            num_samples: Number of weight samples S.

        Returns:
            Dict with:
                mean: predictive mean, (N, out).
                aleatoric_std: mean learned observation noise, (N, out).
                epistemic_std: std of MC means across samples, (N, out).
                total_std: sqrt(aleatoric^2 + epistemic^2), (N, out).
        """
        was_training = self.training
        self.eval()
        means, variances = [], []
        for _ in range(num_samples):
            m, lv = self.forward(x, sample=True)
            means.append(m)
            variances.append(torch.exp(lv))
        self.train(was_training)

        mc_means = torch.stack(means)  # (S, N, out)
        mc_vars = torch.stack(variances)
        mean = mc_means.mean(dim=0)
        epistemic_var = mc_means.var(dim=0, unbiased=False)
        aleatoric_var = mc_vars.mean(dim=0)
        return {
            "mean": mean,
            "aleatoric_std": aleatoric_var.sqrt(),
            "epistemic_std": epistemic_var.sqrt(),
            "total_std": (aleatoric_var + epistemic_var).sqrt(),
        }


class PGBNNLoss(nn.Module):
    """Combined physics-guided evidence lower bound (ELBO) objective.

    L_total = NLL + beta * KL / num_batches + lambda_physics * L_physics

    * NLL — heteroscedastic Gaussian negative log-likelihood, which trains
      the aleatoric (observation-noise) head.
    * KL — Bayes-by-Backprop complexity cost, scaled by ``beta`` and the
      number of minibatches per epoch (KL reweighting).
    * L_physics — any differentiable physics-consistency loss produced by
      the ``src.physics`` modules, supplied per batch by the caller.

    Args:
        beta_kl: KL weight beta.
        lambda_physics: Physics-loss weight.
        num_batches: Minibatches per epoch (KL is amortised across them).
    """

    def __init__(
        self, beta_kl: float = 1e-3, lambda_physics: float = 0.1, num_batches: int = 1
    ) -> None:
        super().__init__()
        self.beta_kl = beta_kl
        self.lambda_physics = lambda_physics
        self.num_batches = max(num_batches, 1)

    @staticmethod
    def gaussian_nll(
        mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Heteroscedastic Gaussian negative log-likelihood.

        NLL = 0.5 * [log_var + (y - mu)^2 / exp(log_var)] (+ const).

        Args:
            mean: Predicted mean.
            log_var: Predicted log observation variance.
            target: Ground-truth targets.

        Returns:
            Scalar mean NLL.
        """
        return 0.5 * (log_var + (target - mean).pow(2) / torch.exp(log_var)).mean()

    def forward(
        self,
        model: PhysicsGuidedBNN,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
        physics_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss and its components.

        Args:
            model: The BNN (used to query the KL divergence).
            mean: Predicted mean from a stochastic forward pass.
            log_var: Predicted log-variance from the same pass.
            target: Ground-truth targets, same shape as ``mean``.
            physics_loss: Optional scalar physics-consistency loss.

        Returns:
            Dict with "total", "nll", "kl", "physics" scalar tensors.
        """
        nll = self.gaussian_nll(mean, log_var, target)
        kl = model.kl_divergence() / self.num_batches
        phys = (
            physics_loss
            if physics_loss is not None
            else torch.zeros((), device=mean.device, dtype=mean.dtype)
        )
        total = nll + self.beta_kl * kl + self.lambda_physics * phys
        return {"total": total, "nll": nll, "kl": kl, "physics": phys}


def train_step(
    model: PhysicsGuidedBNN,
    loss_fn: PGBNNLoss,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    physics_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    num_mc_samples: int = 2,
) -> Dict[str, float]:
    """One optimisation step averaging the ELBO over MC weight samples.

    Args:
        model: BNN to train.
        loss_fn: Combined PG-BNN loss.
        optimizer: Any torch optimizer over ``model.parameters()``.
        x: Batch inputs, (N, in_features).
        y: Batch targets, (N, out_features).
        physics_fn: Optional callable(prediction_mean, x) → scalar physics loss.
        num_mc_samples: MC samples per step for the ELBO estimate.

    Returns:
        Dict of float loss components for logging.
    """
    model.train()
    optimizer.zero_grad()
    totals: Dict[str, torch.Tensor] = {}
    for _ in range(num_mc_samples):
        mean, log_var = model(x, sample=True)
        phys = physics_fn(mean, x) if physics_fn is not None else None
        parts = loss_fn(model, mean, log_var, y, physics_loss=phys)
        for k, v in parts.items():
            totals[k] = totals.get(k, torch.zeros((), device=x.device)) + v / num_mc_samples
    totals["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    return {k: float(v.detach()) for k, v in totals.items()}
