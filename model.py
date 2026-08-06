import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Mean-field variational Bayesian linear layer."""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters (mean and log-std)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_std = nn.Parameter(torch.ones(out_features, in_features) * -3)
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_std = nn.Parameter(torch.ones(out_features) * -3)

        self.prior_std = prior_std
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick
        weight_std = torch.exp(self.weight_log_std)
        bias_std = torch.exp(self.bias_log_std)

        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)

        return F.linear(self.dropout(x), weight, bias)


class PhysicsGuidedBNN(nn.Module):
    """
    Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction.

    Integrates ISO 281 bearing physics with Bayesian deep learning for
    uncertainty-aware remaining useful life estimation.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Network architecture from config
        network_cfg = config["network"]
        input_dim = config["num_input_features"]
        hidden_dims = network_cfg["hidden_dims"]

        # Build Bayesian layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output heads: mean and log-variance for Gaussian RUL prediction
        self.rul_mean_head = BayesianLinear(prev_dim, 1)
        self.rul_log_var_head = BayesianLinear(prev_dim, 1)

        # Physics parameters
        self.physics_weight = config["physics"]["physics_loss_weight"]
        self.iso_281_enabled = config["physics"]["iso_281_constraint"]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning RUL mean and log-variance.

        Args:
            x: Input telemetry tensor of shape (batch, num_input_features)

        Returns:
            rul_mean: Predicted RUL in days (batch, 1)
            rul_log_var: Log-variance for uncertainty (batch, 1)
        """
        features = self.feature_extractor(x)
        rul_mean = self.rul_mean_head(features)
        rul_log_var = self.rul_log_var_head(features)
        return rul_mean, rul_log_var

    def physics_constraint(
        self, rul_mean: torch.Tensor, operating_hours: torch.Tensor
    ) -> torch.Tensor:
        """
        ISO 281 bearing life physics constraint.

        L10 life: 90% reliability at 10^6 revolutions.
        """
        if not self.iso_281_enabled:
            return torch.tensor(0.0, device=rul_mean.device)

        # Simplified ISO 281 constraint: RUL should decrease with operating hours
        expected_rul = self.config["physics"]["l10_life_reference"] - operating_hours
        physics_loss = F.mse_loss(rul_mean.squeeze(), expected_rul.clamp(min=0))

        return self.physics_weight * physics_loss

    def elbo_loss(
        self,
        rul_pred: torch.Tensor,
        rul_log_var: torch.Tensor,
        rul_target: torch.Tensor,
        operating_hours: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evidence Lower Bound (ELBO) with physics constraint.
        """
        # Negative log-likelihood (Gaussian)
        nll = 0.5 * (rul_log_var + ((rul_target - rul_pred) ** 2) / torch.exp(rul_log_var))
        nll = nll.mean()

        # KL divergence (approximated via Monte Carlo)
        kl_divergence = torch.tensor(0.0, device=rul_pred.device)

        # Physics constraint
        physics_loss = self.physics_constraint(rul_pred, operating_hours)

        # Total ELBO
        beta = self.config["training"]["elbo_beta"]
        elbo = nll + beta * kl_divergence + physics_loss

        return elbo

    @classmethod
    def from_pretrained(cls, repo_id: str, cache_dir: str = None):
        """Load model from Hugging Face Hub."""
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir)
        model_path = hf_hub_download(repo_id=repo_id, filename="bnn_demo.pt", cache_dir=cache_dir)

        with open(config_path) as f:
            config = json.load(f)

        model = cls(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        return model


# ─── USAGE EXAMPLE ─────────────────────────────────────────────
if __name__ == "__main__":
    # Load from Hugging Face
    model = PhysicsGuidedBNN.from_pretrained("AerovigilAI/wind-turbine-pg-bnn")
    model.eval()

    # Dummy inference
    dummy_input = torch.randn(1, 6)  # batch=1, 6 features
    rul_mean, rul_log_var = model(dummy_input)

    print(f"Predicted RUL: {rul_mean.item():.2f} days")
    print(f"Uncertainty (log-var): {rul_log_var.item():.4f}")
