"""Flower federated-learning client for multi-farm fleet training.

Each wind farm trains the shared PG-BNN on its local SCADA data; only model
weights leave the site (raw SCADA never does). The evaluation metrics include
the physics-consistency loss, enabling *physics-aware aggregation*: the
server can down-weight clients whose updates violate the shared physics.

Requires the ``flwr`` package on the client machines. The class degrades to
a clear ImportError message when Flower is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.models.bayesian_nn import PGBNNLoss, PhysicsGuidedBNN, train_step

logger = logging.getLogger(__name__)

try:
    import flwr as fl

    _ClientBase = fl.client.NumPyClient
except ImportError:  # pragma: no cover - environment dependent
    fl = None

    class _ClientBase:  # type: ignore[no-redef]
        """Fallback base so the module imports without flwr installed."""


@dataclass
class FederatedConfig:
    """Configuration for fleet-wide federated training.

    Attributes:
        num_rounds: Total federated rounds orchestrated by the server.
        min_clients: Minimum number of farms required per round.
        local_epochs: Local training epochs per round on each farm.
        lr: Local learning rate.
        batch_size: Local minibatch size.
        beta_kl: KL weight of the local ELBO.
        lambda_physics: Physics-loss weight of the local objective.
        server_address: host:port of the Flower server.
    """

    num_rounds: int = 20
    min_clients: int = 2
    local_epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 256
    beta_kl: float = 1e-3
    lambda_physics: float = 0.1
    server_address: str = "0.0.0.0:8080"


class FlowerFederatedClient(_ClientBase):
    """Flower ``NumPyClient`` wrapping a local PG-BNN and farm dataset.

    Args:
        model: The shared-architecture ``PhysicsGuidedBNN``.
        train_data: Tuple (x, y) of local training tensors.
        val_data: Tuple (x, y) of local validation tensors.
        config: Federated training configuration.
        physics_fn: Optional callable(pred_mean, x) → scalar physics loss,
            evaluated on local data for physics-aware aggregation.
    """

    def __init__(
        self,
        model: PhysicsGuidedBNN,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
        config: FederatedConfig | None = None,
        physics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if fl is None:
            raise ImportError("flwr is required for federated training: pip install flwr")
        self.model = model
        self.x_train, self.y_train = train_data
        self.x_val, self.y_val = val_data
        self.config = config or FederatedConfig()
        self.physics_fn = physics_fn

    # ── Flower NumPyClient API ───────────────────────────────────────────
    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """Return current model parameters as a list of numpy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load aggregated parameters received from the server."""
        state = self.model.state_dict()
        for key, array in zip(state.keys(), parameters):
            state[key] = torch.as_tensor(array)
        self.model.load_state_dict(state)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
        """One round of local training on this farm's SCADA data."""
        self.set_parameters(parameters)
        n = self.x_train.shape[0]
        num_batches = max(n // self.config.batch_size, 1)
        loss_fn = PGBNNLoss(self.config.beta_kl, self.config.lambda_physics, num_batches)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        last: dict[str, float] = {}
        for _ in range(int(config.get("local_epochs", self.config.local_epochs))):
            perm = torch.randperm(n)
            for b in range(num_batches):
                idx = perm[b * self.config.batch_size : (b + 1) * self.config.batch_size]
                last = train_step(
                    self.model,
                    loss_fn,
                    optimizer,
                    self.x_train[idx],
                    self.y_train[idx],
                    physics_fn=self.physics_fn,
                )
        logger.info("local fit done: %s", last)
        return self.get_parameters(config), n, {f"train_{k}": v for k, v in last.items()}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
        """Evaluate aggregated weights locally, reporting physics metrics."""
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            mean, log_var = self.model(self.x_val, sample=False)
            nll = float(PGBNNLoss.gaussian_nll(mean, log_var, self.y_val))
            rmse = float(torch.sqrt(torch.mean((mean - self.y_val) ** 2)))
            phys = float(self.physics_fn(mean, self.x_val)) if self.physics_fn is not None else 0.0
        # Physics-aware aggregation: the server strategy can weight clients
        # by 1 / (1 + physics_loss) using this reported metric.
        metrics = {"nll": nll, "rmse": rmse, "physics_loss": phys}
        return nll, int(self.x_val.shape[0]), metrics


def start_client(client: FlowerFederatedClient) -> None:
    """Connect a farm client to the federated server and start training.

    Args:
        client: A configured ``FlowerFederatedClient``.
    """
    if fl is None:
        raise ImportError("flwr is required for federated training: pip install flwr")
    fl.client.start_numpy_client(server_address=client.config.server_address, client=client)
