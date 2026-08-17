"""Physics-Informed Neural Operator (PINO) for wake-field prediction.

Implements:
    * ``SpectralConv2d`` — convolution parameterised directly in Fourier
      space (Li et al., "Fourier Neural Operator for Parametric PDEs", 2021).
    * ``FourierNeuralOperator`` — lifting → FNO blocks → projection, mapping
      input fields (e.g. inflow velocity + turbine forcing) to output fields
      (e.g. wake velocity) in continuous function space.
    * ``PINO`` — wraps the FNO with a simplified steady incompressible
      Navier–Stokes residual (advection–diffusion of the streamwise velocity)
      used as a PDE loss during training.

The operator is resolution-invariant: it can be trained on one grid size and
evaluated on another, because the learned kernels live in Fourier space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2-D Fourier-space convolution layer.

    Applies FFT → complex linear transform on the lowest ``modes1 x modes2``
    frequency modes → inverse FFT. Higher modes are truncated, which acts as
    a learnable low-pass spectral filter.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        modes1: Number of retained Fourier modes along the first spatial dim.
        modes2: Number of retained Fourier modes along the second spatial dim.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication: (B, Ci, X, Y) x (Ci, Co, X, Y) → (B, Co, X, Y)."""
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spectral convolution forward pass.

        Args:
            x: Input field of shape (B, C_in, H, W).

        Returns:
            Output field of shape (B, C_out, H, W).
        """
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2+1) complex

        out_ft = torch.zeros(
            batch, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOBlock(nn.Module):
    """One FNO block: spectral conv + pointwise (1x1) conv + GELU.

    Args:
        width: Channel width of the block.
        modes1: Retained modes along dim 1.
        modes2: Retained modes along dim 2.
    """

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual-style FNO block forward."""
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FourierNeuralOperator(nn.Module):
    """Fourier Neural Operator mapping input fields to output fields.

    Pipeline: lifting (1x1 conv) → N FNO blocks → projection MLP.
    Grid coordinates are concatenated to the input channels so the operator
    is aware of absolute position (standard FNO practice).

    Args:
        in_channels: Physical input channels (e.g. inflow u, v, forcing).
        out_channels: Physical output channels (e.g. wake streamwise velocity).
        width: Latent channel width.
        modes1: Fourier modes retained along height.
        modes2: Fourier modes retained along width.
        num_blocks: Number of FNO blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.lifting = nn.Conv2d(in_channels + 2, width, kernel_size=1)  # +2 for (x, y) grid
        self.blocks = nn.ModuleList(FNOBlock(width, modes1, modes2) for _ in range(num_blocks))
        self.projection = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    @staticmethod
    def _grid(shape: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        """Normalised (x, y) coordinate grid, shape (B, 2, H, W)."""
        b, _, h, w = shape
        gx = torch.linspace(0.0, 1.0, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        gy = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input fields to output fields.

        Args:
            x: Input field tensor of shape (B, in_channels, H, W).

        Returns:
            Output field tensor of shape (B, out_channels, H, W).
        """
        grid = self._grid(x.shape, x.device)
        h = self.lifting(torch.cat([x, grid], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.projection(h)


class PINO(nn.Module):
    """Physics-Informed Neural Operator for turbine wake fields.

    Adds a simplified steady 2-D Navier–Stokes residual (advection–diffusion
    of the streamwise velocity u) to the data loss:

        R(u) = u * du/dx + v * du/dy - nu * (d2u/dx2 + d2u/dy2) - f

    Spatial derivatives are evaluated with central finite differences on the
    output grid. ``f`` is the actuator-disk forcing channel of the input.

    Args:
        operator: The underlying ``FourierNeuralOperator``.
        viscosity: Effective (turbulent) kinematic viscosity nu.
        domain_size: Physical size (Lx, Ly) of the grid in metres.
    """

    def __init__(
        self,
        operator: FourierNeuralOperator | None = None,
        viscosity: float = 1.5e-2,
        domain_size: tuple[float, float] = (1000.0, 500.0),
    ) -> None:
        super().__init__()
        self.operator = operator or FourierNeuralOperator()
        self.viscosity = viscosity
        self.domain_size = domain_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the output field (delegates to the FNO)."""
        return self.operator(x)

    def pde_residual(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Simplified steady Navier–Stokes residual on the predicted field.

        Args:
            x: Input fields (B, C_in, H, W); channel 0 = inflow u, channel 1 =
               inflow v, channel 2 (if present) = actuator-disk forcing f.
            u_pred: Predicted streamwise velocity (B, 1, H, W).

        Returns:
            Residual field (B, 1, H-2, W-2) on interior points.
        """
        b, _, h, w = u_pred.shape
        dx = self.domain_size[0] / max(w - 1, 1)
        dy = self.domain_size[1] / max(h - 1, 1)

        u = u_pred
        v = x[:, 1:2]
        f = x[:, 2:3] if x.shape[1] > 2 else torch.zeros_like(u)

        # Central differences (interior points): axis -1 is x, axis -2 is y.
        du_dx = (u[..., 1:-1, 2:] - u[..., 1:-1, :-2]) / (2.0 * dx)
        du_dy = (u[..., 2:, 1:-1] - u[..., :-2, 1:-1]) / (2.0 * dy)
        d2u_dx2 = (u[..., 1:-1, 2:] - 2.0 * u[..., 1:-1, 1:-1] + u[..., 1:-1, :-2]) / dx**2
        d2u_dy2 = (u[..., 2:, 1:-1] - 2.0 * u[..., 1:-1, 1:-1] + u[..., :-2, 1:-1]) / dy**2

        u_c = u[..., 1:-1, 1:-1]
        v_c = v[..., 1:-1, 1:-1]
        f_c = f[..., 1:-1, 1:-1]
        return u_c * du_dx + v_c * du_dy - self.viscosity * (d2u_dx2 + d2u_dy2) - f_c

    def loss(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        lambda_pde: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combined data + PDE-residual loss.

        Args:
            x: Input fields (B, C_in, H, W).
            y_true: Ground-truth output fields (B, C_out, H, W).
            lambda_pde: Weight of the PDE residual term.

        Returns:
            Tuple (total_loss, data_loss, pde_loss) as scalar tensors.
        """
        y_pred = self.forward(x)
        data_loss = F.mse_loss(y_pred, y_true)
        residual = self.pde_residual(x, y_pred[:, :1])
        pde_loss = residual.pow(2).mean()
        return data_loss + lambda_pde * pde_loss, data_loss, pde_loss
