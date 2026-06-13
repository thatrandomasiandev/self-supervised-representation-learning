"""MLP encoders and projection heads for SSL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Feature encoder returning L2-normalized representations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """Two-layer projection head used in SimCLR / VICReg."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLREncoder(nn.Module):
    """Encoder + projection head; returns both representation and projection."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        repr_dim: int = 64,
        proj_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden_dim, repr_dim, n_layers, dropout)
        self.projector = ProjectionHead(repr_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


class MaskedAutoencoder(nn.Module):
    """MLP encoder–decoder for masked feature reconstruction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden_dim, latent_dim, n_layers, dropout)
        decoder_layers: list[nn.Module] = []
        dim = latent_dim
        for _ in range(n_layers):
            decoder_layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        decoder_layers.append(nn.Linear(dim, in_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reconstruct only masked positions; unmasked inputs pass through."""
        h = self.encode(x)
        recon = self.decoder(h)
        return x * (1.0 - mask) + recon * mask


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (SimCLR)."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    return F.cross_entropy(sim, labels)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> torch.Tensor:
    """VICReg: invariance + variance + covariance regularization."""
    repr_loss = F.mse_loss(z1, z2)

    def _variance_term(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    def _covariance_term(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / max(n - 1, 1)
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        return (off_diag**2).mean()

    var_loss = _variance_term(z1) + _variance_term(z2)
    cov_loss = _covariance_term(z1) + _covariance_term(z2)
    return sim_coeff * repr_loss + var_coeff * var_loss + cov_coeff * cov_loss
