"""Projection and prediction heads for self-supervised learning.

ProjectionHead maps representations to a lower-dimensional space where the
contrastive/redundancy-reduction objective is applied.  PredictionHead adds an
extra asymmetric MLP on top, required by BYOL and SimSiam to break symmetry
and prevent representation collapse without negative pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP projector with batch normalisation.

    Maps encoder representations to a space where the SSL loss is computed.

    Architecture::

        Linear(in_dim, hidden_dim)  →  BN  →  ReLU  →  Linear(hidden_dim, out_dim)

    Args:
        in_dim: Dimensionality of the encoder output.
        hidden_dim: Width of the hidden layer.
        out_dim: Dimensionality of the projected embedding.

    Returns:
        Projected tensor of shape ``(batch, out_dim)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    """Asymmetric predictor MLP for BYOL / SimSiam.

    Sits on top of the *online* projector and introduces asymmetry between the
    online and target branches, which is critical for convergence without
    negative pairs (Grill et al., 2020).

    Architecture::

        Linear(in_dim, hidden_dim)  →  BN  →  ReLU  →  Linear(hidden_dim, out_dim)

    Args:
        in_dim: Dimensionality of the projection (online branch output).
        hidden_dim: Width of the hidden layer.
        out_dim: Output dimensionality (must match target projection dim).

    Returns:
        Predicted tensor of shape ``(batch, out_dim)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
