"""Data augmentations for tabular SSL."""

from __future__ import annotations

import numpy as np
import torch


def gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    return x + std * torch.randn_like(x)


def feature_dropout(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    mask = (torch.rand_like(x) > drop_prob).float()
    return x * mask


def scaling(x: torch.Tensor, sigma: float = 0.15) -> torch.Tensor:
    factor = 1.0 + sigma * torch.randn(x.shape[0], 1, device=x.device)
    return x * factor


def augment_tabular(
    x: torch.Tensor,
    noise_std: float = 0.15,
    dropout_prob: float = 0.1,
    scale_sigma: float = 0.15,
) -> torch.Tensor:
    """Compose standard tabular augmentations for two-view SSL."""
    out = gaussian_noise(x, noise_std)
    out = feature_dropout(out, dropout_prob)
    out = scaling(out, scale_sigma)
    return out
