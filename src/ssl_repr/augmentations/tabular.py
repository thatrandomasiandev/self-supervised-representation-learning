"""Tabular data augmentations for self-supervised pretraining.

These augmentations operate on dense feature tensors and are designed for
two-view contrastive / redundancy-reduction SSL on tabular datasets.

References:
    - SCARF: Self-Supervised Contrastive Learning using Random Feature
      Corruption (Bahri et al., 2022)
    - Yun et al., CutMix (2019) — adapted here for 1-D tabular samples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def feature_corruption(
    x: torch.Tensor,
    corruption_rate: float = 0.3,
    marginals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace a random subset of features with samples from the marginal.

    For each sample in the batch, independently selects a random
    ``corruption_rate`` fraction of features and replaces them with values
    drawn from the empirical marginal distribution of that feature (i.e. a
    randomly chosen value from another row in the same batch).

    Args:
        x: Input tensor of shape ``(batch, features)``.
        corruption_rate: Fraction of features to replace per sample,
            in ``[0, 1]``.
        marginals: Optional pre-computed marginal samples of shape
            ``(batch, features)``.  When *None*, marginals are approximated
            by randomly permuting each feature column within the batch.

    Returns:
        Corrupted tensor with the same shape as *x*.
    """
    batch_size, n_features = x.shape
    mask = torch.rand(batch_size, n_features, device=x.device) < corruption_rate

    if marginals is None:
        perm_idx = torch.argsort(
            torch.rand(batch_size, n_features, device=x.device), dim=0
        )
        marginals = torch.gather(x, 0, perm_idx)

    return torch.where(mask, marginals, x)


def gaussian_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add isotropic Gaussian noise :math:`\\mathcal{N}(0, \\sigma^2 I)`.

    Args:
        x: Input tensor of shape ``(batch, features)``.
        sigma: Standard deviation of the noise distribution.

    Returns:
        Perturbed tensor with the same shape as *x*.
    """
    return x + sigma * torch.randn_like(x)


def cutmix_tabular(
    x: torch.Tensor,
    mix_rate: float = 0.3,
) -> torch.Tensor:
    """Mix two randomly paired samples by swapping a subset of features.

    For each row ``x[i]`` a partner ``x[j]`` is chosen by random permutation.
    A binary mask selects ``mix_rate`` of the features to copy from ``x[j]``
    into ``x[i]``.

    Args:
        x: Input tensor of shape ``(batch, features)``.
        mix_rate: Expected fraction of features taken from the partner.

    Returns:
        Mixed tensor with the same shape as *x*.
    """
    batch_size, n_features = x.shape
    perm = torch.randperm(batch_size, device=x.device)
    partner = x[perm]
    mask = torch.rand(batch_size, n_features, device=x.device) < mix_rate
    return torch.where(mask, partner, x)


@dataclass
class TabularAugmenter:
    """Configurable augmentation pipeline for tabular SSL.

    Composes feature corruption, additive Gaussian noise, and CutMix in a
    single callable.  Each augmentation is applied independently with its own
    probability / strength, and they are composed sequentially:

        ``x  →  feature_corruption  →  gaussian_noise  →  cutmix_tabular``

    Args:
        corruption_rate: Fraction of features replaced by marginal samples.
        noise_sigma: Standard deviation of additive Gaussian noise.
        cutmix_rate: Fraction of features swapped with a random partner.

    Example::

        aug = TabularAugmenter(corruption_rate=0.3, noise_sigma=0.1)
        view1 = aug(batch)
        view2 = aug(batch)
    """

    corruption_rate: float = 0.3
    noise_sigma: float = 0.1
    cutmix_rate: float = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation pipeline.

        Args:
            x: Input tensor of shape ``(batch, features)``.

        Returns:
            Augmented tensor with the same shape as *x*.
        """
        out = x
        if self.corruption_rate > 0.0:
            out = feature_corruption(out, corruption_rate=self.corruption_rate)
        if self.noise_sigma > 0.0:
            out = gaussian_noise(out, sigma=self.noise_sigma)
        if self.cutmix_rate > 0.0:
            out = cutmix_tabular(out, mix_rate=self.cutmix_rate)
        return out
