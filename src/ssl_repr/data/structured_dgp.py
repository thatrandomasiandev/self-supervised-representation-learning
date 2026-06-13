"""Linear latent-factor DGP for masked autoencoding benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssl_repr.data.base import StructuredDataset
from ssl_repr.utils.seed import set_seed


@dataclass
class StructuredDGPConfig:
    n_samples: int = 3000
    observed_dim: int = 48
    latent_dim: int = 8
    noise_std: float = 0.15
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42


def _split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def generate_structured_data(config: StructuredDGPConfig | None = None) -> StructuredDataset:
    """
    Generate observations x = W z + eps with binary labels from first latent.

    DGP:
      z ~ N(0, I_latent)
      x = W z + noise_std * eps
      y = 1[z_0 > 0]  (downstream label from latent, not observed during SSL)
    """
    cfg = config or StructuredDGPConfig()
    rng = set_seed(cfg.seed)

    W = rng.standard_normal((cfg.observed_dim, cfg.latent_dim))
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8

    Z = rng.standard_normal((cfg.n_samples, cfg.latent_dim))
    X = Z @ W.T + cfg.noise_std * rng.standard_normal((cfg.n_samples, cfg.observed_dim))
    y = (Z[:, 0] > 0).astype(np.int64)

    train_idx, val_idx, test_idx = _split_indices(
        cfg.n_samples, cfg.train_ratio, cfg.val_ratio, rng
    )

    return StructuredDataset(
        X_train=X[train_idx].astype(np.float32),
        X_val=X[val_idx].astype(np.float32),
        X_test=X[test_idx].astype(np.float32),
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        metadata={
            "dgp": "linear_latent_factors",
            "n_samples": cfg.n_samples,
            "observed_dim": cfg.observed_dim,
            "latent_dim": cfg.latent_dim,
            "noise_std": cfg.noise_std,
            "seed": cfg.seed,
        },
        ground_truth={
            "W": W.astype(np.float32),
            "Z_unlabeled": np.vstack([Z[train_idx], Z[val_idx], Z[test_idx]]).astype(np.float32),
        },
    )
