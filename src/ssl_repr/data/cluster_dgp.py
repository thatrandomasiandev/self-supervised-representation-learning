"""Gaussian mixture cluster DGP for contrastive and VICReg benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssl_repr.data.base import ClusterDataset
from ssl_repr.utils.seed import set_seed


@dataclass
class ClusterDGPConfig:
    n_samples: int = 3000
    n_features: int = 32
    n_clusters: int = 5
    cluster_separation: float = 3.0
    nuisance_dim: int = 8
    cluster_std: float = 0.8
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


def generate_cluster_data(config: ClusterDGPConfig | None = None) -> ClusterDataset:
    """
    Generate clustered data with signal and nuisance dimensions.

    DGP:
      centroids_k ~ N(0, I) scaled by cluster_separation on first (n_features - nuisance_dim) dims
      x | y=k ~ N(centroid_k, cluster_std^2 I) + nuisance noise on trailing dims
    """
    cfg = config or ClusterDGPConfig()
    rng = set_seed(cfg.seed)

    signal_dim = max(2, cfg.n_features - cfg.nuisance_dim)
    centroids = np.zeros((cfg.n_clusters, cfg.n_features), dtype=np.float64)
    centroids[:, :signal_dim] = (
        rng.standard_normal((cfg.n_clusters, signal_dim)) * cfg.cluster_separation
    )

    labels = rng.integers(0, cfg.n_clusters, size=cfg.n_samples)
    X = centroids[labels] + cfg.cluster_std * rng.standard_normal((cfg.n_samples, cfg.n_features))
    if cfg.nuisance_dim > 0:
        X[:, signal_dim:] += rng.standard_normal((cfg.n_samples, cfg.nuisance_dim))

    train_idx, val_idx, test_idx = _split_indices(
        cfg.n_samples, cfg.train_ratio, cfg.val_ratio, rng
    )

    return ClusterDataset(
        X_train=X[train_idx].astype(np.float32),
        y_train=labels[train_idx].astype(np.int64),
        X_val=X[val_idx].astype(np.float32),
        y_val=labels[val_idx].astype(np.int64),
        X_test=X[test_idx].astype(np.float32),
        y_test=labels[test_idx].astype(np.int64),
        metadata={
            "dgp": "gaussian_clusters",
            "n_samples": cfg.n_samples,
            "n_features": cfg.n_features,
            "n_clusters": cfg.n_clusters,
            "cluster_separation": cfg.cluster_separation,
            "nuisance_dim": cfg.nuisance_dim,
            "seed": cfg.seed,
        },
        ground_truth={
            "centroids": centroids.astype(np.float32),
            "labels": labels.astype(np.int64),
            "signal_dim": signal_dim,
        },
    )
