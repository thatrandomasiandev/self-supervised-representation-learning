"""Tests for DGP invariants and ground-truth accessors."""

import numpy as np

from ssl_repr.data.cluster_dgp import ClusterDGPConfig, generate_cluster_data
from ssl_repr.data.structured_dgp import StructuredDGPConfig, generate_structured_data


def test_cluster_splits_sum_to_total():
    data = generate_cluster_data(ClusterDGPConfig(n_samples=1000, seed=7))
    total = len(data.y_train) + len(data.y_val) + len(data.y_test)
    assert total == 1000
    assert data.X_unlabeled.shape[0] == total


def test_cluster_ground_truth_centroids_shape():
    data = generate_cluster_data(ClusterDGPConfig(n_clusters=4, n_features=20, seed=8))
    centroids = data.ground_truth["centroids"]
    assert centroids.shape == (4, 20)


def test_structured_latent_recovery_rank():
    data = generate_structured_data(
        StructuredDGPConfig(n_samples=500, observed_dim=32, latent_dim=4, noise_std=0.05, seed=9)
    )
    W = data.ground_truth["W"]
    Z = data.ground_truth["Z_unlabeled"]
    X = data.X_unlabeled
    # Low noise => X approximates low-rank
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    energy_top4 = np.sum(s[:4] ** 2) / np.sum(s**2)
    assert energy_top4 > 0.85


def test_structured_unlabeled_alignment():
    data = generate_structured_data(StructuredDGPConfig(n_samples=600, seed=10))
    n = len(data.y_train) + len(data.y_val) + len(data.y_test)
    assert data.X_unlabeled.shape[0] == n
    assert data.ground_truth["Z_unlabeled"].shape[0] == n
