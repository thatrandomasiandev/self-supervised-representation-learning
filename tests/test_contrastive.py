"""Tests that SimCLR learns useful cluster representations."""

import numpy as np

from ssl_repr.contrastive.simclr import SimCLRTrainConfig, fit_simclr
from ssl_repr.data.cluster_dgp import ClusterDGPConfig, generate_cluster_data


def test_simclr_linear_probe_beats_chance():
    data = generate_cluster_data(
        ClusterDGPConfig(
            n_samples=1500,
            n_clusters=4,
            cluster_separation=4.0,
            seed=12,
        )
    )
    result = fit_simclr(
        data,
        SimCLRTrainConfig(epochs=40, batch_size=128, hidden_dim=64, repr_dim=32, seed=12),
    )
    chance = 1.0 / data.n_clusters
    assert result.metrics.linear_probe_acc > chance + 0.15


def test_simclr_embeddings_finite():
    data = generate_cluster_data(ClusterDGPConfig(n_samples=800, seed=13))
    result = fit_simclr(
        data,
        SimCLRTrainConfig(epochs=20, batch_size=64, hidden_dim=32, seed=13),
    )
    assert np.all(np.isfinite(result.embeddings))
    assert result.embeddings.shape[0] == data.X_unlabeled.shape[0]
