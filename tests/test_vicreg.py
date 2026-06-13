"""Tests for VICReg representation learning."""

import numpy as np

from ssl_repr.data.cluster_dgp import ClusterDGPConfig, generate_cluster_data
from ssl_repr.vicreg.trainer import VICRegTrainConfig, fit_vicreg


def test_vicreg_avoids_collapse():
    data = generate_cluster_data(
        ClusterDGPConfig(n_samples=1200, n_clusters=4, cluster_separation=3.5, seed=16)
    )
    result = fit_vicreg(
        data,
        VICRegTrainConfig(epochs=40, batch_size=128, hidden_dim=64, repr_dim=32, seed=16),
    )
    assert result.repr_std > 0.05


def test_vicreg_linear_probe_beats_chance():
    data = generate_cluster_data(
        ClusterDGPConfig(n_samples=1500, n_clusters=5, cluster_separation=4.0, seed=17)
    )
    result = fit_vicreg(
        data,
        VICRegTrainConfig(epochs=50, batch_size=128, hidden_dim=64, seed=17),
    )
    chance = 1.0 / data.n_clusters
    assert result.metrics.linear_probe_acc > chance + 0.1
