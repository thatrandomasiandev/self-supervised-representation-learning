"""Tests for contrastive SSL methods and evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch

from ssl_repr.contrastive.simclr import (
    BYOLTrainConfig,
    BYOLTrainer,
    MoCoTrainConfig,
    MoCoTrainer,
    NTXentLoss,
    SimCLRTrainConfig,
    fit_simclr,
)
from ssl_repr.data.cluster_dgp import ClusterDGPConfig, generate_cluster_data
from ssl_repr.evaluation.metrics import (
    feature_alignment,
    feature_uniformity,
    knn_eval,
    linear_probe_eval,
)
from ssl_repr.vicreg.trainer import (
    covariance_loss,
    vicreg_loss_decomposed,
)


# ---------------------------------------------------------------------------
# NT-Xent loss properties
# ---------------------------------------------------------------------------


def test_ntxent_lower_for_similar_pairs():
    """NT-Xent should produce lower loss when positive pairs are similar."""
    torch.manual_seed(0)
    criterion = NTXentLoss(temperature=0.5)

    z_base = torch.randn(32, 64)
    z_similar = z_base + 0.01 * torch.randn_like(z_base)
    z_dissimilar = torch.randn(32, 64)

    loss_similar = criterion(z_base, z_similar).item()
    loss_dissimilar = criterion(z_base, z_dissimilar).item()

    assert loss_similar < loss_dissimilar, (
        f"Expected loss for similar pairs ({loss_similar:.4f}) "
        f"< dissimilar ({loss_dissimilar:.4f})"
    )


def test_ntxent_gradient_flows():
    """Verify gradients propagate through NT-Xent."""
    z1 = torch.randn(16, 32, requires_grad=True)
    z2 = torch.randn(16, 32, requires_grad=True)
    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(z1, z2)
    loss.backward()
    assert z1.grad is not None
    assert torch.all(torch.isfinite(z1.grad))


# ---------------------------------------------------------------------------
# VICReg covariance collapse penalty
# ---------------------------------------------------------------------------


def test_vicreg_covariance_penalizes_collapse():
    """Covariance term should be higher for correlated than decorrelated dims."""
    torch.manual_seed(1)
    d = 32

    z_decorr = torch.randn(128, d)
    cov_decorr = covariance_loss(z_decorr).item()

    base = torch.randn(128, 1)
    z_corr = base.expand(-1, d) + 0.01 * torch.randn(128, d)
    cov_corr = covariance_loss(z_corr).item()

    assert cov_corr > cov_decorr, (
        f"Collapsed covariance ({cov_corr:.6f}) should exceed "
        f"decorrelated ({cov_decorr:.6f})"
    )


def test_vicreg_decomposed_components_positive():
    """All VICReg loss components should be non-negative."""
    torch.manual_seed(2)
    z1 = torch.randn(64, 32)
    z2 = torch.randn(64, 32)
    total, sim, var, cov = vicreg_loss_decomposed(z1, z2)
    assert total.item() >= 0
    assert sim.item() >= 0
    assert var.item() >= 0
    assert cov.item() >= 0


# ---------------------------------------------------------------------------
# SimCLR end-to-end
# ---------------------------------------------------------------------------


def test_simclr_linear_probe_beats_chance():
    """SimCLR + linear probe should beat random guessing."""
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
    """All SimCLR embeddings should be finite."""
    data = generate_cluster_data(ClusterDGPConfig(n_samples=800, seed=13))
    result = fit_simclr(
        data,
        SimCLRTrainConfig(epochs=20, batch_size=64, hidden_dim=32, seed=13),
    )
    assert np.all(np.isfinite(result.embeddings))
    assert result.embeddings.shape[0] == data.X_unlabeled.shape[0]


# ---------------------------------------------------------------------------
# Linear probe after short training beats chance
# ---------------------------------------------------------------------------


def test_linear_probe_after_5_epochs_beats_chance():
    """Even 5 epochs of SimCLR should produce features better than chance."""
    data = generate_cluster_data(
        ClusterDGPConfig(
            n_samples=1200,
            n_clusters=4,
            cluster_separation=5.0,
            seed=99,
        )
    )
    result = fit_simclr(
        data,
        SimCLRTrainConfig(
            epochs=5,
            batch_size=128,
            hidden_dim=64,
            repr_dim=32,
            temperature=0.5,
            seed=99,
        ),
    )
    chance = 1.0 / data.n_clusters
    assert result.metrics.linear_probe_acc > chance, (
        f"5-epoch linear probe {result.metrics.linear_probe_acc:.3f} "
        f"should exceed chance {chance:.3f}"
    )


# ---------------------------------------------------------------------------
# MoCo
# ---------------------------------------------------------------------------


def test_moco_trains_without_error():
    """MoCo should complete training and produce finite embeddings."""
    data = generate_cluster_data(
        ClusterDGPConfig(n_samples=800, n_clusters=3, cluster_separation=4.0, seed=20)
    )
    trainer = MoCoTrainer(
        data,
        MoCoTrainConfig(
            epochs=10,
            batch_size=64,
            hidden_dim=32,
            repr_dim=16,
            projection_dim=16,
            queue_size=256,
            seed=20,
        ),
    )
    result = trainer.fit()
    assert np.all(np.isfinite(result.embeddings))
    assert result.train_loss > 0


# ---------------------------------------------------------------------------
# BYOL
# ---------------------------------------------------------------------------


def test_byol_trains_without_error():
    """BYOL should complete training and produce finite embeddings."""
    data = generate_cluster_data(
        ClusterDGPConfig(n_samples=800, n_clusters=3, cluster_separation=4.0, seed=21)
    )
    trainer = BYOLTrainer(
        data,
        BYOLTrainConfig(
            epochs=10,
            batch_size=64,
            hidden_dim=32,
            repr_dim=16,
            projection_dim=16,
            seed=21,
        ),
    )
    result = trainer.fit()
    assert np.all(np.isfinite(result.embeddings))
    assert result.embeddings.shape[0] == data.X_unlabeled.shape[0]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def test_linear_probe_eval_basic():
    """Standalone linear probe should return accuracy in [0, 1]."""
    rng = np.random.default_rng(30)
    X_train = rng.standard_normal((200, 16)).astype(np.float32)
    y_train = rng.integers(0, 3, 200)
    X_test = rng.standard_normal((80, 16)).astype(np.float32)
    y_test = rng.integers(0, 3, 80)
    result = linear_probe_eval(X_train, y_train, X_test, y_test, seed=30)
    assert 0.0 <= result.accuracy <= 1.0


def test_knn_eval_basic():
    """k-NN evaluation should return accuracy in [0, 1]."""
    rng = np.random.default_rng(31)
    X_train = rng.standard_normal((200, 16)).astype(np.float32)
    y_train = rng.integers(0, 3, 200)
    X_test = rng.standard_normal((80, 16)).astype(np.float32)
    y_test = rng.integers(0, 3, 80)
    result = knn_eval(X_train, y_train, X_test, y_test, k=200)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.k <= 200


def test_feature_uniformity_finite():
    """Uniformity metric should return a finite scalar."""
    rng = np.random.default_rng(32)
    embeddings = rng.standard_normal((100, 16)).astype(np.float32)
    u = feature_uniformity(embeddings)
    assert np.isfinite(u)


def test_feature_alignment_close_pairs():
    """Alignment should be low for identical pairs, higher for random ones."""
    rng = np.random.default_rng(33)
    z1 = rng.standard_normal((100, 16)).astype(np.float32)
    z2_close = z1 + 0.01 * rng.standard_normal((100, 16)).astype(np.float32)
    z2_random = rng.standard_normal((100, 16)).astype(np.float32)

    align_close = feature_alignment(z1, z2_close)
    align_random = feature_alignment(z1, z2_random)
    assert align_close < align_random
