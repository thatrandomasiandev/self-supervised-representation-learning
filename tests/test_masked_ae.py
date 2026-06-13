"""Tests for masked autoencoder training."""

import numpy as np

from ssl_repr.data.structured_dgp import StructuredDGPConfig, generate_structured_data
from ssl_repr.masked.trainer import MaskedAETrainConfig, fit_masked_ae


def test_masked_ae_reconstruction_improves():
    data = generate_structured_data(
        StructuredDGPConfig(n_samples=1200, observed_dim=32, latent_dim=6, noise_std=0.1, seed=14)
    )
    result = fit_masked_ae(
        data,
        MaskedAETrainConfig(epochs=50, batch_size=128, hidden_dim=64, latent_dim=16, seed=14),
    )
    assert result.metrics.recon_mse < 1.0
    assert result.metrics.linear_probe_acc > 0.55


def test_masked_ae_latent_r2_positive():
    data = generate_structured_data(
        StructuredDGPConfig(n_samples=1500, observed_dim=40, latent_dim=8, noise_std=0.08, seed=15)
    )
    result = fit_masked_ae(
        data,
        MaskedAETrainConfig(epochs=60, batch_size=128, hidden_dim=64, latent_dim=24, seed=15),
    )
    assert result.metrics.latent_r2 > 0.0
    assert np.all(np.isfinite(result.embeddings))
