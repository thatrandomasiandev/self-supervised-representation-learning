"""Masked autoencoder training loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ssl_repr.data.base import StructuredDataset
from ssl_repr.masked.metrics import MaskedAEMetrics, evaluate_masked_ae
from ssl_repr.models.encoder import MaskedAutoencoder
from ssl_repr.utils.seed import set_torch_seed


@dataclass
class MaskedAETrainConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 0.001
    hidden_dim: int = 128
    latent_dim: int = 32
    n_layers: int = 2
    dropout: float = 0.1
    mask_ratio: float = 0.4
    seed: int = 42


@dataclass
class MaskedAEResult:
    model_name: str
    train_loss: float
    metrics: MaskedAEMetrics
    embeddings: np.ndarray


def _random_mask(batch: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    return (torch.rand_like(batch) < mask_ratio).float()


def fit_masked_ae(
    dataset: StructuredDataset,
    config: MaskedAETrainConfig | None = None,
) -> MaskedAEResult:
    """Train masked autoencoder on unlabeled features."""
    cfg = config or MaskedAETrainConfig()
    set_torch_seed(cfg.seed)

    X = torch.as_tensor(dataset.X_unlabeled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = MaskedAutoencoder(
        in_dim=dataset.observed_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    last_loss = 0.0
    for _ in range(cfg.epochs):
        for (batch_x,) in loader:
            mask = _random_mask(batch_x, cfg.mask_ratio)
            masked_input = batch_x * (1.0 - mask)
            recon = model(masked_input, mask)
            loss = F.mse_loss(recon * mask, batch_x * mask, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(X).cpu().numpy()
        full_mask = _random_mask(X, cfg.mask_ratio)
        masked_input = X * (1.0 - full_mask)
        X_recon = model(masked_input, full_mask).cpu().numpy()

    n_train = len(dataset.y_train)
    n_val = len(dataset.y_val)
    n_test = len(dataset.y_test)
    test_start = n_train + n_val

    X_test = dataset.X_test
    X_recon_test = X_recon[test_start : test_start + n_test]
    train_emb = embeddings[:n_train]
    test_emb = embeddings[test_start : test_start + n_test]

    Z_all = dataset.ground_truth.get("Z_unlabeled")
    if Z_all is not None:
        Z_train = Z_all[:n_train].astype(np.float32)
        Z_test = Z_all[test_start : test_start + n_test].astype(np.float32)
    else:
        Z_train = None
        Z_test = None

    metrics = evaluate_masked_ae(
        X_test,
        X_recon_test,
        train_emb,
        test_emb,
        dataset.y_train,
        dataset.y_test,
        Z_train=Z_train,
        Z_test=Z_test,
        seed=cfg.seed,
    )

    return MaskedAEResult(
        model_name="MaskedAE",
        train_loss=last_loss,
        metrics=metrics,
        embeddings=embeddings,
    )
