"""Masked autoencoder training loops.

Implements a basic masked autoencoder (MaskedAE) for tabular data and a
patch-based Masked Autoencoder (MAE) inspired by He et al. (2022).

References:
    - He et al., *Masked Autoencoders Are Scalable Vision Learners*,
      CVPR 2022.  Adapted here for 1-D tabular "patches".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ssl_repr.data.base import StructuredDataset
from ssl_repr.masked.metrics import MaskedAEMetrics, evaluate_masked_ae
from ssl_repr.models.encoder import MaskedAutoencoder
from ssl_repr.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic Masked AE (tabular)
# ---------------------------------------------------------------------------


@dataclass
class MaskedAETrainConfig:
    """Configuration for basic masked autoencoder pretraining.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden_dim: Width of encoder / decoder hidden layers.
        latent_dim: Encoder output (latent) dimensionality.
        n_layers: Number of hidden layers in encoder and decoder.
        dropout: Dropout probability.
        mask_ratio: Fraction of input features to mask.
        seed: Random seed for reproducibility.
    """

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
    """Container for masked autoencoder training outcomes.

    Args:
        model_name: Identifier string.
        train_loss: Final training loss value.
        metrics: Reconstruction and downstream metrics.
        embeddings: Frozen encoder representations for all samples.
    """

    model_name: str
    train_loss: float
    metrics: MaskedAEMetrics
    embeddings: np.ndarray


def _random_mask(batch: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Generate a random binary mask for feature masking."""
    return (torch.rand_like(batch) < mask_ratio).float()


def fit_masked_ae(
    dataset: StructuredDataset,
    config: MaskedAETrainConfig | None = None,
) -> MaskedAEResult:
    """Train masked autoencoder on unlabeled features.

    Args:
        dataset: Structured dataset with unlabeled observations and
            train/val/test labels for downstream evaluation.
        config: Training hyper-parameters.  Uses defaults when *None*.

    Returns:
        :class:`MaskedAEResult` with reconstruction metrics and embeddings.
    """
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
    for epoch in range(cfg.epochs):
        for (batch_x,) in loader:
            mask = _random_mask(batch_x, cfg.mask_ratio)
            masked_input = batch_x * (1.0 - mask)
            recon = model(masked_input, mask)
            loss = F.mse_loss(recon * mask, batch_x * mask, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        logger.debug("MaskedAE epoch %d  loss=%.4f", epoch, last_loss)

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
    X_recon_test = X_recon[test_start: test_start + n_test]
    train_emb = embeddings[:n_train]
    test_emb = embeddings[test_start: test_start + n_test]

    Z_all = dataset.ground_truth.get("Z_unlabeled")
    if Z_all is not None:
        Z_train = Z_all[:n_train].astype(np.float32)
        Z_test = Z_all[test_start: test_start + n_test].astype(np.float32)
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


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """Split a 1-D input into non-overlapping patches and embed each.

    Divides an input vector of length *in_dim* into *num_patches* contiguous
    groups and projects each group to *embed_dim* via a shared linear layer.

    If *in_dim* is not evenly divisible by *num_patches*, the input is
    zero-padded on the right.

    Args:
        in_dim: Total number of input features.
        num_patches: Number of non-overlapping patches.
        embed_dim: Dimensionality of each patch embedding.

    Returns:
        Tensor of shape ``(batch, num_patches, embed_dim)``.
    """

    def __init__(self, in_dim: int, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = int(np.ceil(in_dim / num_patches))
        self.padded_dim = self.patch_size * num_patches
        self.proj = nn.Linear(self.patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify and embed.

        Args:
            x: Input tensor of shape ``(batch, in_dim)``.

        Returns:
            Patch embeddings of shape ``(batch, num_patches, embed_dim)``.
        """
        if x.shape[1] < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - x.shape[1]))
        patches = x.view(x.shape[0], self.num_patches, self.patch_size)
        return self.proj(patches)


# ---------------------------------------------------------------------------
# MAE Trainer (patch-based, He et al. 2022 style)
# ---------------------------------------------------------------------------


@dataclass
class MAETrainConfig:
    """Configuration for patch-based Masked Autoencoder pretraining.

    The encoder processes only *visible* patches (mask_ratio fraction are
    removed) and the decoder reconstructs *all* patches.  The loss is MSE
    computed only on the masked patches.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        num_patches: Number of patches to split the input into.
        embed_dim: Patch embedding dimensionality.
        encoder_depth: Number of encoder MLP blocks.
        decoder_dim: Decoder hidden dimensionality.
        decoder_depth: Number of decoder MLP blocks.
        dropout: Dropout probability.
        mask_ratio: Fraction of patches to mask (He et al. use 0.75).
        seed: Random seed for reproducibility.
    """

    epochs: int = 80
    batch_size: int = 256
    lr: float = 0.001
    num_patches: int = 8
    embed_dim: int = 64
    encoder_depth: int = 2
    decoder_dim: int = 64
    decoder_depth: int = 1
    dropout: float = 0.1
    mask_ratio: float = 0.75
    seed: int = 42


@dataclass
class MAEResult:
    """Container for MAE training outcomes.

    Args:
        model_name: Identifier string.
        train_loss: Final training loss value.
        metrics: Reconstruction and downstream metrics.
        embeddings: Frozen encoder representations for all samples.
    """

    model_name: str
    train_loss: float
    metrics: MaskedAEMetrics
    embeddings: np.ndarray


class _MAEEncoder(nn.Module):
    """Processes only visible (unmasked) patch embeddings."""

    def __init__(self, embed_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MAEDecoder(nn.Module):
    """Reconstructs all patches from encoder output + mask tokens."""

    def __init__(
        self, embed_dim: int, decoder_dim: int, patch_size: int, depth: int, dropout: float
    ) -> None:
        super().__init__()
        self.embed_to_dec = nn.Linear(embed_dim, decoder_dim)
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(decoder_dim, decoder_dim),
                nn.LayerNorm(decoder_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(decoder_dim, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_to_dec(x)
        x = self.net(x)
        return self.head(x)


class _MAEModel(nn.Module):
    """Full MAE: patch embed → random mask → encoder (visible) → decoder (all)."""

    def __init__(
        self,
        in_dim: int,
        num_patches: int,
        embed_dim: int,
        encoder_depth: int,
        decoder_dim: int,
        decoder_depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(in_dim, num_patches, embed_dim)
        self.encoder = _MAEEncoder(embed_dim, encoder_depth, dropout)
        self.decoder = _MAEDecoder(
            embed_dim, decoder_dim, self.patch_embed.patch_size, decoder_depth, dropout
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.num_patches = num_patches
        self.patch_size = self.patch_embed.patch_size
        self.in_dim = in_dim

    def _random_patch_mask(
        self, batch_size: int, mask_ratio: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return indices of visible and masked patches."""
        num_mask = max(1, int(self.num_patches * mask_ratio))
        num_visible = self.num_patches - num_mask
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_visible = ids_shuffle[:, :num_visible]
        ids_masked = ids_shuffle[:, num_visible:]
        return ids_visible, ids_masked

    def forward(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: patchify, mask, encode visible, decode all.

        Args:
            x: Raw input ``(N, in_dim)``.
            mask_ratio: Fraction of patches to mask.

        Returns:
            Tuple of ``(reconstruction, patch_mask)`` where
            *reconstruction* has shape ``(N, num_patches, patch_size)`` and
            *patch_mask* is a boolean tensor ``(N, num_patches)`` that is
            True for masked positions.
        """
        batch_size = x.shape[0]
        patches = self.patch_embed(x)  # (N, P, E)
        ids_visible, ids_masked = self._random_patch_mask(batch_size, mask_ratio, x.device)

        vis_patches = torch.gather(
            patches, 1, ids_visible.unsqueeze(-1).expand(-1, -1, patches.shape[2])
        )
        encoded = self.encoder(vis_patches)  # (N, V, E)

        mask_tokens = self.mask_token.expand(batch_size, ids_masked.shape[1], -1)
        full_tokens = torch.cat([encoded, mask_tokens], dim=1)  # (N, P, E)

        ids_restore = torch.argsort(
            torch.cat([ids_visible, ids_masked], dim=1), dim=1
        )
        full_tokens = torch.gather(
            full_tokens, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[2])
        )

        recon = self.decoder(full_tokens)  # (N, P, patch_size)

        patch_mask = torch.zeros(batch_size, self.num_patches, device=x.device, dtype=torch.bool)
        patch_mask.scatter_(1, ids_masked, True)

        return recon, patch_mask

    def encode_all(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all patches (no masking) and mean-pool for downstream.

        Args:
            x: Raw input ``(N, in_dim)``.

        Returns:
            Pooled representation ``(N, embed_dim)``.
        """
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        return encoded.mean(dim=1)


class MAETrainer:
    """Patch-based Masked Autoencoder trainer (He et al., 2022).

    The encoder processes only *visible* patches, and the lightweight
    decoder reconstructs all patches from encoder output + learnable mask
    tokens.  The reconstruction loss (MSE) is computed only on the masked
    patch positions.

    Args:
        dataset: Structured dataset with unlabeled features and labels.
        config: MAE hyper-parameters.  Uses defaults when *None*.

    Attributes:
        model: The internal ``_MAEModel`` after training.
        result: Training result container (available after :meth:`fit`).
    """

    def __init__(
        self,
        dataset: StructuredDataset,
        config: MAETrainConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.cfg = config or MAETrainConfig()
        self.model: _MAEModel | None = None
        self.result: MAEResult | None = None

    def fit(self) -> MAEResult:
        """Run MAE pretraining and downstream evaluation.

        Returns:
            :class:`MAEResult` with loss, metrics, and embeddings.
        """
        cfg = self.cfg
        set_torch_seed(cfg.seed)

        X = torch.as_tensor(self.dataset.X_unlabeled, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )

        model = _MAEModel(
            in_dim=self.dataset.observed_dim,
            num_patches=cfg.num_patches,
            embed_dim=cfg.embed_dim,
            encoder_depth=cfg.encoder_depth,
            decoder_dim=cfg.decoder_dim,
            decoder_depth=cfg.decoder_depth,
            dropout=cfg.dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        model.train()
        last_loss = 0.0
        for epoch in range(cfg.epochs):
            for (batch_x,) in loader:
                if batch_x.shape[1] < model.patch_embed.padded_dim:
                    batch_padded = F.pad(
                        batch_x, (0, model.patch_embed.padded_dim - batch_x.shape[1])
                    )
                else:
                    batch_padded = batch_x

                target_patches = batch_padded.view(
                    batch_padded.shape[0], cfg.num_patches, model.patch_size
                )

                recon, patch_mask = model(batch_x, mask_ratio=cfg.mask_ratio)

                mask_expanded = patch_mask.unsqueeze(-1).float()
                loss = F.mse_loss(
                    recon * mask_expanded,
                    target_patches * mask_expanded,
                    reduction="sum",
                ) / (mask_expanded.sum() * model.patch_size + 1e-8)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
            logger.debug("MAE epoch %d  loss=%.4f", epoch, last_loss)

        self.model = model

        model.eval()
        with torch.no_grad():
            embeddings = model.encode_all(X).cpu().numpy()

            if X.shape[1] < model.patch_embed.padded_dim:
                X_padded = F.pad(X, (0, model.patch_embed.padded_dim - X.shape[1]))
            else:
                X_padded = X
            target_all = X_padded.view(X.shape[0], cfg.num_patches, model.patch_size)
            recon_all, mask_all = model(X, mask_ratio=cfg.mask_ratio)
            X_recon_flat = recon_all.view(X.shape[0], -1)[:, :self.dataset.observed_dim].cpu().numpy()

        n_train = len(self.dataset.y_train)
        n_val = len(self.dataset.y_val)
        n_test = len(self.dataset.y_test)
        test_start = n_train + n_val

        X_test = self.dataset.X_test
        X_recon_test = X_recon_flat[test_start: test_start + n_test]
        train_emb = embeddings[:n_train]
        test_emb = embeddings[test_start: test_start + n_test]

        Z_all = self.dataset.ground_truth.get("Z_unlabeled")
        if Z_all is not None:
            Z_train = Z_all[:n_train].astype(np.float32)
            Z_test = Z_all[test_start: test_start + n_test].astype(np.float32)
        else:
            Z_train = None
            Z_test = None

        metrics = evaluate_masked_ae(
            X_test,
            X_recon_test,
            train_emb,
            test_emb,
            self.dataset.y_train,
            self.dataset.y_test,
            Z_train=Z_train,
            Z_test=Z_test,
            seed=cfg.seed,
        )

        self.result = MAEResult(
            model_name="MAE",
            train_loss=last_loss,
            metrics=metrics,
            embeddings=embeddings,
        )
        return self.result
