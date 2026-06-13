"""VICReg and Barlow Twins training loops.

Implements variance-invariance-covariance regularisation (VICReg) and Barlow
Twins, both of which learn representations by encouraging embedding
invariance to augmentations while preventing informational collapse through
explicit redundancy-reduction terms.

References:
    - Bardes et al., *VICReg: Variance-Invariance-Covariance Regularization
      for Self-Supervised Learning*, ICLR 2022.
    - Zbontar et al., *Barlow Twins: Self-Supervised Learning via Redundancy
      Reduction*, ICML 2021.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ssl_repr.contrastive.augmentations import augment_tabular
from ssl_repr.contrastive.metrics import RepresentationMetrics, evaluate_representations
from ssl_repr.data.base import ClusterDataset
from ssl_repr.models.encoder import MLPEncoder
from ssl_repr.models.projection_head import ProjectionHead as BNProjectionHead
from ssl_repr.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decomposed VICReg loss
# ---------------------------------------------------------------------------


def invariance_loss(z: torch.Tensor, z_prime: torch.Tensor) -> torch.Tensor:
    r"""Invariance term :math:`S(Z, Z')`.

    Mean-squared distance between paired embeddings, encouraging views of
    the same sample to map to nearby points.

    .. math::

        S(Z, Z') = \frac{1}{d}\|Z - Z'\|_F^2

    Args:
        z: Embeddings from view 1, shape ``(N, d)``.
        z_prime: Embeddings from view 2, shape ``(N, d)``.

    Returns:
        Scalar invariance loss.
    """
    return F.mse_loss(z, z_prime)


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    r"""Variance term :math:`V(Z)`.

    Hinge loss on the per-dimension standard deviation, pushing each
    embedding dimension to maintain at least :math:`\gamma` standard
    deviation and preventing collapse to a point.

    .. math::

        V(Z) = \frac{1}{d}\sum_{i=1}^{d} \max\!\bigl(0,\;
        \gamma - \mathrm{std}(z_i)\bigr)

    Args:
        z: Embedding matrix, shape ``(N, d)``.
        gamma: Target minimum standard deviation per dimension.

    Returns:
        Scalar variance loss.
    """
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    r"""Covariance term :math:`C(Z)`.

    Penalises off-diagonal elements of the covariance matrix, encouraging
    decorrelated embedding dimensions.

    .. math::

        C(Z) = \frac{1}{d(d-1)} \sum_{i \neq j}
        \bigl[\mathrm{Cov}(Z)_{ij}\bigr]^2

    Args:
        z: Embedding matrix, shape ``(N, d)``.

    Returns:
        Scalar covariance loss.
    """
    n, d = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    return (off_diag ** 2).mean()


def vicreg_loss_decomposed(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Full VICReg objective with individual loss components.

    .. math::

        \mathcal{L} = \lambda_s S(Z, Z')
                     + \lambda_v \bigl[V(Z) + V(Z')\bigr]
                     + \lambda_c \bigl[C(Z) + C(Z')\bigr]

    Args:
        z1: Projections from view 1, shape ``(N, d)``.
        z2: Projections from view 2, shape ``(N, d)``.
        sim_coeff: Weight :math:`\lambda_s` for invariance term.
        var_coeff: Weight :math:`\lambda_v` for variance term.
        cov_coeff: Weight :math:`\lambda_c` for covariance term.
        gamma: Target standard deviation for the variance hinge.

    Returns:
        Tuple of ``(total_loss, sim_loss, var_loss, cov_loss)``.
    """
    sim = invariance_loss(z1, z2)
    var = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov = covariance_loss(z1) + covariance_loss(z2)
    total = sim_coeff * sim + var_coeff * var + cov_coeff * cov
    return total, sim, var, cov


# ---------------------------------------------------------------------------
# VICReg Trainer
# ---------------------------------------------------------------------------


@dataclass
class VICRegTrainConfig:
    """Configuration for VICReg pretraining.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden_dim: Width of encoder hidden layers.
        repr_dim: Encoder output dimensionality.
        projection_dim: Projector output dimensionality.
        n_layers: Number of encoder hidden layers.
        dropout: Dropout probability.
        sim_coeff: Weight for invariance (similarity) term.
        var_coeff: Weight for variance term.
        cov_coeff: Weight for covariance term.
        seed: Random seed for reproducibility.
    """

    epochs: int = 80
    batch_size: int = 256
    lr: float = 0.001
    hidden_dim: int = 128
    repr_dim: int = 64
    projection_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    sim_coeff: float = 25.0
    var_coeff: float = 25.0
    cov_coeff: float = 1.0
    seed: int = 42


@dataclass
class VICRegResult:
    """Container for VICReg training outcomes.

    Args:
        model_name: Identifier string.
        train_loss: Final training loss value.
        metrics: Downstream evaluation metrics.
        embeddings: Frozen encoder representations for all samples.
        repr_std: Mean per-dimension standard deviation of embeddings.
    """

    model_name: str
    train_loss: float
    metrics: RepresentationMetrics
    embeddings: np.ndarray
    repr_std: float


class VICRegModel(nn.Module):
    """VICReg encoder + projector.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Hidden layer width.
        repr_dim: Encoder output dimensionality.
        proj_dim: Projector output dimensionality.
        n_layers: Number of encoder hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        repr_dim: int,
        proj_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden_dim, repr_dim, n_layers, dropout)
        self.projector = BNProjectionHead(repr_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def fit_vicreg(
    dataset: ClusterDataset,
    config: VICRegTrainConfig | None = None,
) -> VICRegResult:
    """Pretrain VICReg on unlabeled cluster data.

    Args:
        dataset: Cluster dataset with unlabeled features and labels.
        config: Training hyper-parameters.  Uses defaults when *None*.

    Returns:
        :class:`VICRegResult` with final loss, metrics, and embeddings.
    """
    cfg = config or VICRegTrainConfig()
    set_torch_seed(cfg.seed)

    X = torch.as_tensor(dataset.X_unlabeled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = VICRegModel(
        in_dim=dataset.n_features,
        hidden_dim=cfg.hidden_dim,
        repr_dim=cfg.repr_dim,
        proj_dim=cfg.projection_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    last_loss = 0.0
    for epoch in range(cfg.epochs):
        for (batch_x,) in loader:
            v1 = augment_tabular(batch_x)
            v2 = augment_tabular(batch_x)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss, _, _, _ = vicreg_loss_decomposed(
                z1, z2,
                sim_coeff=cfg.sim_coeff,
                var_coeff=cfg.var_coeff,
                cov_coeff=cfg.cov_coeff,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        logger.debug("VICReg epoch %d  loss=%.4f", epoch, last_loss)

    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(X).cpu().numpy()
        repr_std = float(np.mean(np.std(embeddings, axis=0)))

    n_train = len(dataset.y_train)
    n_val = len(dataset.y_val)
    metrics = evaluate_representations(
        embeddings,
        dataset.y_train,
        dataset.y_val,
        dataset.y_test,
        train_size=n_train,
        val_size=n_val,
        n_clusters=dataset.n_clusters,
        seed=cfg.seed,
    )

    return VICRegResult(
        model_name="VICReg",
        train_loss=last_loss,
        metrics=metrics,
        embeddings=embeddings,
        repr_std=repr_std,
    )


# ---------------------------------------------------------------------------
# Barlow Twins
# ---------------------------------------------------------------------------


@dataclass
class BarlowTwinsTrainConfig:
    """Configuration for Barlow Twins pretraining.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden_dim: Width of encoder hidden layers.
        repr_dim: Encoder output dimensionality.
        projection_dim: Projector output dimensionality.
        n_layers: Number of encoder hidden layers.
        dropout: Dropout probability.
        lambd: Off-diagonal trade-off coefficient :math:`\\lambda`.
        seed: Random seed for reproducibility.
    """

    epochs: int = 80
    batch_size: int = 256
    lr: float = 0.001
    hidden_dim: int = 128
    repr_dim: int = 64
    projection_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    lambd: float = 0.005
    seed: int = 42


@dataclass
class BarlowTwinsResult:
    """Container for Barlow Twins training outcomes.

    Args:
        model_name: Identifier string.
        train_loss: Final training loss value.
        metrics: Downstream evaluation metrics.
        embeddings: Frozen encoder representations for all samples.
        repr_std: Mean per-dimension standard deviation of embeddings.
    """

    model_name: str
    train_loss: float
    metrics: RepresentationMetrics
    embeddings: np.ndarray
    repr_std: float


def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambd: float = 0.005,
) -> torch.Tensor:
    r"""Barlow Twins redundancy-reduction loss.

    Computes the cross-correlation matrix between normalised embeddings
    and penalises deviations from the identity:

    .. math::

        \mathcal{C}_{ij} = \frac{\sum_b z_{b,i}^{(1)}\, z_{b,j}^{(2)}}{N}

    .. math::

        \mathcal{L} = \sum_i (1 - \mathcal{C}_{ii})^2
                     + \lambda \sum_{i \neq j} \mathcal{C}_{ij}^2

    Args:
        z1: Projections from view 1, shape ``(N, d)``.
        z2: Projections from view 2, shape ``(N, d)``.
        lambd: Weight for off-diagonal terms.

    Returns:
        Scalar Barlow Twins loss.
    """
    n = z1.shape[0]
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-4)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-4)

    c = (z1_norm.T @ z2_norm) / n  # (d, d)

    d = c.shape[0]
    diag = torch.diagonal(c)
    on_diag = ((1.0 - diag) ** 2).sum()

    off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=c.device)
    off_diag = (c[off_diag_mask] ** 2).sum()

    return on_diag + lambd * off_diag


class BarlowTwinsTrainer:
    """Barlow Twins trainer for tabular data.

    Learns representations by making the cross-correlation matrix of
    embeddings from two augmented views approach the identity, thereby
    decorrelating embedding dimensions while encouraging invariance.

    Args:
        dataset: Cluster dataset with unlabeled features and labels.
        config: Barlow Twins hyper-parameters.  Uses defaults when *None*.

    Attributes:
        model: The VICRegModel used as backbone after training.
        result: Training result container (available after :meth:`fit`).
    """

    def __init__(
        self,
        dataset: ClusterDataset,
        config: BarlowTwinsTrainConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.cfg = config or BarlowTwinsTrainConfig()
        self.model: VICRegModel | None = None
        self.result: BarlowTwinsResult | None = None

    def fit(self) -> BarlowTwinsResult:
        """Run Barlow Twins pretraining and downstream evaluation.

        Returns:
            :class:`BarlowTwinsResult` with loss, metrics, and embeddings.
        """
        cfg = self.cfg
        set_torch_seed(cfg.seed)

        X = torch.as_tensor(self.dataset.X_unlabeled, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )

        model = VICRegModel(
            in_dim=self.dataset.n_features,
            hidden_dim=cfg.hidden_dim,
            repr_dim=cfg.repr_dim,
            proj_dim=cfg.projection_dim,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        model.train()
        last_loss = 0.0
        for epoch in range(cfg.epochs):
            for (batch_x,) in loader:
                v1 = augment_tabular(batch_x)
                v2 = augment_tabular(batch_x)
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = barlow_twins_loss(z1, z2, lambd=cfg.lambd)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
            logger.debug("BarlowTwins epoch %d  loss=%.4f", epoch, last_loss)

        self.model = model

        model.eval()
        with torch.no_grad():
            embeddings = model.encoder(X).cpu().numpy()
            repr_std = float(np.mean(np.std(embeddings, axis=0)))

        n_train = len(self.dataset.y_train)
        n_val = len(self.dataset.y_val)
        metrics = evaluate_representations(
            embeddings,
            self.dataset.y_train,
            self.dataset.y_val,
            self.dataset.y_test,
            train_size=n_train,
            val_size=n_val,
            n_clusters=self.dataset.n_clusters,
            seed=cfg.seed,
        )

        self.result = BarlowTwinsResult(
            model_name="BarlowTwins",
            train_loss=last_loss,
            metrics=metrics,
            embeddings=embeddings,
            repr_std=repr_std,
        )
        return self.result
