"""Contrastive self-supervised learning trainers.

Implements SimCLR, MoCo v2, and BYOL training loops for tabular data with
corresponding loss functions.

References:
    - Chen et al., *A Simple Framework for Contrastive Learning of Visual
      Representations* (SimCLR), ICML 2020.
    - He et al., *Momentum Contrast for Unsupervised Visual Representation
      Learning* (MoCo), CVPR 2020.
    - Grill et al., *Bootstrap Your Own Latent — A New Approach to
      Self-Supervised Learning* (BYOL), NeurIPS 2020.
"""

from __future__ import annotations

import copy
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
from ssl_repr.models.encoder import MLPEncoder, SimCLREncoder
from ssl_repr.models.projection_head import PredictionHead
from ssl_repr.models.projection_head import ProjectionHead as BNProjectionHead
from ssl_repr.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------


class NTXentLoss(nn.Module):
    r"""Normalised Temperature-scaled Cross-Entropy loss (NT-Xent).

    Given a batch of *2N* embeddings (two augmented views concatenated),
    the loss for a positive pair :math:`(i, j)` is:

    .. math::

        \ell_{i,j} = -\log
        \frac{\exp\bigl(\mathrm{sim}(z_i, z_j) / \tau\bigr)}
             {\sum_{k \neq i} \exp\bigl(\mathrm{sim}(z_i, z_k) / \tau\bigr)}

    where :math:`\mathrm{sim}(u, v) = u^\top v / (\|u\|\,\|v\|)` and
    :math:`\tau` is the temperature.

    Args:
        temperature: Scaling temperature :math:`\tau > 0`.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss over two views.

        Args:
            z1: Projections from view 1, shape ``(N, d)``.
            z2: Projections from view 2, shape ``(N, d)``.

        Returns:
            Scalar loss averaged over all 2N positive pairs.
        """
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        labels = torch.cat(
            [torch.arange(batch_size, device=z.device) + batch_size,
             torch.arange(batch_size, device=z.device)]
        )
        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# SimCLR
# ---------------------------------------------------------------------------


@dataclass
class SimCLRTrainConfig:
    """Configuration for SimCLR pretraining.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size (must be large enough for good negatives).
        lr: Adam learning rate.
        hidden_dim: Width of encoder hidden layers.
        repr_dim: Encoder output / representation dimensionality.
        projection_dim: Projector output dimensionality.
        n_layers: Number of encoder hidden layers.
        dropout: Dropout probability inside encoder.
        temperature: NT-Xent temperature :math:`\\tau`.
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
    temperature: float = 0.5
    seed: int = 42


@dataclass
class SimCLRResult:
    """Container for SimCLR training outcomes.

    Args:
        model_name: Identifier string.
        train_loss: Final training loss value.
        metrics: Downstream evaluation metrics.
        embeddings: Frozen encoder representations for all samples.
    """

    model_name: str
    train_loss: float
    metrics: RepresentationMetrics
    embeddings: np.ndarray


def fit_simclr(
    dataset: ClusterDataset,
    config: SimCLRTrainConfig | None = None,
) -> SimCLRResult:
    """Pretrain SimCLR on unlabeled data, then evaluate with linear probe.

    Args:
        dataset: Cluster dataset providing unlabeled features and labels for
            downstream evaluation.
        config: Training hyper-parameters.  Uses defaults when *None*.

    Returns:
        A :class:`SimCLRResult` with final loss, evaluation metrics, and
        extracted embeddings.
    """
    cfg = config or SimCLRTrainConfig()
    set_torch_seed(cfg.seed)

    X = torch.as_tensor(dataset.X_unlabeled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = SimCLREncoder(
        in_dim=dataset.n_features,
        hidden_dim=cfg.hidden_dim,
        repr_dim=cfg.repr_dim,
        proj_dim=cfg.projection_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = NTXentLoss(temperature=cfg.temperature)

    model.train()
    last_loss = 0.0
    for epoch in range(cfg.epochs):
        for (batch_x,) in loader:
            v1 = augment_tabular(batch_x)
            v2 = augment_tabular(batch_x)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        logger.debug("SimCLR epoch %d  loss=%.4f", epoch, last_loss)

    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(X).cpu().numpy()

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

    return SimCLRResult(
        model_name="SimCLR",
        train_loss=last_loss,
        metrics=metrics,
        embeddings=embeddings,
    )


# ---------------------------------------------------------------------------
# MoCo
# ---------------------------------------------------------------------------


@dataclass
class MoCoTrainConfig:
    """Configuration for Momentum Contrast pretraining.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden_dim: Width of encoder hidden layers.
        repr_dim: Encoder output dimensionality.
        projection_dim: Projector output dimensionality.
        n_layers: Number of encoder hidden layers.
        dropout: Dropout probability.
        temperature: InfoNCE temperature :math:`\\tau`.
        momentum: EMA coefficient for key encoder, :math:`m \\in [0, 1)`.
        queue_size: Number of keys maintained in the FIFO queue :math:`K`.
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
    temperature: float = 0.07
    momentum: float = 0.999
    queue_size: int = 65536
    seed: int = 42


@dataclass
class MoCoResult:
    """Container for MoCo training outcomes.

    Args:
        model_name: Identifier string.
        train_loss: Final training loss value.
        metrics: Downstream evaluation metrics.
        embeddings: Frozen encoder representations for all samples.
    """

    model_name: str
    train_loss: float
    metrics: RepresentationMetrics
    embeddings: np.ndarray


class _MoCoModel(nn.Module):
    """Internal MoCo v2 model with momentum-updated key encoder and queue."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        repr_dim: int,
        proj_dim: int,
        n_layers: int,
        dropout: float,
        momentum: float,
        queue_size: int,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        self.query_encoder = MLPEncoder(in_dim, hidden_dim, repr_dim, n_layers, dropout)
        self.query_projector = BNProjectionHead(repr_dim, hidden_dim, proj_dim)

        self.key_encoder = copy.deepcopy(self.query_encoder)
        self.key_projector = copy.deepcopy(self.query_projector)
        for param in list(self.key_encoder.parameters()) + list(self.key_projector.parameters()):
            param.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self) -> None:
        r"""EMA update: :math:`\theta_k = m \theta_k + (1 - m) \theta_q`."""
        for p_q, p_k in zip(
            list(self.query_encoder.parameters()) + list(self.query_projector.parameters()),
            list(self.key_encoder.parameters()) + list(self.key_projector.parameters()),
        ):
            p_k.data.mul_(self.momentum).add_(p_q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        """Enqueue a batch of keys into the FIFO buffer."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        remaining = self.queue_size - ptr
        if batch_size <= remaining:
            self.queue[:, ptr: ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, : batch_size - remaining] = keys[remaining:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(
        self, x_q: torch.Tensor, x_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute query embeddings, positive-key embeddings, and queue.

        Args:
            x_q: Query view, shape ``(N, D)``.
            x_k: Key view, shape ``(N, D)``.

        Returns:
            Tuple of (queries, positive_keys, negative_queue).
        """
        q = F.normalize(self.query_projector(self.query_encoder(x_q)), dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.key_projector(self.key_encoder(x_k)), dim=1)

        neg_queue = self.queue.clone().detach()
        self._enqueue(k)

        return q, k, neg_queue


def _infonce_with_queue(
    q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float
) -> torch.Tensor:
    r"""InfoNCE loss using a momentum-maintained queue of negatives.

    .. math::

        \mathcal{L} = -\log
        \frac{\exp(q \cdot k_+ / \tau)}
             {\exp(q \cdot k_+ / \tau) + \sum_{k_-} \exp(q \cdot k_- / \tau)}

    Args:
        q: Query embeddings ``(N, d)``.
        k: Positive key embeddings ``(N, d)``.
        queue: Negative keys ``(d, K)``.
        temperature: Scaling temperature :math:`\tau`.

    Returns:
        Scalar InfoNCE loss.
    """
    l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(1)  # (N, 1)
    l_neg = torch.einsum("nc,ck->nk", q, queue)  # (N, K)
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


class MoCoTrainer:
    """Momentum Contrast (MoCo v2) trainer for tabular data.

    Maintains a slowly-updated *key encoder* via exponential moving average
    and a FIFO queue of :math:`K` negative keys, enabling a large effective
    number of negatives independent of batch size.

    Args:
        dataset: Cluster dataset providing unlabeled features and labels.
        config: MoCo hyper-parameters.  Uses defaults when *None*.

    Attributes:
        model: The internal ``_MoCoModel`` after training.
        result: Training result container (available after :meth:`fit`).
    """

    def __init__(
        self,
        dataset: ClusterDataset,
        config: MoCoTrainConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.cfg = config or MoCoTrainConfig()
        self.model: _MoCoModel | None = None
        self.result: MoCoResult | None = None

    def fit(self) -> MoCoResult:
        """Run MoCo pretraining and downstream evaluation.

        Returns:
            :class:`MoCoResult` with loss, metrics, and embeddings.
        """
        cfg = self.cfg
        set_torch_seed(cfg.seed)

        effective_queue = min(
            cfg.queue_size,
            len(self.dataset.X_unlabeled) * 4,
        )

        X = torch.as_tensor(self.dataset.X_unlabeled, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )

        model = _MoCoModel(
            in_dim=self.dataset.n_features,
            hidden_dim=cfg.hidden_dim,
            repr_dim=cfg.repr_dim,
            proj_dim=cfg.projection_dim,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            momentum=cfg.momentum,
            queue_size=effective_queue,
        )
        optimizer = torch.optim.Adam(
            list(model.query_encoder.parameters()) + list(model.query_projector.parameters()),
            lr=cfg.lr,
        )

        model.train()
        last_loss = 0.0
        for epoch in range(cfg.epochs):
            for (batch_x,) in loader:
                x_q = augment_tabular(batch_x)
                x_k = augment_tabular(batch_x)
                q, k, neg_queue = model(x_q, x_k)
                loss = _infonce_with_queue(q, k, neg_queue, temperature=cfg.temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
            logger.debug("MoCo epoch %d  loss=%.4f", epoch, last_loss)

        self.model = model

        model.eval()
        with torch.no_grad():
            embeddings = model.query_encoder(X).cpu().numpy()

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

        self.result = MoCoResult(
            model_name="MoCo",
            train_loss=last_loss,
            metrics=metrics,
            embeddings=embeddings,
        )
        return self.result


# ---------------------------------------------------------------------------
# BYOL
# ---------------------------------------------------------------------------


@dataclass
class BYOLTrainConfig:
    """Configuration for Bootstrap Your Own Latent pretraining.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden_dim: Width of encoder hidden layers.
        repr_dim: Encoder output dimensionality.
        projection_dim: Projector / predictor output dimensionality.
        n_layers: Number of encoder hidden layers.
        dropout: Dropout probability.
        momentum: EMA coefficient for target network.
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
    momentum: float = 0.996
    seed: int = 42


@dataclass
class BYOLResult:
    """Container for BYOL training outcomes.

    Args:
        model_name: Identifier string.
        train_loss: Final training loss value.
        metrics: Downstream evaluation metrics.
        embeddings: Frozen encoder representations for all samples.
    """

    model_name: str
    train_loss: float
    metrics: RepresentationMetrics
    embeddings: np.ndarray


class _BYOLModel(nn.Module):
    """Internal BYOL model with online + target networks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        repr_dim: int,
        proj_dim: int,
        n_layers: int,
        dropout: float,
        momentum: float,
    ) -> None:
        super().__init__()
        self.momentum = momentum

        self.online_encoder = MLPEncoder(in_dim, hidden_dim, repr_dim, n_layers, dropout)
        self.online_projector = BNProjectionHead(repr_dim, hidden_dim, proj_dim)
        self.predictor = PredictionHead(proj_dim, hidden_dim, proj_dim)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            param.requires_grad = False

    @torch.no_grad()
    def _ema_update(self) -> None:
        r"""EMA update: :math:`\theta_t = m \theta_t + (1 - m) \theta_o`."""
        for p_o, p_t in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters()),
        ):
            p_t.data.mul_(self.momentum).add_(p_o.data, alpha=1.0 - self.momentum)

    @staticmethod
    def _byol_loss(p: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        r"""BYOL regression loss.

        .. math::

            \mathcal{L} = 2 - 2 \cdot
            \frac{p \cdot \bar{z}}{\|p\| \, \|\bar{z}\|}

        Args:
            p: Online prediction, shape ``(N, d)``.
            z_target: Target projection (detached), shape ``(N, d)``.

        Returns:
            Scalar mean loss.
        """
        p = F.normalize(p, dim=1)
        z_target = F.normalize(z_target, dim=1)
        return (2.0 - 2.0 * (p * z_target).sum(dim=1)).mean()

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Compute symmetrised BYOL loss for two views.

        Args:
            v1: First augmented view ``(N, D)``.
            v2: Second augmented view ``(N, D)``.

        Returns:
            Scalar BYOL loss.
        """
        p1 = self.predictor(self.online_projector(self.online_encoder(v1)))
        p2 = self.predictor(self.online_projector(self.online_encoder(v2)))

        with torch.no_grad():
            self._ema_update()
            z1_target = self.target_projector(self.target_encoder(v1))
            z2_target = self.target_projector(self.target_encoder(v2))

        loss = self._byol_loss(p1, z2_target.detach()) + self._byol_loss(p2, z1_target.detach())
        return loss / 2.0


class BYOLTrainer:
    """Bootstrap Your Own Latent (BYOL) trainer for tabular data.

    Trains an online network and a slowly-updated target network.  The
    online branch includes a *predictor* MLP that introduces asymmetry,
    allowing learning without negative pairs.  The loss minimises the
    cosine distance between the online prediction and the (stopped-gradient)
    target projection.

    Args:
        dataset: Cluster dataset providing unlabeled features and labels.
        config: BYOL hyper-parameters.  Uses defaults when *None*.

    Attributes:
        model: The internal ``_BYOLModel`` after training.
        result: Training result container (available after :meth:`fit`).
    """

    def __init__(
        self,
        dataset: ClusterDataset,
        config: BYOLTrainConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.cfg = config or BYOLTrainConfig()
        self.model: _BYOLModel | None = None
        self.result: BYOLResult | None = None

    def fit(self) -> BYOLResult:
        """Run BYOL pretraining and downstream evaluation.

        Returns:
            :class:`BYOLResult` with loss, metrics, and embeddings.
        """
        cfg = self.cfg
        set_torch_seed(cfg.seed)

        X = torch.as_tensor(self.dataset.X_unlabeled, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )

        model = _BYOLModel(
            in_dim=self.dataset.n_features,
            hidden_dim=cfg.hidden_dim,
            repr_dim=cfg.repr_dim,
            proj_dim=cfg.projection_dim,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            momentum=cfg.momentum,
        )
        optimizer = torch.optim.Adam(
            list(model.online_encoder.parameters())
            + list(model.online_projector.parameters())
            + list(model.predictor.parameters()),
            lr=cfg.lr,
        )

        model.train()
        last_loss = 0.0
        for epoch in range(cfg.epochs):
            for (batch_x,) in loader:
                v1 = augment_tabular(batch_x)
                v2 = augment_tabular(batch_x)
                loss = model(v1, v2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
            logger.debug("BYOL epoch %d  loss=%.4f", epoch, last_loss)

        self.model = model

        model.eval()
        with torch.no_grad():
            embeddings = model.online_encoder(X).cpu().numpy()

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

        self.result = BYOLResult(
            model_name="BYOL",
            train_loss=last_loss,
            metrics=metrics,
            embeddings=embeddings,
        )
        return self.result
