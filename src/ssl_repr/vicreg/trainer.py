"""VICReg training loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ssl_repr.contrastive.augmentations import augment_tabular
from ssl_repr.contrastive.metrics import RepresentationMetrics, evaluate_representations
from ssl_repr.data.base import ClusterDataset
from ssl_repr.models.encoder import MLPEncoder, ProjectionHead, vicreg_loss
from ssl_repr.utils.seed import set_torch_seed


@dataclass
class VICRegTrainConfig:
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
    model_name: str
    train_loss: float
    metrics: RepresentationMetrics
    embeddings: np.ndarray
    repr_std: float


class VICRegModel(torch.nn.Module):
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
        self.projector = ProjectionHead(repr_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def fit_vicreg(
    dataset: ClusterDataset,
    config: VICRegTrainConfig | None = None,
) -> VICRegResult:
    """Pretrain VICReg on unlabeled cluster data."""
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
    for _ in range(cfg.epochs):
        for (batch_x,) in loader:
            v1 = augment_tabular(batch_x)
            v2 = augment_tabular(batch_x)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = vicreg_loss(
                z1,
                z2,
                sim_coeff=cfg.sim_coeff,
                var_coeff=cfg.var_coeff,
                cov_coeff=cfg.cov_coeff,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

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
