"""Dataset containers for self-supervised learning benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClusterDataset:
    """Gaussian-mixture cluster data with train/val/test splits for linear probing."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(np.concatenate([self.y_train, self.y_val, self.y_test]))))

    @property
    def X_unlabeled(self) -> np.ndarray:
        """All samples for unsupervised pretraining."""
        return np.vstack([self.X_train, self.X_val, self.X_test])


@dataclass
class StructuredDataset:
    """Linear latent-factor observations for masked autoencoding."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def observed_dim(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def X_unlabeled(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_val, self.X_test])
