"""Standalone evaluation metrics for frozen SSL representations.

Provides linear probe, k-NN classification, and geometric quality measures
(uniformity, alignment) that can be applied to any pre-extracted embedding
matrix.

References:
    - Wang & Isola, *Understanding Contrastive Representation Learning
      through Alignment and Uniformity on the Hypersphere*, ICML 2020.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------


@dataclass
class LinearProbeResult:
    """Result of a linear-probe evaluation.

    Args:
        accuracy: Classification accuracy on the test split.
        n_train: Number of training samples used.
        n_test: Number of test samples used.
    """

    accuracy: float
    n_train: int
    n_test: int


def linear_probe_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000,
    seed: int = 42,
) -> LinearProbeResult:
    """Fit logistic regression on frozen features and evaluate.

    Standardises the embeddings, fits an L2-regularised logistic regression,
    and reports test accuracy.  This is the standard linear evaluation
    protocol for self-supervised methods.

    Args:
        X_train: Training embeddings, shape ``(N_train, d)``.
        y_train: Training labels, shape ``(N_train,)``.
        X_test: Test embeddings, shape ``(N_test, d)``.
        y_test: Test labels, shape ``(N_test,)``.
        max_iter: Maximum solver iterations for convergence.
        seed: Random seed for the solver.

    Returns:
        :class:`LinearProbeResult` with accuracy and sample counts.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_test, preds))

    logger.debug("Linear probe accuracy: %.4f  (train=%d, test=%d)", acc, len(y_train), len(y_test))
    return LinearProbeResult(accuracy=acc, n_train=len(y_train), n_test=len(y_test))


# ---------------------------------------------------------------------------
# k-NN evaluation
# ---------------------------------------------------------------------------


@dataclass
class KNNResult:
    """Result of k-NN evaluation.

    Args:
        accuracy: Classification accuracy on the test split.
        k: Number of neighbours used.
    """

    accuracy: float
    k: int


def knn_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 200,
) -> KNNResult:
    """k-NN classification on frozen features.

    Uses cosine-normalised embeddings and a weighted k-NN classifier,
    following the protocol common in SSL evaluation.

    Args:
        X_train: Training embeddings, shape ``(N_train, d)``.
        y_train: Training labels, shape ``(N_train,)``.
        X_test: Test embeddings, shape ``(N_test, d)``.
        y_test: Test labels, shape ``(N_test,)``.
        k: Number of nearest neighbours.  Clamped to
            ``min(k, N_train)`` internally.

    Returns:
        :class:`KNNResult` with accuracy and effective *k*.
    """
    effective_k = min(k, len(y_train))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=effective_k, weights="distance", metric="minkowski")
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_test, preds))

    logger.debug("k-NN (k=%d) accuracy: %.4f", effective_k, acc)
    return KNNResult(accuracy=acc, k=effective_k)


# ---------------------------------------------------------------------------
# Geometric representation quality
# ---------------------------------------------------------------------------


def feature_uniformity(
    embeddings: np.ndarray,
    temperature: float = 2.0,
) -> float:
    r"""Uniformity of representations on the unit hypersphere.

    Measures how uniformly the (L2-normalised) embeddings are distributed.
    Lower values indicate more uniform spread.

    .. math::

        \mathcal{L}_{\text{uniform}} = \log\;
        \mathbb{E}_{(x,y) \sim p_{\text{data}}}
        \bigl[e^{-t \|f(x) - f(y)\|^2}\bigr]

    From Wang & Isola (2020).

    Args:
        embeddings: Feature matrix, shape ``(N, d)``.
        temperature: Temperature parameter :math:`t`.

    Returns:
        Scalar uniformity metric (log-scale; lower is more uniform).
    """
    z = torch.as_tensor(embeddings, dtype=torch.float32)
    z = torch.nn.functional.normalize(z, dim=1)
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    vals = sq_pdist[mask]
    return float(torch.log(torch.exp(-temperature * vals).mean()).item())


def feature_alignment(
    z1: np.ndarray,
    z2: np.ndarray,
) -> float:
    r"""Alignment between positive pairs.

    Measures the expected squared L2 distance between embeddings of
    matched (positive) pairs after normalisation:

    .. math::

        \mathcal{L}_{\text{align}} =
        \mathbb{E}_{(x, x^+)}
        \bigl[\|f(x) - f(x^+)\|^2\bigr]

    Lower values indicate that positive pairs are well-aligned.

    Args:
        z1: Embeddings of anchor samples, shape ``(N, d)``.
        z2: Embeddings of corresponding positive samples, shape ``(N, d)``.

    Returns:
        Mean squared L2 distance between normalised positive pairs.
    """
    a = torch.as_tensor(z1, dtype=torch.float32)
    b = torch.as_tensor(z2, dtype=torch.float32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return float((a - b).pow(2).sum(dim=1).mean().item())
