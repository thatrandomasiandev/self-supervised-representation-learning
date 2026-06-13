"""Representation quality metrics for SSL evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class RepresentationMetrics:
    linear_probe_acc: float
    nmi: float
    silhouette: float
    embedding_std: float


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int = 42,
) -> float:
    """Train logistic regression on frozen embeddings."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ev = scaler.transform(X_eval)
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_ev)
    return float(accuracy_score(y_eval, preds))


def evaluate_representations(
    embeddings: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    train_size: int,
    val_size: int,
    n_clusters: int,
    seed: int = 42,
) -> RepresentationMetrics:
    """Evaluate embeddings with linear probe, NMI, and silhouette."""
    train_emb = embeddings[:train_size]
    val_emb = embeddings[train_size : train_size + val_size]
    test_emb = embeddings[train_size + val_size :]

    probe_acc = fit_linear_probe(train_emb, y_train, test_emb, y_test, seed=seed)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    all_labels = np.concatenate([y_train, y_val, y_test])
    cluster_preds = kmeans.fit_predict(embeddings)
    nmi = float(normalized_mutual_info_score(all_labels, cluster_preds))

    sil = float(silhouette_score(embeddings, all_labels))

    emb_std = float(np.mean(np.std(embeddings, axis=0)))

    return RepresentationMetrics(
        linear_probe_acc=probe_acc,
        nmi=nmi,
        silhouette=sil,
        embedding_std=emb_std,
    )
