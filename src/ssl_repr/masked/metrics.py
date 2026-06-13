"""Masked autoencoder metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class MaskedAEMetrics:
    recon_mse: float
    latent_r2: float
    linear_probe_acc: float


def evaluate_masked_ae(
    X_true: np.ndarray,
    X_recon: np.ndarray,
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    Z_train: np.ndarray | None,
    Z_test: np.ndarray | None,
    seed: int = 42,
) -> MaskedAEMetrics:
    """Evaluate reconstruction quality and downstream probes."""
    recon_mse = float(mean_squared_error(X_true, X_recon))

    scaler = StandardScaler()
    tr = scaler.fit_transform(train_emb)
    te = scaler.transform(test_emb)

    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(tr, y_train)
    probe_acc = float(accuracy_score(y_test, clf.predict(te)))

    latent_r2 = 0.0
    if Z_train is not None and Z_test is not None:
        ridge = Ridge(alpha=1.0)
        ridge.fit(tr, Z_train)
        latent_r2 = float(r2_score(Z_test, ridge.predict(te)))

    return MaskedAEMetrics(
        recon_mse=recon_mse,
        latent_r2=latent_r2,
        linear_probe_acc=probe_acc,
    )
