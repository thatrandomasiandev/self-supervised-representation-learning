# Self-Supervised Representation Learning

A research benchmark suite for **contrastive learning**, **masked autoencoding**, and **non-contrastive regularization** — the three dominant paradigms in self-supervised representation learning (SSL). All experiments use synthetic tabular data with known cluster structure or latent factors, enabling exact evaluation of representation quality via linear probes and clustering metrics.

The central research question: *which SSL objectives recover semantically meaningful representations, and under what data assumptions does each paradigm succeed or fail?*

---

## Research scope

| Module | Problem | Methods | Primary metrics |
|--------|---------|---------|-----------------|
| **Contrastive** | Learn representations invariant to augmentations | SimCLR with NT-Xent loss | Linear probe acc, NMI, silhouette |
| **Masked AE** | Reconstruct masked inputs to learn structure | Feature-masking autoencoder | Reconstruction MSE, latent R² |
| **VICReg** | Prevent collapse without negative pairs | Variance-invariance-covariance regularization | Linear probe acc, NMI, embedding variance |

---

## Module 1: Contrastive learning (SimCLR)

### Problem formulation

**Contrastive learning** trains an encoder f such that augmented views of the same instance are close in embedding space, while views of different instances are far apart. SimCLR (Chen et al., 2020) demonstrated that strong augmentations + a nonlinear projection head + large batch sizes suffice for competitive SSL without memory banks or momentum encoders.

### Loss function

**NT-Xent** (normalized temperature-scaled cross-entropy):

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where z_i, z_j are projections of two augmented views and τ is temperature. This is an instance of **InfoNCE** (Oord et al., 2018).

### Implemented components

- **SimCLR trainer** (`contrastive/simclr.py`) with NT-Xent loss
- **Tabular augmentations** (`contrastive/augmentations.py`): Gaussian noise, feature dropout, scaling
- **MLP encoder + projection head** (`models/encoder.py`)

### Synthetic DGP (`data/cluster_dgp.py`)

Gaussian mixture clusters in a low-dimensional subspace with **nuisance dimensions** — features uncorrelated with cluster identity. Tests whether contrastive learning ignores nuisance variation.

### Evaluation metrics

- **Linear probe accuracy:** Logistic regression on frozen encoder embeddings (standard SSL protocol; Kolesnikov et al., 2019)
- **NMI / silhouette:** Unsupervised clustering quality of embeddings

---

## Module 2: Masked autoencoding

### Problem formulation

**Masked autoencoders** (He et al., 2022; Devlin et al., 2019) learn representations by reconstructing randomly masked input features. The encoder must capture global structure to fill in missing information — analogous to BERT's masked language modeling (Devlin et al., 2019).

### Implemented method

- Random **feature masking** at ratio 0.4 (configurable)
- MLP encoder-decoder architecture
- Reconstruction loss on masked positions only

### Synthetic DGP (`data/structured_dgp.py`)

Observed features are a **noisy linear projection** of low-dimensional latent factors. Ground-truth latents enable R² measurement of representation fidelity.

### Evaluation metrics

- **Reconstruction MSE:** Accuracy of masked feature prediction
- **Latent R²:** Correlation between learned embeddings and true latent factors

---

## Module 3: VICReg

### Problem formulation

Non-contrastive methods avoid collapse (all embeddings → constant) without requiring negative pairs or large batches. **VICReg** (Bardes et al., 2022) uses three explicit regularization terms:

$$\mathcal{L} = \alpha \cdot \underbrace{\|z - z'\|^2}_{\text{invariance}} + \beta \cdot \underbrace{\max(0, \gamma - \text{std}(z))^2}_{\text{variance}} + \lambda \cdot \underbrace{\text{off-diagonal}(C)^2}_{\text{covariance}}$$

- **Invariance:** Augmented views should map to similar embeddings
- **Variance:** Prevent dimensional collapse by maintaining embedding spread
- **Covariance:** Decorrelate embedding dimensions to avoid redundant information

### Evaluation metrics

- **Linear probe accuracy:** Downstream classification on frozen embeddings
- **Embedding variance:** Diagnostic for collapse (should remain above γ)

---

## Benchmark protocol

```bash
pip install -e ".[dev]"

python scripts/run_benchmark.py --config configs/contrastive_benchmark.yaml --module all
python scripts/run_benchmark.py --config configs/contrastive_benchmark.yaml --module contrastive
python scripts/run_benchmark.py --config configs/masked_benchmark.yaml --module masked
python scripts/run_benchmark.py --config configs/vicreg_benchmark.yaml --module vicreg

pytest
```

---

## Project layout

```
src/ssl_repr/
├── data/           # Cluster and structured-factor DGPs
├── models/         # MLP encoders, projection heads, masked autoencoder
├── contrastive/    # SimCLR, augmentations, NT-Xent
├── masked/         # Masked autoencoder training
├── vicreg/         # VICReg with variance/covariance regularization
└── evaluation/     # Benchmark runner and reporting
```

---

## Implementation notes

- Experiments use **tabular data**, not images — augmentations are feature-level, not spatial
- Linear probe evaluation follows the protocol of Chen et al. (2020): train encoder with SSL, freeze, train logistic regression
- Labels are used **only** for linear probe evaluation, never during SSL training
- SimCLR batch size and temperature significantly affect results (Chen et al., 2020)

---

## References

- Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-invariance-covariance regularization for self-supervised learning. *ICLR*.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *CVPR*.
- Kolesnikov, A., Zhai, X., & Beyer, L. (2019). Revisiting self-supervised visual representation learning. *CVPR*.
- Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748*.

---

## Future work

- Vision augmentations on synthetic image patches (CIFAR-style DGP)
- BYOL (Grill et al., 2020) and SimSiam (Chen & He, 2021) non-contrastive baselines
- Multi-modal contrastive learning (CLIP; Radford et al., 2021)
