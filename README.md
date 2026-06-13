# Self-Supervised Representation Learning

PhD-level SSL suite covering **contrastive learning** (SimCLR), **masked autoencoding**, and **non-contrastive regularization** (VICReg) — all evaluated on synthetic data with known ground truth.

## Modules

| Module | Description | Key metrics |
|--------|-------------|-------------|
| **Contrastive** | SimCLR with NT-Xent on augmented Gaussian-cluster views | Linear probe acc, NMI, silhouette |
| **Masked AE** | Feature masking with MLP encoder–decoder on latent-factor data | Reconstruction MSE, latent R² |
| **VICReg** | Variance–invariance–covariance regularization without negatives | Linear probe acc, NMI, collapse variance |

## Assumptions

- **Contrastive / VICReg:** Cluster structure is recoverable from augmentations; labels used only for linear probe evaluation
- **Masked AE:** Observed features are a noisy linear projection of low-dimensional latents; masking is i.i.d. per dimension
- **Linear probe:** Frozen encoder, logistic regression on held-out labels (standard SSL evaluation protocol)

## Setup

```bash
cd 07-self-supervised-representation-learning
pip install -e ".[dev]"
```

## Run benchmarks

```bash
# All modules
python scripts/run_benchmark.py --config configs/contrastive_benchmark.yaml --module all

# Individual modules
python scripts/run_benchmark.py --config configs/contrastive_benchmark.yaml --module contrastive
python scripts/run_benchmark.py --config configs/masked_benchmark.yaml --module masked
python scripts/run_benchmark.py --config configs/vicreg_benchmark.yaml --module vicreg
```

Results are written to `results/{timestamp}/metrics.json` and `summary.md`.

## Run tests

```bash
pytest
```

## Project layout

```
src/ssl_repr/
├── data/           # Cluster and structured-factor DGPs with ground-truth accessors
├── models/         # MLP encoders, projection heads, masked autoencoder
├── contrastive/    # SimCLR training, augmentations, metrics
├── masked/         # Masked autoencoder training and evaluation
├── vicreg/         # VICReg training and collapse diagnostics
└── evaluation/     # Benchmark runner and reporting
```

## Future work

- Vision augmentations on synthetic image patches (CIFAR-style DGP)
- BYOL / SimSiam with stop-gradient and predictor heads
- Multi-modal contrastive (CLIP-style) on paired synthetic modalities
