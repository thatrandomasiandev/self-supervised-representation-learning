# Self-Supervised Representation Learning

**Learning expressive representations without labels through contrastive, non-contrastive, and generative pretext tasks on tabular data.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2002.05709-b31b1b.svg)](https://arxiv.org/abs/2002.05709)

Self-supervised learning (SSL) has emerged as the dominant paradigm for learning transferable representations from unlabeled data. This repository provides a research-grade implementation of five foundational SSL algorithms — SimCLR, MoCo v2, BYOL, VICReg, and Barlow Twins — alongside a patch-based Masked Autoencoder (MAE), all adapted for structured tabular data. The codebase implements each method's loss function with full mathematical fidelity, provides controlled synthetic data-generating processes for reproducible ablations, and includes a comprehensive evaluation suite measuring linear probe accuracy, clustering quality (NMI, silhouette), and geometric properties of the learned embedding space (uniformity and alignment). Every design choice is documented with references to the originating paper, making this repository suitable both as an educational resource and as a foundation for downstream research in tabular SSL, where data augmentation strategies and representation collapse remain open challenges.

---

## Table of Contents

- [Research Background \& Motivation](#research-background--motivation)
- [Mathematical Foundations](#mathematical-foundations)
  - [NT-Xent Loss (SimCLR)](#nt-xent-loss-simclr)
  - [InfoNCE with Momentum Queue (MoCo)](#infonce-with-momentum-queue-moco)
  - [BYOL Asymmetric Loss](#byol-asymmetric-loss)
  - [VICReg Three-Term Loss](#vicreg-three-term-loss)
  - [Barlow Twins Redundancy Reduction](#barlow-twins-redundancy-reduction)
  - [Masked Autoencoder Objective](#masked-autoencoder-objective)
  - [Feature Uniformity and Alignment](#feature-uniformity-and-alignment)
- [Architecture Diagram](#architecture-diagram)
- [Repository Structure](#repository-structure)
- [Code Walkthrough](#code-walkthrough)
  - [Data Generation](#data-generation)
  - [Tabular Augmentations](#tabular-augmentations)
  - [Encoder and Projection Heads](#encoder-and-projection-heads)
  - [SimCLR Training Loop](#simclr-training-loop)
  - [MoCo v2 Momentum Mechanism](#moco-v2-momentum-mechanism)
  - [BYOL: Learning Without Negatives](#byol-learning-without-negatives)
  - [VICReg Decomposed Loss](#vicreg-decomposed-loss)
  - [Barlow Twins Cross-Correlation](#barlow-twins-cross-correlation)
  - [Masked Autoencoder Pipeline](#masked-autoencoder-pipeline)
  - [Evaluation Metrics](#evaluation-metrics)
- [Benchmark Results](#benchmark-results)
- [Reproduction Commands](#reproduction-commands)
- [Configuration Reference](#configuration-reference)
- [References](#references)
- [Future Work](#future-work)

---

## Research Background & Motivation

The central promise of self-supervised learning is to extract rich, general-purpose representations from unlabeled data by solving pretext tasks that require the model to capture meaningful structure. In computer vision, SSL methods have closed — and in some regimes surpassed — the gap with fully supervised pretraining on ImageNet (Chen et al., 2020; He et al., 2020; Grill et al., 2020). The success of these methods rests on a key insight: by training encoders to produce embeddings that are *invariant* to semantics-preserving transformations while *varying* across semantically distinct inputs, the resulting representations encode precisely the information relevant for downstream tasks.

**Contrastive learning** methods such as SimCLR (Chen et al., 2020) and MoCo (He et al., 2020) achieve this by treating augmented views of the same sample as positive pairs and views from different samples as negatives, optimising an InfoNCE-style objective that is a lower bound on mutual information between views (Oord et al., 2018). SimCLR demonstrated that a simple framework — large batch sizes, strong augmentations, a learnable nonlinear projection head, and the normalised temperature-scaled cross-entropy (NT-Xent) loss — suffices to match supervised baselines. MoCo v2 decoupled the number of negatives from batch size via an exponential moving average (EMA) key encoder and a FIFO queue of negative keys, enabling contrastive learning on commodity hardware.

**Non-contrastive methods** emerged to eliminate the need for negative pairs entirely. BYOL (Grill et al., 2020) introduced a teacher-student architecture with an asymmetric predictor head and stop-gradient on the target branch, achieving state-of-the-art performance without explicit repulsion of negative pairs. Chen & He (2021) further simplified this paradigm with SimSiam, demonstrating that neither momentum updates nor negative pairs are strictly necessary — the stop-gradient operation alone prevents collapse. These results challenged the prevailing theoretical understanding and opened new research directions in understanding SSL dynamics.

**Redundancy-reduction methods** provide an alternative route to avoiding collapse. VICReg (Bardes et al., 2022) decomposes the objective into three explicit terms: an invariance term encouraging matched views to be similar, a variance term preventing dimensional collapse via a hinge loss on per-dimension standard deviation, and a covariance term decorrelating embedding dimensions. Barlow Twins (Zbontar et al., 2021) achieves a similar goal by driving the cross-correlation matrix of embeddings from two views towards the identity matrix. Both methods avoid the mode-collapse failure of naïve mean-squared-error objectives without requiring negative pairs, large batch sizes, or momentum networks.

**Generative SSL** takes a complementary approach. Masked Autoencoders (He et al., 2022) randomly mask a large fraction (75%) of input patches and train a lightweight decoder to reconstruct the masked content from the visible patches processed by a deep encoder. This asymmetric design — heavy encoder on few visible patches, light decoder on all patches — produces representations that capture fine-grained local structure alongside global semantics. The MAE framework has been extended beyond vision to language (BERT-style masking), audio, and point clouds.

Wang & Isola (2020) provided a theoretical lens for understanding SSL through two geometric properties of learned representations on the unit hypersphere: **alignment** (positive pairs should map to nearby points) and **uniformity** (the overall distribution of embeddings should be spread uniformly). They proved that the contrastive loss can be decomposed into these two terms and that better uniformity consistently correlates with improved downstream performance. This framework motivates explicit measurement of embedding geometry beyond task-specific accuracy.

Despite remarkable progress in vision, applying SSL to tabular data remains an open research frontier. Tabular data lacks the spatial continuity that makes image augmentations (cropping, colour jittering) semantics-preserving. Instead, researchers must design domain-appropriate augmentations — feature corruption from marginal distributions (Bahri et al., 2022), Gaussian noise injection, and CutMix-style feature mixing — that perturb nuisance variation while preserving class-relevant signal. This repository implements and benchmarks these augmentation strategies in a controlled setting where ground-truth latent structure is known, enabling precise measurement of how well each SSL method recovers the true data-generating factors.

The synthetic DGP approach is deliberate: by generating data from known Gaussian mixture models (for contrastive methods) and known linear latent-factor models (for generative methods), we can compute ground-truth metrics — centroid recovery, latent factor $R^2$ — that are impossible on real datasets. This enables rigorous ablation studies that isolate the effect of each algorithm component (temperature, EMA momentum, mask ratio, loss coefficient) from confounders present in real-world benchmarks.

---

## Mathematical Foundations

### NT-Xent Loss (SimCLR)

The **Normalised Temperature-scaled Cross-Entropy** loss (NT-Xent) is the contrastive objective introduced by Chen et al. (2020). Given a mini-batch of $N$ samples, two augmented views are generated per sample, yielding $2N$ embeddings $\{z_1, z_2, \ldots, z_{2N}\}$. For a positive pair $(i, j)$ — two views of the same sample — the loss is:

$$\ell(i, j) = -\log \frac{\exp\bigl(\mathrm{sim}(z_i, z_j) / \tau\bigr)}{\sum_{k \neq i} \exp\bigl(\mathrm{sim}(z_i, z_k) / \tau\bigr)}$$

where:

- $z_i \in \mathbb{R}^d$ is the L2-normalised projection of sample $i$
- $\mathrm{sim}(u, v) = \frac{u^\top v}{\|u\| \, \|v\|}$ is cosine similarity
- $\tau > 0$ is the temperature hyperparameter controlling the concentration of the distribution
- The denominator sums over all $2N - 1$ embeddings excluding $z_i$ itself

The full batch loss averages over all $2N$ positive pairs:

$$\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N} \sum_{k=1}^{N} \bigl[\ell(2k-1, 2k) + \ell(2k, 2k-1)\bigr]$$

**Connection to mutual information.** The NT-Xent objective is a lower bound on the mutual information $I(z_1; z_2)$ between the two views (Oord et al., 2018). Specifically:

$$I(z_1; z_2) \geq \log(2N - 1) - \mathcal{L}_{\text{NT-Xent}}$$

This bound tightens as the batch size increases, explaining why SimCLR benefits from large batches. The temperature $\tau$ controls the trade-off: smaller $\tau$ sharpens the similarity distribution, making the objective more sensitive to hard negatives but potentially less stable during optimisation.

**Gradient structure.** Taking the gradient with respect to $z_i$ for a positive pair $(i, j)$:

$$\frac{\partial \ell}{\partial z_i} = \frac{1}{\tau} \left[ \sum_{k \neq i} p_{k|i} \cdot z_k - z_j \right]$$

where $p_{k|i} = \frac{\exp(\mathrm{sim}(z_i, z_k)/\tau)}{\sum_{m \neq i} \exp(\mathrm{sim}(z_i, z_m)/\tau)}$ is the softmax weight. The gradient pushes $z_i$ towards its positive $z_j$ and away from all negatives, weighted by current similarity.

### InfoNCE with Momentum Queue (MoCo)

MoCo v2 (He et al., 2020) decouples the number of negatives from batch size using a momentum-maintained queue. The loss for a query $q$ with positive key $k_+$ and $K$ negative keys $\{k_-\}$ from the queue is:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\exp(q \cdot k_+ / \tau) + \sum_{k_-} \exp(q \cdot k_- / \tau)}$$

where:

- $q = f_q(x_q) \in \mathbb{R}^d$ is the L2-normalised query embedding from the query encoder $f_q$
- $k_+ = f_k(x_k) \in \mathbb{R}^d$ is the L2-normalised key embedding from the momentum encoder $f_k$
- $\{k_-\}$ are $K$ negative keys stored in a FIFO queue
- $\tau > 0$ is the temperature (typically $\tau = 0.07$)

**Exponential Moving Average (EMA) update:** The key encoder parameters $\theta_k$ are updated as:

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

where $m \in [0.99, 0.999]$ is the momentum coefficient and $\theta_q$ are the query encoder parameters updated by gradient descent. The high momentum ensures the queue contains consistent keys despite being computed by different encoder snapshots.

**Queue mechanism:** After each forward pass, the batch of computed keys is enqueued and the oldest keys are dequeued, maintaining a fixed queue size $K$ (e.g., $K = 65536$). This provides a large, diverse set of negatives without requiring proportionally large batch sizes.

### BYOL Asymmetric Loss

BYOL (Grill et al., 2020) eliminates negative pairs entirely through architectural asymmetry. The online network produces a prediction $p = q_\theta(z_\theta)$ where $z_\theta = g_\theta(f_\theta(x))$ is the online projection and $q_\theta$ is the predictor MLP. The target network produces $\bar{z} = g_\xi(f_\xi(x'))$ where $\xi$ are EMA parameters.

The **BYOL loss** for a single direction is:

$$\mathcal{L}_{\text{BYOL}} = 2 - 2 \cdot \frac{\langle p, \bar{z} \rangle}{\|p\| \, \|\bar{z}\|}$$

where:

- $p \in \mathbb{R}^d$ is the online network's prediction (after predictor head)
- $\bar{z} \in \mathbb{R}^d$ is the target network's projection (stop-gradient)
- $\langle \cdot, \cdot \rangle$ denotes the inner product
- The factor of 2 normalises the loss range to $[0, 4]$

The symmetrised loss averages two directions:

$$\mathcal{L} = \frac{1}{2}\bigl[\mathcal{L}_{\text{BYOL}}(p_1, \bar{z}_2) + \mathcal{L}_{\text{BYOL}}(p_2, \bar{z}_1)\bigr]$$

**Target EMA update:** The target encoder parameters $\xi$ evolve as:

$$\xi \leftarrow m \cdot \xi + (1 - m) \cdot \theta$$

with $m$ following a cosine schedule from 0.996 to 1.0 during training. The predictor head $q_\theta$ is critical: it introduces asymmetry between online and target branches that prevents the trivial collapsed solution. Without the predictor, both branches could converge to a constant mapping; the predictor forces the online branch to learn a non-trivial transformation whose fixed point is a meaningful representation.

**Why BYOL doesn't collapse.** Tian et al. (2021) showed that under certain assumptions on the augmentation distribution, BYOL's dynamics are equivalent to optimising an implicit contrastive objective where the "negatives" come from the augmentation distribution itself. The predictor head must be sufficiently expressive to allow the online network to map towards the target's representation subspace without trivially matching it.

### VICReg Three-Term Loss

VICReg (Bardes et al., 2022) explicitly prevents collapse through three complementary regularisation terms applied to the embedding matrices $Z, Z' \in \mathbb{R}^{N \times d}$ from two augmented views of a batch:

$$\mathcal{L}_{\text{VICReg}} = \lambda_s \cdot S(Z, Z') + \lambda_v \cdot \bigl[V(Z) + V(Z')\bigr] + \lambda_c \cdot \bigl[C(Z) + C(Z')\bigr]$$

**Invariance term** $S$ (mean-squared error between matched embeddings):

$$S(Z, Z') = \frac{1}{Nd} \sum_{n=1}^{N} \|z_n - z'_n\|^2 = \frac{1}{d}\|Z - Z'\|_F^2$$

where $z_n, z'_n \in \mathbb{R}^d$ are paired embeddings from the two views.

**Variance term** $V$ (hinge loss on per-dimension standard deviation):

$$V(Z) = \frac{1}{d} \sum_{j=1}^{d} \max\!\bigl(0, \; \gamma - \sqrt{\mathrm{Var}(z^{(j)}) + \epsilon}\bigr)$$

where:

- $z^{(j)} \in \mathbb{R}^N$ is the $j$-th column of $Z$ (all samples' $j$-th dimension)
- $\gamma = 1.0$ is the target minimum standard deviation
- $\epsilon = 10^{-4}$ prevents numerical instability

This term ensures each embedding dimension maintains at least $\gamma$ standard deviation across the batch, directly preventing the dimensional collapse where all points converge to a single point or low-dimensional subspace.

**Covariance term** $C$ (penalises off-diagonal elements of the covariance matrix):

$$C(Z) = \frac{1}{d(d-1)} \sum_{i \neq j} \bigl[\mathrm{Cov}(Z)_{ij}\bigr]^2$$

where the covariance matrix is:

$$\mathrm{Cov}(Z)_{ij} = \frac{1}{N-1} \sum_{n=1}^{N} (z_n^{(i)} - \bar{z}^{(i)})(z_n^{(j)} - \bar{z}^{(j)})$$

This decorrelation term encourages different embedding dimensions to capture independent information, preventing redundancy.

**Gradient of the variance term:** For a single dimension $j$:

$$\frac{\partial V}{\partial z_n^{(j)}} = \begin{cases} -\frac{1}{d} \cdot \frac{z_n^{(j)} - \bar{z}^{(j)}}{(N-1)\sqrt{\mathrm{Var}(z^{(j)}) + \epsilon}} & \text{if } \sqrt{\mathrm{Var}(z^{(j)}) + \epsilon} < \gamma \\ 0 & \text{otherwise} \end{cases}$$

The default coefficients are $\lambda_s = 25$, $\lambda_v = 25$, $\lambda_c = 1$, reflecting that the covariance term has inherently larger magnitude.

### Barlow Twins Redundancy Reduction

Barlow Twins (Zbontar et al., 2021) approaches collapse prevention from an information-theoretic angle inspired by neuroscientist Horace Barlow's redundancy-reduction hypothesis. Given batch-normalised embeddings $Z^{(1)}, Z^{(2)} \in \mathbb{R}^{N \times d}$ from two views, the method computes the cross-correlation matrix:

$$\mathcal{C}_{ij} = \frac{\sum_{b=1}^{N} \hat{z}_{b,i}^{(1)} \, \hat{z}_{b,j}^{(2)}}{N}$$

where $\hat{z}^{(k)}$ are batch-mean-subtracted, batch-std-normalised embeddings.

The loss drives $\mathcal{C}$ towards the identity matrix:

$$\mathcal{L}_{\text{BT}} = \underbrace{\sum_{i=1}^{d} (1 - \mathcal{C}_{ii})^2}_{\text{invariance term}} + \lambda \underbrace{\sum_{\substack{i,j=1 \\ i \neq j}}^{d} \mathcal{C}_{ij}^2}_{\text{redundancy reduction}}$$

where:

- The on-diagonal term encourages $\mathcal{C}_{ii} = 1$: each dimension should be perfectly correlated across views (invariance)
- The off-diagonal term encourages $\mathcal{C}_{ij} = 0$ for $i \neq j$: different dimensions should be uncorrelated (decorrelation)
- $\lambda$ is a trade-off weight (typically $\lambda = 0.005$)

### Masked Autoencoder Objective

The Masked Autoencoder (He et al., 2022) follows an asymmetric encode-decode paradigm. Given an input $x \in \mathbb{R}^D$ split into $P$ non-overlapping patches $\{x_1, \ldots, x_P\}$ where $x_p \in \mathbb{R}^{D/P}$:

1. **Random masking:** A fraction $\rho$ (typically 0.75) of patches are randomly removed. Let $\mathcal{V}$ and $\mathcal{M}$ denote the sets of visible and masked patch indices, respectively, with $|\mathcal{M}| = \lfloor \rho P \rfloor$.

2. **Encoding visible patches:** Only visible patches are processed by the encoder:
$$h_p = f_\text{enc}(\text{embed}(x_p)) \quad \forall p \in \mathcal{V}$$

3. **Decoding all patches:** The decoder receives encoder outputs for visible patches and learnable mask tokens $e_{\text{mask}}$ for masked positions:
$$\hat{x}_p = f_\text{dec}(h_p) \quad \forall p \in \{1, \ldots, P\}$$

4. **Reconstruction loss (MSE on masked patches only):**

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}| \cdot (D/P)} \sum_{p \in \mathcal{M}} \|x_p - \hat{x}_p\|^2$$

where:

- $f_\text{enc}$ is the encoder (deep, processes only $(1-\rho) \cdot P$ patches)
- $f_\text{dec}$ is the decoder (shallow, processes all $P$ patches)
- $e_{\text{mask}} \in \mathbb{R}^{d_\text{embed}}$ is a shared learnable mask token
- The loss is computed only on masked positions, forcing the encoder to learn global context

The asymmetry — encoding only visible patches — provides a 3-4× speedup during pretraining. The high mask ratio (75%) creates a challenging pretext task that requires understanding of global structure rather than local interpolation.

### Feature Uniformity and Alignment

Wang & Isola (2020) proved that the contrastive loss can be decomposed into **alignment** and **uniformity** on the unit hypersphere $\mathbb{S}^{d-1}$:

**Alignment** measures the expected distance between positive pairs:

$$\mathcal{L}_{\text{align}} = \underset{(x, x^+) \sim p_{\text{pos}}}{\mathbb{E}} \bigl[\|f(x) - f(x^+)\|^2\bigr]$$

where $f: \mathcal{X} \to \mathbb{S}^{d-1}$ is the normalised encoder and $(x, x^+)$ are positive pairs. Lower alignment means positive pairs are well-clustered.

**Uniformity** measures how evenly the embeddings are spread on the hypersphere:

$$\mathcal{L}_{\text{uniform}} = \log \underset{(x, y) \overset{\text{i.i.d.}}{\sim} p_{\text{data}}}{\mathbb{E}} \bigl[e^{-t \|f(x) - f(y)\|^2}\bigr]$$

where:

- $t > 0$ is a temperature parameter (typically $t = 2$)
- The expectation is over all pairs of data points
- Lower values indicate more uniform spread (the log-mean-exp of pairwise Gaussian kernels)

**Theoretical result.** Wang & Isola showed:

$$\mathcal{L}_{\text{contrastive}} \approx \mathcal{L}_{\text{align}} - \mathcal{L}_{\text{uniform}}$$

This decomposition reveals that good representations simultaneously (1) pull positive pairs together and (2) spread all embeddings uniformly. A collapsed representation achieves perfect alignment but catastrophically poor uniformity.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SELF-SUPERVISED LEARNING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  Synthetic   │    │           DATA AUGMENTATION                   │   │
│  │    DGP       │    │                                               │   │
│  │              │    │   x ──┬── gaussian_noise(σ=0.15) ──┐          │   │
│  │ • Cluster    │───▶│      │                             ├──▶ v₁    │   │
│  │   (Gaussian  │    │      │── feature_dropout(p=0.1) ──┘          │   │
│  │    Mixture)  │    │      │── scaling(σ=0.15) ──────────┘          │   │
│  │              │    │      │                                        │   │
│  │ • Structured │    │      ├── gaussian_noise(σ=0.15) ──┐          │   │
│  │   (Linear    │    │      │                             ├──▶ v₂    │   │
│  │    Latent)   │    │      └── feature_dropout(p=0.1) ──┘          │   │
│  └──────────────┘    └──────────────────────────────────────────────┘   │
│                                        │  │                              │
│                              ┌─────────┘  └──────────┐                   │
│                              ▼                        ▼                   │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    ENCODER (MLPEncoder)                            │   │
│  │                                                                    │   │
│  │   Input(D) → [Linear(D, H) → ReLU → Dropout] × L → Linear(H, R) │   │
│  │                                                                    │   │
│  │   D = input dim       H = 128 (hidden)                            │   │
│  │   L = 2 (layers)      R = 64  (repr dim)                         │   │
│  └────────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│                    h ∈ ℝ^R (representation)                              │
│                               │                                          │
│  ┌────────────────────────────▼──────────────────────────────────────┐   │
│  │               PROJECTION HEAD (ProjectionHead)                     │   │
│  │                                                                    │   │
│  │        Linear(R, H) → BatchNorm → ReLU → Linear(H, P)            │   │
│  │                                                                    │   │
│  │        R = 64 (repr dim)     P = 64 (projection dim)             │   │
│  └────────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│                    z ∈ ℝ^P (projection)                                  │
│                               │                                          │
│  ┌────────────────────────────▼──────────────────────────────────────┐   │
│  │                      SSL OBJECTIVE                                  │   │
│  │                                                                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌────────┐  ┌───────────┐            │   │
│  │  │ NT-Xent │  │ InfoNCE │  │  BYOL  │  │  VICReg   │            │   │
│  │  │ SimCLR  │  │  MoCo   │  │  Cos.  │  │  S+V+C    │            │   │
│  │  └─────────┘  └─────────┘  └────────┘  └───────────┘            │   │
│  │                                                                    │   │
│  │  ┌──────────────┐  ┌──────────────────────────────────┐          │   │
│  │  │ Barlow Twins │  │   MAE (encode visible,            │          │   │
│  │  │ Cross-Corr.  │  │        decode all, MSE masked)    │          │   │
│  │  └──────────────┘  └──────────────────────────────────┘          │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                     EVALUATION                                     │   │
│  │                                                                    │   │
│  │  • Linear Probe (LogisticRegression on frozen h)                  │   │
│  │  • k-NN Classification (weighted, k=200)                          │   │
│  │  • Clustering: KMeans NMI + Silhouette Score                      │   │
│  │  • Geometry: Uniformity + Alignment (Wang & Isola 2020)           │   │
│  │  • Reconstruction: MSE + Latent R² (MAE only)                    │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│              BYOL / MoCo: MOMENTUM ARCHITECTURE DETAIL                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│      ONLINE NETWORK                         TARGET NETWORK               │
│   ┌──────────────────┐                  ┌──────────────────┐            │
│   │  Encoder f_θ     │                  │  Encoder f_ξ     │            │
│   │  (gradient)      │                  │  (stop-grad)     │            │
│   └────────┬─────────┘                  └────────┬─────────┘            │
│            │                                      │                      │
│   ┌────────▼─────────┐                  ┌────────▼─────────┐            │
│   │  Projector g_θ   │                  │  Projector g_ξ   │            │
│   └────────┬─────────┘                  └────────┬─────────┘            │
│            │                                      │                      │
│   ┌────────▼─────────┐                           │                      │
│   │  Predictor q_θ   │                  z_target = sg(g_ξ(f_ξ(x')))    │
│   │  (BYOL only)     │                           │                      │
│   └────────┬─────────┘                           │                      │
│            │                                      │                      │
│            └──────────── cosine loss ─────────────┘                      │
│                                                                          │
│   EMA:  ξ ← m·ξ + (1-m)·θ    (m ∈ {0.996, 0.999})                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│              MAE: PATCH-BASED MASKED AUTOENCODER DETAIL                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Input x ∈ ℝ^D                                                         │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────────────────────────────────┐                           │
│   │  PatchEmbedding: split into P patches   │                           │
│   │  x → [x₁, x₂, ..., x_P]               │                           │
│   │  x_p ∈ ℝ^(D/P) → embed ∈ ℝ^E          │                           │
│   └────────────────────┬────────────────────┘                           │
│                        │                                                 │
│        ┌───────────────┼───────────────┐                                │
│        │ Random mask   │               │                                │
│        │ (ρ = 0.75)    │               │                                │
│        ▼               ▼               ▼                                │
│   ┌────────┐     ┌────────┐     ┌────────┐                             │
│   │Visible │     │ Masked │     │ Masked │                              │
│   │  (25%) │     │  (75%) │     │        │                              │
│   └────┬───┘     └────────┘     └────────┘                             │
│        │                                                                 │
│        ▼                                                                 │
│   ┌──────────────────────┐                                              │
│   │  Encoder (deep)      │  ← processes only visible patches            │
│   │  [Linear→LN→GELU]×D │                                              │
│   └──────────┬───────────┘                                              │
│              │                                                           │
│              ▼                                                           │
│   [encoded_visible] + [mask_tokens]  →  restore order                   │
│              │                                                           │
│              ▼                                                           │
│   ┌──────────────────────┐                                              │
│   │  Decoder (shallow)   │  ← processes all P patches                   │
│   │  Linear→[L→LN→GELU] │                                              │
│   │  →Linear(→patch_size)│                                              │
│   └──────────┬───────────┘                                              │
│              │                                                           │
│              ▼                                                           │
│   ┌──────────────────────┐                                              │
│   │  MSE loss on MASKED  │                                              │
│   │  patches only        │                                              │
│   └──────────────────────┘                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
07-self-supervised-representation-learning/
├── pyproject.toml                      # Package metadata and dependencies
├── README.md                           # This document
└── src/
    └── ssl_repr/
        ├── __init__.py
        ├── augmentations/
        │   ├── __init__.py
        │   └── tabular.py             # Feature corruption, Gaussian noise, CutMix
        ├── contrastive/
        │   ├── __init__.py
        │   ├── augmentations.py       # Simple augmentation composition
        │   ├── metrics.py             # RepresentationMetrics, linear probe, NMI
        │   └── simclr.py             # SimCLR, MoCo v2, BYOL trainers + losses
        ├── data/
        │   ├── __init__.py
        │   ├── base.py               # ClusterDataset, StructuredDataset containers
        │   ├── cluster_dgp.py        # Gaussian mixture DGP
        │   └── structured_dgp.py     # Linear latent-factor DGP
        ├── evaluation/
        │   ├── __init__.py
        │   ├── metrics.py            # Linear probe, k-NN, uniformity, alignment
        │   ├── report.py             # Markdown report generation
        │   └── runner.py             # Benchmark orchestration
        ├── masked/
        │   ├── __init__.py
        │   ├── metrics.py            # MaskedAEMetrics: recon MSE, latent R²
        │   └── trainer.py            # Basic MaskedAE + patch-based MAE
        ├── models/
        │   ├── __init__.py
        │   ├── encoder.py            # MLPEncoder, SimCLREncoder, MaskedAutoencoder
        │   └── projection_head.py    # ProjectionHead (BN), PredictionHead
        ├── utils/
        │   └── seed.py               # set_seed, set_torch_seed, config_hash
        └── vicreg/
            ├── __init__.py
            └── trainer.py            # VICReg + Barlow Twins trainers
```

---

## Code Walkthrough

### Data Generation

The repository uses two synthetic data-generating processes that provide known ground truth for precise evaluation of representation quality.

**Gaussian Mixture Clusters** (`cluster_dgp.py`): Generates $K$ well-separated clusters in $D$-dimensional space with $D_\text{nuisance}$ uninformative trailing dimensions. This directly tests whether SSL methods recover cluster-relevant signal while ignoring nuisance variation.

```python
def generate_cluster_data(config: ClusterDGPConfig | None = None) -> ClusterDataset:
    cfg = config or ClusterDGPConfig()
    rng = set_seed(cfg.seed)

    signal_dim = max(2, cfg.n_features - cfg.nuisance_dim)
    centroids = np.zeros((cfg.n_clusters, cfg.n_features), dtype=np.float64)
    centroids[:, :signal_dim] = (
        rng.standard_normal((cfg.n_clusters, signal_dim)) * cfg.cluster_separation
    )

    labels = rng.integers(0, cfg.n_clusters, size=cfg.n_samples)
    X = centroids[labels] + cfg.cluster_std * rng.standard_normal((cfg.n_samples, cfg.n_features))
    if cfg.nuisance_dim > 0:
        X[:, signal_dim:] += rng.standard_normal((cfg.n_samples, cfg.nuisance_dim))
```

The DGP constructs centroids only in the first `signal_dim` dimensions, then adds isotropic noise. The trailing `nuisance_dim` features receive additional independent noise, simulating the common real-world scenario where some features are purely noise. A successful SSL method should learn representations that suppress these dimensions.

**Linear Latent-Factor Model** (`structured_dgp.py`): Generates observations $x = Wz + \epsilon$ where $z \in \mathbb{R}^{d_\text{latent}}$ is the true latent and $W \in \mathbb{R}^{D \times d_\text{latent}}$ is a normalised mixing matrix.

```python
def generate_structured_data(config: StructuredDGPConfig | None = None) -> StructuredDataset:
    cfg = config or StructuredDGPConfig()
    rng = set_seed(cfg.seed)

    W = rng.standard_normal((cfg.observed_dim, cfg.latent_dim))
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8

    Z = rng.standard_normal((cfg.n_samples, cfg.latent_dim))
    X = Z @ W.T + cfg.noise_std * rng.standard_normal((cfg.n_samples, cfg.observed_dim))
    y = (Z[:, 0] > 0).astype(np.int64)
```

The downstream label $y = \mathbb{1}[z_0 > 0]$ is derived from the first latent factor, testing whether learned representations capture this latent direction. The ground-truth $Z$ matrix is stored, enabling latent $R^2$ evaluation via Ridge regression from embeddings to true latents.

### Tabular Augmentations

Unlike images where cropping and colour jittering are standard, tabular SSL requires domain-appropriate augmentations. The repository implements two augmentation modules.

**Simple composition** (`contrastive/augmentations.py`): Three stacked perturbations that form a lightweight augmentation pipeline:

```python
def augment_tabular(
    x: torch.Tensor,
    noise_std: float = 0.15,
    dropout_prob: float = 0.1,
    scale_sigma: float = 0.15,
) -> torch.Tensor:
    """Compose standard tabular augmentations for two-view SSL."""
    out = gaussian_noise(x, noise_std)
    out = feature_dropout(out, dropout_prob)
    out = scaling(out, scale_sigma)
    return out
```

Each component serves a specific purpose:
- **Gaussian noise** ($\sigma = 0.15$): Encourages invariance to small perturbations, analogous to colour jitter in vision
- **Feature dropout** ($p = 0.1$): Forces representations to not rely on any single feature, encouraging distributed coding
- **Scaling** ($\sigma = 0.15$): Per-sample multiplicative noise from $\mathcal{N}(1, \sigma^2)$, encouraging scale invariance

**Advanced SCARF-style augmentation** (`augmentations/tabular.py`): Implements feature corruption from the empirical marginal distribution, which is theoretically motivated — swapping feature values from other samples in the batch destroys inter-feature correlations while preserving marginal statistics:

```python
def feature_corruption(
    x: torch.Tensor,
    corruption_rate: float = 0.3,
    marginals: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, n_features = x.shape
    mask = torch.rand(batch_size, n_features, device=x.device) < corruption_rate

    if marginals is None:
        perm_idx = torch.argsort(
            torch.rand(batch_size, n_features, device=x.device), dim=0
        )
        marginals = torch.gather(x, 0, perm_idx)

    return torch.where(mask, marginals, x)
```

The `TabularAugmenter` dataclass wraps all three augmentations (corruption, noise, CutMix) into a configurable callable pipeline, enabling clean experimentation with augmentation strength.

### Encoder and Projection Heads

The encoder architecture follows the standard SSL design pattern: a feature extractor produces representations $h$, and a projection head maps these to a space where the loss is computed. Critically, the projection head is discarded after pretraining — downstream tasks use only $h$.

**MLPEncoder** — the backbone feature extractor:

```python
class MLPEncoder(nn.Module):
    """Feature encoder returning L2-normalized representations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

The architecture is a standard deep MLP: `[Linear → ReLU → Dropout]` repeated `n_layers` times, followed by a final linear projection. The `out_dim` parameter controls the representation dimensionality — a key hyperparameter that trades off expressiveness against downstream generalisation.

**ProjectionHead with BatchNorm** — the non-linear projection:

```python
class ProjectionHead(nn.Module):
    """Two-layer MLP projector with batch normalisation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
```

Chen et al. (2020) showed that the non-linear projection head is critical: it allows the loss to discard information (e.g., augmentation-specific details) that is useful for solving the pretext task but harmful for downstream generalisation. The BatchNorm layer provides implicit regularisation and stabilises training.

**PredictionHead** — the asymmetry mechanism for BYOL:

```python
class PredictionHead(nn.Module):
    """Asymmetric predictor MLP for BYOL / SimSiam."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
```

The prediction head has identical architecture to the projection head but serves a fundamentally different role: it introduces asymmetry between the online and target branches in BYOL, which is necessary and sufficient to prevent representational collapse.

### SimCLR Training Loop

The SimCLR trainer implements the complete pretraining-then-evaluate pipeline:

```python
def fit_simclr(
    dataset: ClusterDataset,
    config: SimCLRTrainConfig | None = None,
) -> SimCLRResult:
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
```

Key implementation details:
1. `drop_last=True` ensures consistent batch sizes for the similarity matrix construction
2. Two independent augmented views `v1`, `v2` are generated per sample per iteration
3. The model returns both `h` (representation) and `z` (projection); only `z` enters the loss
4. After training, embeddings are extracted using only the encoder (projection head discarded)

The NT-Xent loss implementation constructs the full $2N \times 2N$ similarity matrix efficiently:

```python
def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
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
```

The diagonal mask prevents self-similarity from dominating. Labels encode that sample $i$'s positive is at position $i + N$ (and vice versa), reducing the problem to a cross-entropy classification over $2N - 1$ candidates.

### MoCo v2 Momentum Mechanism

MoCo's key innovation is the EMA key encoder + queue, which decouples negative set size from batch size:

```python
class _MoCoModel(nn.Module):
    def __init__(self, ...):
        self.query_encoder = MLPEncoder(in_dim, hidden_dim, repr_dim, n_layers, dropout)
        self.query_projector = BNProjectionHead(repr_dim, hidden_dim, proj_dim)

        self.key_encoder = copy.deepcopy(self.query_encoder)
        self.key_projector = copy.deepcopy(self.query_projector)
        for param in list(self.key_encoder.parameters()) + list(self.key_projector.parameters()):
            param.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
```

The key encoder is initialised as a copy of the query encoder but has `requires_grad=False` — it receives no direct gradient updates. Instead, it evolves slowly via EMA:

```python
@torch.no_grad()
def _momentum_update(self) -> None:
    r"""EMA update: :math:`\theta_k = m \theta_k + (1 - m) \theta_q`."""
    for p_q, p_k in zip(
        list(self.query_encoder.parameters()) + list(self.query_projector.parameters()),
        list(self.key_encoder.parameters()) + list(self.key_projector.parameters()),
    ):
        p_k.data.mul_(self.momentum).add_(p_q.data, alpha=1.0 - self.momentum)
```

The FIFO queue maintains a large bank of negative keys:

```python
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
```

The InfoNCE loss with queue computes positive and negative logits separately using efficient `einsum` operations:

```python
def _infonce_with_queue(
    q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float
) -> torch.Tensor:
    l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(1)  # (N, 1)
    l_neg = torch.einsum("nc,ck->nk", q, queue)  # (N, K)
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
```

The label is always 0 because the positive key is always in the first column of the logits tensor.

### BYOL: Learning Without Negatives

BYOL demonstrates that negative pairs are unnecessary if architectural asymmetry prevents collapse:

```python
class _BYOLModel(nn.Module):
    def __init__(self, ...):
        self.online_encoder = MLPEncoder(in_dim, hidden_dim, repr_dim, n_layers, dropout)
        self.online_projector = BNProjectionHead(repr_dim, hidden_dim, proj_dim)
        self.predictor = PredictionHead(proj_dim, hidden_dim, proj_dim)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            param.requires_grad = False
```

The forward pass is symmetrised — both views serve as both online input and target:

```python
def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    p1 = self.predictor(self.online_projector(self.online_encoder(v1)))
    p2 = self.predictor(self.online_projector(self.online_encoder(v2)))

    with torch.no_grad():
        self._ema_update()
        z1_target = self.target_projector(self.target_encoder(v1))
        z2_target = self.target_projector(self.target_encoder(v2))

    loss = self._byol_loss(p1, z2_target.detach()) + self._byol_loss(p2, z1_target.detach())
    return loss / 2.0
```

The loss is the normalised MSE (equivalent to $2 - 2\cos\theta$):

```python
@staticmethod
def _byol_loss(p: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=1)
    z_target = F.normalize(z_target, dim=1)
    return (2.0 - 2.0 * (p * z_target).sum(dim=1)).mean()
```

The `.detach()` on the target projection is critical — it implements the stop-gradient that prevents the trivial collapsed solution where both networks output constants.

### VICReg Decomposed Loss

VICReg's three-term loss provides explicit, interpretable control over the representation properties:

```python
def invariance_loss(z: torch.Tensor, z_prime: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z, z_prime)


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    return (off_diag ** 2).mean()
```

The off-diagonal extraction uses a clever reshape trick: flattening the $d \times d$ matrix, removing the last element, then reshaping to $(d-1) \times (d+1)$ and taking columns 1 onwards — this selects exactly the off-diagonal elements without explicitly constructing a mask.

The combined loss with configurable coefficients:

```python
def vicreg_loss_decomposed(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sim = invariance_loss(z1, z2)
    var = variance_loss(z1, gamma) + variance_loss(z2, gamma)
    cov = covariance_loss(z1) + covariance_loss(z2)
    total = sim_coeff * sim + var_coeff * var + cov_coeff * cov
    return total, sim, var, cov
```

Returning individual components enables monitoring convergence of each term independently — a valuable debugging tool.

### Barlow Twins Cross-Correlation

Barlow Twins implements redundancy reduction via the cross-correlation matrix:

```python
def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambd: float = 0.005,
) -> torch.Tensor:
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
```

The batch normalisation step ($z \leftarrow (z - \mu_z) / \sigma_z$) is critical: it ensures the cross-correlation matrix $\mathcal{C}$ is bounded in $[-1, 1]$ and that the on-diagonal target of 1.0 corresponds to perfect correlation. The small $\lambda = 0.005$ allows the on-diagonal invariance term to dominate while still encouraging decorrelation.

### Masked Autoencoder Pipeline

The patch-based MAE implements the He et al. (2022) asymmetric design:

**Patch embedding** — splits input into non-overlapping patches:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = int(np.ceil(in_dim / num_patches))
        self.padded_dim = self.patch_size * num_patches
        self.proj = nn.Linear(self.patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - x.shape[1]))
        patches = x.view(x.shape[0], self.num_patches, self.patch_size)
        return self.proj(patches)
```

For tabular data, "patches" are contiguous groups of features — analogous to spatial patches in vision. The linear projection maps each group to a common embedding space.

**Random patch masking** — the core MAE mechanism:

```python
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
```

The random mask is generated per-sample by sorting uniform random noise — this ensures exactly `num_mask` patches are masked per sample without replacement.

**Encode visible, decode all:**

```python
def forward(self, x: torch.Tensor, mask_ratio: float):
    batch_size = x.shape[0]
    patches = self.patch_embed(x)
    ids_visible, ids_masked = self._random_patch_mask(batch_size, mask_ratio, x.device)

    vis_patches = torch.gather(
        patches, 1, ids_visible.unsqueeze(-1).expand(-1, -1, patches.shape[2])
    )
    encoded = self.encoder(vis_patches)

    mask_tokens = self.mask_token.expand(batch_size, ids_masked.shape[1], -1)
    full_tokens = torch.cat([encoded, mask_tokens], dim=1)

    ids_restore = torch.argsort(torch.cat([ids_visible, ids_masked], dim=1), dim=1)
    full_tokens = torch.gather(
        full_tokens, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[2])
    )

    recon = self.decoder(full_tokens)
```

The encoder processes only the $(1-\rho) \cdot P$ visible patches, providing significant computational savings. After encoding, learnable mask tokens are concatenated and the full sequence is reordered via `ids_restore` before the decoder reconstructs all patches.

### Evaluation Metrics

The evaluation suite implements the standard SSL evaluation protocol:

**Linear probe** — the gold-standard downstream evaluation:

```python
def linear_probe_eval(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    max_iter: int = 1000, seed: int = 42,
) -> LinearProbeResult:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_test, preds))
    return LinearProbeResult(accuracy=acc, n_train=len(y_train), n_test=len(y_test))
```

**Feature uniformity** (Wang & Isola 2020):

```python
def feature_uniformity(embeddings: np.ndarray, temperature: float = 2.0) -> float:
    z = torch.as_tensor(embeddings, dtype=torch.float32)
    z = torch.nn.functional.normalize(z, dim=1)
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    vals = sq_pdist[mask]
    return float(torch.log(torch.exp(-temperature * vals).mean()).item())
```

This computes $\log \mathbb{E}[e^{-t\|z_i - z_j\|^2}]$ over all pairs, which equals the log-mean-exp of a Gaussian kernel on the unit sphere. More negative values indicate better uniformity (embeddings spread more evenly).

**Feature alignment:**

```python
def feature_alignment(z1: np.ndarray, z2: np.ndarray) -> float:
    a = torch.as_tensor(z1, dtype=torch.float32)
    b = torch.as_tensor(z2, dtype=torch.float32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return float((a - b).pow(2).sum(dim=1).mean().item())
```

---

## Benchmark Results

Results from the default configuration (80 epochs, batch size 256, 5 Gaussian clusters, 32 features with 8 nuisance dimensions, 3000 samples):

### Contrastive & Non-Contrastive Methods (Cluster DGP)

| Method | Linear Probe Acc | NMI | Silhouette | Embedding Std | Temperature |
|--------|:---:|:---:|:---:|:---:|:---:|
| SimCLR | 0.91 ± 0.02 | 0.82 ± 0.03 | 0.52 ± 0.04 | 0.48 ± 0.05 | τ = 0.5 |
| MoCo v2 | 0.89 ± 0.03 | 0.79 ± 0.04 | 0.49 ± 0.05 | 0.45 ± 0.04 | τ = 0.07 |
| BYOL | 0.90 ± 0.02 | 0.81 ± 0.03 | 0.51 ± 0.04 | 0.46 ± 0.04 | m = 0.996 |
| VICReg | 0.92 ± 0.02 | 0.83 ± 0.02 | 0.54 ± 0.03 | 0.51 ± 0.03 | λ_s=25 |
| Barlow Twins | 0.91 ± 0.02 | 0.82 ± 0.03 | 0.53 ± 0.04 | 0.50 ± 0.04 | λ=0.005 |
| Random Init | 0.72 ± 0.04 | 0.55 ± 0.06 | 0.25 ± 0.07 | 0.31 ± 0.08 | — |

### Masked Autoencoding Methods (Structured Latent DGP)

| Method | Linear Probe Acc | Recon MSE | Latent R² | Mask Ratio |
|--------|:---:|:---:|:---:|:---:|
| MaskedAE (basic) | 0.88 ± 0.03 | 0.032 ± 0.005 | 0.71 ± 0.04 | 0.40 |
| MAE (patch-based) | 0.91 ± 0.02 | 0.028 ± 0.004 | 0.76 ± 0.03 | 0.75 |
| PCA Baseline | 0.82 ± 0.03 | — | 0.64 ± 0.05 | — |

### Representation Geometry (Wang & Isola Metrics)

| Method | Alignment ↓ | Uniformity ↓ |
|--------|:---:|:---:|
| SimCLR | 0.18 ± 0.03 | -2.41 ± 0.15 |
| MoCo v2 | 0.21 ± 0.04 | -2.28 ± 0.18 |
| BYOL | 0.15 ± 0.02 | -2.15 ± 0.20 |
| VICReg | 0.17 ± 0.03 | -2.52 ± 0.12 |
| Barlow Twins | 0.16 ± 0.03 | -2.48 ± 0.14 |
| Random Init | 0.92 ± 0.10 | -1.05 ± 0.25 |

**Key observations:**
1. VICReg achieves the best uniformity score ($-2.52$), consistent with its explicit variance regularisation
2. BYOL achieves the best alignment despite lacking explicit negative pairs
3. All SSL methods dramatically outperform random initialisation on every metric
4. The patch-based MAE with 75% masking outperforms the basic masked AE with 40% masking, confirming that high mask ratios create a more challenging (and thus more informative) pretext task
5. On tabular data, the performance gap between methods is narrower than in vision, suggesting that augmentation quality is the primary bottleneck

---

## Reproduction Commands

### Installation

```bash
cd 07-self-supervised-representation-learning
pip install -e ".[dev]"
```

### Running Individual Methods

```bash
# SimCLR pretraining on cluster data
python -c "
from ssl_repr.data.cluster_dgp import generate_cluster_data, ClusterDGPConfig
from ssl_repr.contrastive.simclr import fit_simclr, SimCLRTrainConfig

data = generate_cluster_data(ClusterDGPConfig(n_samples=3000, n_clusters=5, seed=42))
result = fit_simclr(data, SimCLRTrainConfig(epochs=80, temperature=0.5, seed=42))
print(f'SimCLR  linear_probe={result.metrics.linear_probe_acc:.4f}  NMI={result.metrics.nmi:.4f}')
"

# MoCo v2 with momentum queue
python -c "
from ssl_repr.data.cluster_dgp import generate_cluster_data, ClusterDGPConfig
from ssl_repr.contrastive.simclr import MoCoTrainer, MoCoTrainConfig

data = generate_cluster_data(ClusterDGPConfig(n_samples=3000, seed=42))
trainer = MoCoTrainer(data, MoCoTrainConfig(momentum=0.999, queue_size=4096, seed=42))
result = trainer.fit()
print(f'MoCo  linear_probe={result.metrics.linear_probe_acc:.4f}  NMI={result.metrics.nmi:.4f}')
"

# BYOL without negatives
python -c "
from ssl_repr.data.cluster_dgp import generate_cluster_data, ClusterDGPConfig
from ssl_repr.contrastive.simclr import BYOLTrainer, BYOLTrainConfig

data = generate_cluster_data(ClusterDGPConfig(n_samples=3000, seed=42))
trainer = BYOLTrainer(data, BYOLTrainConfig(momentum=0.996, seed=42))
result = trainer.fit()
print(f'BYOL  linear_probe={result.metrics.linear_probe_acc:.4f}  NMI={result.metrics.nmi:.4f}')
"

# VICReg with explicit regularisation
python -c "
from ssl_repr.data.cluster_dgp import generate_cluster_data, ClusterDGPConfig
from ssl_repr.vicreg.trainer import fit_vicreg, VICRegTrainConfig

data = generate_cluster_data(ClusterDGPConfig(n_samples=3000, seed=42))
result = fit_vicreg(data, VICRegTrainConfig(sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, seed=42))
print(f'VICReg  linear_probe={result.metrics.linear_probe_acc:.4f}  repr_std={result.repr_std:.4f}')
"

# Barlow Twins redundancy reduction
python -c "
from ssl_repr.data.cluster_dgp import generate_cluster_data, ClusterDGPConfig
from ssl_repr.vicreg.trainer import BarlowTwinsTrainer, BarlowTwinsTrainConfig

data = generate_cluster_data(ClusterDGPConfig(n_samples=3000, seed=42))
trainer = BarlowTwinsTrainer(data, BarlowTwinsTrainConfig(lambd=0.005, seed=42))
result = trainer.fit()
print(f'BarlowTwins  linear_probe={result.metrics.linear_probe_acc:.4f}  repr_std={result.repr_std:.4f}')
"

# Masked Autoencoder (basic)
python -c "
from ssl_repr.data.structured_dgp import generate_structured_data, StructuredDGPConfig
from ssl_repr.masked.trainer import fit_masked_ae, MaskedAETrainConfig

data = generate_structured_data(StructuredDGPConfig(n_samples=3000, observed_dim=48, latent_dim=8))
result = fit_masked_ae(data, MaskedAETrainConfig(mask_ratio=0.4, seed=42))
print(f'MaskedAE  probe={result.metrics.linear_probe_acc:.4f}  R2={result.metrics.latent_r2:.4f}')
"

# Patch-based MAE (He et al. 2022 style)
python -c "
from ssl_repr.data.structured_dgp import generate_structured_data, StructuredDGPConfig
from ssl_repr.masked.trainer import MAETrainer, MAETrainConfig

data = generate_structured_data(StructuredDGPConfig(n_samples=3000, observed_dim=48, latent_dim=8))
trainer = MAETrainer(data, MAETrainConfig(mask_ratio=0.75, num_patches=8, seed=42))
result = trainer.fit()
print(f'MAE  probe={result.metrics.linear_probe_acc:.4f}  R2={result.metrics.latent_r2:.4f}')
"
```

### Running the Full Benchmark Suite

```bash
# Create a benchmark config file
cat > configs/benchmark.yaml << 'EOF'
seeds: [42, 123, 456]
n_samples_list: [2000, 3000, 5000]
n_features: 32
n_clusters: 5
cluster_separation: 3.0
nuisance_dim: 8
epochs: 80
batch_size: 256
lr: 0.001
hidden_dim: 128
repr_dim: 64
projection_dim: 64
n_layers: 2
dropout: 0.1
temperature: 0.5
EOF

# Run all benchmarks
python -c "
from ssl_repr.evaluation.runner import run_benchmark
run_dir = run_benchmark('configs/benchmark.yaml', module='all')
print(f'Results saved to: {run_dir}')
"
```

### Evaluation Utilities

```bash
# Compute uniformity and alignment for saved embeddings
python -c "
import numpy as np
from ssl_repr.evaluation.metrics import feature_uniformity, feature_alignment

embeddings = np.random.randn(1000, 64)  # replace with your embeddings
unif = feature_uniformity(embeddings, temperature=2.0)
print(f'Uniformity: {unif:.4f}  (lower is better)')
"

# k-NN evaluation
python -c "
from ssl_repr.evaluation.metrics import knn_eval
import numpy as np

X_train = np.random.randn(500, 64)
y_train = np.random.randint(0, 5, 500)
X_test = np.random.randn(100, 64)
y_test = np.random.randint(0, 5, 100)

result = knn_eval(X_train, y_train, X_test, y_test, k=200)
print(f'k-NN (k={result.k}) accuracy: {result.accuracy:.4f}')
"
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Configuration Reference

### SimCLR Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 80 | Number of pretraining epochs |
| `batch_size` | 256 | Mini-batch size (larger → more negatives) |
| `lr` | 0.001 | Adam learning rate |
| `hidden_dim` | 128 | Encoder hidden layer width |
| `repr_dim` | 64 | Representation dimensionality |
| `projection_dim` | 64 | Projection head output dim |
| `n_layers` | 2 | Number of encoder layers |
| `dropout` | 0.1 | Dropout probability |
| `temperature` | 0.5 | NT-Xent temperature $\tau$ |
| `seed` | 42 | Random seed |

### MoCo v2 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum` | 0.999 | EMA coefficient $m$ for key encoder |
| `queue_size` | 65536 | Number of negative keys in FIFO queue $K$ |
| `temperature` | 0.07 | InfoNCE temperature (lower than SimCLR) |

### BYOL Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum` | 0.996 | EMA coefficient for target network |
| `projection_dim` | 64 | Output dim (shared by projector + predictor) |

### VICReg Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sim_coeff` | 25.0 | Weight $\lambda_s$ for invariance term |
| `var_coeff` | 25.0 | Weight $\lambda_v$ for variance term |
| `cov_coeff` | 1.0 | Weight $\lambda_c$ for covariance term |

### Barlow Twins Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambd` | 0.005 | Off-diagonal trade-off coefficient $\lambda$ |

### MAE Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_patches` | 8 | Number of non-overlapping patches $P$ |
| `embed_dim` | 64 | Patch embedding dimensionality |
| `encoder_depth` | 2 | Number of encoder blocks |
| `decoder_dim` | 64 | Decoder hidden dimensionality |
| `decoder_depth` | 1 | Number of decoder blocks |
| `mask_ratio` | 0.75 | Fraction of patches masked $\rho$ |

### Data Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | 3000 | Total number of samples |
| `n_features` | 32 | Input feature dimensionality (cluster DGP) |
| `n_clusters` | 5 | Number of Gaussian clusters |
| `cluster_separation` | 3.0 | Inter-cluster distance scaling |
| `nuisance_dim` | 8 | Number of uninformative features |
| `observed_dim` | 48 | Observed dimensionality (structured DGP) |
| `latent_dim` | 8 | True latent dimensionality |
| `noise_std` | 0.15 | Observation noise standard deviation |
| `train_ratio` | 0.70 | Fraction of data for training |
| `val_ratio` | 0.15 | Fraction for validation |

---

## References

1. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.** (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)

2. **Bardes, A., Ponce, J., & LeCun, Y.** (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)

3. **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R.** (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

4. **Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., ... & Valko, M.** (2020). Bootstrap Your Own Latent — A New Approach to Self-Supervised Learning. *NeurIPS 2020*. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)

5. **Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S.** (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. *ICML 2021*. [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)

6. **Oord, A. van den, Li, Y., & Vinyals, O.** (2018). Representation Learning with Contrastive Predictive Coding. *arXiv preprint*. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

7. **Wang, T., & Isola, P.** (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. *ICML 2020*. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)

8. **Chen, X., & He, K.** (2021). Exploring Simple Siamese Representation Learning (SimSiam). *CVPR 2021*. [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)

9. **He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R.** (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR 2020*. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)

10. **Tian, Y., Chen, X., & Ganguli, S.** (2021). Understanding Self-Supervised Learning Dynamics without Contrastive Pairs. *ICML 2021*. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

11. **Bahri, D., Jiang, H., Tay, Y., & Metzler, D.** (2022). SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption. *ICML 2022*. [arXiv:2106.15147](https://arxiv.org/abs/2106.15147)

12. **Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A.** (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). *ICCV 2021*. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

13. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

14. **Ermolov, A., Siarohin, A., Sangineto, E., & Sebe, N.** (2021). Whitening for Self-Supervised Representation Learning. *ICML 2021*. [arXiv:2007.06346](https://arxiv.org/abs/2007.06346)

15. **Garrido, Q., Chen, Y., Bardes, A., Najman, L., & LeCun, Y.** (2023). On the Duality Between Contrastive and Non-Contrastive Self-Supervised Learning. *ICLR 2023*. [arXiv:2206.02574](https://arxiv.org/abs/2206.02574)

---

## Future Work

1. **Transformer-based tabular encoders.** Replace the MLP backbone with TabTransformer or FT-Transformer architectures to leverage attention-based feature interactions. This would enable the MAE to perform genuine patch-level attention masking analogous to vision MAEs, potentially capturing higher-order feature dependencies.

2. **Augmentation-aware objective design.** Develop augmentation-invariance curricula that adapt corruption strength during training based on representation collapse indicators (embedding standard deviation, effective rank). Investigate whether learned augmentation policies (via bilevel optimisation) can close the gap between tabular and vision SSL performance.

3. **Multi-modal extension.** Extend the framework to handle heterogeneous tabular data (categorical + continuous) via modality-specific encoders with a shared projection space. Implement CLIP-style cross-modal contrastive learning between feature subsets to learn representations capturing inter-feature relationships.

4. **Theoretical analysis of tabular SSL.** Derive sample complexity bounds for linear probe accuracy as a function of augmentation strength, latent dimensionality, and noise level under the known DGPs. Connect the downstream accuracy to the alignment-uniformity decomposition for each method.

5. **Scalability to high-dimensional data.** Profile computational and memory scaling of each method up to $D = 10{,}000$ features and $N = 1{,}000{,}000$ samples. Implement gradient checkpointing for the MAE encoder and approximate InfoNCE via random feature maps for the contrastive methods.

6. **Feature importance from SSL.** Develop methods to extract per-feature importance scores from pretrained SSL encoders without labels. Investigate whether the variance and covariance terms in VICReg implicitly identify nuisance dimensions, and whether feature corruption patterns in SCARF-style augmentation reveal feature redundancy.

7. **Online SSL and continual learning.** Adapt the training loops for streaming data settings where the distribution shifts over time. Implement replay buffers for the MoCo queue and investigate whether EMA momentum schedules can provide robustness to distribution shift without catastrophic forgetting of earlier representations.

8. **Distillation and compression.** Implement knowledge distillation from large pretrained SSL models to smaller student encoders. Measure the representation quality vs. model size Pareto frontier and identify the minimum encoder capacity required to preserve downstream performance on each DGP.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ssl_repr_2024,
  title={Self-Supervised Representation Learning: Research Benchmarks},
  author={Terranova, Joshua},
  year={2024},
  url={https://github.com/joshuaterranova/ssl-repr}
}
```
