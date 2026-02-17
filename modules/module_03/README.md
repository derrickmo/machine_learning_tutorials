# Module 03 — Unsupervised & Statistical Learning

## Introduction

Module 03 explores the world of unsupervised and statistical learning, where the goal shifts from predicting labels to discovering hidden structure, reducing dimensionality, and modeling data distributions without supervision. Building on the linear algebra (Module 1-06), probability (Module 1-07), and supervised learning foundations (Module 2), students implement clustering algorithms, dimensionality reduction techniques, anomaly detectors, kernel methods, matrix factorizations, and Bayesian inference from scratch. By the end of this module, students will be able to cluster data with K-Means and DBSCAN, visualize high-dimensional embeddings with PCA, t-SNE, and UMAP, detect anomalies, decompose matrices for topic modeling, and reason probabilistically with conjugate priors. These techniques underpin critical deep learning components in later modules -- PCA connects to autoencoders (Module 11), kernel methods connect to attention mechanisms (Module 8), and Bayesian inference connects to variational autoencoders (Module 11) and reward modeling (Module 13).

**Folder:** `module_03_unsupervised_and_statistical_learning/`

**GPU Required:** No (classical ML / math foundations)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 3-01 | K-Means Clustering | Lloyd's algorithm implementation; elbow method and silhouette score; K-Means++ initialization; convergence properties; limitations (non-convex clusters, sensitivity to k); mini-batch K-Means | `sklearn.make_blobs` | ~2 min |
| 3-02 | Hierarchical & Density-Based Clustering | Agglomerative clustering (single/complete/Ward linkage); dendrogram visualization and cutting; DBSCAN — core/border/noise points, epsilon-neighborhood; comparison on non-spherical data | `sklearn.make_moons`, `sklearn.make_blobs` | ~3 min |
| 3-03 | Principal Component Analysis | Eigendecomposition derivation; variance explained and scree plots; choosing number of components; PCA for denoising; PCA for visualization; connection to SVD; limitations (linearity) | `sklearn` Digits, MNIST | ~3 min |
| 3-04 | t-SNE, UMAP & Manifold Learning | t-SNE — perplexity parameter, crowding problem, heavy-tailed Student-t; UMAP — graph construction and optimization; side-by-side comparison on MNIST; global vs local structure preservation | MNIST, FashionMNIST | ~5 min |
| 3-05 | Independent Component Analysis | Statistical independence vs uncorrelatedness; cocktail party problem; FastICA algorithm; contrast functions (kurtosis, negentropy); comparison with PCA on mixed signals | Synthetic (mixed signals) | ~2 min |
| 3-06 | Gaussian Mixture Models & EM Algorithm | GMMs as soft clustering; EM algorithm derivation (E-step: responsibilities, M-step: parameter updates); convergence and local optima; choosing number of components (BIC/AIC); connection to K-Means as a special case | `sklearn.make_blobs` | ~3 min |
| 3-07 | Anomaly Detection | Statistical methods (z-score, IQR); Gaussian density estimation (univariate and multivariate); Isolation Forest from scratch; autoencoder-based anomaly detection preview; threshold selection and evaluation (precision-recall for anomalies) | `sklearn.make_classification` | ~3 min |
| 3-08 | Kernel Methods & Feature Maps | Kernel trick revisited — Mercer's theorem; RBF kernel as infinite-dimensional feature map; kernel PCA; kernel density estimation; connection to attention mechanisms (softmax as kernel); random Fourier features | `sklearn.make_moons`, `sklearn.make_circles` | ~3 min |
| 3-09 | Matrix Factorization & Decomposition | SVD for dimensionality reduction and matrix completion; Non-negative Matrix Factorization (NMF) for topic modeling; truncated SVD; connection to recommender systems (Module 19) and LoRA (Module 13) | `sklearn` Digits, Synthetic | ~3 min |
| 3-10 | Bayesian Inference & Probabilistic Thinking | Prior, likelihood, posterior; conjugate priors (Beta-Binomial, Normal-Normal); MAP estimation revisited; predictive distributions; Bayesian linear regression; connection to VAEs (Module 11) and reward modeling (Module 13) | Synthetic (Beta-Binomial, Normal-Normal) | ~2 min |

---

## Topic Details

### 3-01: K-Means Clustering
Students implement Lloyd's algorithm from scratch in NumPy -- random initialization, assignment step, update step, convergence check -- and then improve it with K-Means++ initialization for better starting centroids. The notebook covers cluster quality evaluation using the elbow method and silhouette scores, and explores K-Means' known limitations: failure on non-convex clusters, sensitivity to the choice of k, and dependence on initialization. Mini-batch K-Means is implemented as a scalable alternative for larger datasets. This algorithm is foundational for understanding soft clustering in GMMs (3-06), vector quantization in VQ-VAEs (Module 11), and codebook learning in tokenizers, and its O(nkd) per-iteration complexity is analyzed in Module 1-10.

### 3-02: Hierarchical & Density-Based Clustering
This topic implements two fundamentally different clustering paradigms: agglomerative hierarchical clustering (with single, complete, and Ward linkage) that produces dendrograms for multi-scale cluster analysis, and DBSCAN which discovers arbitrarily-shaped clusters by identifying core points, border points, and noise through epsilon-neighborhood density. Students visualize dendrograms and practice cutting them at different heights, then compare both approaches against K-Means on non-spherical data (make_moons) where centroid-based methods fail. These density-based concepts connect forward to density estimation in anomaly detection (3-07) and the graph-based neighborhood construction in UMAP (3-04). Understanding when different clustering paradigms succeed or fail is essential for the unsupervised feature learning that underlies deep representation learning.

### 3-03: Principal Component Analysis
Students derive PCA from the eigendecomposition of the covariance matrix, implement it from scratch in NumPy, and apply it to the Digits and MNIST datasets for both visualization and denoising. The notebook covers variance-explained ratios, scree plots for choosing the number of components, the mathematical equivalence between eigendecomposition-based PCA and SVD-based PCA, and PCA's fundamental limitation of only capturing linear relationships. PCA is the most widely used dimensionality reduction technique in ML and connects directly to the linear algebra foundations in Module 1-06, while also serving as the conceptual precursor to autoencoders (Module 11) which learn nonlinear analogs of principal components. The denoising application here foreshadows denoising autoencoders and diffusion models.

### 3-04: t-SNE, UMAP & Manifold Learning
This topic implements t-SNE (with its perplexity parameter, crowding problem solution via heavy-tailed Student-t distributions, and gradient-based optimization) and UMAP (with its graph construction and cross-entropy optimization), then runs side-by-side comparisons on MNIST and FashionMNIST to analyze their different strengths. Students learn that t-SNE excels at preserving local neighborhood structure while UMAP better preserves global topology, and explore how hyperparameter choices (perplexity, n_neighbors, min_dist) dramatically affect the resulting embeddings. These visualization techniques are used throughout the deep learning modules to inspect learned representations -- from CNN feature maps (Module 6) to transformer embeddings (Module 8) to LLM hidden states (Module 10). Understanding manifold learning also provides intuition for why deep networks can learn useful representations of high-dimensional data.

### 3-05: Independent Component Analysis
Students implement the FastICA algorithm from scratch to solve the classic cocktail party problem: separating mixed source signals into their independent components using contrast functions based on kurtosis and negentropy. The notebook carefully distinguishes statistical independence (which ICA seeks) from mere uncorrelatedness (which PCA achieves), demonstrating on synthetic mixed signals that PCA and ICA produce fundamentally different decompositions. ICA relies on the non-Gaussianity of source signals, and the notebook explores why Gaussian sources cannot be separated. While less commonly used in modern deep learning than PCA, ICA's focus on statistical independence connects to the disentangled representations sought by beta-VAEs (Module 11) and provides important intuition about what "meaningful features" means in unsupervised learning.

### 3-06: Gaussian Mixture Models & EM Algorithm
This topic derives and implements the Expectation-Maximization (EM) algorithm from scratch for fitting Gaussian Mixture Models, covering the E-step (computing responsibilities -- soft cluster assignments) and M-step (updating means, covariances, and mixing coefficients). Students explore convergence behavior, sensitivity to initialization and local optima, model selection via BIC and AIC, and the mathematical relationship showing K-Means as a special case of GMMs with hard assignments and isotropic covariances. The EM algorithm is one of the most important algorithms in statistical learning, and the variational perspective on EM connects directly to variational autoencoders (Module 11) where the encoder plays the role of the E-step and the decoder plays the role of the M-step. GMMs also provide the density estimation foundation used in anomaly detection (3-07).

### 3-07: Anomaly Detection
Students implement multiple anomaly detection approaches from scratch: statistical methods (z-score, IQR), univariate and multivariate Gaussian density estimation, and Isolation Forest (which detects anomalies as points that require fewer random splits to isolate). The notebook covers threshold selection strategies and evaluation using precision-recall curves, which are more appropriate than ROC curves for the highly imbalanced nature of anomaly detection problems. An autoencoder-based anomaly detection preview connects forward to Module 11 where reconstruction error from autoencoders serves as an anomaly score. These techniques are directly applicable in production ML systems (Module 20) for detecting data drift, model degradation, and out-of-distribution inputs.

### 3-08: Kernel Methods & Feature Maps
This topic revisits the kernel trick from SVMs (2-05) with deeper mathematical rigor -- Mercer's theorem, the interpretation of the RBF kernel as an infinite-dimensional feature map, and the construction of kernel PCA and kernel density estimation from scratch. Students implement random Fourier features as an efficient finite-dimensional approximation to kernel methods and explore the provocative connection between softmax attention (Module 8) and kernel functions. Kernel methods provide a bridge between classical ML and deep learning: they show that feature transformation is the key to learning complex patterns, whether done explicitly (feature engineering), implicitly (kernels), or adaptively (neural networks). The kernel PCA implementation here complements the linear PCA from 3-03.

### 3-09: Matrix Factorization & Decomposition
Students implement SVD-based dimensionality reduction and matrix completion, truncated SVD for efficient computation, and Non-negative Matrix Factorization (NMF) for interpretable topic modeling on the Digits dataset and synthetic text data. The notebook demonstrates how constraining factors to be non-negative yields parts-based representations (e.g., strokes of digits, topics in documents) that are more interpretable than PCA components. Matrix factorization connects forward to two critical applications: recommender systems (Module 19) where user-item matrices are factored to predict preferences, and LoRA (Module 13) where low-rank factorization enables parameter-efficient fine-tuning of large language models. The SVD foundations from Module 1-06 are applied and extended here.

### 3-10: Bayesian Inference & Probabilistic Thinking
This topic implements full Bayesian inference from scratch using conjugate prior pairs: Beta-Binomial for binary data and Normal-Normal for continuous data, covering prior specification, likelihood computation, posterior derivation, and predictive distributions. Students implement Bayesian linear regression and compare point estimates (MLE, MAP) against full posterior distributions that quantify parameter uncertainty. Bayesian thinking is the conceptual foundation for several advanced topics: the ELBO objective in variational autoencoders (Module 11) is derived from Bayesian inference, reward modeling in RLHF (Module 13) benefits from uncertainty quantification, and Gaussian processes (Module 4-10) are a Bayesian approach to function estimation. This topic also revisits MAP estimation from Module 1-07 with deeper treatment.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 03-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 03-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

- `scipy` — statistical tests
- `umap-learn` — UMAP dimensionality reduction

---

## Datasets

- `sklearn.make_blobs` (3-01, 3-02, 3-06)
- `sklearn.make_moons` (3-02, 3-08)
- `sklearn` Digits (3-03, 3-09)
- MNIST (3-03, 3-04)
- FashionMNIST (3-04)
- Synthetic (3-05 mixed signals, 3-09, 3-10 Beta-Binomial/Normal-Normal)
- `sklearn.make_classification` (3-07)
- `sklearn.make_circles` (3-08)

---

## Prerequisites Chain

- **03-01:** Requires 1-01, 1-09
- **03-02:** Requires 3-01
- **03-03:** Requires 1-06
- **03-04:** Requires 3-03
- **03-05:** Requires 1-06
- **03-06:** Requires 1-07, 3-01
- **03-07:** Requires 1-07
- **03-08:** Requires 1-06, 2-05
- **03-09:** Requires 1-06
- **03-10:** Requires 1-07, 3-06

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 3 — Unsupervised and Statistical Learning
| Concept | Owner |
|---------|-------|
| K-Means clustering | 3-01 |
| Hierarchical clustering, DBSCAN | 3-02 |
| PCA (eigendecomposition-based) | 3-03 |
| t-SNE, UMAP | 3-04 |
| ICA (FastICA) | 3-05 |
| Gaussian mixture models, EM algorithm | 3-06 |
| Anomaly detection (statistical, isolation forest) | 3-07 |
| Kernel methods, kernel PCA, kernel density estimation | 3-08 |
| Matrix factorization (NMF, truncated SVD) | 3-09 |
| Bayesian inference, conjugate priors | 3-10 |

---

## Cross-Module Ownership Warnings

No special cross-module warnings for this module.

---

## Special Notes

No special notes for this module.
