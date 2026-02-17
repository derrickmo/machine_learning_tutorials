# Module 19 — ML Applications & Domain Problems

## Introduction

Module 19 applies the machine learning and deep learning techniques from the preceding 18 modules to a diverse set of real-world application domains — from recommender systems and time series forecasting to graph neural networks and algorithmic fairness. This module demonstrates that mastering ML fundamentals is only half the story; each application domain introduces unique data structures, evaluation criteria, and design constraints that require adapting core techniques in domain-specific ways. Students will build recommender engines with matrix factorization and neural collaborative filtering, forecast time series with LSTM and Transformer encoders, construct graph convolutional networks for node classification, and implement SHAP for model-agnostic explainability. After completing this module, students will have hands-on experience with the application areas most commonly encountered in industry and ML system design interviews, and will understand the responsible AI considerations — fairness, bias detection, and ethics — that must accompany any deployed system. Module 19 serves as the practical synthesis layer before the production deployment and MLOps focus of Module 20.

**Folder:** `module_19_ml_applications/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 19-01 | Recommender Systems — Collaborative Filtering | User-item interaction matrix; matrix factorization (SVD, ALS); implicit vs explicit feedback; evaluation metrics (NDCG, MAP, Hit Rate); cold start problem | Synthetic MovieLens-style | ~5 min |
| 19-02 | Recommender Systems — Neural & Hybrid | Embedding-based neural collaborative filtering (NCF); two-tower architecture for retrieval; content features integration; hybrid collaborative + content-based approach; serving at scale concepts | Synthetic MovieLens-style | ~8 min |
| 19-03 | Time Series Forecasting — Classical & Deep | Windowing and stationarity testing; autoregressive features; LSTM encoder-decoder for forecasting; Transformer for time series; evaluation (MAE, MAPE, SMAPE); walk-forward validation | Synthetic time series | ~10 min |
| 19-04 | Search & Ranking Systems | Learning to rank (pointwise, pairwise, listwise approaches); feature engineering for ranking; two-stage retrieval + ranking architecture; NDCG optimization; relevance feedback; common ML system design interview question | AG_NEWS, Synthetic queries | ~8 min |
| 19-05 | Explainability — SHAP & Interpretability | SHAP values — Shapley value derivation and kernel SHAP from scratch; permutation importance; partial dependence plots; LIME concepts; feature importance comparison: tree models vs neural networks; distinct from 10-10 (transformer-specific mechanistic interpretability) — this covers model-agnostic methods | `sklearn` California Housing | ~5 min |
| 19-06 | Graph Neural Networks | Graph representation (adjacency matrix, edge list); message passing framework; Graph Convolutional Network (GCN) from scratch (Kipf & Welling); node classification on citation network; graph attention concepts | Synthetic citation network | ~8 min |
| 19-07 | Audio Classification | Audio classification pipeline using features from 12-08 (mel-spectrograms, MFCCs — reference only, do not re-teach); CNN and Transformer on spectrograms; SpecAugment for audio data augmentation; speech commands classification end-to-end; distinct from 12-08 (audio representations) — this covers the full classification application | SPEECHCOMMANDS | ~10 min |
| 19-08 | Semi-Supervised & Multi-Task Learning | Self-training with pseudo-labels; confidence thresholding; consistency regularization (FixMatch concepts); hard vs soft parameter sharing; task-specific heads on shared backbone; loss balancing strategies (uncertainty weighting, gradient normalization); when each paradigm helps vs hurts performance | FashionMNIST, CIFAR-10 | ~10 min |
| 19-09 | Fairness, Bias & Ethics in ML | Algorithmic fairness definitions (demographic parity, equalized odds, calibration); bias detection in datasets and models; disparate impact analysis; bias mitigation strategies (pre-processing, in-processing, post-processing); fairness-accuracy tradeoffs; model cards and responsible AI documentation; connection to data filtering in 17-04 and guardrails in 18-09 | `sklearn.make_classification` (synthetic biased) | ~5 min |
| 19-10 | Tabular Deep Learning | Why trees often beat neural nets on tabular data; TabNet (attentive feature selection); FT-Transformer (feature tokenization); embedding categorical features for neural networks; head-to-head comparison with XGBoost | `sklearn` California Housing | ~10 min |

---

## Topic Details

### 19-01: Recommender Systems — Collaborative Filtering
Students will build a recommender system from scratch starting with the user-item interaction matrix, implementing matrix factorization via SVD and Alternating Least Squares (ALS) to learn latent user and item embeddings. The notebook covers the distinction between implicit feedback (clicks, views) and explicit feedback (ratings), and how the loss function changes accordingly. Students will implement ranking-specific evaluation metrics — NDCG, MAP, and Hit Rate — and analyze the cold start problem where new users or items have no interaction history. Collaborative filtering is the foundation of modern recommendation systems and directly extends the matrix decomposition techniques from Module 3 (3-09) into a practical application.

### 19-02: Recommender Systems — Neural & Hybrid
This notebook progresses from classical matrix factorization to neural collaborative filtering (NCF), implementing embedding layers for users and items with a multi-layer perceptron that learns non-linear interactions. Students will build the two-tower architecture used in production retrieval systems — separate user and item encoders whose dot product scores candidates efficiently at serving time. The notebook integrates content features (item metadata, user demographics) into a hybrid approach that combines collaborative and content-based signals, and discusses serving considerations at scale including approximate nearest neighbor retrieval for candidate generation. These architectures directly mirror what is deployed at companies like YouTube, Netflix, and Spotify.

### 19-03: Time Series Forecasting — Classical & Deep
Students will implement a complete time series forecasting pipeline, starting with windowing strategies and stationarity testing (ADF test) to properly structure sequential data for ML models. The notebook builds autoregressive features, then implements an LSTM encoder-decoder architecture and a Transformer model adapted for time series, comparing both against classical baselines. Evaluation uses walk-forward validation (expanding or sliding window) with MAE, MAPE, and SMAPE metrics that account for the temporal ordering that standard cross-validation would violate. This topic applies the RNN knowledge from Module 7 and Transformer knowledge from Module 8 to one of the most common industrial ML applications.

### 19-04: Search & Ranking Systems
This notebook implements a learning-to-rank system, covering pointwise (regression on relevance scores), pairwise (learning relative ordering between document pairs), and listwise (directly optimizing NDCG) approaches. Students will build the two-stage retrieval-then-ranking architecture used in production search engines — a fast retrieval stage (connecting to 18-01 and 18-02 for embedding-based retrieval) followed by a more accurate ranking model with rich features. Feature engineering for ranking (query-document similarity, document quality signals, freshness) and relevance feedback loops are implemented. Search and ranking is one of the most common ML system design interview topics and the most revenue-generating ML application at major tech companies.

### 19-05: Explainability — SHAP & Interpretability
Students will derive Shapley values from cooperative game theory and implement kernel SHAP from scratch, understanding how each feature's contribution to a prediction is computed as a weighted average over all possible feature coalitions. The notebook also implements permutation importance and partial dependence plots as complementary model-agnostic interpretability tools, and introduces LIME conceptually. A key comparison shows how feature importance differs between tree models (which have built-in importance) and neural networks (which require post-hoc methods). This topic is distinct from 10-10 (mechanistic interpretability for transformers) by focusing on methods that work for any model type — essential knowledge for deploying models in regulated industries like healthcare and finance.

### 19-06: Graph Neural Networks
This notebook introduces graph-structured data — adjacency matrices, edge lists, and node feature matrices — and builds a Graph Convolutional Network (GCN) from scratch following the Kipf and Welling formulation. Students will implement the message passing framework where each node aggregates information from its neighbors, apply symmetric normalization with the degree matrix, and train the GCN for node classification on a synthetic citation network. Graph attention concepts (GAT) are introduced to show how attention mechanisms weight neighbor contributions adaptively. GNNs are increasingly important for social network analysis, molecule property prediction, and knowledge graphs, and this topic extends the neural network foundations from Module 5 into non-Euclidean data structures.

### 19-07: Audio Classification
Students will build a complete audio classification pipeline using mel-spectrogram and MFCC features from 12-08 (referenced but not re-taught) as input to both CNN and Transformer classifiers. The notebook implements SpecAugment — time warping, frequency masking, and time masking applied to spectrograms as data augmentation — which is one of the most effective techniques for improving audio model robustness. The full speech commands classification pipeline is built end-to-end from raw audio to predicted labels, with careful attention to audio-specific preprocessing (resampling, padding/truncation, windowing). This topic demonstrates how the CNN architectures from Module 6 and Transformer architectures from Module 8 adapt to audio data through the spectrogram representation.

### 19-08: Semi-Supervised & Multi-Task Learning
This notebook combines two data-efficiency paradigms into a single topic. For semi-supervised learning, students will implement self-training with pseudo-labels (using confident predictions on unlabeled data as training targets), confidence thresholding, and consistency regularization following the FixMatch approach. For multi-task learning, students will build shared-backbone architectures with task-specific heads using both hard parameter sharing (shared layers) and soft parameter sharing (regularized separate models), and implement loss balancing strategies including uncertainty weighting and gradient normalization. The notebook analyzes when each paradigm helps versus hurts performance, providing practical guidelines for leveraging unlabeled data and related tasks.

### 19-09: Fairness, Bias & Ethics in ML
Students will implement formal algorithmic fairness definitions — demographic parity, equalized odds, and calibration — and learn to detect bias in both datasets (representation imbalance, label bias) and trained models (disparate impact analysis). The notebook builds bias mitigation strategies at three stages: pre-processing (resampling, reweighting), in-processing (adversarial debiasing, fairness constraints in the loss), and post-processing (threshold adjustment per group). Students will empirically measure the fairness-accuracy tradeoff and create model cards documenting model capabilities, limitations, and ethical considerations. This topic connects to data filtering ethics in 17-04 and safety guardrails in 18-09 while covering the distinct scope of algorithmic fairness — a critical concern for any deployed ML system.

### 19-10: Tabular Deep Learning
This notebook addresses the persistent question of why gradient-boosted trees (XGBoost, LightGBM) often outperform neural networks on tabular data, and implements the two leading deep learning approaches designed to close that gap. Students will build TabNet from scratch — using sequential attention to select relevant features at each decision step — and FT-Transformer, which tokenizes each feature into an embedding and processes them through a Transformer encoder. The notebook implements proper categorical feature embedding for neural networks and conducts a rigorous head-to-head comparison with XGBoost on the California Housing dataset, analyzing when deep learning adds value on tabular data versus when trees remain superior. This topic bridges the tree-based methods from Module 2 with the deep learning architectures from Modules 5 and 8.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 19-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 19-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- Synthetic MovieLens-style (19-01, 19-02)
- Synthetic time series (19-03)
- AG_NEWS (19-04)
- `sklearn` California Housing (19-05, 19-10)
- Synthetic citation network (19-06)
- SPEECHCOMMANDS (19-07)
- FashionMNIST (19-08)
- CIFAR-10 (19-08)
- `sklearn.make_classification` synthetic biased (19-09)

---

## Prerequisites Chain

- **19-01:** Requires 3-09, 1-06
- **19-02:** Requires 19-01, 5-07
- **19-03:** Requires 2-01, 7-04, 8-04
- **19-04:** Requires 2-08, 10-04, 18-01
- **19-05:** Requires 2-03, 2-04
- **19-06:** Requires 1-06, 5-07
- **19-07:** Requires 12-08, 6-03
- **19-08:** Requires 5-07, 10-02, 4-02
- **19-09:** Requires 4-01, 4-05
- **19-10:** Requires 5-07, 8-04, 2-04

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 19 — ML Applications
| Concept | Owner |
|---------|-------|
| Recommender systems (collaborative filtering, MF) | 19-01 |
| Neural recommenders, two-tower architecture | 19-02 |
| Time series forecasting | 19-03 |
| Search and ranking systems | 19-04 |
| SHAP, model-agnostic explainability, interpretability | 19-05 |
| Graph neural networks (GCN) | 19-06 |
| Audio classification pipeline (end-to-end, SpecAugment) | 19-07 |
| Semi-supervised learning (pseudo-labeling, FixMatch), multi-task learning (parameter sharing, loss balancing) | 19-08 |
| Algorithmic fairness, bias detection, responsible AI | 19-09 |
| Tabular deep learning (TabNet, FT-Transformer) | 19-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ SHAP (19-05) is model-agnostic interpretability. Mechanistic interpretability (10-10) is transformer-specific — distinct scope.
- ⚠️ Audio classification (19-07) uses mel-spectrograms and MFCCs from 12-08. Do NOT re-teach audio representations — reference 12-08 with a one-line comment and focus on the classification pipeline and SpecAugment.
- ⚠️ Fairness (19-09) covers algorithmic fairness. Data filtering ethics is in 17-04; guardrails/safety is in 18-09 — distinct scopes.

---

## Special Notes

- Original 19-08 (Semi-Supervised Learning) and 19-09 (Multi-Task Learning) were merged into 19-08 as both are data-efficiency paradigms.
- Freed slot used for 19-09 (Fairness, Bias & Ethics) — a critical gap in the original curriculum.
