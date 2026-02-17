# Module 02 — Supervised Learning

## Introduction

Module 02 introduces the core supervised learning algorithms that form the backbone of practical machine learning, from linear models to ensemble methods. Building directly on the mathematical foundations from Module 01, students implement each algorithm from scratch using NumPy and PyTorch before comparing against scikit-learn's optimized versions, gaining deep understanding of how these models learn decision boundaries, handle different data geometries, and trade off bias against variance. By the end of this module, students will be able to build, train, and evaluate linear regression, logistic regression, decision trees, random forests, gradient boosting, SVMs, k-NN, Naive Bayes, and ensemble methods, and will know how to choose the right algorithm for a given problem. These classical ML techniques are not only powerful in their own right but also provide the conceptual vocabulary -- loss minimization, regularization, kernel methods, boosting -- that recurs throughout the deep learning modules (5 through 20).

**Folder:** `module_02_supervised_learning/`

**GPU Required:** No (classical ML / math foundations)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 2-01 | Linear Regression | Ordinary least squares derivation from MLE; closed-form (normal equation) vs gradient descent; polynomial features and overfitting; regularization preview (ridge motivation); residual analysis | `sklearn` California Housing | ~2 min |
| 2-02 | Logistic Regression & Binary Classification | Sigmoid function derivation; log-loss (cross-entropy) from MLE; gradient descent update rule; L1 (Lasso) vs L2 (Ridge) regularization and their geometric interpretation; decision boundary visualization | `sklearn.make_moons` | ~2 min |
| 2-03 | Decision Trees & Random Forests | Gini impurity and information gain from scratch; recursive splitting; pruning strategies; bagging (bootstrap aggregating); Random Forest — feature subsampling; feature importance via impurity decrease | `sklearn` Iris, `sklearn.make_classification` | ~3 min |
| 2-04 | Gradient Boosting & AdaBoost | AdaBoost — sequential re-weighting of misclassified samples; Gradient Boosting — residual fitting via gradient descent in function space; shrinkage (learning rate); comparison with XGBoost concepts | `sklearn.make_classification` | ~3 min |
| 2-05 | Support Vector Machines | Maximum margin classifier; hard vs soft margin (C parameter); the kernel trick — RBF, polynomial kernels; support vectors and sparsity; dual formulation intuition; decision boundary visualization | `sklearn.make_moons`, `sklearn.make_circles` | ~3 min |
| 2-06 | Generalized Linear Models & Exponential Family | Exponential family distributions (Bernoulli, Gaussian, Poisson, multinomial); sufficient statistics and natural parameters; GLM construction — choose distribution → link function → learning rule; GLM derivation of linear regression, logistic regression, and softmax as special cases; unifying framework for 2-01 and 2-02 | `sklearn` Iris, California Housing | ~3 min |
| 2-07 | k-Nearest Neighbors | Distance metrics (Euclidean, Manhattan, cosine); curse of dimensionality with visualization; weighted voting; choosing k; KD-tree acceleration; comparison: lazy vs eager learners | `sklearn` Digits, `sklearn.make_blobs` | ~2 min |
| 2-08 | Naive Bayes for Text Classification | Bag-of-words and TF-IDF representations; multinomial Naive Bayes derivation; Laplace smoothing; conditional independence assumption and when it breaks; text classification pipeline end-to-end | `sklearn.fetch_20newsgroups` | ~2 min |
| 2-09 | Stacking & Voting Ensembles | Hard voting vs soft voting; meta-learner stacking (level-0 base models → level-1 combiner); blending; comparison of ensemble strategies; when ensembles help vs hurt | `sklearn` Digits | ~3 min |
| 2-10 | Model Comparison & Algorithm Selection | All algorithms on the same dataset; performance table (accuracy, F1, training time); choosing the right algorithm — decision framework based on data size, dimensionality, interpretability, and latency | `sklearn` Digits | ~5 min |

---

## Topic Details

### 2-01: Linear Regression
Students derive the ordinary least squares (OLS) objective from maximum likelihood estimation, then implement both the closed-form normal equation solution and iterative gradient descent in NumPy on the California Housing dataset. The notebook explores polynomial feature expansion to demonstrate overfitting, motivating the need for regularization (ridge regression preview), and includes thorough residual analysis to diagnose model assumptions. Linear regression is the simplest supervised learning model and serves as the conceptual anchor for understanding loss minimization, gradient-based optimization, and regularization -- themes that recur in every model from logistic regression (2-02) through deep neural networks (Module 5). The MLE derivation here also connects forward to the generalized linear model framework in 2-06.

### 2-02: Logistic Regression & Binary Classification
This topic derives the sigmoid function and binary cross-entropy loss from maximum likelihood estimation, then implements logistic regression with gradient descent from scratch. Students visualize decision boundaries on the make_moons dataset and implement both L1 (Lasso) and L2 (Ridge) regularization, exploring their geometric interpretations -- the diamond vs circle constraint regions that lead to sparsity vs shrinkage. The log-loss derived here is the same cross-entropy loss used in neural network classification (Module 5-04), making logistic regression a direct precursor to deep learning. The regularization techniques connect forward to weight decay in neural networks (5-10) and the L1/L2 penalty concepts in feature selection (4-03).

### 2-03: Decision Trees & Random Forests
Students implement decision tree construction from scratch, computing Gini impurity and information gain at each split node, performing recursive partitioning, and applying pruning strategies to control overfitting. The notebook then builds Random Forests by implementing bagging (bootstrap aggregating) with feature subsampling, demonstrating how ensembling reduces variance while maintaining low bias. Feature importance via impurity decrease is implemented and visualized, providing an interpretability tool used throughout the course. Decision trees are the foundation for all tree-based ensemble methods -- gradient boosting (2-04), stacking (2-09) -- and the information gain criterion connects back to entropy from Module 1-08.

### 2-04: Gradient Boosting & AdaBoost
This topic implements two foundational boosting algorithms from scratch: AdaBoost, which sequentially re-weights misclassified samples to focus on hard examples, and Gradient Boosting, which fits each new tree to the negative gradient (residuals) of the loss function. Students explore the shrinkage (learning rate) parameter that controls the contribution of each tree and understand gradient boosting as gradient descent in function space. These algorithms consistently rank among the top performers on tabular data in practice, and the residual-fitting idea in gradient boosting is conceptually related to residual connections in deep networks (Module 6, Module 8). The notebook also introduces XGBoost concepts to connect to production-grade implementations.

### 2-05: Support Vector Machines
Students implement the maximum margin classifier, exploring hard-margin and soft-margin (C parameter) formulations, and then implement the kernel trick to handle non-linearly separable data using RBF and polynomial kernels on make_moons and make_circles datasets. The notebook develops intuition for the dual formulation, support vector sparsity, and how the kernel function implicitly maps data into high-dimensional feature spaces without computing the transformation explicitly. SVMs are a cornerstone of classical ML and introduce the kernel concept that reappears in kernel PCA (3-08), Gaussian processes (4-10), and even attention mechanisms (Module 8) where softmax can be viewed as a kernel. The dual formulation connects forward to convex optimization theory in Module 4-08.

### 2-06: Generalized Linear Models & Exponential Family
This topic presents the exponential family of distributions -- Bernoulli, Gaussian, Poisson, and multinomial -- and shows how each member's sufficient statistics and natural parameters lead to a systematic GLM construction: choose a distribution, derive its canonical link function, and obtain the learning rule. Students implement the GLM framework from scratch and verify that linear regression (2-01), logistic regression (2-02), and softmax classification all emerge as special cases of this single unifying framework. This theoretical unification deepens understanding of why specific loss functions pair with specific activation functions, a pattern that recurs in neural network design (Module 5). The softmax derivation here is the same softmax used in attention mechanisms (Module 8) and language model output layers (Module 10).

### 2-07: k-Nearest Neighbors
Students implement k-NN from scratch with multiple distance metrics (Euclidean, Manhattan, cosine), weighted voting schemes, and KD-tree acceleration for efficient neighbor lookup. The notebook includes a compelling visualization of the curse of dimensionality, showing how distance metrics become less discriminative as dimensionality grows -- a phenomenon that motivates dimensionality reduction techniques in Module 3. k-NN is the canonical "lazy learner" (no training phase), and comparing it against eager learners highlights fundamental tradeoffs between training time and inference time. The distance metric concepts introduced here recur in clustering (Module 3-01, 3-02), retrieval-augmented generation (Module 18), and nearest-neighbor search in embedding spaces.

### 2-08: Naive Bayes for Text Classification
This topic implements a complete text classification pipeline from scratch: bag-of-words and TF-IDF text representations, multinomial Naive Bayes derivation from Bayes' theorem, and Laplace smoothing to handle zero-count words, all evaluated end-to-end on the 20 Newsgroups dataset. Students analyze the conditional independence assumption, understanding when it holds approximately and when it fails, and why the model often performs well despite this strong assumption. Naive Bayes connects back to probability fundamentals (1-07) and information theory (1-08), while the bag-of-words and TF-IDF representations introduced here are precursors to the word embedding approaches in Module 7 (RNNs and NLP) and the tokenization strategies in Module 8 (Transformers).

### 2-09: Stacking & Voting Ensembles
Students implement hard voting, soft voting, and meta-learner stacking (level-0 base models feeding a level-1 combiner) from scratch, along with blending as a simpler alternative. The notebook systematically compares ensemble strategies on the Digits dataset, analyzing when combining diverse models improves performance and when it fails (e.g., when base models are too correlated). Understanding ensemble methods is essential for Module 4 (cross-validation in stacking prevents leakage) and for practical ML deployment where ensembles of neural networks are common. The stacking concept also connects forward to mixture-of-experts architectures in Module 17 (LLM systems) where different "expert" sub-networks specialize on different inputs.

### 2-10: Model Comparison & Algorithm Selection
This capstone comparison notebook trains all algorithms from 2-01 through 2-09 on the same Digits dataset under identical conditions and produces a comprehensive performance table covering accuracy, F1 score, and training time. Students develop a decision framework for algorithm selection based on data size, dimensionality, interpretability requirements, and latency constraints -- the practical reasoning that separates ML engineers from tutorial followers. This topic synthesizes the entire module and connects forward to Module 4 (evaluation metrics, cross-validation) where students will learn rigorous comparison methodology. The comparison framework established here also applies to neural architecture selection in the deep learning modules.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 02-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 02-10 | F — Comparison/Architecture | `TEMPLATE_COMPARISON.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- `sklearn` California Housing (2-01, 2-06)
- `sklearn.make_moons` (2-02, 2-05)
- `sklearn.make_classification` (2-03, 2-04)
- `sklearn` Iris (2-03, 2-06)
- `sklearn.make_circles` (2-05)
- `sklearn` Digits (2-07, 2-09, 2-10)
- `sklearn.make_blobs` (2-07)
- `sklearn.fetch_20newsgroups` (2-08)

---

## Prerequisites Chain

- **02-01:** Requires 1-01, 1-09
- **02-02:** Requires 2-01, 1-07
- **02-03:** Requires 1-08
- **02-04:** Requires 2-03
- **02-05:** Requires 1-06, 1-09
- **02-06:** Requires 2-01, 2-02, 1-07
- **02-07:** Requires 1-01
- **02-08:** Requires 1-07, 1-08
- **02-09:** Requires 2-03, 2-04
- **02-10:** Requires 2-01 through 2-09

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 2 — Supervised Learning
| Concept | Owner |
|---------|-------|
| Linear regression (OLS, gradient descent) | 2-01 |
| Logistic regression, sigmoid, log-loss | 2-02 |
| Decision trees, Gini impurity, information gain | 2-03 |
| Gradient boosting, AdaBoost | 2-04 |
| Support vector machines, kernel trick, dual formulation | 2-05 |
| Generalized linear models, exponential family | 2-06 |
| k-Nearest Neighbors, distance metrics | 2-07 |
| Naive Bayes, bag-of-words, TF-IDF | 2-08 |
| Stacking, voting ensembles | 2-09 |
| Algorithm comparison and selection framework | 2-10 |

---

## Cross-Module Ownership Warnings

No special cross-module warnings for this module.

---

## Special Notes

- Classical ML module — sklearn is the primary tool. PyTorch is used for gradient descent demonstrations.
- 2-06 (GLMs) unifies 2-01 (linear regression) and 2-02 (logistic regression) as special cases.
