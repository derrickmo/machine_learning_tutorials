# Module 01 — Mathematical & Programming Foundations

This module establishes the mathematical and programming toolkit that every subsequent module in the course depends on.

---

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 1-01 | Python, NumPy & Tensor Speed | Vectorization benchmarks, broadcasting, memory layout |
| 1-02 | Advanced NumPy & PyTorch Operations | Reshape, einsum, advanced indexing, in-place ops |
| 1-03 | Pandas for Tabular Data | EDA workflow, missing values, categorical encoding |
| 1-04 | Visualization with Matplotlib | Figure/axes model, plot types, decision boundaries |
| 1-05 | Data Loading with PyTorch | Dataset, DataLoader, transforms, splitting strategies |
| 1-06 | Linear Algebra for Machine Learning | Eigendecomposition, SVD, low-rank approximation |
| 1-07 | Probability & Statistics for ML | Distributions, MLE, MAP, Bayes' theorem |
| 1-08 | Information Theory for ML | Entropy, cross-entropy, KL divergence, mutual information |
| 1-09 | Calculus & Optimization Foundations | Gradients, chain rule, Jacobian, gradient descent |
| 1-10 | Computational Thinking & Complexity | Big-O, vectorization analysis, memory hierarchy |

---

## Prerequisites

None — this is the entry point to the course. High school mathematics is assumed.

---

## Learning Path

Topics 1-01 and 1-02 (NumPy/PyTorch fundamentals) should be completed first. After that:

- **Programming track:** 1-01 → 1-02 → 1-03 → 1-04 → 1-05
- **Mathematics track:** 1-01 → 1-06 → 1-07 → 1-08 → 1-09 → 1-10

Both tracks can be done in parallel after completing 1-01 and 1-02.

---

## Cross-Module Connections

- **Module 2 (Supervised Learning):** Uses Pandas EDA (1-03), visualization (1-04), and probability (1-07) extensively.
- **Module 3 (Unsupervised Learning):** Eigendecomposition and SVD (1-06) power PCA and matrix factorization.
- **Module 5 (Neural Networks):** Chain rule and gradient descent (1-09) are the foundation for backpropagation. Information theory (1-08) connects to loss functions.
- **Module 8 (Transformers):** Einsum notation (1-02) and linear algebra (1-06) are essential for attention mechanisms.
- **Module 16 (Training Optimization):** Computational complexity (1-10) informs distributed training decisions.
