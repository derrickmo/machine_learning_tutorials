# Module 05 — Neural Network Foundations

## Introduction

Module 05 marks the transition from classical machine learning to deep learning, taking students from a raw NumPy neural network through the complete PyTorch ecosystem of autograd, nn.Module, optimizers, and regularization. Starting with a two-layer network built entirely in NumPy, the module systematically covers every component of modern neural network training: activation functions, loss functions, computational graphs, backpropagation, automatic differentiation, optimizer algorithms (SGD through AdamW), learning rate scheduling, and regularization techniques (dropout, batch norm, layer norm). By the end of this module, students will be able to build, train, debug, and regularize feedforward neural networks in PyTorch, with a deep understanding of how each component works under the hood. This module is the foundation for all subsequent deep learning modules -- every CNN (Module 6), RNN (Module 7), transformer (Module 8), and generative model (Module 11) builds on the activation functions, loss functions, optimizers, and regularization techniques taught here.

**Folder:** `module_05_neural_network_foundations/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 5-01 | Neural Network End-to-End Walkthrough | Big-picture overview of deep learning; build and train a 2-layer neural network in pure NumPy; forward pass, loss computation, backward pass, weight update; visualize learning on toy data | `sklearn.make_moons` | ~3 min |
| 5-02 | Perceptron & Multi-Layer Architecture | Single perceptron (AND/OR/XOR); MLP architecture — width vs depth tradeoffs; weight initialization (Xavier/Glorot, He/Kaiming) and why it matters; universal approximation theorem intuition | `sklearn.make_moons`, FashionMNIST | ~5 min |
| 5-03 | Activation Functions Deep Dive | Sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU, Swish/SiLU, softmax; dying ReLU problem; smooth approximations; when to use which; implement each from scratch and visualize gradients | Synthetic (function plots) | ~3 min |
| 5-04 | Loss Functions Deep Dive | MSE, MAE, Huber loss for regression; binary cross-entropy, categorical cross-entropy for classification; hinge loss (SVM connection); focal loss for imbalanced data; KL divergence loss; implement each and analyze gradient behavior | Synthetic, FashionMNIST | ~5 min |
| 5-05 | Forward Pass & Computational Graphs | Representing computation as DAGs; tracing values through the graph; intermediate activations and memory; connection to automatic differentiation; building a simple autograd engine | Synthetic (small computation graphs) | ~3 min |
| 5-06 | Backpropagation from Scratch | Chain rule applied to computational graphs; gradient flow visualization layer-by-layer; numerical gradient checking; vanishing and exploding gradients — causes, diagnosis, mitigation; implement backprop for a 3-layer MLP in NumPy | FashionMNIST | ~5 min |
| 5-07 | PyTorch Fundamentals — Autograd & nn.Module | torch.autograd mechanics (tape-based AD); nn.Module, nn.Sequential, nn.ModuleList; parameter registration; torch.nn.init; converting NumPy networks to PyTorch; custom forward methods | FashionMNIST | ~5 min |
| 5-08 | Optimizers — SGD, Momentum & Nesterov | Vanilla SGD derivation; momentum — exponential moving average of gradients; Nesterov accelerated gradient; mini-batch vs full-batch vs stochastic; learning rate sensitivity; visualize optimizer trajectories on loss surfaces | `sklearn.make_moons`, FashionMNIST | ~5 min |
| 5-09 | Advanced Optimizers & Learning Rate Scheduling | RMSProp — adaptive per-parameter rates; Adam — combining momentum and RMSProp; AdamW — decoupled weight decay (implement from scratch per CS336 A1); learning rate finder (Leslie Smith); one-cycle policy; cosine annealing with warm restarts | FashionMNIST | ~8 min |
| 5-10 | Regularization Techniques | Dropout (standard and inverted); batch normalization (running stats, train vs eval mode); layer normalization; early stopping with patience; weight decay vs L2 regularization; data augmentation as regularization | FashionMNIST | ~8 min |

---

## Topic Details

### 5-01: Neural Network End-to-End Walkthrough
Students build and train a complete two-layer neural network in pure NumPy on the make_moons dataset, implementing the forward pass, loss computation (cross-entropy), backward pass (manual gradient derivation), and weight update loop from scratch without any deep learning framework. The notebook visualizes the decision boundary evolving during training to build intuition for how gradient-based learning shapes the model. This is intentionally the "from zero to working network" experience that demystifies deep learning before introducing the abstractions of PyTorch. It connects forward to every subsequent topic in the module: the forward pass is formalized in 5-05, the backward pass becomes backpropagation in 5-06, and the manual weight updates are replaced by optimizers in 5-08.

### 5-02: Perceptron & Multi-Layer Architecture
This topic starts with the single perceptron, demonstrating its ability to learn AND and OR gates but not XOR (the linearly inseparable case that motivated multi-layer networks), then builds multi-layer perceptrons (MLPs) with varying widths and depths on FashionMNIST. Students implement Xavier/Glorot initialization (for sigmoid/tanh) and He/Kaiming initialization (for ReLU) from scratch, showing empirically how proper initialization prevents signal collapse or explosion in deep networks. The universal approximation theorem is presented intuitively: a sufficiently wide single-hidden-layer network can approximate any continuous function, but depth provides exponentially more efficient representations. These architecture principles and initialization strategies are used in every deep network throughout Modules 6-20.

### 5-03: Activation Functions Deep Dive
Students implement every major activation function from scratch -- sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU, Swish/SiLU, and softmax -- and visualize both the function outputs and their gradient behavior across the input range. The notebook diagnoses the dying ReLU problem (neurons with permanently zero gradients), explains why smooth approximations like GELU and SiLU outperform ReLU in modern architectures, and provides practical guidance on which activation to use in which context. GELU is the activation used in transformers (Module 8) and most modern LLMs (Module 17), SiLU/Swish appears in efficient architectures like EfficientNet, and softmax is the output activation for classification and the core of attention mechanisms. Understanding gradient flow through activations is essential for diagnosing training failures throughout the course.

### 5-04: Loss Functions Deep Dive
This topic implements the major loss functions from scratch -- MSE, MAE, and Huber loss for regression; binary and categorical cross-entropy for classification; hinge loss (connecting back to SVMs in 2-05); focal loss for imbalanced data; and KL divergence loss -- and analyzes each function's gradient behavior to understand training dynamics. Students learn that loss function choice encodes assumptions about the problem: MSE assumes Gaussian noise, cross-entropy derives from MLE for Bernoulli/categorical distributions (connecting to 1-07), focal loss down-weights easy examples (connecting to imbalanced data in 4-05), and KL divergence measures distributional distance (connecting to 1-08). These loss functions appear throughout the deep learning modules: cross-entropy in every classifier, KL divergence in VAEs (Module 11) and RLHF (Module 13), and focal loss in object detection (Module 9).

### 5-05: Forward Pass & Computational Graphs
Students learn to represent neural network computations as directed acyclic graphs (DAGs), trace values through each node, and analyze how intermediate activations consume memory during the forward pass. The notebook builds a simple autograd engine that records operations and supports automatic backward pass computation, providing the conceptual model for how PyTorch's autograd works under the hood (formalized in 5-07). Understanding computational graphs is essential for implementing custom operations, diagnosing memory issues in large models, and techniques like gradient checkpointing (Module 16) that trade compute for memory. This topic bridges the manual gradient computation in 5-01 with the systematic backpropagation algorithm in 5-06.

### 5-06: Backpropagation from Scratch
Students implement the backpropagation algorithm by applying the chain rule to computational graphs, computing gradients layer-by-layer for a 3-layer MLP in NumPy, and verifying correctness with numerical gradient checking (finite differences). The notebook includes detailed gradient flow visualizations that show how gradients propagate backward through each layer, directly demonstrating the vanishing gradient problem (gradients shrink through sigmoid/tanh) and the exploding gradient problem (gradients grow unboundedly in deep networks). Students implement mitigations including gradient clipping and proper initialization. Backpropagation is the single most important algorithm in deep learning -- it is referenced in every module from 6 through 17 -- and this notebook is the sole owner of its derivation. The vanishing/exploding gradient analysis motivates residual connections (Module 6), LSTM gates (Module 7), and layer normalization (5-10).

### 5-07: PyTorch Fundamentals -- Autograd & nn.Module
This topic transitions from NumPy implementations to PyTorch, covering torch.autograd's tape-based automatic differentiation, the nn.Module class hierarchy (nn.Module, nn.Sequential, nn.ModuleList), proper parameter registration, weight initialization with torch.nn.init, and custom forward methods. Students convert their NumPy networks from earlier topics into idiomatic PyTorch code, learning the patterns (`.to(device)`, `.train()` / `.eval()`, `.zero_grad()`) that every PyTorch model uses. This is the gateway to all PyTorch-based development in Modules 6-20: every CNN, RNN, transformer, GAN, and LLM follows the nn.Module pattern established here. The autograd mechanics also connect forward to Module 15 (Advanced PyTorch Internals) where students explore custom autograd functions and JIT compilation.

### 5-08: Optimizers -- SGD, Momentum & Nesterov
Students derive and implement vanilla stochastic gradient descent, momentum (as an exponential moving average of gradients), and Nesterov accelerated gradient from scratch, then visualize optimizer trajectories on 2D loss surfaces to build geometric intuition. The notebook explores mini-batch vs full-batch vs stochastic training, learning rate sensitivity analysis, and the practical impact of batch size on convergence. These first-order optimization methods are the building blocks for the adaptive optimizers in 5-09 (RMSProp, Adam), and SGD with momentum remains competitive with Adam for certain architectures (particularly CNNs with careful learning rate tuning). The loss surface visualization introduced here recurs in Module 16 (training optimization) where students analyze loss landscape geometry for large-scale models.

### 5-09: Advanced Optimizers & Learning Rate Scheduling
This topic implements RMSProp (adaptive per-parameter learning rates), Adam (combining momentum with RMSProp), and AdamW (decoupled weight decay, implemented from scratch following the CS336 Assignment 1 specification) as the modern optimizer toolkit. Students also implement the learning rate finder (Leslie Smith's method for finding optimal learning rate ranges), one-cycle policy, and cosine annealing with warm restarts. AdamW is the default optimizer for virtually all modern deep learning -- transformers (Module 8), language models (Modules 10, 17), and fine-tuning (Module 13) -- and this notebook is the sole owner of its derivation. The learning rate scheduling strategies introduced here are critical for training stability in large-scale models and are applied throughout Modules 6-20.

### 5-10: Regularization Techniques
Students implement the core regularization techniques from scratch: standard and inverted dropout (with the scaling correction), batch normalization (tracking running statistics, train vs eval mode behavior), layer normalization, early stopping with patience, and weight decay. The notebook clarifies the subtle difference between L2 regularization and weight decay (they are equivalent for SGD but different for Adam/AdamW), and frames data augmentation (from 4-04) as another form of regularization. Batch normalization is used in virtually all CNNs (Module 6), layer normalization is the standard in transformers (Module 8) and all subsequent NLP/LLM modules, and dropout remains a key regularization tool across architectures. The train/eval mode distinction introduced here is critical for correct model evaluation and deployment throughout the course.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 05-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 05-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- `sklearn.make_moons` (5-01, 5-02, 5-08)
- FashionMNIST (5-02, 5-04, 5-06, 5-07, 5-08, 5-09, 5-10)
- Synthetic (5-03 function plots, 5-05 computation graphs)

---

## Prerequisites Chain

- **05-01:** Requires 1-01, 1-09
- **05-02:** Requires 5-01
- **05-03:** Requires 5-02
- **05-04:** Requires 5-02, 1-08
- **05-05:** Requires 5-03, 5-04
- **05-06:** Requires 5-05, 1-09
- **05-07:** Requires 5-06
- **05-08:** Requires 5-07, 1-09
- **05-09:** Requires 5-08
- **05-10:** Requires 5-07

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 5 — Neural Network Foundations
| Concept | Owner |
|---------|-------|
| Backpropagation (chain rule on computational graphs) | 5-06 |
| Activation functions (sigmoid, tanh, ReLU, GELU, SiLU/Swish) | 5-03 |
| Loss functions (MSE, BCE, cross-entropy, focal, Huber) | 5-04 |
| Forward pass, computational graphs | 5-05 |
| Autograd, nn.Module, nn.Sequential | 5-07 |
| SGD, momentum, Nesterov | 5-08 |
| Adam, AdamW (from scratch), LR finder, scheduling | 5-09 |
| Dropout, batch norm, layer norm, early stopping | 5-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ Backpropagation (5-06) is referenced extensively in Modules 6-17. Other notebooks must NOT re-derive it — one-line comment only.
- ⚠️ Adam/AdamW (5-09) is used everywhere. Only 5-09 teaches the algorithm.
- ⚠️ Focal loss (5-04) is the canonical implementation. Module 15-04 (Custom Loss Functions) implements additional custom losses (contrastive, label smoothing CE) but must reference 5-04 for focal loss — do not re-implement.
- ⚠️ Gradient clipping is introduced briefly in 5-06 as a vanishing/exploding gradient mitigation. Module 15-04 implements reusable gradient clipping utilities, and Module 16-06 covers clipping strategies for training stability. Each has a distinct scope: 5-06 = concept introduction, 15-04 = utility implementation, 16-06 = production strategies.

---

## Special Notes

No special notes for this module.
