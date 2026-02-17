# Module 04 — ML Theory & Evaluation

## Introduction

Module 04 bridges the gap between building ML models and rigorously evaluating, debugging, and understanding them, covering both practical evaluation methodology and formal learning theory. The first half (4-01 through 4-05) teaches the evaluation and preprocessing toolkit every ML practitioner needs -- metrics, cross-validation, feature engineering, data augmentation, and imbalanced data handling -- while the second half (4-06 through 4-10) provides the theoretical foundations from Stanford CS229 that explain why models generalize, how to diagnose and fix learning failures, and how to quantify uncertainty. By the end of this module, students will be able to properly evaluate any model, build reproducible preprocessing pipelines, understand the bias-variance tradeoff at a mathematical level, and apply Bayesian optimization for hyperparameter search. These skills are prerequisite for all deep learning modules: calibration (4-09) is critical for LLM confidence scoring, data augmentation (4-04) is essential for computer vision, and the debugging framework (4-07) applies to diagnosing neural network training failures throughout Modules 5 through 20.

**Folder:** `module_04_ml_theory_and_evaluation/`

**GPU Required:** No (classical ML / math foundations)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 4-01 | Evaluation Metrics Deep Dive | Precision, recall, F1 (micro/macro/weighted); ROC curve and AUC; precision-recall curves (preferred for imbalanced data); confusion matrix analysis; regression metrics (MSE, MAE, R², MAPE); calibration curves | `sklearn` Digits, California Housing | ~3 min |
| 4-02 | Cross-Validation & Hyperparameter Tuning | k-fold and stratified k-fold; leave-one-out; nested CV for unbiased evaluation; grid search, random search, and Bayesian optimization concepts; learning curves (bias vs variance diagnosis) | `sklearn` Digits | ~5 min |
| 4-03 | Feature Engineering & Pipelines | Numerical scaling (standard, min-max, robust); categorical encoding strategies; sklearn Pipeline and ColumnTransformer for reproducible preprocessing; feature selection (mutual information, L1) | `sklearn` California Housing | ~3 min |
| 4-04 | Data Augmentation & Color Spaces | Geometric transforms (flips, rotations, random crops, affine); color jittering; RGB/HSV/LAB color spaces and normalization strategies; Mixup and CutMix; augmentation policies (AutoAugment concepts) | CIFAR-10 | ~5 min |
| 4-05 | Handling Imbalanced Data | Class imbalance impact on metrics; oversampling (SMOTE concepts) and undersampling from scratch; class weights in loss functions; threshold tuning via precision-recall curves; stratified splitting | `sklearn.make_classification` (imbalanced) | ~3 min |
| 4-06 | Learning Theory — VC Dimension, PAC Learning & Sample Complexity | Empirical risk minimization (ERM); uniform convergence and Hoeffding's inequality; VC dimension and shattering — compute VC for linear classifiers; sample complexity bounds; PAC learning framework; why these matter for understanding generalization guarantees | `sklearn.make_classification` | ~3 min |
| 4-07 | Bias-Variance Decomposition & ML Debugging | Formal bias-variance-noise decomposition with derivation; learning curves for diagnosis (high bias vs high variance); error analysis framework — failure slicing by subgroup; CS229's debugging recipe: high bias → capacity, high variance → regularization/data; double descent phenomenon | `sklearn.make_classification`, `sklearn.make_moons` | ~5 min |
| 4-08 | Convex Optimization Foundations | Convex sets and functions; Lagrangian duality and dual problem construction; KKT conditions; connection to SVM dual formulation (Module 2-05) and constrained optimization in RL; implement Lagrangian dual for simple SVM | Synthetic (optimization surfaces) | ~3 min |
| 4-09 | Calibration & Uncertainty Quantification | Reliability diagrams and Expected Calibration Error (ECE); temperature scaling from scratch; Platt scaling; isotonic regression for calibration; conformal prediction basics — distribution-free prediction intervals; connection to LLM confidence scoring | `sklearn` Digits, `sklearn.make_classification` | ~3 min |
| 4-10 | Gaussian Processes & Bayesian Optimization | Kernel functions as covariance; GP prior and posterior from scratch; prediction with uncertainty bands; GP classification concepts; Bayesian optimization — acquisition functions (Expected Improvement, UCB); connection to hyperparameter tuning (4-02) and kernel methods (3-08) | Synthetic (function optimization) | ~5 min |

---

## Topic Details

### 4-01: Evaluation Metrics Deep Dive
Students implement the full suite of classification and regression evaluation metrics from scratch: precision, recall, F1 score (with micro, macro, and weighted averaging), ROC curves with AUC computation, precision-recall curves, confusion matrix analysis, and regression metrics including MSE, MAE, R-squared, and MAPE. The notebook emphasizes that metric choice depends on the problem -- precision-recall curves are preferred over ROC for imbalanced datasets, and calibration curves reveal whether predicted probabilities are trustworthy. These metrics are used in every subsequent evaluation throughout the course, from comparing classical algorithms (Module 2) to evaluating neural network classifiers (Modules 5-6) to assessing language model quality (Modules 10, 17). This topic establishes the evaluation vocabulary that Module 4 builds upon.

### 4-02: Cross-Validation & Hyperparameter Tuning
This topic implements k-fold cross-validation, stratified k-fold, leave-one-out CV, and the critical nested CV protocol that prevents optimistic bias when tuning hyperparameters. Students build grid search and random search from scratch, learn why random search is often more efficient than grid search in high-dimensional hyperparameter spaces, and get introduced to Bayesian optimization concepts (formalized in 4-10). Learning curves are implemented as the primary diagnostic tool for distinguishing high-bias from high-variance models. Cross-validation is the backbone of honest model evaluation and connects forward to the bias-variance analysis in 4-07, and the hyperparameter tuning concepts are formalized as Bayesian optimization in 4-10. Every model comparison in the deep learning modules relies on the validation methodology established here.

### 4-03: Feature Engineering & Pipelines
Students implement numerical scaling strategies (standard, min-max, robust scalers), categorical encoding methods, and feature selection techniques (mutual information from Module 1-08, L1 regularization from Module 2-02), then compose them into reproducible preprocessing pipelines using sklearn's Pipeline and ColumnTransformer. The notebook emphasizes why fitting transformers on training data only (not validation/test) prevents data leakage -- a subtle but critical point. Feature engineering remains essential even in the deep learning era for tabular data problems, and the Pipeline abstraction introduced here provides the template for the more complex data processing pipelines in Modules 7 (text preprocessing), 9 (image pipelines), and 20 (production deployment). The mutual information-based feature selection connects back to information theory (1-08).

### 4-04: Data Augmentation & Color Spaces
This topic implements geometric transforms (flips, rotations, random crops, affine warps), color jittering, and RGB/HSV/LAB color space conversions from scratch, then introduces advanced augmentation strategies including Mixup (linear interpolation of training pairs) and CutMix (spatial mixing of image patches) on CIFAR-10. Students learn that augmentation is a form of regularization that injects domain-informed invariances, and explore AutoAugment concepts for automated policy search. Data augmentation is arguably the single most impactful technique for improving image classification performance and is used extensively in the CNN module (Module 6), advanced computer vision (Module 9), and multimodal learning (Module 12). The color space knowledge also matters for medical imaging and satellite imagery applications in Module 19.

### 4-05: Handling Imbalanced Data
Students implement oversampling (SMOTE concepts) and undersampling from scratch, explore class-weighted loss functions, and practice threshold tuning via precision-recall curves to handle the pervasive problem of class imbalance. The notebook demonstrates how standard accuracy is misleading on imbalanced datasets and shows that stratified splitting (introduced in 4-02) is necessary to maintain class ratios in train/val/test sets. Imbalanced data handling connects to focal loss (Module 5-04), which was specifically designed for extreme class imbalance in object detection, and to anomaly detection (Module 3-07), which is inherently an imbalanced problem. These techniques are also critical for real-world applications in fraud detection, medical diagnosis, and rare event prediction (Module 19).

### 4-06: Learning Theory -- VC Dimension, PAC Learning & Sample Complexity
This topic formalizes the question "why do ML models generalize?" through the frameworks of empirical risk minimization (ERM), uniform convergence bounds (Hoeffding's inequality), VC dimension, and PAC learning. Students compute the VC dimension of linear classifiers by constructing shattering examples, derive sample complexity bounds that relate model capacity to the amount of data needed for generalization, and implement experiments that empirically validate these theoretical predictions. Understanding VC dimension explains why more complex models (higher VC dimension) need more data, providing the theoretical backbone for the bias-variance tradeoff formalized in 4-07. These learning theory concepts also illuminate why deep neural networks' generalization remains theoretically puzzling -- the double descent phenomenon in 4-07 challenges classical VC-dimension intuitions.

### 4-07: Bias-Variance Decomposition & ML Debugging
Students derive the formal bias-variance-noise decomposition mathematically and then implement experiments that measure each component empirically, using learning curves as the primary diagnostic tool. The notebook presents CS229's systematic debugging recipe: high bias suggests increasing model capacity (more features, larger networks), while high variance suggests adding regularization or more data. Students implement error analysis by slicing failures across subgroups to identify systematic weaknesses. The double descent phenomenon is demonstrated, showing that modern overparameterized models (deep networks) can exhibit a second descent in test error beyond the classical interpolation threshold. This debugging framework is applied throughout Modules 5-20 whenever training fails -- it provides the systematic approach to diagnosing and fixing neural network training problems.

### 4-08: Convex Optimization Foundations
This topic covers convex sets, convex functions, Lagrangian duality (constructing dual problems from primal constrained optimization), and the Karush-Kuhn-Tucker (KKT) conditions that characterize optimal solutions. Students implement the Lagrangian dual for a simple SVM to connect back to the dual formulation introduced in Module 2-05, seeing how duality enables the kernel trick. The KKT conditions formalize the complementary slackness that makes SVMs sparse (only support vectors have non-zero dual variables). Convex optimization theory connects forward to constrained optimization in reinforcement learning (Module 14), where policy optimization must satisfy constraints, and to the trust region methods used in PPO. This topic also provides the theoretical grounding for understanding why convex loss functions (like cross-entropy) are preferred in neural network training.

### 4-09: Calibration & Uncertainty Quantification
Students implement reliability diagrams, Expected Calibration Error (ECE), temperature scaling, Platt scaling, and isotonic regression for model calibration, ensuring that predicted probabilities reflect true likelihoods. The notebook introduces conformal prediction as a distribution-free framework for constructing prediction intervals with guaranteed coverage. Calibration is critical for any application where predicted probabilities inform decisions -- medical diagnosis, autonomous driving, financial risk -- and becomes especially important for LLM confidence scoring in Modules 17-18, where knowing when a language model is uncertain enables better retrieval-augmented generation (Module 18) and safer deployment. Temperature scaling, implemented here from scratch, is the same technique used to control LLM generation randomness.

### 4-10: Gaussian Processes & Bayesian Optimization
This topic implements Gaussian processes from scratch: specifying kernel functions as covariance, computing the GP prior and posterior, generating predictions with uncertainty bands, and introducing GP classification. Students then build Bayesian optimization on top of GPs, implementing acquisition functions (Expected Improvement, Upper Confidence Bound) to efficiently search for optimal hyperparameters. The GP framework connects back to kernel methods (3-08) -- the kernel function determines the GP's inductive bias -- and formalizes the Bayesian optimization concepts previewed in 4-02. Gaussian processes are also the conceptual precursor to neural processes and function-space views of neural networks, and the uncertainty quantification they provide complements the calibration methods in 4-09. Bayesian optimization is used in practice for expensive-to-evaluate neural architecture search and hyperparameter tuning.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 04-01 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 04-02 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 04-03 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 04-04 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 04-05 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 04-06 | B — Theory | `TEMPLATE_THEORY.ipynb` |
| 04-07 | B — Theory | `TEMPLATE_THEORY.ipynb` |
| 04-08 | B — Theory | `TEMPLATE_THEORY.ipynb` |
| 04-09 | B — Theory | `TEMPLATE_THEORY.ipynb` |
| 04-10 | B — Theory | `TEMPLATE_THEORY.ipynb` |

---

## Module-Specific Packages

- `scipy` — optimization (4-08), statistical tests

---

## Datasets

- `sklearn` Digits (4-01, 4-09)
- `sklearn` California Housing (4-01, 4-03)
- `sklearn.make_classification` (4-02, 4-05, 4-06, 4-07, 4-09)
- CIFAR-10 (4-04)
- `sklearn.make_moons` (4-07)
- Synthetic (4-08 optimization surfaces, 4-10 function optimization)

---

## Prerequisites Chain

- **04-01:** Requires 2-01 through 2-10
- **04-02:** Requires 4-01
- **04-03:** Requires 2-01 through 2-10
- **04-04:** Requires 1-04, 1-05
- **04-05:** Requires 4-01, 4-02
- **04-06:** Requires 1-07, 2-01, 2-02
- **04-07:** Requires 4-06, 4-02
- **04-08:** Requires 1-09, 2-05
- **04-09:** Requires 4-01, 1-07
- **04-10:** Requires 1-07, 3-08, 4-02

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 4 — ML Theory and Evaluation
| Concept | Owner |
|---------|-------|
| Precision, recall, F1, ROC-AUC, confusion matrix | 4-01 |
| Cross-validation, hyperparameter tuning | 4-02 |
| Feature engineering, sklearn Pipeline | 4-03 |
| Data augmentation, color spaces | 4-04 |
| Imbalanced data handling | 4-05 |
| Learning theory (VC dimension, PAC, sample complexity) | 4-06 |
| Bias-variance decomposition, ML debugging | 4-07 |
| Convex optimization (Lagrangian duality, KKT) | 4-08 |
| Calibration, uncertainty quantification, conformal prediction | 4-09 |
| Gaussian processes, Bayesian optimization | 4-10 |

---

## Cross-Module Ownership Warnings

No special cross-module warnings for this module.

---

## Special Notes

- Expanded from 5 → 10 topics to cover CS229 formal theory. Theory topics (4-06 through 4-10) MUST be code-driven, not lecture notes.
- 4-09 (Calibration) is critical for LLM confidence scoring in later modules.
