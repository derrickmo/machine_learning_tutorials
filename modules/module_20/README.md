# Module 20 — MLOps & Production Deployment

## Introduction

Module 20 transitions students from building and training ML models to deploying, monitoring, and maintaining them in production — the discipline known as MLOps. While the preceding 19 modules focus on algorithms and architectures, this module addresses the operational reality that most ML projects fail not because the model is bad but because the surrounding infrastructure is missing: no experiment tracking, no automated testing, no drift detection, no deployment pipeline. Students will use MLflow for experiment tracking, export models via TorchScript and ONNX, serve them with FastAPI and Gradio, containerize with Docker, build CI/CD pipelines, implement statistical A/B testing, and design end-to-end ML system architectures. After completing this module, students will possess the full spectrum of skills needed to take an ML model from a Jupyter notebook to a production system with monitoring, testing, and automated retraining triggers. Module 20 is the capstone of the entire 200-topic curriculum, synthesizing technical knowledge from every prior module into production-ready engineering practices.

**Folder:** `module_20_mlops_and_production/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 20-01 | ML Strategy & Error Analysis | Project scoping and feasibility assessment; when NOT to use ML; error analysis methodology (confusion matrix deep dive, failure slicing by subgroup); baseline establishment; iterative improvement workflow | `sklearn` Digits | ~3 min |
| 20-02 | Experiment Tracking with MLflow | Logging hyperparameters, metrics, and artifacts; comparing runs side-by-side; model registry and versioning; MLflow UI walkthrough; organizing experiments by project; reproducibility checklist | FashionMNIST | ~5 min |
| 20-03 | Model Export & Optimization | TorchScript export for deployment; ONNX export and ONNX Runtime inference; dynamic vs static quantization for serving; model size and latency benchmarking; choosing the right export format for target environment | FashionMNIST | ~5 min |
| 20-04 | Model Serving with FastAPI & Gradio | FastAPI prediction endpoint with Pydantic request/response schemas; request batching; Gradio demo UI for rapid prototyping; preprocessing in production (identical transforms as training); health checks and error handling | FashionMNIST | ~5 min |
| 20-05 | Docker Containerization for ML | Dockerfile for ML projects (base image selection, dependency pinning, model copying); multi-stage builds for smaller images; GPU support (nvidia-docker / CUDA base images); reproducible environments; docker-compose for multi-service | FashionMNIST | ~5 min |
| 20-06 | Data Drift & Model Monitoring | Covariate shift and concept drift detection; Kolmogorov-Smirnov test and PSI (Population Stability Index); feature distribution monitoring; performance degradation alerts; retraining trigger strategies; LLM-specific monitoring — prompt/response quality tracking, cost-per-query dashboards, latency monitoring for agent systems | CIFAR-10, Synthetic drift data | ~5 min |
| 20-07 | CI/CD for ML Pipelines | GitHub Actions for ML workflows; automated testing on push; model validation gates (performance threshold checks before deployment); artifact management; deployment automation concepts | FashionMNIST | ~5 min |
| 20-08 | ML Testing, Data Validation & Project Standards | Unit tests for ML code (data loading, preprocessing, model forward pass shapes); data schema validation (Great Expectations concepts); behavioral testing (CheckList methodology); smoke tests for deployed models; repository layout conventions; configuration management (YAML configs, Hydra concepts); Python logging best practices; reproducibility checklist (seeds, dependency versions, data hashes) | FashionMNIST | ~5 min |
| 20-09 | Online Experimentation & A/B Testing | A/B testing framework from scratch; statistical significance testing (t-test, bootstrap confidence intervals); sample size estimation and power analysis; canary deployments and traffic splitting; sequential testing for early stopping; multi-armed bandits for adaptive experiments (connection to 14-04); online vs offline metric gaps | Synthetic experiment data | ~5 min |
| 20-10 | ML System Design Patterns | End-to-end ML system architecture; training vs serving pipelines; feature stores; online vs batch prediction; A/B testing framework; common interview design questions: recommendation system, ad ranking, fraud detection, content moderation | CIFAR-10 | ~10 min |

---

## Topic Details

### 20-01: ML Strategy & Error Analysis
Students will learn to scope ML projects before writing any code — assessing feasibility, estimating potential impact, and determining when NOT to use ML (when simple heuristics or rules suffice). The notebook implements a structured error analysis methodology: starting from the confusion matrix, slicing failures by subgroup (data source, feature range, demographic), and identifying the highest-impact error categories to address. Students will establish baselines (random, majority class, simple heuristic) and implement an iterative improvement workflow that prioritizes data collection, feature engineering, or model changes based on error analysis results. This strategic thinking framework is essential for every subsequent topic in Module 20 and reflects how experienced ML engineers spend more time on problem framing than on model architecture.

### 20-02: Experiment Tracking with MLflow
This notebook introduces MLflow as the experiment tracking tool for organizing and comparing ML experiments systematically. Students will log hyperparameters, training metrics (loss curves, accuracy), and artifacts (model checkpoints, plots) for each training run, then use the MLflow UI to compare runs side-by-side and identify the best configuration. The notebook covers the model registry for versioning production models (staging, production, archived stages), organizing experiments by project, and building a reproducibility checklist that ensures any experiment can be exactly reproduced. Experiment tracking is the single most impactful MLOps practice for teams and is used throughout 20-03 through 20-10 as the backbone for recording all deployment experiments.

### 20-03: Model Export & Optimization
Students will export trained PyTorch models using both TorchScript (torch.jit.trace and torch.jit.script) and ONNX (torch.onnx.export), understanding the tradeoffs between each format — TorchScript for PyTorch-native deployment, ONNX for cross-framework inference with ONNX Runtime. The notebook implements dynamic and static quantization for serving (distinct from the training-focused quantization in 17-06 — here the focus is deployment-ready quantization with latency benchmarking). Students will measure model size, inference latency, and accuracy after export and quantization, and learn to choose the right format for target environments (cloud GPU, CPU server, edge device). This topic is a prerequisite for 20-04 (serving) and 20-05 (containerization).

### 20-04: Model Serving with FastAPI & Gradio
This notebook builds a production-ready model serving system using FastAPI for REST API endpoints and Gradio for rapid prototyping UIs. Students will implement a prediction endpoint with Pydantic request/response schemas for type-safe API contracts, add request batching to improve GPU utilization under concurrent load, and ensure that preprocessing in production is identical to training (a common source of train-serving skew). Health check endpoints and structured error handling make the service production-ready. Gradio provides an instant demo UI for stakeholder presentations. The serving infrastructure built here is containerized in 20-05 and tested in 20-08.

### 20-05: Docker Containerization for ML
Students will write Dockerfiles for ML projects, covering base image selection (Python slim vs CUDA for GPU), dependency pinning with exact versions, multi-stage builds to minimize image size, and proper model copying into the container. The notebook implements GPU support using nvidia-docker / CUDA base images and builds a docker-compose configuration for multi-service deployments (model server + monitoring). Students will learn how containers ensure reproducible environments across development, CI/CD, and production. Containerization is the deployment backbone for the CI/CD pipeline in 20-07 and represents the industry-standard approach to shipping ML models.

### 20-06: Data Drift & Model Monitoring
This notebook implements statistical methods for detecting when production data diverges from training data, causing model performance to degrade. Students will build covariate shift detection using the Kolmogorov-Smirnov test and Population Stability Index (PSI), implement feature distribution monitoring dashboards, and set up performance degradation alerts with configurable thresholds. Retraining trigger strategies — periodic, performance-based, and drift-based — are compared. The notebook also covers LLM-specific monitoring patterns: prompt/response quality tracking, cost-per-query dashboards, and latency monitoring for agent systems (connecting back to 18-09). Monitoring is what separates a deployed model from a production model, and this topic ensures students build systems that remain reliable over time.

### 20-07: CI/CD for ML Pipelines
Students will build a complete CI/CD pipeline for ML using GitHub Actions, implementing automated testing on every push, model validation gates that check whether a newly trained model meets performance thresholds before allowing deployment, and artifact management for model versioning. The notebook covers the unique challenges of CI/CD for ML — data dependencies, training time constraints, and non-determinism — and implements practical solutions including cached datasets, lightweight smoke tests, and performance regression checks. Deployment automation concepts (blue-green deployment, canary releases) connect forward to the A/B testing framework in 20-09.

### 20-08: ML Testing, Data Validation & Project Standards
This notebook builds a comprehensive testing and quality framework for ML projects. Students will write unit tests for data loading, preprocessing transforms, and model forward pass shape consistency using pytest. Data schema validation (Great Expectations-style) catches data quality issues before they reach the model. Behavioral testing using the CheckList methodology (minimum functionality, invariance, directional expectations) validates model behavior beyond aggregate metrics. The notebook also covers project standards: repository layout conventions, configuration management with YAML and Hydra concepts, Python logging best practices, and a reproducibility checklist covering seeds, dependency versions, and data hashes. These practices transform individual notebooks into maintainable, team-ready ML projects.

### 20-09: Online Experimentation & A/B Testing
Students will build a complete A/B testing framework from scratch, implementing statistical significance testing (two-sample t-test and bootstrap confidence intervals), sample size estimation using power analysis, and sequential testing for early stopping of underperforming experiments. The notebook covers canary deployments (routing a small percentage of traffic to the new model before full rollout) and traffic splitting mechanisms. Multi-armed bandit approaches for adaptive experimentation connect back to the exploration-exploitation concepts from Module 14 (14-04). Students will analyze the common gap between offline metrics (accuracy on held-out data) and online metrics (user engagement, revenue), understanding why a model that improves offline accuracy can hurt online performance.

### 20-10: ML System Design Patterns
The capstone notebook synthesizes all Module 20 topics into end-to-end ML system architectures, covering the patterns that appear in production systems and system design interviews. Students will design training pipelines (data ingestion, feature engineering, model training, validation, registry) and serving pipelines (feature stores, online vs batch prediction, model serving, monitoring) as coherent systems. The notebook walks through common ML system design questions — recommendation system, ad click-through rate prediction, fraud detection, and content moderation — showing how to apply the right combination of retrieval, ranking, serving, and monitoring patterns. This topic represents the culmination of the entire 200-topic curriculum, demonstrating that production ML requires not just algorithmic knowledge but systems thinking across data, modeling, serving, and operations.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 20-01 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 20-02 | D — Tool/Library | `TEMPLATE_TOOL.ipynb` |
| 20-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 20-04 | D — Tool/Library | `TEMPLATE_TOOL.ipynb` |
| 20-05 | D — Tool/Library | `TEMPLATE_TOOL.ipynb` |
| 20-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 20-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 20-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 20-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 20-10 | E — Capstone/Integration | `TEMPLATE_CAPSTONE.ipynb` |

---

## Module-Specific Packages

- `mlflow` — experiment tracking (20-02)
- `gradio` — model demo UI (20-04)
- `fastapi` — model serving (20-04)
- `docker` — containerization (20-05)
- `pytest` — ML testing (20-08)

---

## Datasets

- FashionMNIST (deployment demos, testing)
- CIFAR-10 (monitoring, system design)
- `sklearn` Digits (error analysis)
- Synthetic drift data (20-06)
- Synthetic experiment data (A/B testing)

---

## Prerequisites Chain

- **20-01:** Requires 4-01, 4-07
- **20-02:** Requires All prior training notebooks
- **20-03:** Requires 15-05, 5-07
- **20-04:** Requires 20-03, 5-07
- **20-05:** Requires 20-04
- **20-06:** Requires 4-01, 20-02
- **20-07:** Requires 20-02, 20-05
- **20-08:** Requires 20-04
- **20-09:** Requires 4-01, 1-07, 14-04
- **20-10:** Requires 20-01 through 20-09

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 20 — MLOps and Production Deployment
| Concept | Owner |
|---------|-------|
| ML strategy, error analysis | 20-01 |
| Experiment tracking (MLflow) | 20-02 |
| Model export (TorchScript, ONNX) | 20-03 |
| Model serving (FastAPI, Gradio) | 20-04 |
| Docker containerization | 20-05 |
| Data drift, model monitoring, LLM monitoring | 20-06 |
| CI/CD for ML | 20-07 |
| ML testing, data validation, project standards | 20-08 |
| Online experimentation, A/B testing, statistical significance | 20-09 |
| ML system design patterns | 20-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ CheckList behavioral testing (20-08): Module 10-08 owns the CheckList methodology for NLP behavioral testing. 20-08 references CheckList as one testing approach among many but must NOT re-teach the methodology — focus on unit tests, data validation (Great Expectations), smoke tests, and project standards.
- ⚠️ Error analysis (20-01) vs ML debugging (4-07): Module 4-07 owns the bias-variance decomposition and CS229 debugging recipe. 20-01 covers strategic error analysis for production systems (failure slicing by subgroup, iterative improvement workflow). Distinct scopes: 4-07 = theoretical debugging framework, 20-01 = production ML strategy.

---

## Special Notes

- LLM-specific monitoring added to 20-06 (prompt/response quality, cost-per-query dashboards).
- Original 20-08 (ML Testing) and 20-09 (Project Structure) merged into 20-08 as both address quality assurance and project standards.
- Freed slot used for 20-09 (Online Experimentation & A/B Testing) — critical production ML topic.
- 20-01 changed from Algorithm to Evaluation category — it's a strategic methodology notebook, not an algorithm implementation.
