# Machine Learning & Deep Learning — From Scratch to Production

**20 modules · 200 hands-on notebooks · Every algorithm built from first principles**

---

## About This Project

This is a personal learning project born from a desire to truly understand machine learning — not just use APIs, but know what happens inside the box. I wanted to consolidate everything I've learned into a single, cohesive self-study guide that any motivated and aspiring machine learning engineer can follow from start to finish.

The result is a 200-notebook curriculum that starts with NumPy tensor operations and linear algebra, progresses through classical ML and deep learning, and ends with LLM fine-tuning, agentic systems, and production deployment. Every core algorithm is implemented from scratch before comparing against library implementations. No black boxes.

This is also a collaboration with [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) (Anthropic), which was used to develop the notebook contents. The curriculum design, topic selection, pedagogical ordering, and quality standards are human-driven; the code generation, validation, and iterative refinement are a joint effort between human judgment and AI capabilities. Think of it as a case study in what's possible when you pair deep domain knowledge with AI-assisted development at scale.

### Who This Is For

- Engineers transitioning into ML/AI roles who want first-principles understanding, not just framework knowledge
- Self-taught practitioners filling gaps between "I can call model.fit()" and "I can explain why that works"
- Anyone preparing for ML engineering interviews at companies that test fundamentals (Google, Meta, OpenAI, Anthropic, etc.)

### What Makes This Different

- **From scratch first, library second.** Every notebook builds the core algorithm using raw NumPy or PyTorch tensor ops, then compares against the library version. You understand the internals before trusting the abstraction.
- **200 self-contained notebooks.** Each runs independently with "Restart & Run All." No hidden dependencies, no shared utility files, no setup beyond installing the packages.
- **Consistent quality standards.** Every function has Google-style docstrings and type hints. Every notebook follows the same structure. Every training loop plots learning curves. Code quality is enforced by automated validation scripts.

---

## Curriculum Overview

| # | Module | Topics | Category |
|---|--------|--------|----------|
| 1 | Mathematical & Programming Foundations | 10 | Foundations |
| 2 | Supervised Learning | 10 | Classical ML |
| 3 | Unsupervised & Statistical Learning | 10 | Classical ML |
| 4 | ML Theory & Evaluation | 10 | Classical ML |
| 5 | Neural Network Foundations | 10 | DL Core |
| 6 | Convolutional Neural Networks | 10 | DL Core |
| 7 | Recurrent Networks & NLP Foundations | 10 | DL Core |
| 8 | Transformers — Architecture to Attention | 10 | DL Core |
| 9 | Advanced Computer Vision | 10 | Advanced DL |
| 10 | Advanced NLP — Pretrained Language Models | 10 | Advanced DL |
| 11 | Generative Deep Learning | 10 | Advanced DL |
| 12 | Multimodal & Cross-Modal Learning | 10 | Advanced DL |
| 13 | Fine-Tuning & Alignment | 10 | Adaptation |
| 14 | Reinforcement Learning | 10 | Theory |
| 15 | Advanced PyTorch Internals | 10 | Engineering |
| 16 | Training Optimization & Distributed Systems | 10 | Engineering |
| 17 | Large Language Models — Systems & Scaling | 10 | LLMs & Systems |
| 18 | RAG & Agentic AI Systems | 10 | LLMs & Systems |
| 19 | ML Applications & Domain Problems | 10 | Applications |
| 20 | MLOps & Production Deployment | 10 | Production |

**Total: 200 notebooks · ~400 hours · ~13 months at one notebook every 2 days**

---

## Module Details

<details>
<summary><strong>Module 1 — Mathematical & Programming Foundations</strong></summary>

Build fluency with Python's numerical stack and establish the mathematical prerequisites for ML/DL. No prior math courses assumed beyond high school.

| # | Topic |
|---|-------|
| 1-01 | Python, NumPy & Tensor Speed |
| 1-02 | Advanced NumPy & PyTorch Operations |
| 1-03 | Pandas for Tabular Data |
| 1-04 | Visualization with Matplotlib |
| 1-05 | Data Loading with PyTorch |
| 1-06 | Linear Algebra for Machine Learning |
| 1-07 | Probability & Statistics for ML |
| 1-08 | Information Theory for ML |
| 1-09 | Calculus & Optimization Foundations |
| 1-10 | Computational Thinking & Complexity |
</details>

<details>
<summary><strong>Module 2 — Supervised Learning</strong></summary>

Master core supervised algorithms from scratch using sklearn and NumPy. Every algorithm is implemented before being compared against library versions.

| # | Topic |
|---|-------|
| 2-01 | Linear Regression |
| 2-02 | Logistic Regression & Binary Classification |
| 2-03 | Decision Trees & Random Forests |
| 2-04 | Gradient Boosting & AdaBoost |
| 2-05 | Support Vector Machines |
| 2-06 | Generalized Linear Models & Exponential Family |
| 2-07 | k-Nearest Neighbors |
| 2-08 | Naive Bayes for Text Classification |
| 2-09 | Stacking & Voting Ensembles |
| 2-10 | Model Comparison & Algorithm Selection |
</details>

<details>
<summary><strong>Module 3 — Unsupervised & Statistical Learning</strong></summary>

Discover structure in unlabeled data. EM connects to VAEs, kernel methods connect to attention, matrix factorization connects to recommenders and LoRA.

| # | Topic |
|---|-------|
| 3-01 | K-Means Clustering |
| 3-02 | Hierarchical & Density-Based Clustering |
| 3-03 | Principal Component Analysis |
| 3-04 | t-SNE, UMAP & Manifold Learning |
| 3-05 | Independent Component Analysis |
| 3-06 | Gaussian Mixture Models & EM Algorithm |
| 3-07 | Anomaly Detection |
| 3-08 | Kernel Methods & Feature Maps |
| 3-09 | Matrix Factorization & Decomposition |
| 3-10 | Bayesian Inference & Probabilistic Thinking |
</details>

<details>
<summary><strong>Module 4 — ML Theory & Evaluation</strong></summary>

Production-quality evaluation pipelines and formal theory. Covers VC dimension, PAC learning, convex optimization, calibration, and Gaussian processes.

| # | Topic |
|---|-------|
| 4-01 | Evaluation Metrics Deep Dive |
| 4-02 | Cross-Validation & Hyperparameter Tuning |
| 4-03 | Feature Engineering & Pipelines |
| 4-04 | Data Augmentation & Color Spaces |
| 4-05 | Handling Imbalanced Data |
| 4-06 | Learning Theory — VC Dimension, PAC Learning |
| 4-07 | Bias-Variance Decomposition & ML Debugging |
| 4-08 | Convex Optimization Foundations |
| 4-09 | Calibration & Uncertainty Quantification |
| 4-10 | Gaussian Processes & Bayesian Optimization |
</details>

<details>
<summary><strong>Module 5 — Neural Network Foundations</strong></summary>

Build neural networks from first principles in NumPy, implement backprop by hand, master the optimization toolkit, and transition to PyTorch.

| # | Topic |
|---|-------|
| 5-01 | Neural Network End-to-End Walkthrough |
| 5-02 | Perceptron & Multi-Layer Architecture |
| 5-03 | Activation Functions Deep Dive |
| 5-04 | Loss Functions Deep Dive |
| 5-05 | Forward Pass & Computational Graphs |
| 5-06 | Backpropagation from Scratch |
| 5-07 | PyTorch Fundamentals — Autograd & nn.Module |
| 5-08 | Optimizers — SGD, Momentum & Nesterov |
| 5-09 | Advanced Optimizers & Learning Rate Scheduling |
| 5-10 | Regularization Techniques |
</details>

<details>
<summary><strong>Module 6 — Convolutional Neural Networks</strong></summary>

From convolution mechanics through landmark architectures to segmentation, adversarial robustness, and non-image domains.

| # | Topic |
|---|-------|
| 6-01 | Fully Connected Networks for Images |
| 6-02 | Convolution from Scratch |
| 6-03 | CNN Architectures — LeNet to ResNet |
| 6-04 | Transfer Learning & Fine-Tuning |
| 6-05 | U-Net & Encoder-Decoder Architecture |
| 6-06 | Depthwise Separable Convolutions & Efficient Architectures |
| 6-07 | Semantic & Instance Segmentation |
| 6-08 | Neural Style Transfer |
| 6-09 | 1D & 3D Convolutions |
| 6-10 | Adversarial Examples & Robustness |
</details>

<details>
<summary><strong>Module 7 — Recurrent Networks & NLP Foundations</strong></summary>

Tokenization, word vectors, RNNs, LSTMs, attention, parsing, and CRFs — the building blocks that lead to transformers.

| # | Topic |
|---|-------|
| 7-01 | Tokenization — BPE, WordPiece & SentencePiece |
| 7-02 | Word Vectors — Word2Vec, GloVe & FastText |
| 7-03 | Recurrent Neural Networks from Scratch |
| 7-04 | LSTMs & GRUs |
| 7-05 | Sequence-to-Sequence with Attention |
| 7-06 | Text Generation & Decoding Strategies |
| 7-07 | Classical Language Models & Perplexity |
| 7-08 | Dependency Parsing |
| 7-09 | Sequence Labeling & CRFs |
| 7-10 | Contextual Embeddings — ELMo |
</details>

<details>
<summary><strong>Module 8 — Transformers — Architecture to Attention</strong></summary>

The most critical module. Build every transformer component from scratch: attention, RoPE, Flash Attention, KV cache — the backbone of everything in Modules 9–18.

| # | Topic |
|---|-------|
| 8-01 | Self-Attention Mechanism |
| 8-02 | Multi-Head Attention |
| 8-03 | Positional Encoding — Sinusoidal & Learned |
| 8-04 | The Transformer Block |
| 8-05 | Full Transformer — Encoder-Decoder |
| 8-06 | RMSNorm, SwiGLU & Modern Building Blocks |
| 8-07 | Grouped-Query & Multi-Query Attention |
| 8-08 | Rotary Position Embeddings (RoPE) |
| 8-09 | Flash Attention — Algorithm & Concepts |
| 8-10 | KV Cache & Autoregressive Inference |
</details>

<details>
<summary><strong>Module 9 — Advanced Computer Vision</strong></summary>

Grad-CAM, object detection, YOLO, Vision Transformers, self-supervised learning, video understanding, OCR, and a CIFAR-100 training deep dive.

| # | Topic |
|---|-------|
| 9-01 | Grad-CAM & Saliency Maps |
| 9-02 | Object Detection Fundamentals |
| 9-03 | YOLO Detection |
| 9-04 | MediaPipe Real-Time Vision |
| 9-05 | Vision Transformers (ViT) |
| 9-06 | Self-Supervised Vision — DINO & MAE |
| 9-07 | Image Retrieval & Visual Similarity Search |
| 9-08 | Video Understanding |
| 9-09 | OCR & Document AI |
| 9-10 | CNN Training Deep Dive — CIFAR-100 |
</details>

<details>
<summary><strong>Module 10 — Advanced NLP — Pretrained Language Models</strong></summary>

GPT, BERT, fine-tuning, NER, NLI, QA, chain-of-thought, and mechanistic interpretability.

| # | Topic |
|---|-------|
| 10-01 | GPT-Style Autoregressive Language Modeling |
| 10-02 | BERT-Style Masked Language Modeling |
| 10-03 | Encoder vs Decoder vs Encoder-Decoder |
| 10-04 | Pretrained Transformer Fine-Tuning |
| 10-05 | Named Entity Recognition |
| 10-06 | Natural Language Inference |
| 10-07 | Question Answering |
| 10-08 | NLP Evaluation |
| 10-09 | Chain-of-Thought & In-Context Learning |
| 10-10 | Mechanistic Interpretability |
</details>

<details>
<summary><strong>Module 11 — Generative Deep Learning</strong></summary>

Autoencoders, VAEs, GANs, diffusion models, normalizing flows, energy-based models, and autoregressive generation.

| # | Topic |
|---|-------|
| 11-01 | Autoencoders |
| 11-02 | Variational Autoencoders (VAEs) |
| 11-03 | GANs — DCGAN & WGAN |
| 11-04 | Conditional Generation |
| 11-05 | DDPM Diffusion from Scratch |
| 11-06 | Latent Diffusion Models |
| 11-07 | Diffusion Guidance & Evaluation |
| 11-08 | Normalizing Flows & Flow Matching |
| 11-09 | Energy-Based Models & Score Matching |
| 11-10 | Autoregressive Generative Models |
</details>

<details>
<summary><strong>Module 12 — Multimodal & Cross-Modal Learning</strong></summary>

CLIP, contrastive learning, vision-language models, VQA, audio representations, and speech pipelines.

| # | Topic |
|---|-------|
| 12-01 | CLIP — Contrastive Image-Text Pretraining |
| 12-02 | Zero-Shot & Few-Shot Classification |
| 12-03 | Siamese Networks & Triplet Loss |
| 12-04 | Contrastive Self-Supervised Learning — SimCLR & BYOL |
| 12-05 | Vision-Language Models & Image Captioning |
| 12-06 | Visual Question Answering |
| 12-07 | Multi-Modal Fusion Architectures |
| 12-08 | Audio & Speech Representations |
| 12-09 | STT & TTS Foundations |
| 12-10 | Multimodal Evaluation & Alignment Metrics |
</details>

<details>
<summary><strong>Module 13 — Fine-Tuning & Alignment</strong></summary>

LoRA, QLoRA, instruction tuning, reward modeling, RLHF, DPO, and efficient fine-tuning with Unsloth.

| # | Topic |
|---|-------|
| 13-01 | Full Fine-Tuning vs Feature Extraction |
| 13-02 | LoRA — Low-Rank Adaptation from Scratch |
| 13-03 | QLoRA — 4-Bit Quantization + LoRA |
| 13-04 | Adapter Methods Comparison |
| 13-05 | Prompt Tuning & Prefix Tuning |
| 13-06 | Instruction Tuning & SFT |
| 13-07 | Reward Modeling (Bradley-Terry) |
| 13-08 | RLHF with PPO |
| 13-09 | DPO, GRPO & Modern Alignment |
| 13-10 | Efficient Fine-Tuning with Unsloth |
</details>

<details>
<summary><strong>Module 14 — Reinforcement Learning</strong></summary>

MDPs, TD learning, Q-learning, bandits, DQN, policy gradients, actor-critic, model-based RL, offline RL, and imitation learning.

| # | Topic |
|---|-------|
| 14-01 | MDPs, Bellman Equations & Value/Policy Iteration |
| 14-02 | Monte Carlo & TD Learning |
| 14-03 | Q-Learning & SARSA |
| 14-04 | Exploration, Bandits & UCB |
| 14-05 | Deep Q-Networks (DQN) |
| 14-06 | Policy Gradient & REINFORCE |
| 14-07 | Actor-Critic, A2C & PPO |
| 14-08 | Model-Based RL & MCTS |
| 14-09 | Offline RL — CQL & IQL |
| 14-10 | Imitation Learning & Inverse RL |
</details>

<details>
<summary><strong>Module 15 — Advanced PyTorch Internals</strong></summary>

Custom autograd, advanced data pipelines, state_dict surgery, TorchScript, torch.fx, CUDA memory, distributed primitives, and profiling.

| # | Topic |
|---|-------|
| 15-01 | Custom Autograd Functions & Hooks |
| 15-02 | Advanced Data Pipelines |
| 15-03 | Advanced nn.Module Patterns |
| 15-04 | Custom Loss Functions & Gradient Utilities |
| 15-05 | TorchScript & JIT Compilation |
| 15-06 | torch.fx Graph Transformations |
| 15-07 | CUDA Memory Management |
| 15-08 | Distributed Communication Primitives |
| 15-09 | PyTorch Debugging & Profiling |
| 15-10 | Mini Training Framework |
</details>

<details>
<summary><strong>Module 16 — Training Optimization & Distributed Systems</strong></summary>

Mixed precision, torch.compile, gradient checkpointing, DDP, FSDP, and end-to-end optimized training.

| # | Topic |
|---|-------|
| 16-01 | Mixed Precision Training |
| 16-02 | torch.compile — Dynamo & Inductor |
| 16-03 | Gradient Checkpointing |
| 16-04 | Gradient Accumulation & Large Batch Training |
| 16-05 | Data Loading at Scale |
| 16-06 | Training Stability & NaN Recovery |
| 16-07 | Profiling & Bottleneck Analysis |
| 16-08 | Distributed Data Parallel (DDP) |
| 16-09 | FSDP, ZeRO & Model Parallelism |
| 16-10 | End-to-End Optimized Training Pipeline |
</details>

<details>
<summary><strong>Module 17 — Large Language Models — Systems & Scaling</strong></summary>

LLM architectures, scaling laws, MoE, quantization, efficient inference, long context, structured output, and evaluation.

| # | Topic |
|---|-------|
| 17-01 | LLM Architecture Patterns |
| 17-02 | Scaling Laws & Chinchilla |
| 17-03 | Mixture of Experts (MoE) |
| 17-04 | Training Data Pipelines |
| 17-05 | Knowledge Distillation |
| 17-06 | Quantization — PTQ, QAT, GPTQ, AWQ |
| 17-07 | Efficient Inference & Speculative Decoding |
| 17-08 | Long Context — RoPE Scaling & Sliding Window |
| 17-09 | Structured Output & Function Calling |
| 17-10 | LLM Evaluation & Benchmarks |
</details>

<details>
<summary><strong>Module 18 — RAG & Agentic AI Systems</strong></summary>

Embeddings, retrieval, RAG pipelines, agent loops, multi-agent orchestration, voice agents, and guardrails.

| # | Topic |
|---|-------|
| 18-01 | Embeddings & Vector Stores |
| 18-02 | Chunking, BM25 & Dense Retrieval |
| 18-03 | Advanced RAG — HyDE & Reranking |
| 18-04 | RAG Pipeline End-to-End |
| 18-05 | RAG Evaluation Metrics |
| 18-06 | Agent Loops, Tool Use & Planning |
| 18-07 | Multi-Agent Orchestration |
| 18-08 | Voice Agents — STT→LLM→TTS |
| 18-09 | Guardrails & Agent Evaluation |
| 18-10 | Domain-Specific AI Assistant (Capstone) |
</details>

<details>
<summary><strong>Module 19 — ML Applications & Domain Problems</strong></summary>

Recommenders, time series, search ranking, explainability, GNNs, audio classification, and tabular deep learning.

| # | Topic |
|---|-------|
| 19-01 | Recommender Systems — Collaborative Filtering |
| 19-02 | Neural Recommenders & Two-Tower Architecture |
| 19-03 | Time Series Forecasting |
| 19-04 | Search & Ranking Systems |
| 19-05 | SHAP & Model-Agnostic Explainability |
| 19-06 | Graph Neural Networks |
| 19-07 | Audio Classification |
| 19-08 | Semi-Supervised Learning |
| 19-09 | Multi-Task & Multi-Output Learning |
| 19-10 | Tabular Deep Learning |
</details>

<details>
<summary><strong>Module 20 — MLOps & Production Deployment</strong></summary>

ML strategy, experiment tracking, model export, serving, Docker, monitoring, CI/CD, testing, and system design.

| # | Topic |
|---|-------|
| 20-01 | ML Strategy & Error Analysis |
| 20-02 | Experiment Tracking with MLflow |
| 20-03 | Model Export — TorchScript & ONNX |
| 20-04 | Model Serving — FastAPI & Gradio |
| 20-05 | Docker Containerization |
| 20-06 | Data Drift & Model Monitoring |
| 20-07 | CI/CD for ML |
| 20-08 | ML Testing & Data Validation |
| 20-09 | ML Project Structure & Best Practices |
| 20-10 | ML System Design Patterns |
</details>

---

## Getting Started

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Git** for cloning the repository
- A CUDA-compatible **GPU** is recommended for Modules 5–20 but not required — every notebook works on CPU

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/machine-learning-tutorial.git
cd machine-learning-tutorial
```

### 2. Set Up the Environment

**Option A: conda (recommended)**

```bash
conda create -n ml-tutorial python=3.11 -y
conda activate ml-tutorial
pip install -r requirements.txt
```

**Option B: venv**

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Install PyTorch with GPU Support

If you have an NVIDIA GPU, install the CUDA version of PyTorch. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the command matching your system. For example:

```bash
# CUDA 12.4 (check yours with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If you're on CPU only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### 5. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

Navigate to any module folder and open the first notebook (e.g., `module_01_math_and_programming_foundations/01-01_python_numpy_tensor_speed.ipynb`).

---

## How to Use This Repository

### Notebook Structure

Every notebook follows a consistent 6-part structure:

| Part | What It Does |
|------|-------------|
| **Part 0 — Setup** | Imports, seeds, device setup, dataset loading, and a brief EDA so you understand the data before modeling |
| **Part 1 — From Scratch** | Mathematical explanation followed by a step-by-step implementation using only NumPy or raw PyTorch tensors |
| **Part 2 — Assembly** | Combine Part 1 components into a reusable `nn.Module` (DL) or class with `fit`/`predict` (classical ML) |
| **Part 3 — Training** | Full training loop on a real dataset, plus comparison against the library equivalent |
| **Part 4 — Evaluation** | Metrics, visualizations, error analysis, and ablation studies |
| **Part 5 — Summary** | Key takeaways and a link to the next notebook |

### Running a Notebook

Each notebook is fully self-contained. Open any notebook and run **Restart & Run All** — it will download any required datasets automatically to the `data/` directory.

```
# From the repo root, any notebook works independently:
jupyter notebook module_08_transformers/08-01_self_attention_mechanism.ipynb
```

### Datasets

All datasets download automatically on first run via PyTorch's built-in download mechanisms (torchvision, torchtext, torchaudio) or scikit-learn. They are saved to a shared `data/` directory at the repository root.

No manual downloads required. No external API keys are needed for any module. Every notebook is fully self-contained and runs independently on both local machines and Google Colab.

---

## Learning Paths

Not everyone needs all 200 notebooks. Choose the path that matches your goals:

### Fast Track — ML Engineer with LLM Focus

**~120 topics · ~8 months**

Modules 1–5 → 8 → 10 → 13 → 15–18 → 20

Covers the top skills in job postings: Python, PyTorch, transformers, LLMs, RAG, agents, and MLOps.

### Full Track

**200 topics · ~13 months at one notebook every 2 days**

All 20 modules in order. The complete curriculum.

### Specialization Tracks

After completing Modules 1–5 and 8 (the shared foundation), pick a specialization:

| Track | Modules | Target Role |
|-------|---------|-------------|
| **CV Specialist** | 6, 9, 11 (partial), 12 (partial) | Computer vision engineer |
| **NLP & LLM Specialist** | 7, 10, 13, 17, 18 | NLP / LLM engineer |
| **ML Infrastructure** | 15, 16, 17, 20 | ML platform engineer |
| **Research Engineer** | 11, 14, 16 (partial), 17 (partial) | Research / applied scientist |
| **AI Engineer** | 13 (partial), 18, 19, 20 | Full-stack AI engineer |
| **Multimodal & Voice** | 12, 18 (partial), 9 (partial) | Multimodal / voice AI |

---

## Repository Structure

```
machine-learning-tutorial/
│
├── module_01_math_and_programming_foundations/
│   ├── README.md                            # Module landing page
│   ├── 01-01_python_numpy_tensor_speed.ipynb
│   ├── 01-02_advanced_numpy_pytorch_ops.ipynb
│   └── ...
├── module_02_supervised_learning/
├── module_03_unsupervised_and_statistical_learning/
├── module_04_ml_theory_and_evaluation/
├── module_05_neural_network_foundations/
├── module_06_convolutional_neural_networks/
├── module_07_recurrent_networks_and_nlp/
├── module_08_transformers/
├── module_09_advanced_computer_vision/
├── module_10_advanced_nlp/
├── module_11_generative_deep_learning/
├── module_12_multimodal_and_cross_modal_learning/
├── module_13_fine_tuning_and_alignment/
├── module_14_reinforcement_learning/
├── module_15_advanced_pytorch_internals/
├── module_16_training_optimization_and_distributed/
├── module_17_large_language_models/
├── module_18_rag_and_agentic_systems/
├── module_19_ml_applications/
├── module_20_mlops_and_production/
│
├── data/                     # Downloaded datasets (gitignored)
├── docs/                     # Curriculum docs, rules, templates
│   ├── rules/                # Split rule files for notebook generation
│   ├── modules/              # Per-module context files (20 files)
│   └── templates/            # Notebook skeleton templates (6 categories)
├── scripts/                  # Validation, execution, and review scripts
│
├── CLAUDE.md                 # Instructions for Claude Code generation
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

---

## Required Packages

### Core (all modules)

```
numpy
pandas
matplotlib
scikit-learn
torch
torchvision
torchtext
torchaudio
tqdm
jupyter
```

### Module-Specific (installed as needed)

| Package | Module(s) | Install |
|---------|-----------|---------|
| `scipy` | 1, 3, 4, 17 | `pip install scipy` |
| `umap-learn` | 3 | `pip install umap-learn` |
| `ultralytics` | 9 | `pip install ultralytics` |
| `mediapipe` | 9 | `pip install mediapipe` |
| `faiss-cpu` | 9, 18 | `pip install faiss-cpu` |
| `sentencepiece` | 7 | `pip install sentencepiece` |
| `openai-whisper` | 12 | `pip install openai-whisper` |
| `unsloth` | 13 | See [unsloth docs](https://github.com/unslothai/unsloth) |
| `gymnasium` | 14 | `pip install gymnasium` |
| `mlflow` | 20 | `pip install mlflow` |
| `gradio` | 20 | `pip install gradio` |
| `fastapi` + `uvicorn` | 20 | `pip install fastapi uvicorn` |

Each notebook lists its specific imports in the first code cell. If a module-specific package is missing, the import cell will tell you what to install.

---

## Validation & Quality Scripts

The repository includes scripts to validate notebook quality:

```bash
# Validate structure, imports, docstrings, and type hints
python scripts/validate_notebook.py module_05_neural_network_foundations/05-01_nn_end_to_end.ipynb

# Execute a notebook end-to-end (verifies it runs without errors)
python scripts/execute_notebook.py module_05_neural_network_foundations/05-01_nn_end_to_end.ipynb

# Quality review — code metrics, checklist, improvement suggestions
python scripts/review_notebook.py module_05_neural_network_foundations/05-01_nn_end_to_end.ipynb

# Check dataset consistency across all notebooks
python scripts/check_dataset_reuse.py

# Validate all notebooks in a module
python scripts/validate_notebook.py --module 5

# Validate everything
python scripts/validate_notebook.py --all
```

---

## Job Market Alignment

Curriculum topics mapped against 7,000+ ML engineer job postings (2025–2026):

| Module | Job Market Signal | Criticality |
|--------|------------------|-------------|
| 1 | Python (72–75% of postings) | Universal |
| 2–4 | sklearn, evaluation, ML theory | Universal |
| 5 | Deep learning fundamentals (28%) | Core |
| 6–7 | CNN, RNN, NLP foundations | Core |
| **8** | **Transformers — backbone of modern AI** | **Critical** |
| 9 | ViT, detection, self-supervised | Specialist |
| 10 | GPT/BERT, NLI, QA, evaluation | Core |
| 11 | Generative AI, diffusion | Differentiating |
| 12 | CLIP, multimodal, voice | Emerging |
| **13** | **Fine-tuning (LoRA, RLHF) — #1 in-demand** | **Critical** |
| 14 | RL, RLHF connection | Specialist |
| **15** | **PyTorch depth (37–47% of postings)** | **Critical** |
| **16** | **Distributed training — hardest to hire** | **Critical** |
| **17** | **LLMs — fastest-growing specialization** | **Critical** |
| **18** | **RAG (65% of LLM postings), agents** | **Critical** |
| 19 | Recommenders, time series, ranking | Core |
| **20** | **MLOps — 70% of MLE daily work** | **Critical** |

---

## Acknowledgments

- **Stanford CS336** (Language Modeling from Scratch) — the backbone reference for transformer implementation and training
- **Stanford CS224N, CS229, CS230, CS234, CS236** — curriculum cross-references for NLP, classical ML, deep learning, RL, and generative models
- **Claude Code** (Anthropic) — AI-assisted notebook development, validation, and iterative refinement
- The PyTorch, scikit-learn, and Hugging Face open-source communities

---

## License

This project is for educational purposes. Individual notebook implementations are original work. Referenced papers, algorithms, and datasets retain their original licenses.

---

## Contributing

This is primarily a personal learning project, but if you find errors, have suggestions, or want to improve a notebook, feel free to open an issue or pull request.

---

*Built with curiosity, persistence, and a good AI collaborator.*
