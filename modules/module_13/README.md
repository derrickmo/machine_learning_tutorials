# Module 13 — Fine-Tuning & Alignment

## Introduction

Module 13 covers the complete spectrum of techniques for adapting pretrained models to specific tasks and aligning them with human preferences, from parameter-efficient fine-tuning methods like LoRA and QLoRA through the full RLHF pipeline including reward modeling, PPO, and DPO. This module is critically important because the modern ML workflow rarely trains from scratch -- instead, practitioners adapt foundation models through fine-tuning and alignment, making these techniques essential for anyone working with large language models or other pretrained systems. After completing this module, students will be able to implement LoRA from raw matrix decomposition, construct instruction-tuning datasets, train reward models on human preferences, and run complete RLHF and DPO alignment pipelines from scratch. Within the 20-module curriculum, Module 13 builds on the transformer architectures from Module 8 and the pretrained language models from Module 10, while connecting forward to reinforcement learning theory (Module 14 for general PPO), LLM systems at scale (Module 17), and production deployment with fine-tuned models (Module 20).

**Folder:** `module_13_fine_tuning_and_alignment/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 13-01 | Full Fine-Tuning vs Feature Extraction | Catastrophic forgetting — causes and measurement; learning rate strategies for fine-tuning (warmup, discriminative LRs); when full fine-tuning is worth the compute cost; comparison experiments | SST-2 | ~10 min |
| 13-02 | LoRA — Low-Rank Adaptation from Scratch | Low-rank decomposition (W + BA); mathematical derivation — why rank-r adaptation works; where to apply LoRA (attention projections, FFN); rank selection experiments; merge weights at inference; implement from scratch | SST-2 | ~10 min |
| 13-03 | QLoRA & Memory-Efficient Fine-Tuning | 4-bit NormalFloat quantization; double quantization; paged optimizers (offloading to CPU); memory profiling before/after; QLoRA training loop; comparison: LoRA vs QLoRA quality and speed | SST-2 | ~10 min |
| 13-04 | Adapter Methods Comparison | LoRA vs Prefix Tuning vs IA³ — benchmark all on the same task; parameter-efficiency metrics (trainable params / total params); task-specific selection guidance; adapter merging and stacking | SST-2 | ~15 min |
| 13-05 | Prompt Tuning & Prefix Tuning | Soft prompts — learnable continuous vectors prepended to input; prefix tuning — learnable key-value pairs in each attention layer; comparison with LoRA; when prompt tuning outperforms fine-tuning | SST-2 | ~8 min |
| 13-06 | Instruction Tuning & Dataset Construction | Alpaca, Dolly, OpenAssistant dataset formats; quality filtering strategies; data mixing and curriculum; instruction-response pair construction; supervised fine-tuning (SFT) training loop (CS336 A5 core deliverable) | Synthetic instruction data | ~12 min |
| 13-07 | Reward Modeling | Human preference data (chosen vs rejected pairs); Bradley-Terry model for pairwise preferences; reward model architecture (LM backbone + scalar value head); training from scratch; reward hacking and overoptimization risks | Synthetic preference pairs | ~10 min |
| 13-08 | RLHF — PPO for Language Models | PPO mechanics — clipped surrogate objective, KL penalty against reference model; value head and advantage estimation (GAE); RLHF training loop implementation; stability techniques; connection to general RL PPO in Module 14 | Synthetic preference data | ~15 min |
| 13-09 | DPO, GRPO & Modern Alignment | DPO — direct preference optimization loss derivation (no explicit reward model needed); GRPO — group relative policy optimization (CS336 A5); expert iteration and RL from verifiable rewards (RLVR); comparison of all alignment methods on same task | Synthetic preference data | ~12 min |
| 13-10 | Efficient Fine-Tuning with Unsloth | Unsloth + QLoRA end-to-end pipeline; speed and memory benchmarks vs standard training; chat template formatting; export formats (GGUF, merged weights); inference with the fine-tuned model | SST-2, Synthetic instruction data | ~10 min |

---

## Topic Details

### 13-01: Full Fine-Tuning vs Feature Extraction
Students will implement and compare two fundamental adaptation strategies: full fine-tuning, where all model parameters are updated, and feature extraction, where only a classification head is trained on frozen pretrained features. The notebook provides a rigorous treatment of catastrophic forgetting -- measuring how fine-tuning on a new task degrades performance on the original pretraining distribution -- and implements learning rate strategies to mitigate it, including warmup schedules and discriminative learning rates that apply smaller updates to early layers. Through controlled comparison experiments on SST-2, students quantify when the computational cost of full fine-tuning is justified versus when feature extraction suffices. This topic establishes the baseline that all subsequent parameter-efficient methods (LoRA, prefix tuning, adapters) are measured against.

### 13-02: LoRA -- Low-Rank Adaptation from Scratch
This topic implements Low-Rank Adaptation entirely from scratch, starting from the mathematical insight that weight updates during fine-tuning have low intrinsic rank and can be decomposed as W + BA where B and A are low-rank matrices. Students derive why rank-r adaptation captures most of the fine-tuning benefit with a fraction of the parameters, implement LoRA layers that inject trainable low-rank decompositions into attention projections and feed-forward networks, and run rank selection experiments to find the optimal tradeoff between expressiveness and parameter count. The merge-at-inference technique is implemented, showing how LoRA weights can be folded back into the original model for zero-overhead deployment. This is the foundational PEFT method that QLoRA (13-03) extends and that the adapter comparison (13-04) benchmarks against.

### 13-03: QLoRA & Memory-Efficient Fine-Tuning
Building on LoRA (13-02), this topic tackles the memory bottleneck of fine-tuning large models by combining 4-bit NormalFloat quantization of the base model with LoRA adapters trained in higher precision. Students implement the full QLoRA pipeline: 4-bit quantization with the NormalFloat data type, double quantization that further compresses the quantization constants, and paged optimizers that offload optimizer states to CPU when GPU memory is exhausted. Memory profiling before and after quantization demonstrates the dramatic memory savings, and a systematic comparison of LoRA vs QLoRA measures the quality-speed tradeoff to determine when the approximation cost is acceptable. This topic connects forward to quantization theory in Module 17-06, which is listed as a recommended (not required) prerequisite.

### 13-04: Adapter Methods Comparison
This comparison topic benchmarks all major parameter-efficient fine-tuning methods on the same SST-2 task under controlled conditions: LoRA (from 13-02), Prefix Tuning (from 13-05), and IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations). Students measure parameter-efficiency metrics -- trainable parameters as a fraction of total parameters -- alongside task performance, training speed, and memory footprint for each method. The notebook provides task-specific selection guidance: when LoRA's weight-space adaptation is preferred, when prefix tuning's input-space approach excels, and when IA3's lightweight rescaling is sufficient. Adapter merging and stacking techniques are explored, showing how multiple task-specific adapters can coexist on a single base model, which is essential for practical multi-task deployment scenarios.

### 13-05: Prompt Tuning & Prefix Tuning
Students implement two input-space adaptation techniques: prompt tuning, where learnable continuous vectors are prepended to the input embeddings, and prefix tuning, where learnable key-value pairs are injected into every attention layer. Unlike LoRA which modifies the weight matrices, these methods modify the model's input and intermediate representations while keeping all parameters frozen. The notebook derives why soft prompts can be more effective than discrete prompt engineering, and implements the prefix tuning mechanism that provides richer conditioning by intervening at every attention layer rather than just the input. A comparison with LoRA on the same task reveals complementary strengths: prefix tuning excels with very few trainable parameters while LoRA scales better to larger adaptation budgets.

### 13-06: Instruction Tuning & Dataset Construction
This topic covers the data-side of fine-tuning: constructing high-quality instruction-response pairs and training the supervised fine-tuning (SFT) loop that turns a base language model into an instruction-following assistant. Students examine the Alpaca, Dolly, and OpenAssistant dataset formats, implement quality filtering strategies including length filtering, deduplication, and toxicity detection, and design data mixing curricula that balance different instruction categories. The SFT training loop is implemented as a core deliverable aligned with CS336 Assignment 5, with careful attention to chat template formatting, loss masking on instruction tokens, and sequence packing for efficiency. This topic is the behavioral foundation for the alignment pipeline: SFT produces the initial instruction-following behavior that reward modeling (13-07) and RLHF (13-08) subsequently refine.

### 13-07: Reward Modeling
Students implement a reward model from scratch that learns to predict human preferences between response pairs, the critical component that provides the training signal for RLHF. The notebook covers the Bradley-Terry model for pairwise preferences, which converts chosen/rejected response pairs into a maximum likelihood training objective, and builds the reward model architecture by adding a scalar value head to a language model backbone. Training on synthetic preference data demonstrates how the model learns to assign higher scalar rewards to preferred responses. The notebook also addresses reward hacking and overoptimization -- where the policy learns to exploit artifacts in the reward model rather than genuinely improving -- establishing the motivation for the KL penalty in RLHF (13-08) and the reward-model-free approach of DPO (13-09).

### 13-08: RLHF -- PPO for Language Models
This topic implements the full Reinforcement Learning from Human Feedback pipeline, using the reward model from 13-07 to optimize language model outputs via Proximal Policy Optimization. Students implement PPO mechanics tailored for the RLHF setting: the clipped surrogate objective that prevents destructive policy updates, the KL divergence penalty against the reference (SFT) model that prevents reward hacking, a value head for advantage estimation using Generalized Advantage Estimation (GAE), and the complete RLHF training loop with rollout collection, reward scoring, and policy updates. Stability techniques including reward normalization, gradient clipping, and learning rate scheduling are implemented to address the notoriously fragile RLHF training dynamics. This notebook teaches PPO in the RLHF-specific context, while Module 14-07 later covers general PPO for RL environments.

### 13-09: DPO, GRPO & Modern Alignment
Students will implement the cutting-edge alignment methods that simplify or improve upon the RLHF pipeline. The DPO (Direct Preference Optimization) loss is derived from first principles, showing how the reward model can be implicitly defined through the policy itself, eliminating the need for separate reward model training and RL optimization. GRPO (Group Relative Policy Optimization) is implemented as an alignment technique from CS336 Assignment 5 that uses group-relative scoring rather than an explicit reward model. Expert iteration and RL from Verifiable Rewards (RLVR) are covered as complementary approaches for domains where correctness can be automatically verified. A comparison of all alignment methods -- SFT, RLHF, DPO, GRPO -- on the same task quantifies the tradeoffs between complexity, computational cost, and alignment quality.

### 13-10: Efficient Fine-Tuning with Unsloth
This tool-focused topic provides a practical end-to-end pipeline using Unsloth, an optimized library for efficient LLM fine-tuning that delivers significant speedups over standard training implementations. Students set up the Unsloth + QLoRA pipeline, format training data with proper chat templates, run the fine-tuning loop with speed and memory benchmarks compared to standard PyTorch training, and export the fine-tuned model in multiple formats including GGUF for llama.cpp inference and merged weights for standard deployment. Inference with the fine-tuned model is demonstrated end-to-end, closing the loop from raw data to deployed model. This topic bridges the from-scratch understanding built throughout the module with the practical tooling used in production fine-tuning workflows.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 13-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-04 | F — Comparison/Architecture | `TEMPLATE_COMPARISON.ipynb` |
| 13-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 13-10 | D — Tool/Library | `TEMPLATE_TOOL.ipynb` |

---

## Module-Specific Packages

- `unsloth` — efficient fine-tuning (13-10)

---

## Datasets

- SST-2 (13-01, 13-02, 13-03, 13-04, 13-05, 13-10)
- Synthetic instruction data (13-06, 13-10)
- Synthetic preference pairs (13-07)
- Synthetic preference data (13-08, 13-09)

---

## Prerequisites Chain

- **13-01:** Requires 6-04, 10-04
- **13-02:** Requires 13-01, 1-06
- **13-03:** Requires 13-02 | Recommended: 17-06 (quantization — enhances understanding but not required)
- **13-04:** Requires 13-02, 13-05
- **13-05:** Requires 8-04, 10-02
- **13-06:** Requires 10-01, 13-01
- **13-07:** Requires 13-06, 1-07
- **13-08:** Requires 13-07 | Recommended: 14-07 (PPO algorithm — enhances understanding but not required)
- **13-09:** Requires 13-07
- **13-10:** Requires 13-02, 13-06

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 13 — Fine-Tuning and Alignment
| Concept | Owner |
|---------|-------|
| Full fine-tuning vs feature extraction | 13-01 |
| LoRA (low-rank adaptation from scratch) | 13-02 |
| QLoRA (4-bit quantization + LoRA) | 13-03 |
| Adapter methods comparison (LoRA vs prefix vs IA³) | 13-04 |
| Prompt tuning, prefix tuning | 13-05 |
| Instruction tuning, SFT training loop | 13-06 |
| Reward modeling (Bradley-Terry) | 13-07 |
| RLHF with PPO | 13-08 |
| DPO, GRPO, expert iteration, RLVR | 13-09 |
| Unsloth end-to-end fine-tuning | 13-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ PPO in RLHF (13-08): This notebook teaches PPO mechanics in the RLHF context (clipped surrogate objective, KL penalty against reference model). Module 14-07 later covers general PPO for RL environments with full advantage estimation, value networks, and environment rollouts. 13-08 must be self-contained — teach enough PPO to understand RLHF without requiring 14-07. Do NOT re-teach RL-specific concepts (environment interaction, reward shaping, episodic returns).
- ⚠️ SFT (13-06) is behavioral cloning for LLMs — connection to imitation learning in 14-10.

---

## Special Notes

No special notes for this module.
