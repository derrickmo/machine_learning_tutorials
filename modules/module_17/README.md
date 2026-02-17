# Module 17 — Large Language Models — Systems & Scaling

## Introduction

Module 17 takes students from understanding individual Transformer components (Module 8) to understanding how those components are assembled, scaled, and optimized into the large language models that define modern AI. This module covers the full lifecycle of LLMs — from architectural design choices across GPT-2, LLaMA, Mistral, and Qwen, through scaling laws that determine how to allocate compute budgets, to the compression and serving techniques that make billion-parameter models practical to deploy. Students will implement scaling law experiments, build Mixture of Experts layers, construct Common Crawl data pipelines, apply quantization algorithms, and design efficient inference systems. After completing this module, students will understand not just how LLMs work architecturally (covered in Module 8) but how they are trained at scale, compressed for deployment, and evaluated rigorously — providing the foundation for the RAG and agentic systems in Module 18.

**Folder:** `module_17_large_language_models/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 17-01 | LLM Architecture & Design Patterns | GPT-2/LLaMA/Mistral/Qwen architectural choices side-by-side (pre-norm, RMSNorm, SwiGLU, GQA, RoPE — tying back to Module 8); parameter counting formulas; FLOPs estimation (6ND rule); architecture scaling heuristics | WikiText-2 | ~10 min |
| 17-02 | Scaling Laws & Compute-Optimal Training | Kaplan et al. (2020) power-law relationships; Chinchilla (Hoffmann et al., 2022) — compute-optimal allocation; IsoFLOP analysis methodology (CS336 A3 core deliverable); predicting optimal model size N* and data size D* for given compute C; D* ≈ 20N* rule of thumb; inference cost considerations beyond Chinchilla | WikiText-2, WikiText-103 | ~15 min |
| 17-03 | Mixture of Experts (MoE) | Sparse MoE layers — only top-k experts activated per token; gating/routing network; load balancing auxiliary loss; expert parallelism; Switch Transformer and Mixtral architecture; compute efficiency vs equivalent dense models | WikiText-2 | ~10 min |
| 17-04 | Training Data Pipelines for LLMs | Common Crawl pipeline end-to-end (CS336 A4): WARC/WET extraction → HTML-to-text (trafilatura) → language filtering → quality classification → harmful content removal → PII scrubbing (regex + NER) → deduplication (MinHash locality-sensitive hashing); data mixing strategies; Paloma evaluation; ablation studies showing impact of each filtering step | Common Crawl subset | ~15 min |
| 17-05 | Knowledge Distillation | Teacher-student training framework; soft targets and temperature scaling; distillation loss (KL divergence on logits); feature-level and attention-level distillation; task-specific vs task-agnostic; practical distillation recipes | WikiText-2 | ~12 min |
| 17-06 | Quantization Deep Dive | Post-training quantization (PTQ) — INT8, INT4 weight quantization; quantization-aware training (QAT); GPTQ (one-shot, Hessian-based); AWQ (activation-aware weight quantization); GGUF format for llama.cpp deployment; quality vs compression tradeoff benchmarks | WikiText-2 | ~10 min |
| 17-07 | Efficient Inference & Serving | Speculative decoding (draft model proposes, target model verifies); continuous batching for throughput; PagedAttention (vLLM) — virtual memory for KV cache; model parallelism for inference (tensor parallel); latency vs throughput optimization; TGI/vLLM serving architecture; cost optimization patterns — model routing (cheap model for easy queries, expensive for hard), semantic caching, token budgeting | WikiText-2 | ~8 min |
| 17-08 | Long Context Techniques | RoPE scaling methods (NTK-aware interpolation, YaRN, dynamic scaling); ALiBi (attention with linear biases); sliding window attention (Mistral); context length extension evaluation methodology; memory implications of very long sequences | WikiText-2 | ~8 min |
| 17-09 | Structured Output & Function Calling | Constrained decoding (JSON schema enforcement via grammar-guided generation); function calling protocol design (tool name, parameters, results); output parsing and validation; tool-use formatting conventions; code generation concepts (Codex, HumanEval, code-specific patterns); connection to agents in Module 18 | WikiText-2 | ~8 min |
| 17-10 | LLM Evaluation & Benchmarks | Perplexity and its limitations; multiple-choice benchmarks (MMLU, HellaSwag, ARC, WinoGrande); generation benchmarks (HumanEval, MATH, GSM8K); contamination detection and benchmark gaming; TruthfulQA; holistic evaluation methodology; Chatbot Arena (CS336 Lecture 12); distinct from 10-08 (NLU-scale benchmarks: GLUE, behavioral testing) | WikiText-2 | ~8 min |

---

## Topic Details

### 17-01: LLM Architecture & Design Patterns
Students will conduct a detailed side-by-side comparison of four major LLM architectures — GPT-2, LLaMA, Mistral, and Qwen — examining how each makes different choices about normalization (pre-norm vs post-norm, LayerNorm vs RMSNorm), activation functions (GELU vs SwiGLU), attention mechanisms (MHA vs GQA), and positional encoding (learned vs RoPE). The notebook ties each component back to its Module 8 origin while showing how these choices interact at scale. Students will implement parameter counting formulas and FLOPs estimation using the 6ND rule, learning to quickly assess the computational cost of any architecture. This comparison notebook establishes the architectural vocabulary used throughout the rest of Module 17 and is a prerequisite for every subsequent topic.

### 17-02: Scaling Laws & Compute-Optimal Training
This theory-focused notebook derives the power-law scaling relationships from Kaplan et al. (2020) and the Chinchilla-optimal compute allocation from Hoffmann et al. (2022), then validates them empirically using IsoFLOP analysis on small models trained on WikiText. Students will fit scaling law curves using scipy, predict optimal model size N* and data size D* for a given compute budget C, and verify the D* = 20N* rule of thumb. The notebook also covers inference cost considerations that go beyond the original Chinchilla analysis — why inference-heavy deployments favor smaller, more overtrained models. Understanding scaling laws is essential for any practitioner making decisions about model size, dataset size, and training budget, and this knowledge informs the efficiency techniques covered in 17-05 through 17-07.

### 17-03: Mixture of Experts (MoE)
Students will build a sparse Mixture of Experts layer from scratch, implementing a gating network that routes each token to only the top-k experts (out of many more available), achieving higher model capacity without proportional compute cost. The notebook covers the load balancing auxiliary loss that prevents expert collapse (all tokens routed to one expert), and connects to Switch Transformer and Mixtral architectures. Students will compare a dense model against an MoE model with equivalent compute budget, demonstrating the efficiency advantages. Expert parallelism concepts show how MoE layers map to distributed training. This topic extends the Transformer architecture knowledge from Module 8 into the sparse computation paradigm that underpins many modern LLMs.

### 17-04: Training Data Pipelines for LLMs
This notebook builds a complete Common Crawl data pipeline from raw web data to training-ready tokens, following the CS336 A4 methodology. Students will implement each filtering stage — WARC/WET extraction, HTML-to-text conversion with trafilatura, language identification, quality classification, harmful content removal, PII scrubbing using regex and NER, and deduplication with MinHash locality-sensitive hashing. Ablation studies demonstrate the impact of each filtering step on downstream model quality measured via Paloma evaluation. This topic is critical because data quality is the single largest determinant of LLM capability, and understanding the full pipeline helps practitioners diagnose data-related issues in model behavior. The ethics of web-scraped training data connect forward to fairness discussions in 19-09.

### 17-05: Knowledge Distillation
Students will implement the complete teacher-student training framework, starting with soft targets and temperature scaling to transfer dark knowledge from a large teacher model to a smaller student. The notebook derives the distillation loss (KL divergence between teacher and student logit distributions) and implements both logit-level and feature-level distillation. Students will compare task-specific distillation (fine-tuned teacher on a single task) versus task-agnostic distillation (general-purpose compression) and measure the quality-size tradeoff. Practical distillation recipes provide guidelines for choosing temperature, loss weighting, and training schedule. Distillation connects directly to the quantization techniques in 17-06 as complementary model compression strategies.

### 17-06: Quantization Deep Dive
This notebook implements multiple quantization algorithms from scratch — post-training quantization (PTQ) at INT8 and INT4 precision, quantization-aware training (QAT) with simulated quantization during forward passes, GPTQ (one-shot Hessian-based weight quantization), and AWQ (activation-aware weight quantization that preserves salient channels). Students will benchmark each method on perplexity degradation versus compression ratio and inference speedup, and learn the GGUF format used for llama.cpp deployment. Understanding quantization is essential for deploying LLMs on consumer hardware and edge devices, and this topic is a direct prerequisite for the efficient inference techniques in 17-07.

### 17-07: Efficient Inference & Serving
Students will implement speculative decoding — where a small draft model proposes token sequences that a larger target model verifies in parallel, achieving significant latency reduction without quality loss. The notebook covers continuous batching for maximizing throughput, PagedAttention (the vLLM innovation that applies virtual memory concepts to KV cache management), and tensor parallel inference for models that do not fit on a single GPU. Cost optimization patterns — model routing (dispatching easy queries to cheap models), semantic caching, and token budgeting — are implemented as practical deployment strategies. This topic bridges the gap between training an LLM and serving it to users at scale, directly preparing students for the RAG serving scenarios in Module 18.

### 17-08: Long Context Techniques
This notebook explores techniques for extending the context window of pretrained LLMs beyond their original training length. Students will implement RoPE scaling methods — NTK-aware interpolation, YaRN, and dynamic scaling — understanding how each modifies the rotary position embedding to extrapolate to longer sequences. ALiBi (attention with linear biases) is implemented as an alternative position encoding that generalizes to unseen lengths, and sliding window attention (Mistral-style) provides a computationally efficient approach for very long contexts. Students will measure perplexity across different context lengths and analyze the quadratic memory implications of long sequences. Long context capability is increasingly important for RAG systems (Module 18) where large documents must be processed in their entirety.

### 17-09: Structured Output & Function Calling
Students will implement constrained decoding that forces an LLM to produce valid JSON conforming to a specified schema, using grammar-guided generation that masks invalid tokens at each decoding step. The notebook covers function calling protocol design — defining tool schemas, parsing model outputs into structured action objects, executing tools, and feeding results back into the generation context. Output validation with Pydantic models ensures type safety, and code generation patterns (Codex-style) show how structured output extends to programming tasks. This topic is the direct bridge to Module 18, where agent loops rely on function calling to interact with tools, and structured output ensures reliable tool invocation.

### 17-10: LLM Evaluation & Benchmarks
This evaluation-focused notebook builds a comprehensive LLM evaluation pipeline covering perplexity measurement (and its limitations as a standalone metric), multiple-choice benchmarks (MMLU, HellaSwag, ARC, WinoGrande), and generation benchmarks (HumanEval for code, MATH and GSM8K for reasoning). Students will implement contamination detection to identify benchmark leakage in training data and discuss benchmark gaming concerns. The notebook covers TruthfulQA for measuring factual accuracy and introduces Chatbot Arena as an Elo-based human evaluation methodology. This topic is distinct from 10-08 (which covers NLU-scale benchmarks like GLUE and behavioral testing) by focusing specifically on LLM-scale evaluation challenges, providing the metrics framework used to assess models throughout Modules 17-20.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 17-01 | F — Comparison/Architecture | `TEMPLATE_COMPARISON.ipynb` |
| 17-02 | B — Theory | `TEMPLATE_THEORY.ipynb` |
| 17-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 17-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 17-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 17-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 17-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 17-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 17-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 17-10 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |

---

## Module-Specific Packages

- `scipy` — curve fitting for scaling laws (17-02)

---

## Datasets

- WikiText-2 / WikiText-103 (scaling laws, architecture comparison)
- Common Crawl subset (17-04 data pipeline)

---

## Prerequisites Chain

- **17-01:** Requires 8-05, 8-06, 8-07, 8-08
- **17-02:** Requires 17-01
- **17-03:** Requires 17-01, 8-04
- **17-04:** Requires 17-01
- **17-05:** Requires 5-04, 17-01
- **17-06:** Requires 17-01
- **17-07:** Requires 8-10, 17-06
- **17-08:** Requires 8-08, 8-09
- **17-09:** Requires 17-01, 10-01
- **17-10:** Requires 17-01, 10-08

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 17 — Large Language Models
| Concept | Owner |
|---------|-------|
| LLM architecture patterns (GPT/LLaMA/Mistral comparison) | 17-01 |
| Scaling laws, IsoFLOP, Chinchilla | 17-02 |
| Mixture of Experts (MoE) | 17-03 |
| Training data pipelines (Common Crawl, filtering, dedup) | 17-04 |
| Knowledge distillation | 17-05 |
| Quantization (PTQ, QAT, GPTQ, AWQ, GGUF) | 17-06 |
| Efficient inference (speculative decoding, vLLM, PagedAttention), cost optimization | 17-07 |
| Long context (RoPE scaling, ALiBi, sliding window) | 17-08 |
| Structured output, function calling, code generation | 17-09 |
| LLM evaluation and benchmarks (MMLU, HumanEval, MATH) | 17-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ 17-01 ties back to Module 8 components (RMSNorm, SwiGLU, GQA, RoPE). Reference 8-xx, don't re-teach.
- ⚠️ Cost optimization in 17-07 (model routing, caching, token budgeting) is new content, not in Module 8.

---

## Special Notes

No special notes for this module.
