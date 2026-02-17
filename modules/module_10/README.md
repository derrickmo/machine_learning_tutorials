# Module 10 — Advanced NLP — Pretrained Language Models

## Introduction

This module covers the complete landscape of pretrained language models, from training GPT-style and BERT-style architectures from scratch to applying them across the core NLP tasks that define modern natural language understanding. Students will implement autoregressive (decoder-only) and masked (encoder-only) language modeling objectives, compare them against encoder-decoder architectures, and then fine-tune pretrained transformers for text classification, named entity recognition, natural language inference, and question answering. The module also advances into chain-of-thought prompting, in-context learning, and mechanistic interpretability -- the techniques that reveal how and why large language models work at a mechanistic level. Module 10 is the culmination of the NLP pipeline built across Modules 07 and 08, applying transformer architectures to real language tasks, and it lays the groundwork for fine-tuning and alignment (Module 13), LLM systems and scaling (Module 17), and RAG/agentic architectures (Module 18).

**Folder:** `module_10_advanced_nlp/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 10-01 | Decoder-Only Language Model (GPT-Style) | Autoregressive language modeling objective; causal masking; train a small GPT on text data (connection to CS336 A1 full pipeline); perplexity evaluation; sampling and generation; next-token prediction mechanics | WikiText-2 | ~15 min |
| 10-02 | Encoder-Only Masked Language Model (BERT-Style) | Masked LM objective (15% masking strategy); [CLS] and [SEP] tokens; next sentence prediction; train a small BERT; extract contextual embeddings | WikiText-2 | ~15 min |
| 10-03 | Encoder-Decoder & Architecture Comparison | T5/BART — prefix LM and denoising objectives; systematic comparison: BERT vs GPT vs T5; when to use encoder-only, decoder-only, encoder-decoder; parameter efficiency tradeoffs | WikiText-2 | ~15 min |
| 10-04 | Text Classification with Pretrained Transformers | Loading pretrained weights; adding classification head; warmup scheduling; fine-tuning on SST-2 or AG_NEWS; freezing strategies; comparison with training from scratch | SST-2, AG_NEWS | ~10 min |
| 10-05 | Named Entity Recognition & Token Classification | Token-level classification heads; BIO/BILOU tagging schemes; subword-to-word alignment; handling tokenization mismatches; sequence labeling evaluation (entity-level F1); distinct from BiLSTM-CRF in 7-09 — this uses transformer architecture | CoNLL2000Chunking | ~10 min |
| 10-06 | Natural Language Inference & Textual Entailment | Entailment/contradiction/neutral classification; cross-encoder architecture for NLI; transfer learning from NLI to downstream tasks; connection to fact verification and RAG faithfulness (Module 18) | Synthetic NLI pairs | ~10 min |
| 10-07 | Question Answering Systems | Extractive QA (SQuAD-style — span prediction with start/end logits); open-domain QA (retriever + reader); dense passage retrieval concepts; connection to RAG in Module 18; evaluation (EM, F1) | Synthetic QA pairs | ~10 min |
| 10-08 | NLP Evaluation & Behavioral Testing | GLUE/SuperGLUE benchmarks; behavioral testing (CheckList methodology); adversarial evaluation (ANLI concepts); dataset artifacts and spurious correlations; dynamic benchmarks; distinct from 17-10 (which covers LLM-scale benchmarks: MMLU, HumanEval, MATH) | AG_NEWS, SST-2 | ~5 min |
| 10-09 | Chain-of-Thought & In-Context Learning | Few-shot prompting; chain-of-thought (CoT); zero-shot CoT ("let's think step by step"); self-consistency (majority voting over CoT samples); in-context learning theory; why ICL works — induction heads and task identification | WikiText-2 (model from 10-01) | ~8 min |
| 10-10 | Mechanistic Interpretability for NLP | Probing classifiers (what information do layers encode?); attention pattern analysis; activation patching; causal abstraction concepts; circuit discovery; distinct from 19-05 (model-agnostic SHAP/LIME) — this is transformer-specific | WikiText-2 (model from 10-01) | ~8 min |

---

## Topic Details

### 10-01: Decoder-Only Language Model (GPT-Style)
Students train a small GPT-style language model from scratch on WikiText-2, implementing the autoregressive language modeling objective where the model predicts each next token conditioned on all previous tokens using causal masking. The notebook covers the complete training pipeline -- tokenization (using the BPE tokenizer from 7-01), causal attention masking, cross-entropy loss computation, and perplexity evaluation -- following the CS336 Assignment 1 specification for end-to-end language model training. Students implement sampling and generation using the decoding strategies from 7-06 (greedy, temperature, top-k, nucleus) and analyze how model capacity affects generation quality. This is the foundational GPT implementation that serves as the reference architecture for all decoder-only models discussed in the course, and the trained model is reused directly in 10-09 (in-context learning) and 10-10 (mechanistic interpretability).

### 10-02: Encoder-Only Masked Language Model (BERT-Style)
Students implement the BERT pretraining objective from scratch: randomly masking 15% of input tokens (with the 80/10/10 mask/random/keep strategy), training the model to reconstruct masked positions, and optionally performing next sentence prediction (NSP). The notebook covers [CLS] and [SEP] token conventions, segment embeddings for sentence-pair inputs, and the extraction of contextual embeddings from different layers for downstream use. Students train a small BERT on WikiText-2 and demonstrate that the bidirectional encoder produces richer contextualized representations than the unidirectional GPT model for understanding tasks. This implementation provides the encoder model used in 10-04 through 10-07 for fine-tuning on downstream NLP tasks, and establishes the contrast between masked LM and autoregressive LM that is systematically compared in 10-03.

### 10-03: Encoder-Decoder & Architecture Comparison
This comparison notebook implements the T5/BART encoder-decoder paradigm -- including prefix LM and denoising pretraining objectives -- and then conducts a systematic three-way comparison between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) architectures. Students evaluate each architecture on classification, generation, and sequence-to-sequence tasks, measuring where each excels: BERT for understanding tasks requiring bidirectional context, GPT for generation tasks requiring autoregressive output, and T5 for tasks requiring both input understanding and output generation. The notebook analyzes parameter efficiency tradeoffs and provides concrete decision guidelines for when to use each architecture family. This comparison crystallizes the architectural landscape that students will navigate throughout Modules 13 (fine-tuning), 17 (LLM scaling), and 18 (RAG systems).

### 10-04: Text Classification with Pretrained Transformers
Students implement the full fine-tuning pipeline for text classification: loading pretrained transformer weights, replacing the final layer with a task-specific classification head, applying linear warmup learning rate scheduling, and fine-tuning on SST-2 (sentiment) and AG_NEWS (topic classification). The notebook compares frozen feature extraction versus full fine-tuning versus training from scratch, quantifying how pretrained representations provide a massive head start especially on small datasets. Different freezing strategies (freeze all but last N layers, gradual unfreezing) are tested systematically. This is the canonical transfer learning application for NLP, using the workflow established in 6-04 but adapted for transformer architectures, and it establishes the fine-tuning patterns reused in 10-05, 10-06, and 10-07.

### 10-05: Named Entity Recognition & Token Classification
Students implement token-level classification by adding per-token classification heads to a pretrained transformer encoder, handling the critical challenge of subword-to-word alignment where tokenizers split words into multiple subword tokens that must be mapped back to word-level entity labels. The notebook covers BIO and BILOU tagging schemes, implements the alignment logic for handling tokenization mismatches, and evaluates using entity-level F1 (not token-level) to match standard NER evaluation practice. This transformer-based approach is explicitly distinguished from the BiLSTM-CRF in 7-09 -- both are valid approaches to sequence labeling, but this notebook uses transformer representations while 7-09 uses LSTM representations with structured CRF decoding. The subword alignment challenge addressed here is fundamental to any token-level task with pretrained transformers.

### 10-06: Natural Language Inference & Textual Entailment
Students build a cross-encoder architecture for natural language inference (NLI), classifying sentence pairs as entailment, contradiction, or neutral by encoding both sentences jointly through a pretrained transformer and classifying the [CLS] representation. The notebook demonstrates transfer learning from NLI to other downstream tasks, showing that NLI-trained models serve as powerful general-purpose sentence understanding systems. Students connect NLI to practical applications: fact verification (determining if a claim is supported by evidence) and RAG faithfulness checking (determining if a generated answer is faithful to retrieved passages, as covered in Module 18). This topic builds the conceptual foundation for the retrieval-augmented generation evaluation pipeline in 18-05 and the fact-checking applications in Module 19.

### 10-07: Question Answering Systems
Students implement extractive question answering in the SQuAD style, training a model to predict start and end token positions that span the answer within a given passage. The notebook covers the span prediction architecture (two classification heads for start and end logits), no-answer detection, and evaluation using exact match (EM) and token-level F1 metrics. Open-domain QA is introduced as the retriever-reader pipeline: a retriever finds relevant passages from a large corpus, then a reader extracts answers from the retrieved passages. Dense passage retrieval concepts are covered as the modern replacement for sparse BM25 retrieval. This topic is the direct prerequisite for RAG systems in Module 18, where the retriever-reader pattern is extended into a full retrieval-augmented generation pipeline.

### 10-08: NLP Evaluation & Behavioral Testing
This evaluation-focused notebook covers the GLUE and SuperGLUE benchmark suites, providing students with the standard framework for evaluating NLP models across diverse tasks (sentiment, NLI, paraphrase, QA). Beyond aggregate accuracy, the notebook implements behavioral testing using the CheckList methodology -- testing models on minimum functionality, invariance, and directional expectation tests to uncover systematic failures that aggregate metrics miss. Students explore adversarial evaluation (ANLI concepts), dataset artifacts and spurious correlations (hypothesis-only baselines for NLI), and the limitations of static benchmarks. This notebook is distinct from 17-10 (which covers LLM-scale benchmarks like MMLU, HumanEval, and MATH) -- the scope here is encoder-model evaluation at the GLUE level, not generative LLM evaluation.

### 10-09: Chain-of-Thought & In-Context Learning
Students explore how language models can solve tasks through prompting without gradient updates: few-shot in-context learning (providing examples in the prompt), chain-of-thought prompting (asking the model to show intermediate reasoning steps), zero-shot CoT (the "let's think step by step" prompt), and self-consistency (generating multiple CoT samples and taking the majority vote). The notebook uses the small GPT model trained in 10-01 to demonstrate these mechanisms at a scale where the behavior is observable, with clear caveats that production-quality CoT requires much larger models. Students explore theoretical explanations for why in-context learning works, including induction heads (attention patterns that copy from earlier context) and task identification (the model identifying which latent task to perform from the examples). This topic connects forward to the prompting engineering and alignment techniques in Module 13.

### 10-10: Mechanistic Interpretability for NLP
Students implement transformer-specific interpretability techniques: probing classifiers that test what linguistic information (POS tags, syntax, semantics) is encoded at each layer, attention pattern analysis that reveals how different heads specialize, activation patching that identifies which components are causally responsible for specific model behaviors, and causal abstraction concepts for discovering computational circuits within the network. The notebook uses the small GPT model from 10-01 as the analysis target, enabling students to trace information flow through a model they built themselves. This is explicitly distinct from model-agnostic interpretability methods (SHAP, LIME) covered in 19-05 -- mechanistic interpretability examines the internal computations of transformers specifically, not the input-output relationship of arbitrary models. These techniques are essential for understanding LLM behavior and connect to the safety and alignment discussions in Module 13.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 10-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 10-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 10-03 | F — Comparison/Architecture | `TEMPLATE_COMPARISON.ipynb` |
| 10-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 10-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 10-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 10-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 10-08 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 10-09 | B — Theory | `TEMPLATE_THEORY.ipynb` |
| 10-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- WikiText-2 (language modeling)
- SST-2 (classification)
- AG_NEWS (classification)
- CoNLL (NER, 10-05)
- Synthetic NLI pairs (10-06)
- Synthetic QA pairs (10-07)

---

## Prerequisites Chain

- **10-01:** Requires 8-05, 7-01
- **10-02:** Requires 8-05, 7-01
- **10-03:** Requires 10-01, 10-02
- **10-04:** Requires 10-02
- **10-05:** Requires 10-02, 7-09
- **10-06:** Requires 10-02, 10-04
- **10-07:** Requires 10-02 | Recommended: 18-01 (embeddings/vector stores — enhances understanding but not required)
- **10-08:** Requires 10-04
- **10-09:** Requires 10-01
- **10-10:** Requires 10-02, 8-01

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 10 — Advanced NLP
| Concept | Owner |
|---------|-------|
| GPT-style autoregressive language modeling | 10-01 |
| BERT-style masked language modeling | 10-02 |
| Encoder vs decoder vs encoder-decoder comparison | 10-03 |
| Pretrained transformer fine-tuning for classification | 10-04 |
| Named entity recognition (transformer-based token classification) | 10-05 |
| Natural language inference (NLI, textual entailment) | 10-06 |
| Question answering (extractive QA, open-domain QA) | 10-07 |
| NLP evaluation (GLUE, behavioral testing, adversarial eval) | 10-08 |
| Chain-of-thought, in-context learning (ICL) | 10-09 |
| Mechanistic interpretability (probing, activation patching, circuits) | 10-10 |

---

## Cross-Module Ownership Warnings

- GPT-style LM (10-01): Module 17 discusses LLM-scale patterns but must NOT re-teach the autoregressive objective.
- NLP Eval (10-08) covers GLUE/behavioral testing. LLM-scale eval (MMLU, HumanEval) is in 17-10 -- distinct scope.
- ⚠️ CheckList behavioral testing (10-08) is the canonical teaching location. Module 20-08 (ML Testing) references CheckList but must not re-teach the methodology — focus on unit tests, data validation, and project standards instead.

---

## Special Notes

- Expanded from 5 → 10 topics. Chain-of-thought (10-09) and mechanistic interpretability (10-10) are major additions.
- 10-05 (NER) uses transformer architecture. 7-09 (BiLSTM-CRF) is pre-transformer -- both are valid, distinct approaches.
- **Feasibility note:** Topics 10-09 (ICL) and 10-10 (interpretability) require a functioning language model. Since the course prohibits external API keys, these notebooks use the small GPT model trained in 10-01. The goal is to demonstrate the mechanism clearly, not to achieve production-quality results. Set appropriate expectations in the markdown narrative.
