# Module 08 — Transformers — Architecture to Attention

## Introduction

This module is the most critical in the entire 200-topic curriculum: it covers the transformer architecture from first principles, building every component from raw tensor operations without relying on high-level library abstractions. Students will implement self-attention with explicit query-key-value matrix multiplications, multi-head attention, positional encodings, the complete transformer block, and the full encoder-decoder transformer -- following the CS336 Assignment 1 spirit of building everything from scratch. The module then advances to modern architectural innovations used in production LLMs: RMSNorm, SwiGLU activations, grouped-query attention, rotary position embeddings (RoPE), Flash Attention, and KV caching for efficient autoregressive inference. By the end of Module 08, students will have a complete, from-scratch transformer implementation that serves as the foundation for every subsequent module in the course -- Modules 9 through 18 all build directly on the architecture taught here.

**Folder:** `module_08_transformers/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 8-01 | Self-Attention Mechanism | Query, Key, Value matrices from scratch (no nn.Linear — raw weight matrices and matmuls per CS336 A1); scaled dot-product attention; attention weights as soft dictionary lookup; visualizing attention patterns; complexity analysis — O(n²d) | Synthetic sequences | ~3 min |
| 8-02 | Multi-Head Attention | Parallel attention heads with different learned projections; concatenation and output projection; why multiple heads capture different relationships; head pruning intuition; implement from scratch | Synthetic sequences | ~3 min |
| 8-03 | Positional Encoding — Sinusoidal & Learned | Why position information is needed (permutation invariance of attention); sinusoidal encoding (Vaswani et al.); learned position embeddings; relative position encoding concepts; limitations of fixed-length position encodings | Synthetic sequences | ~3 min |
| 8-04 | The Transformer Block | Layer normalization (pre-norm vs post-norm); residual connections; feed-forward network (expand → activate → project); full encoder block assembly; full decoder block with causal masking and cross-attention | WikiText-2 | ~8 min |
| 8-05 | Full Transformer — Encoder-Decoder | Complete architecture assembly (CS336 A1); encoder self-attention, decoder causal self-attention, cross-attention; padding masks and causal masks; transformer for sequence-to-sequence; comparison with LSTM seq2seq | WikiText-2 | ~15 min |
| 8-06 | RMSNorm, SwiGLU & Modern Building Blocks | RMSNorm vs LayerNorm — simpler, faster, used in LLaMA/Mistral (CS336 A2 fused kernel target); SwiGLU activation in FFN (Shazeer 2020); pre-normalization pattern; modern transformer recipe (GPT-2 vs LLaMA/Mistral differences) | WikiText-2 | ~8 min |
| 8-07 | Grouped-Query & Multi-Query Attention | Multi-Query Attention (MQA) — shared KV heads for inference speed; Grouped-Query Attention (GQA) — compromise between MHA and MQA (used in LLaMA 2/3); implement both; memory and throughput comparison | WikiText-2 | ~5 min |
| 8-08 | Rotary Position Embeddings (RoPE) | Complex number formulation of position encoding; rotation matrices applied in 2D subspaces; how RoPE encodes relative position through dot products; RoPE implementation from scratch; connection to RoPE scaling for long context in Module 17 | Synthetic sequences | ~3 min |
| 8-09 | Flash Attention — Algorithm & Concepts | Standard attention memory bottleneck (O(n²) materialization); tiling strategy and online softmax trick; Flash Attention 1 & 2 algorithm walkthrough; IO-complexity analysis; memory-efficient PyTorch implementation; Triton kernel concepts (CS336 A2 core deliverable) | Synthetic (attention benchmarks) | ~5 min |
| 8-10 | KV Cache & Autoregressive Inference | Why autoregressive generation is memory-bound, not compute-bound; KV cache implementation from scratch; incremental decoding loop; cache memory analysis (batch × layers × heads × seq_len × d_head); connection to PagedAttention concepts (vLLM) | WikiText-2 | ~5 min |

---

## Topic Details

### 8-01: Self-Attention Mechanism
Students implement scaled dot-product self-attention entirely from scratch using raw weight matrices and matrix multiplications -- no `nn.Linear` or `nn.MultiheadAttention` -- following the CS336 Assignment 1 specification. The notebook constructs Query, Key, and Value projections from raw parameter tensors, computes attention scores via the scaled dot product, applies softmax to obtain attention weights, and produces the weighted sum of values. Students visualize attention patterns to build intuition for attention as a soft dictionary lookup where queries retrieve relevant values through key similarity. The O(n^2 * d) complexity analysis establishes the fundamental computational bottleneck that motivates Flash Attention (8-09) and efficient attention variants throughout the course. This is the single most important building block in the curriculum: every architecture from Module 9 through Module 18 depends on self-attention.

### 8-02: Multi-Head Attention
Building on the single-head attention from 8-01, students implement multi-head attention by running parallel attention computations with different learned projection matrices, concatenating the outputs, and applying a final output projection. The notebook demonstrates empirically that different heads learn to attend to different types of relationships -- some capture positional patterns, others capture semantic similarity, and others capture syntactic dependencies. Students implement the full multi-head mechanism from scratch and analyze the parameter count compared to equivalent single-head attention with the same total dimension. Head pruning is introduced conceptually to show that not all heads are equally important. This module is the direct prerequisite for the transformer block (8-04), grouped-query attention (8-07), and Flash Attention (8-09).

### 8-03: Positional Encoding -- Sinusoidal & Learned
This topic addresses why transformers need explicit position information -- the self-attention mechanism is permutation invariant, meaning it produces identical outputs regardless of token order -- and implements two solutions. Students build sinusoidal position encodings (Vaswani et al.) from the frequency formula and visualize the wave patterns that encode position as a unique signature of sine and cosine functions at different frequencies. Learned position embeddings are implemented as an alternative where position vectors are trained parameters. The notebook also covers relative position encoding concepts and analyzes the limitations of fixed-length position encodings for sequences longer than training length. This topic sets up the contrast with RoPE (8-08), which solves the length generalization problem, and is prerequisite for the transformer block (8-04).

### 8-04: The Transformer Block
Students assemble the complete transformer block by combining multi-head attention, layer normalization, residual connections, and the position-wise feed-forward network (expand-activate-project pattern). The notebook implements both pre-norm (modern standard, used in GPT-2+) and post-norm (original Vaswani) configurations, comparing training stability. Students build a full encoder block (self-attention + FFN) and a full decoder block (causal self-attention + cross-attention + FFN), implementing causal masking to prevent the decoder from attending to future tokens. This is where all the individual components from 8-01 through 8-03 come together into a functional building block, and it is the direct prerequisite for the full transformer assembly (8-05) and modern architectural variants (8-06).

### 8-05: Full Transformer -- Encoder-Decoder
This topic assembles the complete transformer architecture by stacking encoder and decoder blocks, adding token embeddings and position encodings, and implementing the full forward pass for sequence-to-sequence tasks -- following the CS336 Assignment 1 pipeline. Students implement padding masks (to ignore pad tokens in the encoder) and causal masks (to prevent future information leakage in the decoder), and build the cross-attention connections where decoder layers attend to encoder outputs. The notebook trains the full transformer on WikiText-2 and compares it directly against the LSTM seq2seq model from 7-05, demonstrating the transformer's superior performance and parallelizability. This is the architectural capstone of the module and serves as the reference implementation for decoder-only (10-01), encoder-only (10-02), and encoder-decoder (10-03) language model variants.

### 8-06: RMSNorm, SwiGLU & Modern Building Blocks
Students implement the modern building blocks that distinguish production LLMs (LLaMA, Mistral) from the original transformer: RMSNorm (simpler and faster than LayerNorm, removing the mean-centering step), SwiGLU activation in the feed-forward network (Shazeer 2020, replacing ReLU with a gated swish), and the pre-normalization pattern. The notebook provides a side-by-side comparison of the GPT-2 recipe (post-norm LayerNorm + GELU FFN) versus the LLaMA/Mistral recipe (pre-norm RMSNorm + SwiGLU FFN), measuring the training speed and stability differences. RMSNorm is also introduced as the CS336 Assignment 2 fused kernel target, connecting to GPU optimization concepts. These modern blocks are the standard in all LLMs discussed in Modules 13 and 17.

### 8-07: Grouped-Query & Multi-Query Attention
This topic implements two inference-optimized attention variants that reduce memory bandwidth during autoregressive generation. Multi-Query Attention (MQA) shares a single set of key-value heads across all query heads, dramatically reducing the KV cache size. Grouped-Query Attention (GQA) -- used in LLaMA 2 and LLaMA 3 -- is a compromise that groups query heads into clusters sharing KV heads. Students implement both from scratch, measure memory usage and throughput compared to standard multi-head attention, and analyze the accuracy tradeoff. The memory savings are quantified in terms of KV cache size reduction, directly connecting to the KV cache analysis in 8-10 and the inference optimization discussion in Module 17.

### 8-08: Rotary Position Embeddings (RoPE)
Students implement RoPE from scratch, starting from the complex number formulation where positions are encoded as rotation angles applied to pairs of embedding dimensions. The notebook derives how rotation matrices in 2D subspaces encode absolute position such that the dot product between two rotated vectors depends only on their relative distance -- elegantly solving the relative position encoding problem. Students verify this mathematical property empirically and compare RoPE against sinusoidal and learned position embeddings on sequence modeling tasks. The connection to RoPE scaling for long-context extension (used in LLaMA and discussed in Module 17) is established, showing how the frequency base can be adjusted to extrapolate beyond training-length sequences.

### 8-09: Flash Attention -- Algorithm & Concepts
This topic addresses the O(n^2) memory bottleneck of standard attention, where materializing the full attention matrix becomes prohibitive for long sequences. Students implement the Flash Attention algorithm conceptually: tiling the attention computation into blocks that fit in SRAM, using the online softmax trick to compute exact attention without materializing the full matrix, and analyzing the IO-complexity reduction from O(n^2) HBM reads to O(n^2 / SRAM_size). The notebook walks through Flash Attention 1 and 2 algorithm details, implements a memory-efficient PyTorch version that demonstrates the tiling strategy, and introduces Triton kernel concepts as the CS336 Assignment 2 core deliverable. This is the key inference and training optimization that enables long-context models and is referenced in Modules 16 and 17.

### 8-10: KV Cache & Autoregressive Inference
Students implement the KV cache from scratch, understanding why autoregressive generation is memory-bound rather than compute-bound: at each decoding step, only the new token's query needs to attend to all previous keys and values, so caching previously computed KV pairs eliminates redundant computation. The notebook builds a complete incremental decoding loop with KV cache management and analyzes cache memory requirements as a function of batch size, number of layers, number of heads, sequence length, and head dimension. PagedAttention concepts (used in vLLM) are introduced as the solution to memory fragmentation when serving multiple sequences with different lengths. This topic is essential for understanding LLM inference in Module 17 and production serving in Module 20.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 08-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 08-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- Synthetic sequences (8-01, 8-02, 8-03, 8-08, 8-09)
- WikiText-2 (8-04, 8-05, 8-06, 8-07, 8-10)

---

## Prerequisites Chain

- **08-01:** Requires 1-06, 5-07
- **08-02:** Requires 8-01
- **08-03:** Requires 8-02
- **08-04:** Requires 8-02, 8-03, 5-10
- **08-05:** Requires 8-04
- **08-06:** Requires 8-04
- **08-07:** Requires 8-02
- **08-08:** Requires 8-03, 1-09
- **08-09:** Requires 8-02 | Recommended: 15-07 (CUDA memory — enhances understanding but not required)
- **08-10:** Requires 8-05

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 8 — Transformers
| Concept | Owner |
|---------|-------|
| Self-attention (QKV from raw matrices) | 8-01 |
| Multi-head attention | 8-02 |
| Sinusoidal and learned position encodings | 8-03 |
| Transformer block (pre-norm, residual, FFN) | 8-04 |
| Full transformer (encoder-decoder, causal masking) | 8-05 |
| RMSNorm, SwiGLU, modern transformer recipe | 8-06 |
| Grouped-query attention (GQA), multi-query attention | 8-07 |
| Rotary position embeddings (RoPE) | 8-08 |
| Flash Attention algorithm and IO-complexity | 8-09 |
| KV cache, autoregressive inference loop | 8-10 |

---

## Cross-Module Ownership Warnings

- Self-attention (8-01) is the foundation for ALL of Modules 9-13, 17-18. Never re-teach the QKV mechanics.
- Flash Attention (8-09) is referenced in Module 16 and 17 but only taught here.

---

## Special Notes

- This is the MOST CRITICAL module — transformers are the backbone of everything in Modules 9-18.
- CS336 A1 requires implementing a full Transformer without nn.Linear. Follow that spirit.
