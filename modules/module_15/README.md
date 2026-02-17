# Module 15 — Advanced PyTorch Internals

## Introduction

Module 15 takes students deep into the internals of PyTorch, covering the advanced APIs, extension points, and systems-level knowledge needed to build production-quality training infrastructure and debug complex issues that arise in real-world deep learning projects. This module is important because while earlier modules use PyTorch as a tool for implementing algorithms, practitioners need to understand what happens beneath the nn.Module surface -- autograd mechanics, data pipeline optimization, JIT compilation, graph transformations, GPU memory management, and distributed communication primitives. After completing this module, students will be able to write custom autograd functions, build advanced data pipelines with custom samplers and collate functions, apply TorchScript and torch.fx transformations, diagnose GPU memory issues, and construct a mini training framework with callbacks and checkpointing. Within the 20-module curriculum, Module 15 bridges the gap between algorithm implementation (Modules 5-14) and the distributed training and deployment infrastructure covered in Module 16 (distributed systems) and Module 20 (MLOps), giving students the systems fluency to scale their models to production.

**Folder:** `module_15_advanced_pytorch_internals/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 15-01 | Custom Autograd Functions & Hooks | torch.autograd.Function — custom forward and backward; forward hooks for activation extraction; backward hooks for gradient modification; gradient accumulation hooks; practical use cases (e.g., custom gradient clipping) | FashionMNIST | ~5 min |
| 15-02 | Advanced Data Pipelines | Custom Sampler and BatchSampler; weighted random sampling for imbalanced data; collate_fn for variable-length sequences (padding, packing); IterableDataset for streaming large datasets; pin_memory and prefetching | FashionMNIST | ~5 min |
| 15-03 | Advanced nn.Module Patterns | Dynamic architecture (conditional layers, skip connections); parameter groups for different learning rates; state_dict surgery (rename, remove, reshape keys); lazy modules; module hooks for logging and debugging | FashionMNIST | ~5 min |
| 15-04 | Custom Loss Functions & Training Utilities | Focal loss, label smoothing cross-entropy, contrastive losses implemented from scratch; custom metrics with running averages; gradient clipping utilities; EMA (exponential moving average) of model weights | CIFAR-10 | ~8 min |
| 15-05 | TorchScript & JIT Compilation | Tracing vs scripting — when to use which; exporting models for C++ runtime; limitations (dynamic control flow, unsupported ops); debugging traced models; comparison with torch.export (newer API) | FashionMNIST | ~5 min |
| 15-06 | torch.fx & Graph Transformations | Symbolic tracing of nn.Module to graph IR; graph inspection and manipulation; custom transformation passes (e.g., fuse conv+bn); quantization-aware transforms; connection to torch.compile internals | FashionMNIST | ~5 min |
| 15-07 | CUDA Memory Management & GPU Internals | GPU memory hierarchy (registers, shared memory, L2 cache, HBM); PyTorch caching allocator; memory fragmentation and OOM debugging; pinned memory and async transfers; CUDA streams for compute/transfer overlap (CS336 Lecture 5) | CIFAR-10 | ~5 min |
| 15-08 | Distributed Primitives & Communication | Point-to-point (send/recv) and collective ops (all_reduce, broadcast, all_gather, reduce_scatter); NCCL backend; ring all-reduce algorithm walkthrough; process groups; building blocks for DDP/FSDP in Module 16 | CIFAR-10 | ~5 min |
| 15-09 | Debugging PyTorch Models | autograd.detect_anomaly for NaN/Inf tracking; numerical gradient checking; anomalous gradient detection; common pitfalls (in-place ops breaking autograd, detached tensors, wrong device); systematic debugging workflow; memory leak detection with gc and torch.cuda.memory_stats | CIFAR-10 | ~5 min |
| 15-10 | Building a Mini Training Framework | Trainer class with callback system (on_epoch_start, on_batch_end, etc.); automatic checkpointing and resumption; logging (loss, lr, grad norm per step); EarlyStopping callback; understanding what Lightning/Fabric abstract away | FashionMNIST | ~8 min |

---

## Topic Details

### 15-01: Custom Autograd Functions & Hooks
Students will extend PyTorch's automatic differentiation engine by implementing custom `torch.autograd.Function` subclasses with explicit forward and backward methods, enabling operations that cannot be expressed as compositions of existing autograd-tracked operations or that require custom gradient computation for numerical stability. The notebook covers forward hooks for extracting intermediate activations (useful for feature visualization and transfer learning), backward hooks for modifying gradients during backpropagation (useful for gradient-based debugging and custom regularization), and gradient accumulation hooks for implementing techniques like custom gradient clipping at the parameter level. Practical use cases demonstrate when custom autograd functions are necessary versus when standard PyTorch operations suffice. This topic provides the low-level autograd understanding that underpins debugging (15-09) and the custom training utilities (15-04) later in the module.

### 15-02: Advanced Data Pipelines
This topic covers the advanced data loading and preprocessing APIs that become essential when working with real-world datasets that are imbalanced, variable-length, or too large to fit in memory. Students implement custom `Sampler` and `BatchSampler` classes for controlling how indices are drawn, including weighted random sampling that oversamples minority classes to address class imbalance. A custom `collate_fn` is built for variable-length sequences, implementing both padding to a maximum length and pack_padded_sequence for efficient RNN processing. `IterableDataset` is implemented for streaming data from disk or network sources when the full dataset cannot be loaded into memory. Performance optimizations including `pin_memory` for faster GPU transfers and prefetching with `num_workers` are benchmarked, giving students the toolkit for building data pipelines that are not the training bottleneck.

### 15-03: Advanced nn.Module Patterns
Students will learn advanced patterns for building and manipulating PyTorch modules beyond the basic Sequential and subclassing patterns from Module 5. The notebook covers dynamic architectures with conditional layers that activate based on input properties and skip connections that route information past intermediate layers, parameter groups that assign different learning rates and weight decay to different parts of the model (critical for fine-tuning), and `state_dict` surgery for renaming, removing, and reshaping checkpoint keys when loading weights across architecturally different models. Lazy modules that infer their shapes from the first forward pass are implemented for rapid prototyping, and module hooks for logging and debugging enable systematic inspection of intermediate computations. These patterns are essential for the model manipulation tasks in TorchScript export (15-05) and the training framework (15-10).

### 15-04: Custom Loss Functions & Training Utilities
This topic implements a collection of advanced loss functions and training utilities from scratch that are frequently needed in practice but not part of PyTorch's default library. Students build focal loss (which down-weights easy examples to focus training on hard cases, critical for object detection), label smoothing cross-entropy (which prevents overconfident predictions), and contrastive losses for metric learning. Custom metrics with efficient running averages are implemented for tracking training progress without storing full prediction histories. Gradient clipping utilities implement both norm-based and value-based clipping with proper gradient scaling. Exponential Moving Average (EMA) of model weights is built from scratch, maintaining a shadow copy of parameters updated via exponential smoothing for better generalization at test time. These utilities recur throughout later modules and the training framework capstone (15-10).

### 15-05: TorchScript & JIT Compilation
Students will export PyTorch models for optimized inference using TorchScript, learning both tracing (which records operations from a sample forward pass) and scripting (which compiles Python code directly to TorchScript IR). The notebook systematically explores when to use each approach: tracing for models with static control flow and scripting for models with data-dependent branching, loops, or conditionals. Limitations are covered hands-on -- unsupported operations, dynamic control flow that tracing misses, and type annotation requirements for scripting -- along with debugging techniques for diagnosing export failures. Models are exported to the TorchScript format for deployment in C++ runtimes. A comparison with `torch.export` (the newer export API) shows the evolving landscape of model serialization, connecting forward to deployment concerns in Module 20.

### 15-06: torch.fx & Graph Transformations
This topic introduces `torch.fx`, PyTorch's framework for symbolic tracing and graph-level transformations of nn.Module models. Students perform symbolic tracing to convert a model into a graph intermediate representation (IR) consisting of placeholder inputs, call_function nodes, call_module nodes, and output, then inspect and manipulate this graph programmatically. Custom transformation passes are implemented, including fusing Conv2d and BatchNorm layers into a single optimized convolution -- a common optimization for inference that torch.fx automates. Quantization-aware transforms demonstrate how graph manipulation enables automatic insertion of quantization nodes. The connection to `torch.compile` internals is drawn, showing how torch.fx serves as the foundation for PyTorch's modern compilation pipeline, giving students insight into what happens when they call `torch.compile(model)`.

### 15-07: CUDA Memory Management & GPU Internals
Students will develop a deep understanding of GPU memory hierarchy and PyTorch's memory management system, essential knowledge for training large models without running out of memory. The notebook covers the GPU memory hierarchy from fastest/smallest (registers, shared memory) through L2 cache to slowest/largest (HBM high-bandwidth memory), and explains how these levels affect kernel performance and data movement patterns. PyTorch's caching allocator is examined in detail: how it pools freed GPU memory for reuse rather than returning it to CUDA, why `torch.cuda.memory_allocated()` differs from `nvidia-smi`, and how memory fragmentation leads to OOM errors even when total free memory appears sufficient. Pinned memory for asynchronous CPU-to-GPU transfers and CUDA streams for overlapping compute and data transfer are implemented following CS336 Lecture 5 principles.

### 15-08: Distributed Primitives & Communication
This topic covers the low-level distributed communication operations that form the building blocks for distributed training systems like DDP and FSDP in Module 16. Students implement point-to-point operations (send/recv between specific processes) and collective operations (all_reduce for gradient synchronization, broadcast for parameter initialization, all_gather for collecting tensors from all processes, and reduce_scatter for FSDP-style sharded communication) using PyTorch's distributed package. The ring all-reduce algorithm is walked through step by step, showing how the bandwidth-optimal algorithm distributes gradient synchronization across all workers in O(1) time relative to the number of workers. NCCL backend specifics and process group management are covered, giving students the concrete understanding needed to implement and debug the distributed training strategies in Module 16.

### 15-09: Debugging PyTorch Models
Students will build a systematic debugging workflow for diagnosing the most common and frustrating failures in PyTorch model training. The notebook covers `autograd.detect_anomaly` for tracking down NaN and Inf values to the exact operation that produced them, numerical gradient checking for verifying custom backward implementations against finite differences, and anomalous gradient detection for identifying exploding or vanishing gradients in specific layers. Common pitfalls are demonstrated and diagnosed: in-place operations silently breaking the autograd graph, `.detach()` accidentally stopping gradient flow, tensors on the wrong device causing cryptic errors, and incorrect broadcasting leading to shape mismatches. Memory leak detection using Python's garbage collector and `torch.cuda.memory_stats` rounds out the debugging toolkit, connecting to the memory management concepts from 15-07.

### 15-10: Building a Mini Training Framework
The capstone topic brings together all the module's concepts by building a complete mini training framework from scratch -- a simplified version of what PyTorch Lightning and Fabric provide. Students implement a `Trainer` class with a callback system supporting hooks at every training lifecycle point (on_epoch_start, on_batch_end, on_backward, etc.), automatic checkpointing and seamless training resumption from the latest checkpoint, and per-step logging of loss, learning rate, and gradient norms. An EarlyStopping callback is implemented as a concrete example of the callback pattern. By building the abstraction layer themselves, students understand exactly what Lightning/Fabric abstract away: device management, gradient accumulation, mixed precision, logging, and checkpoint management. This topic synthesizes autograd hooks (15-01), data pipelines (15-02), nn.Module patterns (15-03), custom losses and EMA (15-04), and debugging tools (15-09) into a cohesive training infrastructure.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 15-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 15-10 | E — Capstone/Integration | `TEMPLATE_CAPSTONE.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- FashionMNIST (framework demos)
- CIFAR-10 (profiling and optimization)

---

## Prerequisites Chain

- **15-01:** Requires 5-06, 5-07
- **15-02:** Requires 1-05, 5-07
- **15-03:** Requires 5-07
- **15-04:** Requires 5-04, 5-07
- **15-05:** Requires 5-07, 15-03
- **15-06:** Requires 15-05, 5-07
- **15-07:** Requires 5-07
- **15-08:** Requires 5-07
- **15-09:** Requires 5-07, 15-07
- **15-10:** Requires 15-01 through 15-09

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 15 — Advanced PyTorch Internals
| Concept | Owner |
|---------|-------|
| Custom autograd functions and hooks | 15-01 |
| Advanced data pipelines (samplers, collate_fn) | 15-02 |
| Advanced nn.Module patterns (state_dict surgery) | 15-03 |
| Custom loss functions, EMA, gradient utilities | 15-04 |
| TorchScript and JIT compilation | 15-05 |
| torch.fx graph transformations | 15-06 |
| CUDA memory management, GPU internals | 15-07 |
| Distributed communication primitives (all_reduce, NCCL) | 15-08 |
| PyTorch debugging (detect_anomaly, gradient checking, memory leaks) | 15-09 |
| Mini training framework (Trainer pattern) | 15-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ Focal loss (15-04): Module 5-04 owns the canonical focal loss implementation. 15-04 should reference 5-04 for focal loss and focus on contrastive losses, label smoothing CE, and EMA as its unique contributions.
- ⚠️ Gradient clipping (15-04): Module 5-06 introduces gradient clipping as a concept. 15-04 implements reusable gradient clipping utilities. Module 16-06 covers gradient clipping strategies for training stability. Scope: 15-04 = utility implementation, 16-06 = production strategies.
- ⚠️ EMA (15-04): EMA is used contextually in 9-06 (DINO) and 12-04 (BYOL) before being formally taught here. 15-04 is the canonical owner — those earlier notebooks implement EMA inline with brief explanations.
- ⚠️ CUDA memory (15-07) vs memory optimization (16-03): 15-07 teaches GPU memory hierarchy and the PyTorch caching allocator. 16-03 teaches gradient checkpointing and activation memory optimization. Distinct scopes: 15-07 = understanding memory, 16-03 = optimizing memory usage.
- ⚠️ Debugging (15-09) vs training stability (16-06): 15-09 covers PyTorch-specific debugging (detect_anomaly, gradient checking, in-place ops). 16-06 covers training stability at scale (loss spikes, NaN recovery, warmup schedules). Distinct scopes: 15-09 = finding bugs, 16-06 = preventing divergence.

---

## Special Notes

No special notes for this module.
