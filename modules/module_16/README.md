# Module 16 — Training Optimization & Distributed Systems

## Introduction

Module 16 addresses the critical gap between training a model that works and training a model efficiently at scale. Students will learn the full stack of performance optimization techniques that modern ML engineers rely on daily — from mixed-precision arithmetic and compiler-driven graph optimization to distributed training across multiple GPUs. This module is essential because even a well-architected model is useless if it takes weeks to train when it could take hours; these techniques reduce wall-clock time, memory consumption, and cost simultaneously. After completing this module, students will be able to profile training bottlenecks, apply mixed precision and gradient checkpointing, launch distributed training jobs with DDP and FSDP, and combine all of these optimizations into a production-grade training pipeline. Module 16 builds directly on the PyTorch internals covered in Module 15 and provides the systems foundation needed for training large language models in Module 17.

**Folder:** `module_16_training_optimization_and_distributed/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 16-01 | Mixed Precision Training (torch.amp) | FP32/FP16/BF16 number formats and dynamic range; GradScaler for FP16 loss scaling; when BF16 is preferable (no scaler needed, larger dynamic range); speed and memory benchmarks; numerical stability edge cases | CIFAR-10 | ~8 min |
| 16-02 | torch.compile & Compiler Optimization | Dynamo (graph capture via bytecode analysis) + Inductor (optimized code generation); compilation modes (default, reduce-overhead, max-autotune); dynamic shapes handling; speedup measurement methodology; Triton kernel generation by the compiler | CIFAR-10 | ~8 min |
| 16-03 | Memory Optimization & Gradient Checkpointing | Activation memory analysis (why it dominates for large models); torch.utils.checkpoint — recompute vs store tradeoff; selective checkpointing strategies; memory profiling with torch.cuda.memory_stats; CPU offloading concepts | CIFAR-10 | ~8 min |
| 16-04 | Gradient Accumulation & Large Batch Training | Simulating large effective batch sizes without more GPU memory; gradient sync frequency; linear scaling rule for learning rate; LARS and LAMB optimizers for very large batches; batch size warmup strategies | CIFAR-10 | ~8 min |
| 16-05 | Data Loading at Scale | Multi-worker DataLoader optimization (num_workers tuning); prefetch_factor and persistent_workers; memory-mapped datasets (numpy memmap, safetensors); WebDataset/streaming patterns; avoiding data-starved GPU training | CIFAR-10 | ~5 min |
| 16-06 | Training Stability & Debugging Divergence | Loss spike taxonomy (data corruption, numerical overflow, LR too high); gradient norm monitoring and clipping strategies (max-norm, value clipping); warmup schedules (linear, cosine); NaN/Inf recovery; z-loss regularization | CIFAR-10 | ~8 min |
| 16-07 | Profiling & Bottleneck Analysis | End-to-end wall-clock benchmarking (CS336 A2 methodology); torch.profiler with Chrome trace viewer and TensorBoard; CUDA event timing; CPU-GPU overlap analysis; identifying data loading vs compute vs communication bottlenecks; roofline model concepts; flame graphs (CS336 A2 benchmarking harness) | CIFAR-10 | ~5 min |
| 16-08 | Distributed Data Parallel (DDP) | DistributedDataParallel — gradient all-reduce synchronization after backward; launch scripts (torchrun, torch.distributed.launch); DistributedSampler for data sharding; fault tolerance; scaling efficiency measurement (CS336 A2 core deliverable) | WikiText-2 | ~10 min |
| 16-09 | Model Parallelism — FSDP & ZeRO | FSDP — parameter sharding, gradient sharding, optimizer state sharding (maps to CS336 A2 optimizer sharding); ZeRO Stage 1/2/3 comparison; tensor parallelism concepts; pipeline parallelism (GPipe); DeepSpeed integration overview | WikiText-2 | ~10 min |
| 16-10 | End-to-End Optimized Training Pipeline | Combining all optimizations: mixed precision + compile + DDP + gradient accumulation + optimized data loading; train a real model with each optimization added incrementally; measure speedup at each stage; best practices checklist | CIFAR-10 | ~15 min |

---

## Topic Details

### 16-01: Mixed Precision Training (torch.amp)
Students will implement mixed-precision training from scratch using PyTorch's automatic mixed precision (AMP) module, gaining a deep understanding of how FP32, FP16, and BF16 number formats differ in dynamic range and precision. The notebook covers the mechanics of GradScaler — why FP16 training requires loss scaling to prevent gradient underflow, and why BF16's larger dynamic range eliminates this need. Students will build benchmarks comparing training speed and peak memory across all three precisions on a CNN trained on CIFAR-10, and will deliberately trigger numerical instability edge cases (overflow in FP16 normalization layers, underflow in small gradients) to understand when mixed precision fails. This topic is foundational for every subsequent optimization in the module and is a prerequisite for efficient LLM training in Module 17.

### 16-02: torch.compile & Compiler Optimization
This topic explores PyTorch's compiler stack — TorchDynamo for graph capture via Python bytecode analysis and TorchInductor for generating optimized GPU kernels. Students will compile a model under all three modes (default, reduce-overhead, max-autotune), measure wall-clock speedups rigorously, and examine the Triton kernels that the compiler generates to understand operator fusion and memory access optimization. The notebook also covers dynamic shapes handling, which is critical for NLP workloads with variable-length sequences. Understanding torch.compile is increasingly important as it becomes the default way to accelerate PyTorch models, and this knowledge directly applies to the optimized training pipeline in 16-10.

### 16-03: Memory Optimization & Gradient Checkpointing
Students will analyze why activation memory — not parameters or gradients — dominates GPU memory for large models, and implement gradient checkpointing using torch.utils.checkpoint to trade compute for memory. The notebook covers selective checkpointing strategies (which layers benefit most from checkpointing vs which are too cheap to bother), and students will use torch.cuda.memory_stats to profile peak memory before and after applying checkpointing. CPU offloading concepts are introduced as an additional memory reduction technique. This topic is essential for training models that would otherwise not fit in GPU memory, directly enabling the large-model training scenarios in 16-09 (FSDP) and Module 17.

### 16-04: Gradient Accumulation & Large Batch Training
This notebook teaches students to simulate large effective batch sizes by accumulating gradients over multiple forward-backward passes before calling optimizer.step(). Students will implement the linear scaling rule (scale learning rate proportionally with batch size) and understand why it breaks down for very large batches, motivating LARS and LAMB optimizers that they will implement from scratch. The notebook covers gradient sync frequency considerations for distributed settings and batch size warmup strategies. Gradient accumulation is a prerequisite for 16-08 (DDP), where understanding gradient synchronization timing becomes critical for distributed efficiency.

### 16-05: Data Loading at Scale
Students will systematically tune PyTorch's DataLoader parameters — num_workers, prefetch_factor, and persistent_workers — and measure how each affects GPU utilization by identifying data-loading bottlenecks. The notebook introduces memory-mapped datasets using numpy memmap and safetensors format for zero-copy tensor loading, as well as WebDataset streaming patterns for datasets too large to fit on disk. Students will build benchmarks showing how an under-tuned DataLoader can leave the GPU idle for the majority of training time. Efficient data loading is a prerequisite for the profiling work in 16-07 and the end-to-end pipeline in 16-10.

### 16-06: Training Stability & Debugging Divergence
This topic provides a systematic framework for diagnosing and fixing training failures — loss spikes, NaN gradients, and slow convergence. Students will implement gradient norm monitoring hooks, build gradient clipping (both max-norm and value clipping) from scratch, and construct warmup schedules (linear and cosine). The notebook includes a "failure lab" where students deliberately induce common failure modes (learning rate too high, numerical overflow, corrupted data batch) and practice recovery strategies including NaN/Inf detection and z-loss regularization. These debugging skills are critical for every training scenario in the remainder of the course, particularly the complex distributed setups in 16-08 and 16-09.

### 16-07: Profiling & Bottleneck Analysis
Students will learn to use torch.profiler with Chrome trace viewer and TensorBoard integration to produce detailed flame graphs of training iterations, identifying exactly where time is spent across data loading, forward pass, backward pass, and optimizer step. The notebook covers CUDA event timing for precise GPU measurement, CPU-GPU overlap analysis to detect synchronization stalls, and the roofline model for understanding whether a workload is compute-bound or memory-bound. This profiling methodology (based on the CS336 A2 benchmarking harness) gives students the diagnostic skills needed to know which optimization from 16-01 through 16-06 will have the biggest impact on their specific workload.

### 16-08: Distributed Data Parallel (DDP)
This notebook implements DistributedDataParallel training, covering the all-reduce algorithm that synchronizes gradients across processes after each backward pass. Students will write launch scripts using torchrun, configure DistributedSampler for correct data sharding across workers, and measure scaling efficiency (how close to linear speedup they achieve with 2 and 4 simulated processes). The notebook also covers fault tolerance — handling worker failures gracefully — and the relationship between gradient accumulation (from 16-04) and gradient synchronization frequency. DDP is the most common form of distributed training and is a prerequisite for understanding FSDP in 16-09.

### 16-09: Model Parallelism — FSDP & ZeRO
Students will implement FullyShardedDataParallel (FSDP) concepts, understanding how parameter sharding, gradient sharding, and optimizer state sharding each reduce per-GPU memory (mapping to ZeRO Stages 1, 2, and 3). The notebook compares the memory and communication tradeoffs of each sharding level and introduces tensor parallelism (splitting individual layers across GPUs) and pipeline parallelism (GPipe-style micro-batching). A DeepSpeed integration overview shows how these concepts map to a production framework. This topic completes the distributed training picture and is essential for understanding how models too large for a single GPU — like those in Module 17 — are trained in practice.

### 16-10: End-to-End Optimized Training Pipeline
The capstone notebook combines every optimization from the module into a single training pipeline, adding each technique incrementally — mixed precision, torch.compile, gradient accumulation, optimized data loading, gradient checkpointing, and DDP — and measuring the cumulative speedup at each stage. Students will train a real model on CIFAR-10 with this fully optimized pipeline and produce a summary table showing wall-clock time, peak memory, and throughput (samples/second) with and without each optimization. The notebook concludes with a best practices checklist that students can apply to any future training project, directly preparing them for the large-scale training scenarios in Modules 17 and beyond.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 16-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 16-10 | E — Capstone/Integration | `TEMPLATE_CAPSTONE.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- CIFAR-10 (optimization benchmarks)
- WikiText-2 (distributed training demos)

---

## Prerequisites Chain

- **16-01:** Requires 5-07, 15-07
- **16-02:** Requires 5-07, 15-06
- **16-03:** Requires 5-06, 15-07
- **16-04:** Requires 5-07, 5-09
- **16-05:** Requires 1-05, 15-02
- **16-06:** Requires 5-09, 5-10
- **16-07:** Requires 15-09
- **16-08:** Requires 15-08, 16-04
- **16-09:** Requires 16-08, 15-08
- **16-10:** Requires 16-01 through 16-09

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 16 — Training Optimization and Distributed Systems
| Concept | Owner |
|---------|-------|
| Mixed precision (FP16/BF16, GradScaler) | 16-01 |
| torch.compile (Dynamo + Inductor) | 16-02 |
| Gradient checkpointing | 16-03 |
| Gradient accumulation, large batch training | 16-04 |
| Data loading at scale | 16-05 |
| Training stability, gradient clipping, NaN recovery | 16-06 |
| Profiling and bottleneck analysis (torch.profiler, Chrome trace, roofline model) | 16-07 |
| Distributed Data Parallel (DDP) | 16-08 |
| FSDP, ZeRO stages, model parallelism | 16-09 |
| End-to-end optimized training pipeline | 16-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ Memory optimization (16-03) vs CUDA memory (15-07): 15-07 teaches GPU memory hierarchy and the caching allocator. 16-03 teaches gradient checkpointing and activation memory optimization as training techniques. Distinct scopes: 15-07 = understanding memory, 16-03 = optimizing memory for training.
- ⚠️ Training stability (16-06) vs debugging (15-09): 15-09 covers PyTorch-specific debugging tools (detect_anomaly, gradient checking). 16-06 covers training stability at scale (loss spikes, gradient clipping strategies, NaN recovery, warmup). Distinct scopes: 15-09 = finding bugs, 16-06 = preventing divergence.
- ⚠️ Gradient clipping (16-06) vs (15-04): 15-04 implements reusable gradient clipping utilities. 16-06 covers gradient clipping strategies in the context of training stability and large-scale optimization. Reference 15-04 for the utility implementations.
- ⚠️ Profiling (16-07): This is the sole owner of profiling and bottleneck analysis. Module 15-09 (debugging) does NOT cover profiling — it was explicitly scoped to debugging only.

---

## Special Notes

No special notes for this module.
