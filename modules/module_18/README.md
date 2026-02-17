# Module 18 — RAG & Agentic AI Systems

## Introduction

Module 18 brings together everything students have learned about language models, embeddings, and structured output to build the retrieval-augmented generation (RAG) and agentic AI systems that represent the current frontier of applied AI. Starting from vector search fundamentals and progressing through chunking strategies, advanced retrieval techniques, and full agent loops with tool use, this module teaches students to build AI systems that can reason over external knowledge, take actions in the world, and coordinate multiple specialized agents. Every component is built from scratch without external API keys, using the small GPT model from Module 10 to demonstrate architectural patterns clearly. After completing this module, students will be able to build end-to-end RAG pipelines, implement ReAct-style agent loops, design multi-agent orchestration systems, and evaluate these systems rigorously — skills that directly map to production AI engineering roles. Module 18 synthesizes knowledge from Modules 8 (Transformers), 10 (NLP), 12 (multimodal), and 17 (LLM systems) into practical, deployable architectures.

**Folder:** `module_18_rag_and_agentic_systems/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 18-01 | Embeddings & Vector Stores from Scratch | Sentence embedding generation (mean pooling, [CLS] token); cosine similarity and dot-product search; brute-force index in NumPy; approximate nearest neighbor concepts (HNSW, IVF); FAISS integration | WikiText-2, AG_NEWS | ~5 min |
| 18-02 | Chunking Strategies & Retrieval | Fixed-size, sentence-level, semantic chunking; chunk overlap and context window management; BM25 keyword retrieval from scratch; dense retrieval with bi-encoders; hybrid retrieval (BM25 + dense score fusion) | AG_NEWS | ~5 min |
| 18-03 | Advanced RAG — Query Transformation & Reranking | HyDE (hypothetical document embeddings); query expansion and decomposition; cross-encoder reranking (more accurate but slower than bi-encoder); multi-hop retrieval for complex questions; contextual compression | AG_NEWS | ~8 min |
| 18-04 | RAG Pipeline End-to-End | Document ingestion → chunking → embedding → indexing → retrieval → context injection → generation; using our Module 10 language model; prompt engineering for RAG (cite sources, handle no-answer) | Custom document corpus | ~10 min |
| 18-05 | RAG Evaluation & Quality Metrics | Faithfulness (is the answer grounded in retrieved context?); relevance (did we retrieve the right chunks?); answer correctness; hallucination detection; RAGAS-style automated evaluation pipeline | Custom document corpus | ~5 min |
| 18-06 | Agent Loops, Tool Use & Planning | ReAct pattern (Reason + Act); function calling dispatch; structured action parsing (JSON output → tool execution → observation); task decomposition and chain-of-thought planning; agent memory types (conversation buffer, summarization, entity memory, persistent state); MCP protocol for tool integration; multi-turn agent with persistent state; error handling and retry logic | Synthetic tool-use tasks | ~8 min |
| 18-07 | Multi-Agent Orchestration Patterns | Role-based agent design (researcher, coder, reviewer roles); sequential and hierarchical orchestration (CrewAI-style patterns); state machine workflows (LangGraph-style); inter-agent communication protocols; supervisor/worker patterns; when multi-agent beats single-agent; implement a simple multi-agent system from scratch | Synthetic multi-agent tasks | ~8 min |
| 18-08 | Voice Agents & Conversational AI Pipelines | STT → LLM reasoning → TTS pipeline architecture; real-time streaming considerations; turn-taking and interruption handling; voice activity detection concepts; latency budget breakdown (STT ~200ms, LLM ~500ms, TTS ~200ms); using open-source STT/TTS models; connection to 12-09 (STT/TTS foundations) | SPEECHCOMMANDS | ~8 min |
| 18-09 | Guardrails, Evaluation & Agent Debugging | Input validation and content filtering; output schema enforcement (Pydantic models); safety checks and refusal detection; success rate and task completion metrics; trajectory analysis (reasoning trace inspection); failure mode taxonomy (tool errors, reasoning loops, hallucination); cost-per-task profiling; latency breakdown | Synthetic agent traces | ~5 min |
| 18-10 | Building a Domain-Specific AI Assistant | End-to-end capstone: document corpus ingestion → retrieval pipeline → agent reasoning → tool use → multi-agent coordination → voice interface option → evaluation; combining all Module 18 concepts into a complete system | Custom document corpus | ~15 min |

---

## Topic Details

### 18-01: Embeddings & Vector Stores from Scratch
Students will generate sentence embeddings using mean pooling and [CLS] token extraction from a pretrained encoder, then build a brute-force vector search index in pure NumPy using cosine similarity and dot-product scoring. The notebook progresses from exact search to approximate nearest neighbor (ANN) concepts — HNSW graph-based indexing and IVF cluster-based indexing — explaining why exact search becomes impractical at scale and how ANN trades recall for speed. Students will integrate FAISS for efficient similarity search and benchmark query latency and recall across different index types. This topic provides the retrieval foundation for the entire RAG pipeline built in 18-02 through 18-05, and the embedding skills connect back to the representation learning covered in Module 10.

### 18-02: Chunking Strategies & Retrieval
This notebook addresses the critical preprocessing step in RAG: how to split documents into chunks that are meaningful for retrieval. Students will implement fixed-size chunking, sentence-level chunking, and semantic chunking (splitting at topic boundaries detected by embedding similarity drops), comparing retrieval quality across strategies. The notebook builds BM25 keyword retrieval from scratch (including TF-IDF scoring and inverted index construction), implements dense retrieval with bi-encoder models, and combines both into a hybrid retrieval system using reciprocal rank fusion. Understanding chunk size, overlap, and retrieval method tradeoffs is essential for building RAG systems that retrieve the right context, directly feeding into the advanced techniques in 18-03.

### 18-03: Advanced RAG — Query Transformation & Reranking
Students will implement advanced retrieval techniques that go beyond simple query-to-document matching. HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to the query, then retrieves documents similar to that answer — often outperforming direct query embedding. The notebook covers query expansion and decomposition for complex multi-part questions, cross-encoder reranking (which scores query-document pairs jointly for higher accuracy than bi-encoder retrieval), and multi-hop retrieval that chains multiple retrieval steps for questions requiring information synthesis. Contextual compression reduces retrieved chunk length to only the relevant portions. These techniques significantly improve RAG quality and are integrated into the full pipeline in 18-04.

### 18-04: RAG Pipeline End-to-End
This notebook assembles all retrieval components from 18-01 through 18-03 into a complete RAG pipeline: document ingestion, chunking, embedding, indexing, retrieval (hybrid with reranking), context injection into the prompt, and generation using the Module 10 language model. Students will implement prompt engineering techniques specific to RAG — instructing the model to cite retrieved sources, handle cases where no relevant context was found, and distinguish between supported and unsupported claims. The pipeline demonstrates the full information flow from raw documents to generated answers, providing the system that will be evaluated in 18-05 and extended with agent capabilities in 18-06.

### 18-05: RAG Evaluation & Quality Metrics
Students will build a comprehensive RAG evaluation pipeline implementing the key metrics: faithfulness (is the generated answer grounded in the retrieved context, or does it hallucinate?), relevance (did the retrieval stage find the right chunks?), and answer correctness (does the final answer match the ground truth?). The notebook implements RAGAS-style automated evaluation that uses the language model itself to assess these dimensions, along with heuristic-based hallucination detection. Students will evaluate the pipeline built in 18-04 across different configurations and learn how to diagnose whether retrieval or generation is the bottleneck when quality is poor. These evaluation skills are essential for iterating on RAG systems in production.

### 18-06: Agent Loops, Tool Use & Planning
This notebook builds a complete ReAct (Reason + Act) agent from scratch — a system that interleaves chain-of-thought reasoning with tool invocation in a loop until a task is complete. Students will implement the function calling dispatch mechanism (parsing structured JSON actions from model output, executing the corresponding tool, and feeding observations back into the context), build multiple memory types (conversation buffer, summarization memory, entity memory, and persistent state), and implement the MCP (Model Context Protocol) pattern for standardized tool integration. Error handling and retry logic ensure the agent recovers gracefully from tool failures and reasoning loops. This topic transforms a language model from a text generator into an autonomous problem solver, directly enabling the multi-agent systems in 18-07.

### 18-07: Multi-Agent Orchestration Patterns
Students will design and implement multi-agent systems where specialized agents with distinct roles (researcher, coder, reviewer) collaborate to solve complex tasks. The notebook covers sequential orchestration (agents execute in a fixed pipeline), hierarchical orchestration with supervisor/worker patterns (CrewAI-style), and state machine workflows where agent transitions depend on task state (LangGraph-style). Students will implement inter-agent communication protocols and analyze when multi-agent systems outperform single-agent approaches (task decomposability, specialization benefits) versus when they add unnecessary complexity. The complete multi-agent system built from scratch demonstrates these patterns concretely and connects forward to the capstone assistant in 18-10.

### 18-08: Voice Agents & Conversational AI Pipelines
This notebook extends the agent framework into voice interaction by building a complete STT (speech-to-text) to LLM reasoning to TTS (text-to-speech) pipeline. Students will implement the full architecture — audio input processing, transcription, agent reasoning, and speech synthesis — with careful attention to the latency budget (STT ~200ms, LLM ~500ms, TTS ~200ms for acceptable real-time performance). The notebook covers real-time streaming considerations, turn-taking and interruption handling, and voice activity detection concepts. Open-source STT/TTS models are used to keep the notebook self-contained. This topic builds on the speech and audio foundations from 12-09 while applying the agent architecture from 18-06, demonstrating how multimodal capabilities extend agentic systems.

### 18-09: Guardrails, Evaluation & Agent Debugging
Students will implement a comprehensive quality assurance framework for agent systems, covering input validation and content filtering (blocking harmful or off-topic requests), output schema enforcement using Pydantic models, and safety checks including refusal detection. The notebook builds evaluation metrics specific to agents — success rate, task completion rate, and cost-per-task profiling — and implements trajectory analysis tools that let developers inspect the full reasoning trace to diagnose failures. A failure mode taxonomy categorizes common agent failures (tool execution errors, infinite reasoning loops, hallucinated tool names) with specific mitigation strategies for each. These debugging and evaluation skills are essential for making agent systems reliable enough for production deployment.

### 18-10: Building a Domain-Specific AI Assistant
The capstone notebook integrates every component from Module 18 into a complete domain-specific AI assistant. Students will build the full system end-to-end: document corpus ingestion and indexing (18-01/18-02), retrieval pipeline with advanced techniques (18-03), RAG generation (18-04), agent reasoning with tool use (18-06), multi-agent coordination for complex queries (18-07), optional voice interface (18-08), and comprehensive evaluation (18-05/18-09). The notebook demonstrates how to scope an assistant to a specific domain, configure retrieval and generation for domain-specific needs, and measure system quality holistically. This capstone represents the culmination of the entire NLP and LLM track (Modules 7, 8, 10, 17, 18) and produces a system architecture that students can adapt for real-world applications.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 18-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 18-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 18-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 18-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 18-05 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 18-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 18-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 18-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 18-09 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 18-10 | E — Capstone/Integration | `TEMPLATE_CAPSTONE.ipynb` |

---

## Module-Specific Packages

- `faiss-cpu` — ANN search (18-01, 18-02)

---

## Datasets

- WikiText-2 (18-01)
- AG_NEWS (18-01, 18-02, 18-03)
- Custom document corpus (18-04, 18-05, 18-10)
- Synthetic tool-use tasks (18-06)
- Synthetic multi-agent tasks (18-07)
- SPEECHCOMMANDS (18-08)
- Synthetic agent traces (18-09)

---

## Prerequisites Chain

- **18-01:** Requires 1-06, 10-02
- **18-02:** Requires 18-01, 2-08
- **18-03:** Requires 18-02, 10-02
- **18-04:** Requires 18-01 through 18-03
- **18-05:** Requires 18-04
- **18-06:** Requires 17-09, 10-09
- **18-07:** Requires 18-06
- **18-08:** Requires 12-09, 18-06
- **18-09:** Requires 18-06, 18-07
- **18-10:** Requires 18-01 through 18-09

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 18 — RAG and Agentic Systems
| Concept | Owner |
|---------|-------|
| Embeddings, vector stores, ANN search | 18-01 |
| Chunking strategies, BM25, dense retrieval | 18-02 |
| Advanced RAG (HyDE, reranking) | 18-03 |
| RAG pipeline end-to-end | 18-04 |
| RAG evaluation metrics | 18-05 |
| Agent loops, tool use, ReAct, planning, memory, MCP | 18-06 |
| Multi-agent orchestration patterns | 18-07 |
| Voice agents, STT→LLM→TTS pipeline | 18-08 |
| Guardrails, agent evaluation and debugging | 18-09 |
| Domain-specific AI assistant (capstone) | 18-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ Agent loops (18-06) consolidated agent tool use + multi-step planning into one topic.
- ⚠️ Guardrails + agent eval (18-09) consolidated into one topic. Both are agent quality assurance.

---

## Special Notes

- 18-06 and 18-07 (old) consolidated into 18-06. 18-08 and 18-09 (old) consolidated into 18-09.
- Freed slots used for multi-agent orchestration (18-07) and voice agents (18-08).
- **Feasibility note:** Topics 18-06 (agents), 18-07 (multi-agent), and 18-08 (voice agents) require a functioning language model. Since the course prohibits external API keys, these notebooks use the small GPT model trained in Module 10-01. The goal is to demonstrate the architectural patterns and reasoning loops clearly, not to achieve production-quality agent behavior. Set appropriate expectations in the markdown narrative.
