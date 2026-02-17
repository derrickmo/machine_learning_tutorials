# Module 07 — Recurrent Networks & NLP Foundations

## Introduction

This module covers the foundational techniques for processing sequential and textual data, spanning tokenization, word embeddings, recurrent architectures, and classical NLP tasks that remain essential knowledge for understanding modern language models. Students will build BPE tokenizers, Word2Vec embeddings, vanilla RNNs, LSTMs, GRUs, and sequence-to-sequence models with attention entirely from scratch, gaining deep understanding of how text is represented, encoded, and generated before the transformer era. By the end of this module, students will have implemented decoding strategies (beam search, top-k, nucleus sampling), classical language models with perplexity evaluation, dependency parsers, CRF sequence labelers, and contextual embeddings (ELMo) -- establishing the full pre-transformer NLP toolkit. Module 07 is the critical bridge between the neural network foundations of Module 05 and the transformer architectures of Module 08, and the concepts taught here (BPE tokenization, perplexity, attention mechanisms) are reused throughout Modules 8, 10, 13, and 17.

**Folder:** `module_07_recurrent_networks_and_nlp/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 7-01 | Tokenization — BPE, WordPiece & SentencePiece | Character, word, subword tokenization; byte-level BPE from scratch (CS336 A1 core deliverable — pre-tokenization, merges, encoding/decoding); WordPiece (BERT); SentencePiece unigram model concepts; tokenizer evaluation (fertility, coverage, OOV rate) | WikiText-2 | ~5 min |
| 7-02 | Word Vectors — Word2Vec, GloVe & FastText | Word2Vec CBOW and skip-gram with negative sampling from scratch; GloVe (co-occurrence matrix + weighted least squares); FastText (subword n-grams for OOV handling); evaluation: analogy tasks, similarity benchmarks; embedding visualization (PCA/t-SNE); embedding arithmetic | WikiText-2 | ~8 min |
| 7-03 | Recurrent Neural Networks from Scratch | Vanilla RNN forward pass implemented in NumPy; hidden state as memory; backpropagation through time (BPTT); vanishing gradient problem — mathematical analysis and visualization | Synthetic sequences | ~5 min |
| 7-04 | LSTMs & GRUs | LSTM cell — forget/input/output gates implemented from scratch; GRU — update/reset gates; gating as gradient highway; bidirectional RNNs; comparison on sequence modeling tasks | WikiText-2 | ~8 min |
| 7-05 | Sequence-to-Sequence with Attention | Encoder-decoder architecture for translation; Bahdanau (additive) attention mechanism; attention weights visualization; information bottleneck problem → motivation for self-attention and transformers | Synthetic (translation pairs) | ~10 min |
| 7-06 | Text Generation & Decoding Strategies | Character-level and word-level language models; greedy decoding; temperature scaling; top-k and nucleus (top-p) sampling; beam search implementation; evaluation: perplexity (CS336 A1), BLEU, ROUGE | WikiText-2 | ~8 min |
| 7-07 | Classical Language Models & Perplexity | N-gram language models from scratch; Markov assumption; smoothing techniques (Laplace, Kneser-Ney concepts); perplexity derivation and computation; connection to neural LMs; why perplexity is the foundational metric for all of Module 10/17 | WikiText-2 | ~3 min |
| 7-08 | Dependency Parsing | Transition-based dependency parsing; arc-standard system (shift, left-arc, right-arc); neural transition parser (feed-forward network on stack/buffer features); UAS/LAS metrics; treebank data; connection to syntactic structure in NLP | Synthetic treebank | ~5 min |
| 7-09 | Sequence Labeling & CRFs | CRF layer for sequence labeling from scratch; Viterbi decoding; BiLSTM-CRF architecture; comparison with softmax-per-token independent classification; why structured prediction beats independent classification for sequential labels | CoNLL2000Chunking | ~8 min |
| 7-10 | Contextual Embeddings — ELMo & the Road to BERT | ELMo (bidirectional LSTM contextual embeddings); contextualized vs static embeddings; feature-based vs fine-tuning transfer; the conceptual bridge from RNNs to BERT; polysemy resolution through context | WikiText-2 | ~10 min |

---

## Topic Details

### 7-01: Tokenization -- BPE, WordPiece & SentencePiece
Students implement byte-level Byte Pair Encoding (BPE) entirely from scratch, covering pre-tokenization, iterative merge learning, encoding, and decoding -- following the CS336 Assignment 1 specification that requires this as a core deliverable. The notebook also covers WordPiece (used by BERT) and SentencePiece unigram model concepts, giving students a complete view of the three dominant subword tokenization strategies in modern NLP. Tokenizer quality is evaluated using fertility (tokens per word), vocabulary coverage, and out-of-vocabulary rate, building the analytical framework for understanding tokenizer design choices. This is the single most reused concept in the course: every language model in Modules 8, 10, 13, and 17 depends on the BPE tokenizer built here, making it the entry point for all text-based deep learning.

### 7-02: Word Vectors -- Word2Vec, GloVe & FastText
This topic implements the three foundational word embedding algorithms from scratch: Word2Vec (both CBOW and skip-gram with negative sampling), GloVe (co-occurrence matrix factorization with weighted least squares), and FastText (subword n-gram embeddings for handling out-of-vocabulary words). Students evaluate embeddings on analogy tasks (king - man + woman = queen) and similarity benchmarks, then visualize embedding spaces using PCA and t-SNE to build geometric intuition for how meaning is encoded as direction and distance in vector space. The embedding arithmetic operations demonstrate that word vectors capture semantic and syntactic relationships in their geometry. These static embeddings provide the contrast point for contextual embeddings in 7-10 and the transformer-based representations in Module 10, and the vector space concepts recur in retrieval systems (Module 18).

### 7-03: Recurrent Neural Networks from Scratch
Students implement a vanilla RNN forward pass entirely in NumPy, building the hidden state recurrence equation from raw matrix multiplications and understanding how the hidden state serves as a compressed memory of all previous time steps. The notebook derives and implements backpropagation through time (BPTT), unrolling the computation graph and computing gradients through the full sequence length. The critical insight is the vanishing gradient problem: students mathematically analyze how repeated multiplication by the recurrence weight matrix causes gradients to shrink exponentially, and visualize this decay across sequence positions. This mathematical grounding motivates the gated architectures (LSTM, GRU) in 7-04 and ultimately explains why attention mechanisms (7-05, Module 8) became necessary for long-range dependencies.

### 7-04: LSTMs & GRUs
Students implement the LSTM cell from scratch -- forget gate, input gate, output gate, and cell state -- understanding how gating creates a gradient highway that allows information to flow across many time steps without vanishing. The GRU is implemented as a simplified alternative with update and reset gates, and students compare both architectures on sequence modeling tasks to understand the accuracy-efficiency tradeoff. Bidirectional RNNs are built by running forward and backward LSTMs and concatenating hidden states, enabling models to use both past and future context. The LSTM and GRU are the workhorses of pre-transformer sequence modeling and remain important for understanding architectural evolution; they are direct prerequisites for seq2seq (7-05), text generation (7-06), dependency parsing (7-08), BiLSTM-CRF (7-09), and ELMo (7-10).

### 7-05: Sequence-to-Sequence with Attention
This topic implements the encoder-decoder architecture for sequence-to-sequence tasks (machine translation) and introduces the Bahdanau additive attention mechanism that allows the decoder to selectively focus on different encoder hidden states at each decoding step. Students build the attention module from scratch, compute attention weight distributions, and visualize alignment matrices showing which source words the model attends to when generating each target word. The information bottleneck problem -- where a fixed-length encoder hidden state must compress an entire input sequence -- is demonstrated empirically as the motivation for attention. This notebook is the conceptual bridge to Module 8: the attention mechanism introduced here evolves into self-attention (8-01), and the encoder-decoder structure becomes the full transformer (8-05).

### 7-06: Text Generation & Decoding Strategies
Students implement a complete text generation pipeline with multiple decoding strategies: greedy decoding, temperature-scaled softmax, top-k sampling, nucleus (top-p) sampling, and beam search. Each strategy is built from scratch and compared on generation quality, diversity, and coherence, giving students practical understanding of the quality-diversity tradeoff in language generation. The notebook also implements perplexity as the primary evaluation metric for language models, along with BLEU and ROUGE for comparing generated text against references. These decoding strategies are used throughout Modules 10 (GPT generation), 13 (fine-tuned model inference), and 17 (LLM generation), making this the canonical reference for all text generation in the course.

### 7-07: Classical Language Models & Perplexity
This topic implements n-gram language models from scratch, covering the Markov assumption, maximum likelihood estimation of n-gram probabilities, and smoothing techniques (Laplace smoothing and Kneser-Ney concepts) that address the zero-probability problem for unseen n-grams. The central theoretical contribution is the derivation and computation of perplexity -- from information theory through cross-entropy to the exponentiated per-token loss -- establishing why perplexity is the foundational evaluation metric for all language models. Students connect the n-gram framework to neural language models by showing how a neural LM is simply a better probability estimator with shared parameters. This theoretical grounding is essential for Modules 10 and 17, where perplexity is the primary metric for evaluating GPT-style and LLM-scale language models.

### 7-08: Dependency Parsing
Students implement a transition-based dependency parser using the arc-standard system, building the shift, left-arc, and right-arc operations from scratch and learning to predict transitions using a feed-forward neural network that reads features from the stack and buffer. The notebook covers UAS (unlabeled attachment score) and LAS (labeled attachment score) as evaluation metrics, and uses synthetic treebank data to demonstrate how syntactic structure is recovered from linear word sequences. Dependency parsing reveals the hierarchical structure of natural language that flat sequence models miss, connecting to the broader theme of structured prediction. This topic provides linguistic foundations that enhance understanding of why attention patterns in transformers (Module 8) often align with syntactic dependencies.

### 7-09: Sequence Labeling & CRFs
This topic implements a Conditional Random Field (CRF) layer from scratch for sequence labeling, including the forward algorithm for computing the partition function and Viterbi decoding for finding the optimal label sequence. Students build the full BiLSTM-CRF architecture and compare it against naive softmax-per-token independent classification, demonstrating empirically why structured prediction (where the model considers label dependencies) outperforms treating each position independently. The BIO tagging scheme for chunking is applied to CoNLL2000Chunking data with entity-level F1 evaluation. This pre-transformer approach to sequence labeling provides the contrast point for the transformer-based NER system in 10-05, and the CRF's ability to model label transitions remains relevant even in modern architectures that add CRF layers on top of transformer encoders.

### 7-10: Contextual Embeddings -- ELMo & the Road to BERT
Students implement ELMo-style contextual embeddings using bidirectional LSTMs, where the same word receives different vector representations depending on its surrounding context -- solving the polysemy problem that static embeddings (7-02) cannot handle. The notebook contrasts feature-based transfer (extracting ELMo vectors as fixed features for downstream models) with fine-tuning transfer (updating the pretrained model's weights on the downstream task), establishing the two paradigms that define modern NLP. Students analyze how different layers of the bidirectional LSTM capture different linguistic properties -- syntax in lower layers, semantics in higher layers. This topic is the conceptual bridge from RNNs to BERT: it demonstrates that pretrained contextualized representations dramatically improve downstream task performance, motivating the shift to transformer-based pretraining in Module 10.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 07-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 07-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

- `sentencepiece` — SentencePiece tokenizer demonstration (7-01)

---

## Datasets

- WikiText-2 (7-01, 7-02, 7-04, 7-06, 7-07, 7-10)
- Synthetic sequences (7-03, 7-05)
- Synthetic treebank (7-08)
- CoNLL2000Chunking (7-09)

---

## Prerequisites Chain

- **07-01:** Requires 1-08
- **07-02:** Requires 7-01, 1-06
- **07-03:** Requires 5-06, 5-07
- **07-04:** Requires 7-03
- **07-05:** Requires 7-03, 7-04
- **07-06:** Requires 7-03, 7-04
- **07-07:** Requires 1-07, 1-08
- **07-08:** Requires 7-04, 5-07
- **07-09:** Requires 7-04, 1-07
- **07-10:** Requires 7-04, 7-02

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 7 — Recurrent Networks and NLP Foundations
| Concept | Owner |
|---------|-------|
| BPE tokenization from scratch, WordPiece, SentencePiece | 7-01 |
| Word2Vec (skip-gram, CBOW), GloVe, FastText | 7-02 |
| RNN from scratch, BPTT | 7-03 |
| LSTM, GRU gates | 7-04 |
| Seq2seq with Bahdanau attention | 7-05 |
| Beam search, top-k, nucleus sampling, perplexity | 7-06 |
| N-gram language models, smoothing, perplexity theory | 7-07 |
| Dependency parsing (transition-based, arc-standard) | 7-08 |
| CRF sequence labeling, Viterbi decoding, BiLSTM-CRF | 7-09 |
| ELMo, contextual embeddings | 7-10 |

---

## Cross-Module Ownership Warnings

- BPE tokenization (7-01) is used in Modules 8, 10, 13, 17. Only 7-01 builds it from scratch.
- Perplexity (7-06, 7-07) is used in Modules 10, 17. Theory is owned by 7-07; practical evaluation by 7-06.

---

## Special Notes

- Expanded from 5 → 10 topics. Original 7-01 was split into 7-01 (tokenization) and 7-02 (word vectors).
- CS224N dedicates 2 lectures + 2 assignments to word vectors alone — give them proper depth.
