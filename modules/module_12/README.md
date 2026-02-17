# Module 12 — Multimodal & Cross-Modal Learning

## Introduction

Module 12 explores the rapidly evolving field of multimodal and cross-modal learning, where models must process, align, and reason across different data modalities including images, text, audio, and speech. This module is essential in the learning journey because modern AI systems increasingly operate across modalities -- from CLIP-based zero-shot classifiers to vision-language models and speech systems -- and understanding how to bridge the representational gap between modalities is a core skill for ML/DL practitioners. After completing this module, students will be able to implement contrastive image-text pretraining (CLIP), build vision-language models for captioning and VQA, design multi-modal fusion architectures, and work with audio and speech representations including STT/TTS foundations. Within the broader curriculum, Module 12 builds on the vision foundations from Module 6, the transformer and attention mechanisms from Module 8, and the NLP models from Module 10, while feeding forward into fine-tuning workflows (Module 13), RAG and agentic systems with voice capabilities (Module 18), and domain applications (Module 19).

**Folder:** `module_12_multimodal_and_cross_modal_learning/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 12-01 | CLIP — Contrastive Image-Text Pretraining | Dual encoder architecture (image encoder + text encoder); contrastive loss (InfoNCE); implement a mini-CLIP from scratch; temperature parameter; zero-shot transfer capabilities; cross-modal retrieval via cosine similarity | CIFAR-10 | ~12 min |
| 12-02 | Zero-Shot & Few-Shot Classification | CLIP for zero-shot image classification via text prompts; prompt engineering for vision tasks; few-shot learning with prototypical networks; comparison with fine-tuned classifiers | CIFAR-10 | ~8 min |
| 12-03 | Siamese Networks & Metric Learning | Contrastive loss for pairs; triplet loss with hard negative mining; face verification pipeline; embedding space visualization; applications in image retrieval and similarity search | FashionMNIST | ~8 min |
| 12-04 | Contrastive Self-Supervised Learning — SimCLR & BYOL | SimCLR — positive/negative pairs, NT-Xent loss, augmentation pipeline from scratch; BYOL — non-contrastive, stop-gradient, teacher-student with EMA; representation quality evaluation on downstream tasks; connection to CLIP (12-01) | CIFAR-10 | ~12 min |
| 12-05 | Vision-Language Models & Image Captioning | Cross-modal attention (image features attend to text); encoder-decoder for captioning; BLIP/LLaVA architecture concepts; generating descriptions from images | Flickr8k (synthetic fallback) | ~12 min |
| 12-06 | Visual Question Answering | VQA task formulation; attention over image regions conditioned on question; multi-modal transformer for VQA; VQA evaluation metrics (accuracy, open-ended vs multiple-choice); connection to captioning (12-05) and CLIP (12-01) | Synthetic VQA pairs | ~10 min |
| 12-07 | Multi-Modal Fusion Architectures | Early fusion (concatenate features), late fusion (separate encoders → merge), cross-attention fusion; when to use which; Flamingo/LLaVA architecture patterns; bottleneck tokens (Perceiver concepts) | CIFAR-10, Synthetic text | ~10 min |
| 12-08 | Audio & Speech Representations | Audio loading (torchaudio); mel-spectrograms, MFCCs; wav2vec 2.0 (self-supervised speech embeddings); CLAP (audio-text contrastive learning); distinct from 19-07 (audio classification application) — this covers representation learning | SPEECHCOMMANDS | ~8 min |
| 12-09 | Speech-to-Text & Text-to-Speech Foundations | STT pipeline — CTC decoding, Whisper architecture concepts, open-source STT inference; TTS pipeline — Tacotron/VITS concepts, vocoder basics; latency considerations; provides modality foundations for voice agents in Module 18-08 | SPEECHCOMMANDS, YESNO | ~8 min |
| 12-10 | Multimodal Evaluation & Alignment Metrics | CLIPScore for image-text alignment; image captioning metrics (CIDEr, METEOR, SPICE); VQA accuracy; evaluating multimodal hallucination; cross-modal retrieval metrics (recall@k); mirrors evaluation depth in NLP (10-08) and LLM (17-10) | CIFAR-10, Synthetic multimodal | ~5 min |

---

## Topic Details

### 12-01: CLIP -- Contrastive Image-Text Pretraining
Students will implement a mini-CLIP model from scratch, building a dual encoder architecture with separate image and text encoders whose outputs are projected into a shared embedding space. The notebook covers the InfoNCE contrastive loss that pulls matching image-text pairs together while pushing non-matching pairs apart, along with the learned temperature parameter that controls the sharpness of the similarity distribution. Students will demonstrate zero-shot transfer by classifying images using only text prompts describing each class, and implement cross-modal retrieval via cosine similarity in the shared space. This foundational topic establishes the contrastive alignment paradigm used throughout the module, connecting forward to zero-shot classification (12-02), vision-language models (12-05), and multimodal evaluation via CLIPScore (12-10).

### 12-02: Zero-Shot & Few-Shot Classification
This topic demonstrates the power of aligned multimodal representations by using CLIP for zero-shot image classification, where no task-specific training data is needed -- only natural language descriptions of the target classes. Students will explore prompt engineering for vision tasks, discovering how different text prompts (e.g., "a photo of a {class}" vs just "{class}") dramatically affect zero-shot performance. Few-shot learning with prototypical networks is implemented from scratch, computing class prototypes from a handful of examples and classifying via nearest-centroid in embedding space. A systematic comparison with fine-tuned classifiers quantifies when zero-shot and few-shot approaches are competitive, helping students understand the practical tradeoffs between data efficiency and task-specific optimization.

### 12-03: Siamese Networks & Metric Learning
Students will build Siamese networks that learn embedding spaces where similar items are close and dissimilar items are far apart, implementing both contrastive loss for pairs and triplet loss with hard negative mining from scratch. The notebook constructs a face verification pipeline that determines whether two images depict the same identity by thresholding their embedding distance, demonstrating the practical workflow of metric learning systems. Embedding space visualization reveals how the network organizes inputs into meaningful clusters, and applications in image retrieval and similarity search show how learned metrics enable efficient nearest-neighbor lookups. This topic provides the metric learning foundations that generalize to CLIP's contrastive framework (12-01) and self-supervised methods (12-04).

### 12-04: Contrastive Self-Supervised Learning -- SimCLR & BYOL
This topic implements the two most influential self-supervised learning frameworks: SimCLR, which learns representations by contrasting augmented views of the same image against views of different images using the NT-Xent loss, and BYOL, which achieves competitive performance without negative pairs through a teacher-student architecture with exponential moving average updates and a stop-gradient operation. Students build the full data augmentation pipeline from scratch (random crops, color jitter, Gaussian blur) that is critical to SimCLR's success, and evaluate representation quality by training linear probes on frozen features for downstream classification. The connection to CLIP (12-01) is made explicit: SimCLR/BYOL align views within a single modality while CLIP aligns across modalities, but the underlying contrastive principle is shared.

### 12-05: Vision-Language Models & Image Captioning
Students will implement a vision-language model that generates natural language descriptions of images, building an encoder-decoder architecture where image features from a CNN or ViT encoder attend to text tokens through cross-modal attention. The notebook covers the full captioning pipeline: encoding images into spatial feature maps, generating captions token-by-token with teacher forcing during training and beam search or greedy decoding at inference. Architecture concepts from BLIP and LLaVA are discussed to connect the from-scratch implementation to state-of-the-art systems, showing how modern vision-language models scale the same cross-attention principle. This topic feeds directly into VQA (12-06) and fusion architectures (12-07), and establishes the image-to-text generation pattern used in multimodal evaluation (12-10).

### 12-06: Visual Question Answering
This topic formulates VQA as a multimodal reasoning task where the model must attend to relevant image regions conditioned on a natural language question and produce an answer. Students implement attention mechanisms that allow the question to selectively focus on spatial image features, and build a multi-modal transformer that fuses visual and textual information to predict answers. VQA evaluation metrics are covered, including accuracy scoring for both open-ended and multiple-choice formats, with discussion of the specific challenges in evaluating free-form visual reasoning. The notebook connects back to captioning (12-05) as a complementary vision-language task and to CLIP (12-01) as a shared representation backbone, showing how different multimodal tasks compose the same building blocks.

### 12-07: Multi-Modal Fusion Architectures
Students will implement and compare three fundamental fusion strategies -- early fusion (concatenating features before processing), late fusion (processing modalities independently then merging decisions), and cross-attention fusion (allowing modalities to attend to each other at intermediate layers). Through controlled experiments on the same data, the notebook reveals when each strategy excels: early fusion for tightly coupled modalities, late fusion when modalities are independently informative, and cross-attention for rich inter-modal reasoning. Architecture patterns from Flamingo and LLaVA are analyzed, along with bottleneck tokens inspired by the Perceiver that compress multimodal inputs into a fixed-size representation. This comparison topic synthesizes the fusion decisions implicit in earlier notebooks (12-05, 12-06) into an explicit architectural framework.

### 12-08: Audio & Speech Representations
This topic introduces audio as a new modality, covering the full pipeline from raw waveform loading with torchaudio to feature extraction via mel-spectrograms and MFCCs, and extending to modern self-supervised speech representations. Students implement mel-spectrogram computation from scratch to understand the time-frequency transform that makes audio amenable to CNN and transformer processing, and explore wav2vec 2.0's approach of learning speech embeddings through contrastive self-supervised pretraining on unlabeled audio. CLAP (audio-text contrastive learning) extends the CLIP paradigm to the audio domain, aligning audio and text representations in a shared space. This topic provides the representation foundations that Module 19-07 (audio classification) builds upon, while remaining distinct by focusing on representation learning rather than downstream application.

### 12-09: Speech-to-Text & Text-to-Speech Foundations
Students will implement the core pipelines for bidirectional speech-text conversion: speech-to-text (STT) using CTC decoding with a discussion of the Whisper architecture and open-source STT inference, and text-to-speech (TTS) covering Tacotron/VITS architecture concepts and vocoder basics for waveform synthesis. CTC decoding is implemented from scratch, showing how the Connectionist Temporal Classification loss handles alignment between variable-length audio and text without explicit alignment labels. Latency considerations for real-time applications are analyzed, connecting to the practical requirements of voice-enabled systems. This topic provides essential modality foundations that Module 18-08 (voice agents) directly depends on, establishing the STT/TTS building blocks needed for multimodal agentic AI systems.

### 12-10: Multimodal Evaluation & Alignment Metrics
This evaluation-focused topic provides a comprehensive toolkit for measuring multimodal model quality, implementing CLIPScore for image-text alignment, image captioning metrics (CIDEr, METEOR, SPICE), VQA accuracy scoring, and cross-modal retrieval metrics (recall@k). Students will learn to evaluate multimodal hallucination -- cases where models generate plausible-sounding but factually incorrect descriptions of images -- a critical failure mode in vision-language systems. The notebook mirrors the evaluation depth provided for NLP in Module 10-08 and for LLMs in Module 17-10, ensuring students can rigorously assess model quality across all modalities. By applying these metrics to outputs from earlier notebooks in the module, students develop a unified evaluation perspective that connects generation quality to measurable outcomes.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 12-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-07 | F — Comparison/Architecture | `TEMPLATE_COMPARISON.ipynb` |
| 12-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 12-10 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |

---

## Module-Specific Packages

- `openai-whisper` — STT pipeline (12-09)

---

## Datasets

- CIFAR-10 (12-01, 12-02, 12-04, 12-07, 12-10)
- FashionMNIST (12-03)
- Flickr8k or synthetic fallback (12-05)
- Synthetic VQA pairs (12-06)
- Synthetic text (12-07)
- SPEECHCOMMANDS (12-08)
- YESNO (12-09)
- Synthetic multimodal (12-10)

---

## Prerequisites Chain

- **12-01:** Requires 6-04, 8-02, 10-02
- **12-02:** Requires 12-01
- **12-03:** Requires 5-04, 5-07
- **12-04:** Requires 12-01, 11-01
- **12-05:** Requires 12-01, 8-04, 7-05
- **12-06:** Requires 12-05, 12-01
- **12-07:** Requires 12-01, 8-04
- **12-08:** Requires 1-06, 12-04
- **12-09:** Requires 12-08
- **12-10:** Requires 12-01 through 12-09

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 12 — Multimodal and Cross-Modal Learning
| Concept | Owner |
|---------|-------|
| CLIP (contrastive image-text pretraining) | 12-01 |
| Zero-shot and few-shot classification | 12-02 |
| Siamese networks, triplet loss | 12-03 |
| Contrastive self-supervised learning (SimCLR, BYOL) | 12-04 |
| Vision-language models, image captioning | 12-05 |
| Visual question answering (VQA) | 12-06 |
| Multi-modal fusion architectures (early/late/cross-attention) | 12-07 |
| Audio and speech representations (wav2vec, CLAP) | 12-08 |
| STT and TTS foundations (CTC, Whisper, vocoder) | 12-09 |
| Multimodal evaluation and alignment metrics | 12-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ SimCLR/BYOL (12-04) moved IN from Module 11. Distinct from DINO/MAE (9-06): contrastive vs non-contrastive.
- ⚠️ Audio representations (12-08) vs audio classification (19-07): 12-08 teaches representations; 19-07 is the application.
- ⚠️ Audio representations (12-08) owns mel-spectrograms, MFCCs, and wav2vec. Module 19-07 (audio classification) uses these features but must NOT re-teach them — reference 12-08 with a one-line comment and focus on the classification pipeline.
- ⚠️ CTC decoding (12-09) is the canonical from-scratch implementation. Module 9-09 (OCR) uses CTC conceptually but must reference 12-09 for the algorithm details.
- ⚠️ EMA (12-04, BYOL): EMA is used contextually in BYOL's teacher-student architecture. The general EMA technique is formally owned by 15-04 — implement inline with a brief explanation, noting 15-04 provides the full treatment.

---

## Special Notes

- Renamed from 'Multimodal & Representation Learning.' STT/TTS (12-09) enables voice agents in 18-08.
- Cross-modal retrieval was cut — folded into CLIP (12-01) and image retrieval (9-07).
