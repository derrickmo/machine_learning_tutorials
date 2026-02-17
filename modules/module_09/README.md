# Module 09 — Advanced Computer Vision

## Introduction

This module covers the frontier of computer vision, moving beyond the CNN foundations of Module 06 to encompass interpretability, object detection, Vision Transformers, self-supervised visual pretraining, and specialized applications like OCR and video understanding. Students will implement Grad-CAM for model interpretability, build anchor box and NMS pipelines for detection, construct a Vision Transformer (ViT) from scratch, and implement self-supervised methods (DINO, MAE) that learn powerful visual representations without labeled data. The module also covers practical tool-based workflows using YOLO and MediaPipe, image retrieval with FAISS, video temporal modeling, and document AI pipelines. Module 09 ties together everything from Modules 05, 06, and 08 -- combining CNN architectures, transfer learning, and transformer self-attention into a comprehensive treatment of modern computer vision that prepares students for multimodal learning in Module 12 and production deployment in Module 20.

**Folder:** `module_09_advanced_computer_vision/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 9-01 | Visualizing What CNNs Learn | Filter visualization; activation maximization; Grad-CAM from scratch — gradient-weighted class activation maps; saliency maps; occlusion sensitivity; interpreting what each layer captures | CIFAR-10 | ~5 min |
| 9-02 | Object Detection Fundamentals | Anchor boxes and aspect ratios; intersection-over-union (IoU); non-maximum suppression (NMS) from scratch; mean average precision (mAP) metric; two-stage vs one-stage detector concepts | VOCDetection | ~8 min |
| 9-03 | YOLO Object Detection | Single-pass detection philosophy; grid cells and bounding box prediction; pretrained YOLOv8 inference pipeline; real-time detection on images and video; confidence thresholds and NMS tuning | VOCDetection | ~5 min |
| 9-04 | MediaPipe for Real-Time Vision | Hand landmark detection; pose estimation (33 keypoints); face mesh; gesture recognition pipeline; processing webcam/video streams; practical applications | Sample images | ~3 min |
| 9-05 | Vision Transformers (ViT) | Image as sequence of patches; patch embedding layer; [CLS] token for classification; position embeddings for patches; ViT architecture implementation; comparison with CNN on same task; DeiT distillation concepts | CIFAR-10 | ~15 min |
| 9-06 | Self-Supervised Vision — DINO, MAE & BEiT | DINO (self-distillation with no labels — teacher-student with EMA); MAE (masked autoencoder — mask 75% of patches, reconstruct); BEiT concepts; self-supervised pretraining vs supervised on downstream; distinct from contrastive methods (Module 12-04) — these are non-contrastive/masking-based | CIFAR-10 | ~15 min |
| 9-07 | Image Retrieval & Visual Similarity Search | Feature extraction from pretrained CNNs/ViTs; approximate nearest neighbor search (FAISS — HNSW, IVF); building a visual search pipeline; evaluation (recall@k, mAP); connection to text retrieval in Module 18 | CIFAR-10 | ~8 min |
| 9-08 | Video Understanding & Temporal Models | Video as sequence of frames; 3D convolutions (C3D, I3D concepts); video transformers (TimeSformer, ViViT concepts); temporal action detection; optical flow concepts; video classification pipeline | Synthetic video (CIFAR-10 frame sequences) | ~10 min |
| 9-09 | OCR & Document AI | Scene text detection concepts; CTC decoding concepts (from-scratch implementation in 12-09); TrOCR architecture walkthrough (ViT encoder + GPT decoder); document layout analysis; Donut (document understanding transformer) concepts; practical OCR pipeline on document images | Synthetic document images | ~8 min |
| 9-10 | CNN Training Deep Dive — CIFAR-100 | Train a full ResNet on CIFAR-100 applying all modern tricks: Mixup, CutMix, CutOut, cosine annealing, label smoothing, progressive resizing concepts; ablation study measuring impact of each technique; systematic comparison producing a leaderboard; connects Module 4 (augmentation), Module 5 (optimization), Module 6 (architectures) | CIFAR-100 | ~25 min |

---

## Topic Details

### 9-01: Visualizing What CNNs Learn
Students implement multiple CNN interpretability techniques from scratch: filter visualization (rendering the learned convolution kernels), activation maximization (optimizing input pixels to maximally activate specific neurons), Grad-CAM (computing gradient-weighted class activation maps that highlight which image regions drive a classification decision), saliency maps (input gradients), and occlusion sensitivity (systematically occluding image patches and measuring prediction changes). The Grad-CAM implementation is the centerpiece, requiring students to hook into intermediate layers, compute class-specific gradients, and produce heatmaps overlaid on the original image. These interpretability tools are essential for debugging CNN failures, building trust in model predictions, and understanding the hierarchical feature learning that occurs across layers -- from edges in early layers to object parts in later layers. This topic connects forward to the mechanistic interpretability methods in 10-10 and the model-agnostic explainability tools (SHAP, LIME) in 19-05.

### 9-02: Object Detection Fundamentals
This topic implements the core building blocks of object detection from scratch: generating anchor boxes with multiple scales and aspect ratios, computing intersection-over-union (IoU) between bounding boxes, performing non-maximum suppression (NMS) to eliminate redundant detections, and calculating mean average precision (mAP) across object categories. Students learn the conceptual difference between two-stage detectors (region proposal followed by classification, as in Faster R-CNN) and one-stage detectors (direct prediction on a grid, as in YOLO and SSD). The from-scratch NMS and IoU implementations ensure students understand the post-processing pipeline that converts raw model outputs into final detections. This topic is the prerequisite for the YOLO tool notebook (9-03) and provides the evaluation framework (mAP) used throughout detection-related tasks in the course.

### 9-03: YOLO Object Detection
Students work with the YOLO (You Only Look Once) detection framework using the ultralytics library, running pretrained YOLOv8 models for real-time inference on images and video. The notebook explains the single-pass detection philosophy -- dividing the image into a grid where each cell predicts bounding boxes and class probabilities simultaneously -- and how this differs from the two-stage approach. Students tune confidence thresholds and NMS parameters to balance precision and recall, and evaluate detection quality using the mAP metric from 9-02 on VOCDetection data. This is a tool-focused notebook (Category D) that emphasizes practical deployment rather than from-scratch implementation, giving students experience with production-quality detection systems. The YOLO workflow connects forward to real-time vision applications discussed in Module 19 and deployment pipelines in Module 20.

### 9-04: MediaPipe for Real-Time Vision
This tool-focused notebook introduces Google's MediaPipe framework for real-time pose estimation, hand landmark detection, face mesh generation, and gesture recognition. Students build processing pipelines that extract 33-point body pose keypoints, 21-point hand landmarks, and 468-point face mesh coordinates from images and video streams. The notebook demonstrates practical gesture recognition by mapping hand landmark geometry to discrete gesture classes. MediaPipe represents the production end of the computer vision pipeline -- highly optimized, pre-trained models designed for real-time inference on edge devices -- contrasting with the from-scratch approach used elsewhere in the course. This topic gives students experience with deployed vision APIs and connects forward to the human-computer interaction and domain applications discussed in Module 19.

### 9-05: Vision Transformers (ViT)
Students implement the Vision Transformer architecture from scratch, converting images into sequences of patches via a patch embedding layer, adding a learnable [CLS] token for classification, applying position embeddings, and processing the patch sequence through transformer encoder blocks built in Module 08. The notebook trains ViT on CIFAR-10 and compares it head-to-head against a CNN (ResNet) on the same task, analyzing the accuracy-efficiency tradeoffs and the data requirements where ViTs outperform CNNs. DeiT (Data-efficient Image Transformers) distillation concepts are introduced to address ViT's data hunger on small datasets. This is the pivotal topic that demonstrates transformers are not just for text -- they achieve state-of-the-art results on vision tasks when given sufficient data or proper training recipes. ViT is the foundation for self-supervised vision (9-06), image retrieval (9-07), video transformers (9-08), and multimodal models (Module 12).

### 9-06: Self-Supervised Vision -- DINO, MAE & BEiT
This topic implements two powerful self-supervised visual pretraining methods that learn strong representations without any labeled data. DINO (self-distillation with no labels) uses a teacher-student framework with exponential moving average (EMA) where the student learns to match the teacher's output distributions across different augmented views. MAE (masked autoencoder) masks 75% of image patches and trains an encoder-decoder to reconstruct the missing pixels, forcing the encoder to learn meaningful visual features. BEiT is covered conceptually as a visual BERT analog. Students compare self-supervised pretraining versus supervised pretraining on downstream classification, demonstrating that self-supervised methods can match or exceed supervised baselines. These are explicitly non-contrastive/masking-based methods -- contrastive self-supervised learning (SimCLR) is covered separately in 12-04 with different theoretical foundations.

### 9-07: Image Retrieval & Visual Similarity Search
Students build a complete visual search pipeline: extracting feature vectors from pretrained CNNs and ViTs, indexing them using Facebook's FAISS library with approximate nearest neighbor algorithms (HNSW for graph-based search, IVF for cluster-based search), and retrieving similar images given a query. The notebook evaluates retrieval quality using recall@k and mean average precision, and analyzes the accuracy-speed tradeoffs of different FAISS index types. Students understand why exact nearest neighbor search is infeasible at scale and how approximate methods achieve sub-linear query time through clever data structures. This topic directly connects to dense passage retrieval and text retrieval in Module 18 (RAG), where the same FAISS-based vector search pipeline is applied to text embeddings for retrieval-augmented generation.

### 9-08: Video Understanding & Temporal Models
Students extend vision models from static images to video by treating video as a sequence of frames and implementing temporal modeling approaches. The notebook covers 3D convolutions (C3D, I3D) that apply convolution kernels across both spatial and temporal dimensions, building on the Conv3d concepts from 6-09. Video transformers (TimeSformer with divided space-time attention, ViViT with tubelet embeddings) are covered conceptually, showing how the ViT architecture from 9-05 extends to video by adding a temporal dimension. Students build a video classification pipeline on synthetic frame sequences and implement temporal action detection concepts. Optical flow is introduced as a complementary motion representation. This topic bridges the gap between single-frame analysis and temporal reasoning, connecting forward to multimodal video-text understanding in Module 12.

### 9-09: OCR & Document AI
This topic covers the end-to-end pipeline for extracting text and structure from document images, covering CTC (Connectionist Temporal Classification) decoding conceptually as the sequence prediction mechanism for recognizing variable-length text without requiring per-character alignment (the from-scratch CTC implementation is in Module 12-09). Students walk through the TrOCR architecture (ViT encoder + GPT decoder) that applies the encoder-decoder transformer pattern from 8-05 to document image recognition. The notebook also covers document layout analysis (detecting text blocks, tables, figures) and introduces Donut (Document Understanding Transformer) as an end-to-end approach that skips explicit OCR in favor of direct document understanding. The practical OCR pipeline on synthetic document images demonstrates real-world applicability. This topic connects forward to document processing applications in Module 19 and the multimodal document understanding systems in Module 12.

### 9-10: CNN Training Deep Dive -- CIFAR-100
This capstone notebook trains a full ResNet on CIFAR-100 (100 fine-grained classes) while systematically applying and ablating every modern training technique: Mixup (interpolating training examples), CutMix (cutting and pasting image patches), CutOut (randomly masking image regions), cosine annealing learning rate schedule, label smoothing, and progressive resizing. Students conduct a rigorous ablation study that measures the individual and combined impact of each technique, producing a leaderboard that ranks configurations by test accuracy. This integrates knowledge from Module 4 (data augmentation, evaluation), Module 5 (optimization and scheduling), and Module 6 (CNN architectures) into a single comprehensive training pipeline. The ablation methodology and systematic experimental design demonstrated here are the practices that distinguish rigorous ML practitioners from those who simply run default configurations.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 09-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 09-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 09-03 | D — Tool/Library | `TEMPLATE_TOOL.ipynb` |
| 09-04 | D — Tool/Library | `TEMPLATE_TOOL.ipynb` |
| 09-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 09-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 09-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 09-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 09-09 | D — Tool/Library | `TEMPLATE_TOOL.ipynb` |
| 09-10 | E — Capstone/Integration | `TEMPLATE_CAPSTONE.ipynb` |

---

## Module-Specific Packages

- `ultralytics` — YOLO detection (9-03)
- `mediapipe` — real-time vision (9-04)
- `faiss-cpu` — ANN search (9-07)

---

## Datasets

- CIFAR-10 (CNN interpretability, adversarial)
- VOCDetection (object detection, 9-02/9-03)
- CIFAR-100 (9-10 deep dive)
- Sample images (9-04 MediaPipe demos)
- Synthetic video/document images (9-08, 9-09)

---

## Prerequisites Chain

- **09-01:** Requires 6-03, 6-04
- **09-02:** Requires 6-03
- **09-03:** Requires 9-02
- **09-04:** Requires 6-03
- **09-05:** Requires 8-02, 8-03
- **09-06:** Requires 9-05 | Recommended: 11-01 (autoencoders — enhances understanding but not required)
- **09-07:** Requires 6-04, 9-05
- **09-08:** Requires 6-09, 9-05
- **09-09:** Requires 9-05, 8-05
- **09-10:** Requires 6-03, 6-04, 5-09, 4-04

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 9 — Advanced Computer Vision
| Concept | Owner |
|---------|-------|
| Grad-CAM, saliency maps | 9-01 |
| Anchor boxes, NMS, mAP | 9-02 |
| YOLO detection | 9-03 |
| MediaPipe real-time vision | 9-04 |
| Vision Transformers (ViT) | 9-05 |
| Self-supervised vision (DINO, MAE, BEiT) | 9-06 |
| Image retrieval, visual similarity search (FAISS) | 9-07 |
| Video understanding (3D conv, video transformers) | 9-08 |
| OCR, document AI (TrOCR, Donut) | 9-09 |
| Advanced CNN training (CIFAR-100 with modern tricks) | 9-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ CTC decoding (9-09 vs 12-09): Module 9-09 covers CTC decoding conceptually in the OCR context. Module 12-09 owns the from-scratch CTC implementation for speech-to-text. 9-09 must NOT re-implement CTC — reference 12-09.
- ⚠️ Label smoothing (9-10) and EMA (9-06): These techniques are used in Module 9 before their formal introduction in 15-04. Implement them contextually with brief inline explanations, noting that Module 15-04 provides the full treatment.

---

## Special Notes

- Expanded from 5 → 10 topics. 9-10 is a training deep dive on CIFAR-100 applying all modern tricks (Mixup, CutMix, cosine annealing, etc.).
- Self-supervised (9-06) is DINO/MAE (non-contrastive). SimCLR (contrastive) is in 12-04.
