# Module 06 — Convolutional Neural Networks

## Introduction

This module covers the design, implementation, and analysis of convolutional neural networks, the architecture family that revolutionized image recognition and remains foundational to modern computer vision. Starting from the limitations of fully connected networks on spatial data, students will build convolution operations from raw tensors, implement landmark architectures from LeNet to ResNet, and learn the transfer learning workflow that underpins nearly all practical vision systems. By the end of this module, students will have implemented depthwise separable convolutions, U-Net encoder-decoder architectures, semantic and instance segmentation pipelines, neural style transfer, and adversarial robustness techniques -- all from scratch. Module 06 serves as the bridge between the general neural network foundations of Module 05 and the advanced vision topics in Module 09, while its transfer learning workflow (6-04) is reused across Modules 9, 10, 12, and 13.

**Folder:** `module_06_convolutional_neural_networks/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 6-01 | Fully Connected Networks for Images | Flatten → FC layers on MNIST/CIFAR; limitations (parameter explosion, no spatial invariance); motivation for weight sharing and local connectivity; FC baseline for comparison | CIFAR-10 | ~5 min |
| 6-02 | Convolution from Scratch | 2D convolution operation implemented without nn.Conv2d; filters/kernels, padding (same/valid), stride, dilation; pooling (max, average, global average); output size formula; feature map visualization | CIFAR-10 | ~8 min |
| 6-03 | CNN Architectures — LeNet to ResNet | LeNet-5 (1998), AlexNet (2012), VGG (2014), GoogLeNet/Inception (2014), ResNet (2015) — skip connections and residual learning; receptive field computation; architectural trends and design principles | CIFAR-10 | ~15 min |
| 6-04 | Transfer Learning & Fine-Tuning | Loading pretrained models (torchvision); frozen feature extraction vs full fine-tuning; discriminative learning rates (lower for early layers); progressive resizing; when to fine-tune vs retrain | CIFAR-10 | ~10 min |
| 6-05 | U-Net & Encoder-Decoder Architecture | Encoder path (downsampling), decoder path (upsampling), skip connections; bottleneck as latent representation; application to segmentation; connection to autoencoders (Module 11) and diffusion U-Nets (Module 11) | VOCSegmentation | ~12 min |
| 6-06 | Depthwise Separable Convolutions & Efficient Architectures | Depthwise separable convolution from scratch; MobileNet V1/V2 (inverted residual blocks); EfficientNet compound scaling; channel and spatial attention (SE-Net, CBAM); compute and parameter comparison vs standard convolutions | CIFAR-10 | ~10 min |
| 6-07 | Semantic & Instance Segmentation | Fully Convolutional Networks (FCN); atrous/dilated convolutions (DeepLab concepts); Mask R-CNN for instance segmentation concepts; panoptic segmentation overview; mIoU metric; extends U-Net foundation from 6-05 | VOCSegmentation | ~15 min |
| 6-08 | Neural Style Transfer | Gram matrices as style representation; content and style loss from VGG features; optimization-based generation (Gatys et al.); fast style transfer concepts; style-content tradeoff; implementation from scratch | CIFAR-10 (content images) | ~10 min |
| 6-09 | 1D & 3D Convolutions | Conv1d for sequence and time-series data; temporal convolutions (TCN — causal conv, dilated stacks, WaveNet-style); Conv3d for video; bridging CNNs to non-image domains; comparison with RNNs for sequences | SPEECHCOMMANDS, Synthetic time series | ~8 min |
| 6-10 | Adversarial Examples & Robustness | FGSM and PGD attacks implemented from scratch; adversarial training as defense; robustness evaluation on CIFAR-10; transferability of adversarial examples; certified defense concepts; why DNNs are vulnerable | CIFAR-10 | ~10 min |

---

## Topic Details

### 6-01: Fully Connected Networks for Images
This topic establishes the baseline by applying standard fully connected (dense) networks to image classification on CIFAR-10, exposing why naive flattening of pixel grids fails at scale. Students will implement an FC network from scratch and quantify its parameter count, demonstrating the explosion in weights when spatial dimensions grow and the complete loss of translational invariance. The notebook provides the concrete motivation for weight sharing and local connectivity -- the two principles that define convolutional layers. This baseline model serves as the comparison target throughout the module, giving students an empirical reference for measuring how much convolutional architectures improve over dense networks. The topic directly connects forward to 6-02, where students replace dense layers with convolution operations built from raw tensors.

### 6-02: Convolution from Scratch
Students implement the 2D convolution operation entirely from scratch using raw tensor operations -- no `nn.Conv2d` -- covering kernel sliding, same and valid padding, stride, and dilation. The notebook also builds max pooling, average pooling, and global average pooling from first principles, then derives and verifies the output size formula for arbitrary combinations of padding, stride, and dilation. Feature maps are visualized at each layer to build intuition for how learned filters detect edges, textures, and higher-level patterns. This from-scratch implementation is the foundation for every subsequent topic in Module 06 and ensures students understand exactly what happens inside `nn.Conv2d` before relying on it. The output size formula and feature map concepts recur throughout Modules 9 (ViT patch embeddings), 11 (generative models), and 12 (multimodal architectures).

### 6-03: CNN Architectures -- LeNet to ResNet
This topic traces the evolution of CNN architectures from LeNet-5 (1998) through AlexNet (2012), VGG (2014), GoogLeNet/Inception (2014), to ResNet (2015), implementing key components of each and analyzing the design decisions that drove accuracy improvements on ImageNet. The central concept is ResNet's skip connections and residual learning, which students implement from scratch and validate by showing how residual paths mitigate vanishing gradients in deep networks. Students also compute receptive fields for each architecture and extract general design principles -- increasing channels while decreasing spatial dimensions, batch normalization placement, and 1x1 convolution for dimensionality reduction. This topic is the prerequisite for nearly everything else in Module 06 and beyond: transfer learning (6-04), efficient architectures (6-06), style transfer (6-08), adversarial robustness (6-10), and the CIFAR-100 deep dive (9-10).

### 6-04: Transfer Learning & Fine-Tuning
Students learn the complete transfer learning workflow using pretrained torchvision models: loading ImageNet weights, replacing the classification head, choosing between frozen feature extraction and full fine-tuning, and applying discriminative learning rates that update early layers more slowly than later ones. The notebook implements progressive resizing as a practical technique and provides decision guidelines for when to fine-tune versus retrain from scratch based on dataset size and domain similarity. This is the single notebook that owns the transfer learning concept for the entire 200-topic course -- Modules 9, 10, 12, and 13 all reference this workflow but never re-teach it. The techniques introduced here are essential for practical ML work, where training from scratch is rarely feasible and leveraging pretrained representations is the default approach.

### 6-05: U-Net & Encoder-Decoder Architecture
This topic implements the U-Net architecture from scratch, covering the encoder path (successive downsampling with convolution and pooling), the decoder path (upsampling with transposed convolutions), and the skip connections that concatenate encoder features with decoder features at each resolution level. Students will understand the bottleneck as a compressed latent representation and apply the architecture to pixel-wise segmentation on VOCSegmentation. The encoder-decoder pattern introduced here is one of the most reusable architectural motifs in deep learning -- it reappears in autoencoders (Module 11), diffusion model U-Nets (Module 11), and sequence-to-sequence models (Modules 7-8). This topic also provides the foundation for the more advanced segmentation methods covered in 6-07.

### 6-06: Depthwise Separable Convolutions & Efficient Architectures
Students implement depthwise separable convolutions from raw tensor operations, factoring standard convolutions into depthwise (per-channel spatial filtering) and pointwise (1x1 cross-channel mixing) components, then measure the compute and parameter savings. The notebook builds MobileNet V1 and V2 (with inverted residual blocks and linear bottlenecks) and explains EfficientNet's compound scaling method that jointly optimizes depth, width, and resolution. Channel attention (SE-Net) and spatial attention (CBAM) mechanisms are implemented from scratch and integrated into the efficient architectures. These lightweight architectures are critical for deploying models on mobile devices and edge hardware, and the attention mechanisms introduced here foreshadow the self-attention concepts that dominate Modules 8-10.

### 6-07: Semantic & Instance Segmentation
Building on the U-Net foundation from 6-05, this topic covers the full landscape of image segmentation: Fully Convolutional Networks (FCN), atrous/dilated convolutions as used in DeepLab, Mask R-CNN for instance-level segmentation, and the concept of panoptic segmentation that unifies semantic and instance approaches. Students implement the mIoU (mean Intersection over Union) metric from scratch as the standard evaluation measure for segmentation tasks. The notebook applies these techniques to VOCSegmentation and provides side-by-side comparisons of segmentation quality across architectures. Segmentation is one of the most important practical applications of CNNs, used in autonomous driving, medical imaging, and satellite analysis, and these techniques connect forward to the object detection pipeline in Module 9.

### 6-08: Neural Style Transfer
Students implement the Gatys et al. neural style transfer algorithm from scratch, using VGG feature maps to extract content representations (from deep layers) and Gram matrices to capture style representations (correlations across feature channels). The notebook constructs the combined content-style loss function and optimizes a generated image through gradient descent on pixel values -- a fundamentally different optimization target than training network weights. Students experiment with the style-content tradeoff by adjusting loss weights and visualize how different VGG layers capture different levels of abstraction. This topic was placed in Module 06 (rather than Module 12) because it is fundamentally a CNN feature-space technique that depends on understanding VGG internals, and it connects forward to generative models in Module 11 that also optimize in image space.

### 6-09: 1D & 3D Convolutions
This topic extends convolution beyond 2D images, implementing Conv1d for sequence and time-series data and Conv3d for volumetric and video data. The central implementation is a Temporal Convolutional Network (TCN) with causal convolutions (no future leakage), dilated stacks for exponentially growing receptive fields, and residual connections -- following the WaveNet architecture pattern. Students compare TCN performance against RNNs on sequence modeling tasks, establishing when convolutional approaches outperform recurrent ones. Conv3d is applied to synthetic video data, demonstrating how the spatial convolution concept naturally extends to the temporal dimension. This topic bridges CNNs to non-image domains and connects forward to video understanding in 9-08 and the RNN/transformer comparisons in Modules 7-8.

### 6-10: Adversarial Examples & Robustness
Students implement FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) adversarial attacks from scratch, generating imperceptible perturbations that cause trained classifiers to misclassify with high confidence. The notebook demonstrates adversarial training as a defense -- augmenting the training set with adversarial examples -- and evaluates robustness on CIFAR-10 using accuracy under attack as the metric. Students investigate the transferability of adversarial examples across different model architectures and explore why deep neural networks are fundamentally vulnerable to these perturbations (linear hypothesis). Certified defense concepts provide a theoretical perspective on provable robustness bounds. This topic is essential for understanding model reliability in safety-critical applications and connects to the broader theme of model evaluation and trustworthiness covered in Module 4.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 06-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-07 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 06-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- CIFAR-10 (6-01, 6-02, 6-03, 6-04, 6-06, 6-08, 6-10)
- VOCSegmentation (6-05, 6-07)
- SPEECHCOMMANDS (6-09)
- Synthetic time series (6-09)

---

## Prerequisites Chain

- **06-01:** Requires 5-07, 5-10
- **06-02:** Requires 6-01
- **06-03:** Requires 6-02
- **06-04:** Requires 6-03
- **06-05:** Requires 6-03
- **06-06:** Requires 6-03
- **06-07:** Requires 6-05
- **06-08:** Requires 6-03
- **06-09:** Requires 6-02
- **06-10:** Requires 6-03, 5-06

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 6 — Convolutional Neural Networks
| Concept | Owner |
|---------|-------|
| 2D convolution from scratch (filters, padding, stride) | 6-02 |
| CNN architectures (LeNet → ResNet, skip connections) | 6-03 |
| Transfer learning workflow | 6-04 |
| U-Net, encoder-decoder architecture | 6-05 |
| Depthwise separable convolutions, MobileNet, EfficientNet | 6-06 |
| Semantic/instance segmentation (FCN, Mask R-CNN) | 6-07 |
| Neural style transfer (Gram matrices, VGG features) | 6-08 |
| 1D/3D convolutions, temporal convolutions | 6-09 |
| Adversarial examples and robustness (FGSM, PGD) | 6-10 |

---

## Cross-Module Ownership Warnings

- Transfer learning (6-04) is used in Modules 9, 10, 12, 13. Only 6-04 teaches the workflow.
- Neural style transfer (6-08) was moved FROM Module 12 -- it is a CNN feature-space technique.

---

## Special Notes

- Expanded from 5 → 10 topics. Includes practical deployment architectures (MobileNet, EfficientNet) and adversarial robustness.
- Neural style transfer (6-08) was moved here from Module 12.
