# Module 11 — Generative Deep Learning

## Introduction

Module 11 covers the full landscape of generative deep learning, taking students from classical autoencoders through modern diffusion models, normalizing flows, and autoregressive architectures. This module is critical in the learning journey because generative modeling represents one of the most active and impactful areas of modern AI, underpinning image synthesis, data augmentation, drug discovery, and creative applications. After completing this module, students will be able to implement every major generative paradigm from scratch using PyTorch -- including VAEs, GANs, DDPM diffusion, latent diffusion, normalizing flows, score-based models, and autoregressive generators -- and understand the mathematical foundations that unify them. Within the 20-module curriculum, Module 11 builds on the neural network foundations from Modules 5-6 and the attention mechanisms from Module 8, while providing essential generative building blocks that feed into multimodal learning (Module 12), fine-tuning pipelines (Module 13), and LLM systems (Module 17).

**Folder:** `module_11_generative_deep_learning/`

**GPU Required:** Yes (device cell mandatory)

---

## Topics

| # | Topic | Key Content | Dataset | Time |
|---|-------|------------|---------|------|
| 11-01 | Autoencoders | Encoder-decoder compression; bottleneck representation; reconstruction loss (MSE, BCE); denoising autoencoders; sparse autoencoders; latent space visualization | MNIST | ~5 min |
| 11-02 | Variational Autoencoders (VAEs) | Reparameterization trick derivation; KL divergence regularization; ELBO objective; latent space interpolation and traversal; posterior collapse problem; connection to Bayesian inference (Module 3); VQ-VAE concepts | MNIST | ~8 min |
| 11-03 | Generative Adversarial Networks (GANs) | Generator/discriminator minimax game; DCGAN architecture; mode collapse and training instability; Wasserstein distance and WGAN concepts; spectral normalization | FashionMNIST | ~10 min |
| 11-04 | Conditional Generation & Class-Guided Models | Conditional GANs (concatenating label embeddings); conditional VAEs; class-conditioned generation pipeline; connection to classifier-free guidance in 11-07 | FashionMNIST | ~10 min |
| 11-05 | Diffusion Models — DDPM from Scratch | Forward noising process (variance schedule); reverse denoising process; noise prediction network (U-Net backbone); DDPM training loop implementation; sampling and image generation; mathematical derivation of the loss; discrete diffusion (SEDD) as emerging alternative for language | CIFAR-10 | ~20 min |
| 11-06 | Latent Diffusion Models | VAE encoder → latent space → diffusion in latent space; cross-attention conditioning (text → image); mini Stable Diffusion architecture walkthrough; why latent space is more efficient than pixel space | CIFAR-10 | ~20 min |
| 11-07 | Classifier-Free Guidance & Advanced Sampling | CFG — training with and without conditioning, interpolation at inference; DDIM sampling (deterministic, fewer steps); sampling speed vs quality tradeoff; progressive distillation concepts; generative model evaluation: FID, Inception Score, precision/recall for generated samples, CLIP score | CIFAR-10 | ~15 min |
| 11-08 | Normalizing Flows | Invertible transforms and change of variables formula; coupling layers (RealNVP) from scratch; training via exact log-likelihood; comparison with VAEs (exact vs approximate likelihood); latent space visualization and sampling | Synthetic 2D, MNIST | ~8 min |
| 11-09 | Score Matching, Energy-Based Models & Flow Matching | Energy function formulation; score function (gradient of log-density); denoising score matching from scratch; Langevin dynamics for sampling; SDE/ODE continuous-time diffusion framework — unifying DDPM, NCSN, and score-based models; flow matching objective — simpler training than diffusion; Rectified Flow concepts; connection between score matching and diffusion models | Synthetic 2D, MNIST | ~10 min |
| 11-10 | Autoregressive Generative Models — MADE, PixelCNN & WaveNet | Autoregressive factorization of joint distributions; MADE (masked autoregressive for parallel training); PixelCNN (masked convolutions for images); WaveNet (dilated causal convolutions for audio); causal masking as the unifying principle; comparison with VAEs/flows/diffusion on same data | MNIST (PixelCNN), Synthetic audio (WaveNet) | ~12 min |

---

## Topic Details

### 11-01: Autoencoders
Students will implement the standard autoencoder architecture from scratch, building an encoder that compresses input data into a low-dimensional bottleneck representation and a decoder that reconstructs the original input. The notebook covers reconstruction losses (MSE and BCE), and extends the basic model to denoising autoencoders that learn robust features by reconstructing clean inputs from corrupted versions, as well as sparse autoencoders that encourage activation sparsity via L1 regularization. Students will visualize the learned latent space using dimensionality reduction techniques to understand how the encoder organizes data. This topic establishes the foundational encoder-decoder pattern that recurs throughout the module in VAEs (11-02), GANs (11-03), and the VAE component of latent diffusion (11-06).

### 11-02: Variational Autoencoders (VAEs)
This topic builds on the autoencoder foundation (11-01) by introducing probabilistic latent variables and the variational inference framework. Students will derive and implement the reparameterization trick that enables backpropagation through stochastic sampling, construct the Evidence Lower Bound (ELBO) objective combining reconstruction and KL divergence terms, and train a complete VAE from scratch. The notebook explores latent space interpolation and traversal to demonstrate the smoothness of the learned representation, and addresses the posterior collapse problem where the decoder ignores the latent code. VQ-VAE concepts are introduced as a bridge to discrete latent spaces used in modern architectures, and the connection to Bayesian inference from Module 3 is made explicit, grounding the generative framework in statistical theory.

### 11-03: Generative Adversarial Networks (GANs)
Students will implement the GAN framework from scratch, building both a generator network that produces synthetic samples and a discriminator network that distinguishes real from fake, training them in the minimax adversarial game. The notebook progresses from a basic fully-connected GAN to the DCGAN architecture with convolutional layers, batch normalization, and architectural best practices. Key failure modes are explored hands-on: mode collapse where the generator produces limited diversity, and training instability where one network dominates the other. Students learn about the Wasserstein distance as a more stable training objective and spectral normalization as a regularization technique, providing the mathematical and practical toolkit for understanding why GAN training is notoriously difficult and how modern approaches address these challenges.

### 11-04: Conditional Generation & Class-Guided Models
This topic extends both GANs and VAEs to conditional generation, where the model produces outputs corresponding to a specified class label or attribute. Students will implement conditional GANs by concatenating label embeddings to both generator inputs and discriminator inputs, and build conditional VAEs that condition both the prior and the encoder/decoder networks. A complete class-conditioned generation pipeline is constructed, demonstrating how to generate targeted outputs on demand. This notebook establishes the conditioning paradigm that scales to text-conditioned generation in latent diffusion (11-06) and connects directly to classifier-free guidance (11-07), where the model learns to generate both conditionally and unconditionally during training.

### 11-05: Diffusion Models -- DDPM from Scratch
This is the module's centerpiece topic, where students implement the full Denoising Diffusion Probabilistic Model pipeline from raw mathematics to working image generation. The notebook covers the forward noising process with a linear variance schedule that progressively corrupts data into Gaussian noise, then derives and implements the reverse denoising process using a U-Net noise prediction network trained with the simplified MSE loss. Students build the complete DDPM training loop and sampling procedure, generating images from pure noise. The mathematical derivation of the variational bound and its simplification to the noise prediction objective is worked through in detail. Discrete diffusion (SEDD) is introduced as an emerging alternative for language domains, connecting to the broader trend of applying diffusion beyond continuous data.

### 11-06: Latent Diffusion Models
Building directly on both the VAE (11-02) and DDPM (11-05), this topic implements diffusion in the compressed latent space of a pretrained VAE encoder rather than in pixel space, dramatically reducing computational cost. Students will construct the full latent diffusion pipeline: encoding images to latent representations, running the diffusion forward and reverse processes in latent space, and decoding back to pixel space. Cross-attention conditioning is implemented to enable text-to-image generation, where text embeddings attend to spatial latent features at each denoising step. A mini Stable Diffusion architecture walkthrough ties the from-scratch components to the real-world system, helping students understand why latent diffusion became the dominant paradigm for high-resolution image generation.

### 11-07: Classifier-Free Guidance & Advanced Sampling
This evaluation-focused topic covers the techniques that make diffusion models practical for high-quality conditional generation and fast inference. Students implement classifier-free guidance (CFG), training a single model with random conditioning dropout and interpolating between conditional and unconditional predictions at inference time to amplify the conditioning signal. DDIM sampling is implemented as a deterministic alternative to DDPM that achieves comparable quality in far fewer steps, and the speed-quality tradeoff is quantified empirically. The notebook also covers generative model evaluation metrics -- FID, Inception Score, precision/recall for generated samples, and CLIP score -- giving students a rigorous framework for comparing any generative models from this module or beyond.

### 11-08: Normalizing Flows
Students will implement normalizing flows as an alternative generative paradigm that provides exact log-likelihood computation, contrasting with the approximate inference of VAEs. The notebook derives the change of variables formula for invertible transformations and implements RealNVP coupling layers from scratch, where affine transforms are applied to alternating partitions of the input dimensions. Training via maximum likelihood is straightforward since the exact log-probability is tractable through the chain of invertible transforms and their Jacobian determinants. Students compare flows with VAEs on the same data, observing the tradeoff between exact likelihood and expressiveness, and visualize how the base Gaussian distribution is warped through successive layers into the target data distribution.

### 11-09: Score Matching, Energy-Based Models & Flow Matching
This topic unifies the theoretical foundations underlying multiple generative approaches by introducing the score function (gradient of the log-density) and energy-based formulations. Students implement denoising score matching from scratch, learning to estimate the score at multiple noise levels, and use Langevin dynamics to generate samples by following the learned score field. The SDE/ODE continuous-time framework is presented as a unifying perspective that connects DDPM, Noise-Conditional Score Networks (NCSN), and score-based models under one mathematical umbrella. Flow matching is introduced as a simpler training objective that directly regresses velocity fields, and Rectified Flow concepts show how to straighten the generative trajectories for faster sampling. This topic provides the theoretical depth to understand why diffusion works and where the field is heading.

### 11-10: Autoregressive Generative Models -- MADE, PixelCNN & WaveNet
The final topic covers autoregressive generation, where the joint distribution is factored into a product of conditionals and each output dimension is generated sequentially conditioned on all previous dimensions. Students implement MADE (Masked Autoencoder for Distribution Estimation) which uses masked weight matrices to enable parallel training while maintaining the autoregressive property, PixelCNN with masked convolutions for image generation, and WaveNet with dilated causal convolutions for audio waveform synthesis. Causal masking is highlighted as the unifying principle across all three architectures -- the same idea that powers GPT-style language models in Module 10. A comparison experiment on the same data across VAEs, flows, and diffusion provides a comprehensive view of generative model tradeoffs to close out the module.

---

## Topic Categories

| Topic | Category | Template |
|-------|----------|----------|
| 11-01 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-02 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-03 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-04 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-05 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-06 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-07 | C — Evaluation/Pipeline | `TEMPLATE_EVALUATION.ipynb` |
| 11-08 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-09 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |
| 11-10 | A — Algorithm | `TEMPLATE_ALGORITHM.ipynb` |

---

## Module-Specific Packages

Core packages only — no module-restricted exceptions.

---

## Datasets

- MNIST (11-01, 11-02, 11-08, 11-09, 11-10)
- FashionMNIST (11-03, 11-04)
- CIFAR-10 (11-05, 11-06, 11-07)
- Synthetic 2D (11-08, 11-09)
- Synthetic audio (11-10)

---

## Prerequisites Chain

- **11-01:** Requires 5-07, 5-10
- **11-02:** Requires 11-01, 1-07, 1-08
- **11-03:** Requires 11-01, 5-07
- **11-04:** Requires 11-02, 11-03
- **11-05:** Requires 11-01, 5-07, 1-07
- **11-06:** Requires 11-05, 11-02
- **11-07:** Requires 11-05
- **11-08:** Requires 11-02, 1-09
- **11-09:** Requires 11-05, 1-09
- **11-10:** Requires 11-02, 8-05

---

## Concept Ownership

These concepts are **taught in this module**. Other modules may use them but must not re-teach them.

### Module 11 — Generative Deep Learning
| Concept | Owner |
|---------|-------|
| Autoencoders (denoising, sparse) | 11-01 |
| VAEs (reparameterization trick, ELBO, VQ-VAE) | 11-02 |
| GANs (DCGAN, WGAN, mode collapse) | 11-03 |
| Conditional generation (cGAN, cVAE) | 11-04 |
| DDPM diffusion from scratch, discrete diffusion (SEDD) | 11-05 |
| Latent diffusion models | 11-06 |
| Classifier-free guidance, DDIM, FID/IS/precision-recall eval | 11-07 |
| Normalizing flows (RealNVP, change of variables) | 11-08 |
| Score matching, energy-based models, flow matching, SDE/ODE framework | 11-09 |
| Autoregressive generative models (MADE, PixelCNN, WaveNet) | 11-10 |

---

## Cross-Module Ownership Warnings

- ⚠️ SimCLR/BYOL moved to Module 12-04. Module 11 no longer owns contrastive self-supervised learning.
- ⚠️ 11-10 (Autoregressive Generative) vs 10-01 (GPT): 11-10 covers MADE/PixelCNN/WaveNet as generative theory; 10-01 covers GPT as NLP application.

---

## Special Notes

No special notes for this module.
