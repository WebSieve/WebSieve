<div align="center">

# 🧠 Machine Learning & AI Engineering Portfolio

### _A Comprehensive Collection of Deep Learning, Transformers, and Neural Network Implementations_

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

_From state-of-the-art transformer architectures and classical neural networks to Probabilistic Models - a journey through modern AI_

[🚀 Projects](#featured-projects) • [🏗️ Architecture](#architectures-implemented) • [📊 Applications](#applications--use-cases) • [🛠️ Technologies](#technologies--frameworks)

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Featured Projects](#featured-projects)
  - [Probabilistic Models](#probabilistic-models)
  - [CIFAR-100 Vision Transformer](#cifar-100-vision-transformer)
  - [DeepSeek V3 Recreation](#deepseek-v3-recreation)
  - [Transformer Summarizer](#transformer-summarizer)
  - [Decoder-Only Transformer](#decoder-only-transformer)
  - [Neural Network Projects](#neural-network-projects)
- [Architectures Implemented](#architectures-implemented)
- [Applications & Use Cases](#applications--use-cases)
- [Technologies & Frameworks](#technologies--frameworks)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Highlights & Achievements](#highlights--achievements)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

<div align="center">

![AI Banner](https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=1200&q=80)

</div>

This repository represents a comprehensive exploration of modern machine learning and artificial intelligence, featuring **cutting-edge transformer architectures**, **classical neural networks**, and **production-ready implementations** of state-of-the-art models.

### 🎯 What You'll Find Here

- 🎲 **From-Scratch Probabilistic Models**: The flagship achievement — a complete Bayesian suite in `Probabilistic-Models/` (Beta-Binomial sequential updating, Dirichlet-Multinomial text classification, Dirichlet Compound Multinomial spam classification on the UCI SMS corpus, Bayesian Naive Bayes with mutual-information feature selection, and LDA/QDA generative classifiers) built on exact conjugate posteriors and log-space numerics, with zero reliance on sklearn or deep learning frameworks
- 🏆 **Benchmarked, Not Just Built**: One-command evaluation pipelines producing decision-boundary visualizations, confusion matrices, and full accuracy/precision/recall/F1 reporting — LDA/QDA scoring up to **99.44%**, DCM scoring **98.75%** (95.27% spam F1) on real SMS data, backed by 17 passing unit tests plus a CI accuracy-regression gate
- 🧠 **Advanced Transformer Models**: From-scratch implementations of GPT-style decoders, BERT-style encoders, and innovative architectures
- 🔬 **Research Reproductions**: Faithful recreations of papers like DeepSeek V3 with Multi-Head Latent Attention and Mixture of Experts
- 📊 **Diverse Applications**: Computer vision, NLP, time series, classification, detection, and more
- 🎓 **Educational Code**: Extensively documented implementations perfect for learning and teaching
- ⚡ **Production-Ready**: Complete training pipelines, checkpointing, evaluation metrics, and deployment-ready code

---

## 🚀 Featured Projects

### 🎲 Probabilistic Models

<div align="center">

[![Repository](https://img.shields.io/badge/Status-Active-success?style=flat-square)](https://github.com/WebSieve/Probabilistic-Models)
[![Framework](https://img.shields.io/badge/Framework-NumPy%20%2F%20SciPy-blueviolet?style=flat-square)](https://github.com/WebSieve/Probabilistic-Models)

</div>

Bayesian probabilistic models implemented from scratch using **NumPy** and **SciPy**, featuring conjugate priors for fully analytical posterior inference without MCMC.

**📦 Models:**

- **Beta-Binomial** — Sequential Bayesian updating for binary outcomes. `beta_binomial_model` class with prior, likelihood, posterior update, posterior predictive, and full moment computation.
- **Dirichlet-Multinomial (Bayesian Naive Bayes)** — End-to-end text classifier using the Dirichlet-Multinomial compound distribution. `dmm` class with `fit`/`predict`/`predict_proba`, posterior mean/mode/variance per class, marginal density plotting, synthetic data generation, and built-in comparison against classic Multinomial Naive Bayes.
- **Dirichlet Compound Multinomial (DCM)** — Standalone document-classification package on the UCI SMS Spam Collection (5,574 messages). `DCM` class with `fit`/`predict`/`predict_log_proba` (θ integrated out analytically via the compound likelihood), `dataClass` pipeline with from-scratch tokenization, vocabulary, count-matrix construction (`np.add.at`) and stratified splitting, and an `Evaluator` with accuracy/precision/recall/F1 reporting, confusion matrices, learning curves, and top discriminative words. Scores **98.75%** accuracy (**95.27%** spam F1) on a 7,921-word vocabulary, guarded by a CI accuracy-regression gate.
- **Bayesian Naive Bayes** — Binary-feature classifier with Beta-Bernoulli likelihoods and a Dirichlet class prior. `BayesianNaiveBayes` class with `fit`/`predict_proba`/`predict`, posterior-mean estimation, and top-k mutual-information feature selection.
- **Gaussian Discriminant Analysis (LDA / QDA)** — Generative Gaussian classifiers for tabular data. `QDA` class with `fit`/`predict`/`predict_proba`/`predict_log_proba` (per-class covariances, Mahalanobis + `slogdet` log-likelihoods normalized via `logsumexp`), `LDA` subclass with pooled-covariance `fit`, a shared `generate_synthetic_classification_data` generator (unequal/equal covariance, multiclass, high-dim profiles), and a one-command `run_pipeline` producing decision-boundary plots, confusion matrices, and accuracy/precision/recall/F1 scores. Benchmarks (600 samples/profile, 70/30 split): QDA **99.44%** on unequal covariances, LDA **99.44%** on equal covariances, ≥96.6% multiclass, ~90% in 5-D.

**📁 Files:**

```
Probabilistic-Models/
├── README.md                                         # Suite overview (all five models)
├── pyproject.toml                                  # Project configuration
├── BayesianNaiveBayes/
│   ├── bayesian_nb.py                             # Bayesian Naive Bayes classifier + demo
│   └── bayesian_nb_notes.md                       # Mathematical derivation
├── Beta_Binomial_model/
│   └── beta_bin_model.py                           # Beta-Binomial implementation + demo
├── Dirichlet_Compound_Multinomial/
│   ├── pyproject.toml                              # Package config (NumPy, SciPy, Matplotlib only)
│   ├── LICENSE                                     # MIT
│   ├── README.md                                   # Math, usage, benchmark results
│   ├── .gitignore                                  # Python / venv ignores
│   ├── .github/workflows/ci.yml                   # CI with accuracy-regression gate
│   ├── assets/
│   │   ├── SMSSpamCollection                       # UCI SMS Spam Collection v.1 (5,574 messages)
│   │   └── readme                                  # Dataset description / license
│   └── src/dirichlet_compound_multinomial/
│       ├── __init__.py                             # Public API (dataClass, DCM, Evaluator)
│       ├── data.py                                 # Tokenization, vocab, count matrix, split
│       ├── model.py                                # DCM: compound log-likelihood, fit, predict
│       ├── eval.py                                 # Metrics, confusion matrix, learning curve
│       └── __main__.py                             # Module entry point
├── Dirichlet_Multinomial_Model/
│   ├── dirichlet_multinomial_model.py              # Dirichlet-Multinomial classifier
│   ├── doc.md                                      # Full mathematical derivation
│   └── demo.txt                                    # Sample data format
└── Gaussian_Prob_Models/
    ├── src/gaussian_prob_models/
    │   ├── qda.py                                 # QDA: fit, predict/proba
    │   ├── lda.py                                 # LDA: pooled-covariance fit
    │   ├── datasets.py                            # Synthetic data generator
    │   ├── evaluate.py                            # Metrics + figures
    │   ├── pipeline.py                            # One-command pipeline
    │   ├── __init__.py                            # Public API
    │   └── __main__.py                            # Module entry point
    ├── data/generate_data.py                      # Data CLI shim
    ├── tests/
    │   ├── __init__.py                              # Test package marker
    │   └── test_models.py                           # 17 unittest tests
    ├── pyproject.toml                             # Package config
    ├── requirements.txt                           # Pinned versions
    └── README.md                                  # Math, usage, results
```

**💡 Technical Highlights:**

- Pure NumPy/SciPy — no sklearn, PyMC, or deep learning frameworks used
- Conjugate priors enable closed-form posterior updates — no MCMC or variational inference
- Log-space computations via `scipy.special.gammaln` for numerical stability
- Posterior predictive distributions integrate out latent parameters analytically
- Visualization of marginal posterior densities and error-rate comparisons
- Gaussian class-conditionals with Mahalanobis/`slogdet` log-likelihoods normalized via `logsumexp`
- Decision-boundary contours, confusion-matrix heatmaps, and macro precision/recall/F1 reporting
- Stratified train/test splits preserving class proportions with train-only vocabulary (no data leakage)
- CI benchmark smoke test with an accuracy-regression gate on every push/PR

---

### 🖼️ CIFAR-100 Vision Transformer

<div align="center">

[![Repository](https://img.shields.io/badge/Status-Complete-success?style=flat-square)](https://github.com/WebSieve/Cifar-100_with_Vision_Transformer)
[![Architecture](https://img.shields.io/badge/Architecture-ViT%20%2B%20MoE-blueviolet?style=flat-square)](https://github.com/WebSieve/Cifar-100_with_Vision_Transformer)
</div>

> Note : Built entirely from scratch to develop deep core knowledge of modern transformer mechanisms rather than relying on high-level APIs.

A state-of-the-art **Vision Transformer (ViT)** implemented from the ground up and optimized for CIFAR-100 image classification, featuring advanced sparsity and compression mechanisms.

**⭐ Standout Features:**

| Feature | Description |
| :--- | :--- |
| 🧠 **Mixture of Experts (MoE)** | Sparse computation with 8 experts, top-2 routing, and load balancing auxiliary loss |
| 🗜️ **Multi-Head Latent Attention** | MLA implementation for 4x KV cache compression and memory efficiency |
| ⚡ **Modern Activations** | SwiGLU feed-forward networks and RMSNorm |
| 🎛️ **Advanced Training** | Built-in support for Mixup, CutMix, label smoothing, and AMP |

**📁 Core Components:**

```text
Cifar-100_with_Vision_Transformer/
├── src/
│   ├── attention.py       # Multi-Head Latent Attention mechanics
│   ├── moe.py             # Expert routing and load balancing
│   ├── transformer.py     # Core ViT block architecture
│   ├── embeddings.py      # Patch + position embeddings
│   ├── normalization.py   # RMSNorm / LayerNorm
│   ├── data.py            # Augmentation incl. Mixup and CutMix
│   └── config.py          # Model + training configs
├── train.py               # Full training loop with cosine LR
└── evaluate.py            # Inference and metric evaluation
```

**💡 Technical Highlights:**

- Custom patch embedding mechanism tailored for 32x32 image resolutions (4x4 patches).
- Clean, modular codebase reflecting a strict "First Principles" engineering methodology.
- Designed to scale from ~75% (Baseline ViT) to ~79% Top-1 Accuracy with MLA and MoE techniques (expected results; weights not yet trained — see project README).

---

### 🧬 DeepSeek V3 Recreation

<div align="center">

[![DeepSeek](https://img.shields.io/badge/Status-Active-success?style=flat-square)](https://github.com/WebSieve/DeepSeek_V3_Recreation)
[![Paper](https://img.shields.io/badge/Paper-Implementation-blue?style=flat-square)](https://arxiv.org/abs/2412.19437)

</div>

A clean, educational PyTorch implementation of **DeepSeek V3's core architectural innovations**, pushing the boundaries of efficient transformer design.

**🔑 Key Innovations:**

- **Multi-Head Latent Attention (MLA)**: Low-rank key-value compression reducing KV cache by 4-8x
- **Mixture of Experts (MoE)**: Sparse computation with top-k routing and load balancing
- **RMS Normalization**: Efficient normalization without mean-centering
- **Modular Design**: Clean separation of concerns for easy experimentation

**📁 Components:**

```
DeepSeek_V3_Recreation/
├── src/deepseek_v3/
│   ├── mla.py       # Multi-head latent attention (incl. Multi_Head_Latent_Attention)
│   ├── moe.py       # Sparse expert routing (incl. Mixture_Of_Experts)
│   ├── block.py     # Complete DeepSeek transformer block (DS_Block)
│   ├── rmsnorm.py   # Root Mean Square normalization (RMSNorm)
│   ├── rope.py      # Rotary embeddings + YaRN scaling
│   ├── config.py    # Model configs (tiny + full V3)
│   ├── model.py     # Full model: embeddings → N blocks → output head
│   └── mtp.py       # Multi-token prediction head
├── scripts/         # Training, generation, and benchmark scripts
└── tests/           # Shape / RoPE / cache / routing checks
```

**💡 Technical Highlights:**

- LoRA-style K/V decomposition for memory efficiency
- Separate RoPE and non-RoPE component handling
- SwiGLU activation in expert networks
- Auxiliary loss-free load balancing

---

### ✍️ Transformer Summarizer

<div align="center">

[![Summarizer](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)](https://github.com/WebSieve/Transformer_Summarizer)
[![Scaffold](https://img.shields.io/badge/Type-Scaffold-informational?style=flat-square)](https://github.com/WebSieve/Transformer_Summarizer)

</div>

> Note : Early scaffold — a single attention module so far. Encoder, training, and summarization logic are still to come.

An **encoder-based transformer project** for text summarization and understanding tasks, currently at the scaffolding stage with its first attention building block in place.

**🎯 Planned Purpose:**

- Extractive and abstractive summarization
- Document understanding and encoding
- Semantic text representation

**📦 Current Components:**

```
Transformer_Summarizer/
└── src/
    └── MLA.py    # Multi-head latent attention block (work in progress)
```

**🔧 Features (planned):**

- Multi-head latent attention for efficiency
- Flexible encoder stack architecture
- Support for various text encoding tasks

---

### 🤖 Decoder-Only Transformer

<div align="center">

[![Transformer](https://img.shields.io/badge/Status-Complete-success?style=flat-square)](https://github.com/WebSieve/DecoderOnlyTransformer_Basic)
[![GPT](https://img.shields.io/badge/Style-GPT--like-blueviolet?style=flat-square)](https://github.com/WebSieve/DecoderOnlyTransformer_Basic)

</div>

A **production-ready GPT-style transformer** built from scratch with modern architectural improvements and complete training infrastructure.

**⭐ Standout Features:**

| Feature | Description |
|---------|-------------|
| 🎯 **Rotary Position Embeddings (RoPE)** | Superior position encoding with better extrapolation |
| 🔥 **SwiGLU Activation** | Modern activation function used in LLaMA and PaLM |
| 📏 **RMS Normalization** | Efficient normalization without mean-centering |
| 🎲 **Advanced Sampling** | Temperature and top-k sampling |
| 💾 **Complete Training Pipeline** | Checkpointing, logging, learning rate scheduling |

**📊 Training Capabilities:**

- Custom sliding window tokenization
- Efficient DataLoader implementation
- AdamW optimization with cosine annealing
- Gradient clipping
- Validation loss tracking and best-checkpointing

**🎨 Generation Methods:**

```python
# Tokenize with the built-in GPT-2 tokenizer, then generate
input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids

# Temperature sampling
output_ids = model.generate(input_ids, temperature=0.8)

# Top-k sampling
output_ids = model.generate(input_ids, top_k=50)

# Longer continuations
output_ids = model.generate(input_ids, max_new_tokens=512)

print(model.tokenizer.decode(output_ids[0]))
```

---

### 🧮 Neural Network Projects

<div align="center">

[![NN](https://img.shields.io/badge/Projects-50+-red?style=flat-square)](https://github.com/WebSieve/Neural_Nets_and_Transformers)
[![Diverse](https://img.shields.io/badge/Domains-CV%20%7C%20NLP%20%7C%20TS-orange?style=flat-square)](https://github.com/WebSieve/Neural_Nets_and_Transformers)

</div>

A vast collection of **neural network implementations** spanning computer vision, natural language processing, time series, and more.

> Note: the 50+ count below spans the linked `Neural_Nets_and_Transformers` repo plus a companion notebook collection (object-detection walkthroughs, VAE variants, and extra classification studies).

#### 🖼️ Computer Vision

**Image Classification:**

- **MNIST Digit Recognition**: CNN-based digit classifier with 99%+ accuracy
- **CIFAR-10 Classification**: Multi-class image classification with data augmentation
- **CIFAR-100 Classification**: Fine-grained image classification with 100 classes
- **Transfer Learning**: Pre-trained model fine-tuning for custom datasets

**Object Detection:**

- Custom object detection implementations
- Pretrained YOLO inference (ultralytics)

#### 📝 Natural Language Processing

**Text Classification:**

- **Fake News Detection**: Multi-model ML approach with ensemble methods
- **Sentiment Analysis**: LSTM and transformer-based classifiers
- **Text Categorization**: Multi-class document classification

**Advanced NLP:**

- Positional embedding implementations
- Custom tokenizer demonstrations (GPT-2)
- Sequence-to-sequence models

#### 🔢 Classical ML & Specialized Tasks

**Classification Tasks:**

- **Iris Classification**: Multi-layer perceptron for species classification
- **Wine Classification**: Neural network for wine quality prediction
- **Cover Type Classification**: Forest cover type prediction
- **Higgs Boson Detection**: Binary classification for particle physics

**Specialized Applications:**

- **DNA Sequence Prediction**: Biological sequence modeling
- **Sin Wave Modeling**: Time series regression with neural networks
- **Cosine Function Approximation**: Function learning demonstrations

#### 🏗️ Architecture Implementations

**State-of-the-Art Models:**

- Variational Autoencoders (VAE) (Keras notebook + standalone PyTorch package)
- Gaussian Mixture Models (GMM)
- Backpropagation from scratch
- Custom layer implementations

---

## 🏛️ Architectures Implemented

<div align="center">

| Architecture | Type | Key Features | Status |
|-------------|------|--------------|--------|
| **Bayesian Naive Bayes** | Bayesian | Beta-Bernoulli Likelihood, Dirichlet Prior, MI Feature Selection | ✅ Complete |
| **Beta-Binomial / Dirichlet-Multinomial** | Bayesian | Conjugate Priors, Analytical Posterior, Predictive Distributions | ✅ Complete |
| **Dirichlet Compound Multinomial (DCM)** | Bayesian | Compound Likelihood, Stratified Eval, SMS Spam 98.75% | ✅ Complete |
| **Cifar-100_with_Vision_Transformer** | Vision | MoE, MLA, SwiGLU, RMSNorm | ✅ Complete |
| **DeepSeek V3** | Decoder | MLA, MoE, RMSNorm | ✅ Active |
| **Gaussian Discriminant Analysis (LDA/QDA)** | Generative | Gaussian Class-Conditionals, Pooled/Per-Class Covariance, Log-Space Posteriors | ✅ Complete |
| **GPT-Style Decoder** | Decoder | RoPE, SwiGLU, Causal Attention | ✅ Complete |
| **Transformer Encoder** | Encoder | Multi-Head Attention, Feed-Forward | 🟡 In Progress |
| **CNN Architectures** | Vision | Conv Layers, Pooling, Batch Norm | ✅ Complete |
| **RNN/LSTM** | Sequential | Time Series, Text Generation | ✅ Complete |
| **VAE** | Generative | Latent Space, Reconstruction | ✅ Complete |
| **Mixture of Experts** | Hybrid | Sparse Routing, Load Balancing | ✅ Complete |

</div>

### 🔬 Advanced Components

- **Attention Mechanisms**: Multi-head, multi-head latent, cross-attention, self-attention
- **Position Encodings**: Sinusoidal, learned, rotary (RoPE)
- **Normalization**: LayerNorm, BatchNorm, RMSNorm, GroupNorm
- **Activations**: ReLU, GELU, SwiGLU, Mish, Swish
- **Regularization**: Dropout, weight decay, gradient clipping, label smoothing

---

## 📊 Applications & Use Cases

### 🎯 Computer Vision

```
✓ Image Classification (MNIST, CIFAR-10, CIFAR-100)
✓ Object Detection (Custom implementations)
✓ Transfer Learning (Pre-trained models)
✓ Feature Extraction
```

### 💬 Natural Language Processing

```
✓ Text Generation (GPT-style models)
✓ Text Summarization (Encoder-based)
✓ Text Classification (Sentiment, Fake News)
✓ Language Modeling
```

### 📈 Time Series & Regression

```
✓ Function Approximation (Sin, Cosine)
✓ Sequence Prediction
✓ Regression Tasks
```

### 🔬 Research & Experimentation

```
✓ Architecture Ablation Studies
✓ Hyperparameter Optimization
✓ Novel Component Testing
✓ Paper Implementations
```

---

## 🛠️ Technologies & Frameworks

<div align="center">

### Core Frameworks

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

### Scientific Computing

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

### Visualization & Analysis

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

### Machine Learning

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## 📂 Project Structure

```
📁 Machine Learning Portfolio
│
├── 🥉Cifar-100_with_Vision_Transformer/
│   ├── src/                    
│   │   ├── attention.py         
│   │   ├── moe.py               
│   │   └── transformer.py       
│   ├── train.py                 
│   └── evaluate.py              
│
├── 🧬 DeepSeek_V3_Recreation/
│   ├── Multi-Head Latent Attention
│   ├── Mixture of Experts
│   ├── Complete Transformer Blocks
│   └── RMS Normalization
│
├── ✍️ Transformer_Summarizer/          [🟡 In Progress]
│   ├── Encoder Architecture
│   ├── Advanced Attention Mechanisms
│   └── Summarization Models
│
├── 🤖 DecoderOnlyTransformer_Basic/
│   ├── GPT-style Architecture
│   ├── Training Pipeline
│   ├── Text Generation
│   └── Sampling Strategies
│
├── 🎲 Probabilistic-Models/
│   ├── Beta-Binomial Model
│   ├── Dirichlet-Multinomial Model
│   ├── Dirichlet Compound Multinomial (SMS Spam Classifier)
│   ├── Bayesian Naive Bayes
│   ├── Gaussian Discriminant Analysis (LDA/QDA)
│   ├── Bayesian Inference Pipeline
│   └── Mathematical Documentation
│
├── 🧠 Neural_Net_Projects/
│   ├── Computer Vision
│   │   ├── Image Classification
│   │   ├── Object Detection
│   │   └── Transfer Learning
│   │
│   └── NLP & Text
│       ├── Text Classification
│       ├── Sequence Models
│       └── Custom Tokenizers
│
├── 🎓 Neural_Nets_and_Transformers/
│   ├── Complete Implementations
│   ├── Training Scripts
│   ├── Evaluation Tools
│   └── Tutorial Notebooks
│
├── 🔬 Core Components/
│   ├── Encoder.py / Decoder.py
│   ├── Positional Embeddings
│   ├── Attention Mechanisms
│   └── Custom Layers
│
└── 📓 Jupyter Notebooks/
    ├── Anatomy Explorations
    ├── Architecture Visualizations
    ├── Training Demonstrations
    └── Interactive Tutorials
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
TensorFlow 2.0+
CUDA (optional, for GPU acceleration)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/WebSieve/WebSieve.git
cd WebSieve

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install tensorflow
pip install numpy pandas scikit-learn matplotlib seaborn
pip install jupyter notebook ipykernel
```

### Quick Start Examples

#### 1️⃣ Train a GPT-Style Model

```python
from DecoderOnlyTransformer_Basic.Transformer import Transformer
from DecoderOnlyTransformer_Basic.model_training import train_one_epoch

# Initialize model (vocabulary comes from the built-in GPT-2 tokenizer)
model = Transformer(
    embed_dim=512,
    num_heads=8,
    intermediate_dim=2048,
    num_layers=6,
    max_seq_len=256,
)

# Train one epoch (see model_training.main for the full
# AdamW + cosine-schedule loop with best-checkpointing)
avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=1)
```

#### 2️⃣ Use DeepSeek V3 Components

```python
import torch
from deepseek_v3 import DS_Block

# Create a DeepSeek transformer block
block = DS_Block(
    hidden_size=512,
    num_heads=8,
    kv_lora_rank=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    intermediate_dim=1024,
    num_experts=8,
    num_experts_per_token=2,
)

# Forward pass (returns hidden states, updated KV cache, routing stats)
x = torch.randn(2, 10, 512)
out, _, _ = block(x)
```

#### 3️⃣ Image Classification with CNN

```python
from tensorflow.keras import layers, models

# MNIST classifier in the style of MNIST_digit_classification_with_CNN
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(784,)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# X_train: (n, 784) float32 in [0, 1]; y_train: one-hot via to_categorical(..., 10)
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

#### 4️⃣ Bayesian Inference with Probabilistic Models

```python
from Beta_Binomial_model.beta_bin_model import beta_binomial_model

# Beta-Binomial sequential updating
model = beta_binomial_model(alpha=2, beta=2, error_coef=1e-10)

# Observe 7 successes in 10 trials → update posterior
model.posterior_updation(k_successes=7, n_trials=10)
print(f"Posterior mean: {model.posterior_mean():.3f}")
print(f"Posterior variance: {model.posterior_variance():.5f}")

# Predict next 5 trials
pred_mean = model.posterior_predictive_mean(n_future_trials=5)
print(f"Expected successes in next 5 trials: {pred_mean:.2f}")
```

#### 5️⃣ Bayesian Naive Bayes Classification

```python
from BayesianNaiveBayes.bayesian_nb import BayesianNaiveBayes

model = BayesianNaiveBayes(alpha=1.0, beta_0=1.0, beta_1=1.0)
model.fit(X_train, y_train, top_k_features=10)

# Soft class probabilities and hard labels
proba = model.predict_proba(X_test)
labels = model.predict(X_test)
```

#### 6️⃣ Gaussian Discriminant Analysis (LDA/QDA)

```python
from Gaussian_Prob_Models.src.gaussian_prob_models.qda import QDA
from Gaussian_Prob_Models.src.gaussian_prob_models.datasets import generate_synthetic_classification_data

# Synthetic data with distinct class covariances (QDA showcase)
X, y = generate_synthetic_classification_data(dataset_type="unequal_cov")

model = QDA()
model.fit(X, y)

print(f"Train accuracy: {(model.predict(X) == y).mean():.3f}")
print(model.predict_proba(X)[:3])  # posterior class probabilities
```

> Note: run from the `Probabilistic-Models` repo root. Inside `Gaussian_Prob_Models/` itself the packaged form applies: `from gaussian_prob_models import QDA, generate_synthetic_classification_data` (with `PYTHONPATH=src`).

#### 7️⃣ Dirichlet Compound Multinomial (SMS Spam Classification)

```python
from Dirichlet_Compound_Multinomial.src.dirichlet_compound_multinomial import (
    dataClass, DCM, Evaluator,
)

# Load + tokenize the UCI SMS Spam Collection (5,574 messages)
data = dataClass()
messages, labels = data.getMsgLabel()
y = data.labels_to_vector(labels)

# Stratified split (ham:spam 87:13), vocabulary built on train only
train_idx, test_idx = data.split_indices(y)
vocab, _ = data.buildVocab([messages[i] for i in train_idx])
X = data.docs_to_matrix(messages, vocab)

model = DCM(alpha_init=1.0)
model.fit(X[train_idx], y[train_idx])

pred = model.predict(X[test_idx])
Evaluator.report(y[test_idx], pred)  # 98.75% accuracy, 95.27% spam F1
```

---

## 🏆 Highlights & Achievements

<div align="center">

### 📊 By The Numbers

| Metric | Value |
|--------|-------|
| **Total Projects** | 50+ |
| **Architectures Implemented** | 15+ |
| **Lines of Code** | 25,000+ |
| **Jupyter Notebooks** | 30+ |
| **Research Papers Implemented** | 5+ |
| **Frameworks Mastered** | PyTorch, TensorFlow, Keras |

</div>

### 🌟 Key Achievements

- ✅ **Complete DeepSeek V3 Recreation**: Faithful implementation of cutting-edge architecture
- ✅ **Production-Ready Transformer**: Full training pipeline with state-of-the-art components
- ✅ **Diverse Application Portfolio**: CV, NLP, time series, and specialized tasks
- ✅ **Educational Quality**: Extensively documented code with clear explanations
- ✅ **Modern Best Practices**: RoPE, SwiGLU, RMSNorm, and more
- ✅ **Bayesian Inference Pipelines**: Five conjugate-prior models implemented from scratch with NumPy/SciPy
- ✅ **Bayesian Naive Bayes with Feature Selection**: Beta-Bernoulli classifier with conjugate priors and mutual-information feature ranking
- ✅ **Dirichlet Compound Multinomial Spam Classifier**: Standalone package scoring 98.75% accuracy (95.27% spam F1) on the UCI SMS Spam Collection, with stratified evaluation and a CI accuracy-regression gate
- ✅ **Gaussian Discriminant Analysis from Scratch**: LDA/QDA generative classifiers with log-space posteriors, one-command evaluation pipeline, and full precision/recall/F1 reporting
- ✅ **Modular Design**: Reusable components for rapid experimentation

### 🎯 Notable Implementations

1. **Multi-Head Latent Attention**: Memory-efficient attention with 4-8x KV cache reduction
2. **Mixture of Experts**: Sparse computation with intelligent routing
3. **Rotary Position Embeddings**: Superior position encoding for transformers
4. **Bayesian Probabilistic Models**: From-scratch conjugate inference with Beta-Binomial, Dirichlet-Multinomial, Dirichlet Compound Multinomial, and Bayesian Naive Bayes
5. **Gaussian Discriminant Analysis (LDA/QDA)**: Generative Gaussian classifiers with per-class and pooled covariances, decision-boundary visualization, and precision/recall/F1 evaluation
6. **Complete Training Infrastructure**: Checkpointing, logging, and evaluation pipelines
7. **Diverse Neural Architectures**: CNNs, RNNs, VAEs, and more

---

## 🤝 Contributing

Contributions are welcome! Whether you want to:

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Add new models or architectures
- ✨ Enhance existing implementations

Please feel free to open an issue or submit a pull request.

### Development Guidelines

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation as needed
5. Maintain modular, reusable code

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Research Papers**: DeepSeek V3, GPT, BERT, and countless others
- **Open Source Community**: PyTorch, TensorFlow, and Hugging Face teams
- **Educational Resources**: Stanford CS231n, CS224n, and fast.ai
- **Inspiration**: The broader ML/AI research community

---

## 📬 Contact & Links

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github)](https://github.com/WebSieve)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:msahil2603@gmail.com)

### ⭐ Star this repo if you find it helpful

</div>

---

<div align="center">

### 🚀 Built with passion for AI and Machine Learning

**"The only way to do great work is to love what you do."**

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)

</div>
