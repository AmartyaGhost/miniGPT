# 🤖 GPT from Scratch

A complete implementation of the GPT (Generative Pre-trained Transformer) architecture built from scratch using PyTorch. This project demonstrates the fundamental concepts behind modern language models without relying on pre-built transformer libraries.

## 🎯 Project Overview

This implementation creates a character-level language model that learns to generate Shakespeare-style text. The model is trained on the Tiny Shakespeare dataset and demonstrates core transformer concepts including self-attention, multi-head attention, and positional encodings.

### ✨ Key Features

- **Pure PyTorch Implementation**: Built from first principles without using pre-built transformer libraries
- **Complete Transformer Architecture**: Includes all essential components (attention, embeddings, layer normalization)
- **Character-Level Tokenization**: Learns patterns at the character level for fine-grained text generation
- **Training Pipeline**: Full training loop with loss tracking and model checkpointing
- **Text Generation**: Inference pipeline for generating new text sequences

## 🏗️ Architecture

The model implements a decoder-only transformer architecture with the following specifications:

| Component | Details |
|-----------|---------|
| **Parameters** | 10.8M total parameters |
| **Layers** | 6 transformer blocks |
| **Attention Heads** | 6 heads per layer |
| **Embedding Dimensions** | 384 |
| **Context Length** | 256 tokens |
| **Vocabulary Size** | 65 unique characters |
| **Dropout Rate** | 0.2 |

### 🧠 Model Components

1. **Token Embeddings**: Maps character indices to dense vectors
2. **Positional Embeddings**: Encodes sequence position information
3. **Multi-Head Self-Attention**: Allows tokens to attend to previous context
4. **Feed Forward Networks**: Processes information at each position
5. **Layer Normalization**: Stabilizes training and improves convergence
6. **Residual Connections**: Enables training of deeper networks

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/gpt-from-scratch.git
cd gpt-from-scratch
```

2. Download the dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Training

Run the complete training pipeline:

```python
python train.py
```

Or use the Jupyter notebook for step-by-step exploration:

```bash
jupyter notebook minigpt-model.ipynb
```

### Generation

Generate text with a trained model:

```python
from model import GPTLanguageModel
import torch

# Load trained model
model = GPTLanguageModel()
model.load_state_dict(torch.load('model.pth'))

# Generate text
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
```

## 📊 Training Results

The model demonstrates clear learning progression:

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| **Training Loss** | 4.87 | 0.86 | 82% reduction |
| **Validation Loss** | 4.23 | 1.56 | 63% reduction |
| **Training Steps** | - | 5,000 | - |
| **Training Time** | - | ~1 hour (GPU) | - |

### Sample Generated Text

```
DUKE VINCENTIO:
Worthy Prince forth from Lord Claudio!

KING HENRY VI:
To prevent it, as I love this country's cause.

HENRY BOLINGBROKE:
I thank you for my follow. Walk ye were so?
```

## 📁 Project Structure

```
gpt-from-scratch/
├── minigpt-model.ipynb      # Main notebook with step-by-step implementation
├── model.py                 # Clean model implementation
├── train.py                 # Training script
├── utils.py                 # Helper functions (tokenization, data loading)
├── config.py                # Model and training configurations
├── input.txt                # Training data (Tiny Shakespeare)
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── examples/                # Generated text examples
    ├── training_progress.txt
    └── sample_outputs.txt
```

⭐ **If you find this project helpful, please give it a star!** ⭐

**Built with curiosity and a love for understanding AI from first principles** 🚀
