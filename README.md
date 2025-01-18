# **PyTorch GPT Implementation From Scratch** 🧠
* This repository contains a from-scratch implementation of the GPT (Generative Pre-trained Transformer) architecture using PyTorch. 
* The implementation focuses on modularity, readability, and educational purposes, breaking down the complex architecture into understandable components.

## **Project Structure**
* The project is organized into several key modules, each handling specific aspects of the GPT architecture:

### 1) Core Components
* Attention Mechanism (Attention.py)

    * Implementation of three attention variants:
      
      * Basic self-attention mechanism
      * Causal self-attention with masking
      * Multi-head attention with parallel attention heads


    * Includes query, key, and value transformations
    * Supports optional dropout and bias configurations


* Transformer Block (Transformers.py)

  * Complete transformer block implementation
  * Combines multi-head attention, feed-forward network, and layer normalization
  * Implements residual connections for better gradient flow
  * Configurable dropout rates

* Feed-Forward Network (FFN.py)

  * Two-layer feed-forward network with GELU activation
  * Implements the standard expansion-reduction architecture
  * Custom GELU activation implementation


* Layer Normalization (LayerNormalisation.py)

  * Custom implementation of layer normalization
  * Includes learnable scale and shift parameters
  * Configurable epsilon value for numerical stability

### 2) Model Architecture
* GPT Model (GPTModel.py)

  * Complete GPT architecture implementation including:
  
    * Token embedding layer
    * Positional embedding layer
    * Multiple transformer layers
    * Final layer normalization
    * Output projection layer
  
  * Configurable architecture parameters through gpt_config

### 3) Data Preprocessing
* Tokenization (Tokenizer.py)

  * BPE (Byte-Pair Encoding) tokenizer implementation
  * Utilizes the tiktoken library for compatibility with OpenAI's tokenization
  * Supports special tokens like <|endoftext|>


* Dataset Handling (Dataset.py)

  * Custom PyTorch Dataset implementation for text data
  * Sliding window approach for context creation
  * Configurable context size and stride
  * Efficient data loading with PyTorch's DataLoader
 
### 4) Utilities
* Model Checkpointing (Checkpoint.py)

  * Functions for saving and loading model states
  * Support for saving both model and optimizer states
  * Utility functions for model state management


* Training Utilities (Utils.py)

  * Training and evaluation functions
  * Loss calculation utilities
  * Text generation functions with various sampling strategies
  * Performance plotting utilities


* Pre-trained Model Loading (gpt_download.py)

  * Utilities for downloading and loading pre-trained GPT-2 weights
  * Support for various model sizes (124M, 355M, 774M, 1558M)
  * Weight conversion from TensorFlow checkpoint format
 
### 5) Setup and Usage
1. **Installation**
```bash
pip install torch tiktoken numpy matplotlib tensorflow tqdm requests
```

2. **Basic Usage**
```python
from GPTModel import GPTModelV1
from Tokenizer import BPE_Tokenizer

# Configure model parameters
gpt_config = {
    "vocab_size": 50257,  # GPT-2 vocabulary size
    "embed_dim": 768,     # Embedding dimension
    "context_length": 1024,
    "num_layers": 12,
    "num_heads": 12,
    "dropout_rate": 0.1,
    "qkv_bias": True
}

# Initialize model and tokenizer
model = GPTModelV1(gpt_config)
tokenizer = BPE_Tokenizer("p50k_base")
```

3. **Training Data Preparation**
```python
from Dataset import TextDataLoader

# Create data loader
data_loader = TextDataLoader(
    text_data=your_text_data,
    tokenizer=tokenizer,
    context_size=1024,
    stride=512,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=4
).get_dataLoader()
```

### 6) Key Features
* Modular Architecture

  * Each component is independently implemented and tested
  * Easy to modify or extend individual components
  * Clear separation of concerns


* Training Flexibility

  * Support for both training from scratch and fine-tuning
  * Configurable training parameters
  * Built-in evaluation metrics


* Text Generation

  * Multiple text generation strategies:
  
    * Simple greedy sampling
    * Temperature-controlled sampling
    * Top-k sampling
  
  * Configurable generation parameters


* Pre-trained Model Support

  * Ability to load pre-trained GPT-2 weights
  * Support for multiple model sizes
  * Weight conversion utilities

***Contributing:***
Contributions are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.
