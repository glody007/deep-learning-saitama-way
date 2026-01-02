# Deep Learning from Scratch: The Saitama Way

> Building a deep learning framework from first principles, one level at a time.

## Philosophy: The Saitama Way

This repository documents my journey of reconstructing deep learning from the ground up using the **Saitama way** methodology: building from basic through complex. Just as Saitama trained with fundamental exercises to achieve mastery, this project starts with the most fundamental operations and builds upward.

**Core Principles:**

- **First-principles understanding**: Every concept is derived from mathematical foundations, not memorized
- **Progressive complexity**: Start simple, add complexity only when fundamentals are mastered
- **Hands-on learning**: Paper and spreadsheet work before code
- **No shortcuts**: Build everything from scratch before using optimized libraries

## Methodology

### The Workflow

1. **Paper & Spreadsheet First**: Work through mathematical operations manually

   - Understand the theory on paper
   - Verify calculations in spreadsheets
   - Build intuition before coding

2. **Python Implementation**: Code the concepts in pure Python

   - No NumPy initially (add only after Level 6 if needed)
   - No external deep learning libraries
   - Focus on correctness and understanding over performance

3. **Optional Optimization** (Later): Consider performance improvements
   - C++ or NumPy for tensor operations only (optional)
   - Only after the pure Python version works correctly
   - Performance is secondary to understanding

### Why This Approach?

Deep learning frameworks like PyTorch and TensorFlow are powerful, but they abstract away the fundamentals. This project is about **understanding**, not just using. By building from scratch, you develop:

- Intuition for how neural networks actually work
- Ability to debug issues at the mathematical level
- Confidence to modify and extend frameworks
- Deep understanding that enables research-level work

## Project Structure

```
deep-learning-saitama-way/
├── saitama_dl_framework/     # Core framework implementation
│   ├── tensor/               # Tensor operations (Level 1)
│   ├── autograd/             # Automatic differentiation (Levels 2-4)
│   ├── nn/                   # Neural network primitives (Levels 5-6)
│   ├── optim/                # Optimizers (Level 7)
│   ├── regularization/       # Regularization techniques (Level 8)
│   ├── attention/            # Attention mechanisms (Levels 9-11)
│   ├── transformer/          # Transformer architecture (Levels 12-13)
│   └── training/             # Training utilities (Level 14)
├── exercises/                # Manual exercises and theory
│   ├── 01_linear_algebra_computation.md
│   └── ...                   # Additional exercise files
├── tests/                    # Unit tests for each component
└── README.md                 # This file
```

## Language Strategy

### Primary: Python

Python is the primary language for this project because:

- Widely used in deep learning research
- Fast iteration for mathematical experiments
- Clear, readable code that matches mathematical notation
- Pure Python first (no NumPy until after Level 6)

### Optional: C++ or NumPy (Later)

After the pure Python implementation is complete and correct:

- **C++**: For tensor operation kernels (optional performance optimization)
- **NumPy**: Alternative optimization path for tensor operations
- **When**: Only after understanding is solid, and only for tensor operations
- **Why**: Performance optimization, not for learning

The philosophy remains: **understand first, optimize later**.

## Curriculum: Phase I - III

This repository covers **Phase I through Phase III**. Phase IV and beyond (alignment, evals, safety) will be implemented in separate repositories using PyTorch, as they require the scale and infrastructure that established frameworks provide.

---

## PHASE I — Foundations (Levels 1-4)

### Goal: Think like a neural network

### Level 1 — Linear Algebra & Computation

**Manual Work:**

- Scalars, vectors, matrices
- Dot product, matrix multiplication
- Broadcasting (by hand)
- Element-wise operations

**Code:**

- Implement `Tensor` class with:
  - Shape tracking
  - Data storage (list of lists)
  - `add`, `mul`, `matmul` operations
  - Shape validation

**Papers:**

- Focus on mathematical foundations

**Status:** ✅ In Progress - Basic tensor operations implemented

---

### Level 2 — Calculus for Learning

**Manual Work:**

- Derivatives and partial derivatives
- Chain rule
- Gradient computation by hand

**Code:**

- Add `.grad` attribute to tensors
- Implement backpropagation manually for:
  - `y = ax + b`
  - `y = x²`
  - Simple compositions

**Papers:**

- Focus on understanding gradient flow

---

### Level 3 — Autograd Engine

**Manual Work:**

- Computational graphs
- Reverse-mode automatic differentiation
- Forward vs. backward pass

**Code:**

- Build `Value` object (similar to micrograd, but from scratch)
- Implement `backward()` method
- Support for basic operations

**Papers:**

- Implementation-focused level

---

### Level 4 — Tensors + Autograd

**Manual Work:**

- Jacobians
- Vectorized gradients
- Broadcasting in gradients

**Code:**

- Tensor autograd system
- Broadcasting gradients correctly
- Operations: `sum`, `mean`, `reshape`
- Full computational graph for tensors

**Papers:**

- _Automatic Differentiation in Machine Learning_ (Baydin et al.)

---

## PHASE II — Neural Networks (Levels 5-8)

### Goal: Build PyTorch mentally

### Level 5 — Neural Network Primitives

**Manual Work:**

- Linear layer mathematics
- Bias addition
- Activation functions (ReLU, Tanh, Sigmoid)

**Code:**

- `Linear` layer class
- `ReLU`, `Tanh`, `Sigmoid` activations
- `Module` base class
- `Parameter` class for trainable weights

**Papers:**

- _Understanding LSTM Networks_ (Olah) - for intuition

---

### Level 6 — Losses & Training

**Manual Work:**

- Mean Squared Error (MSE)
- Cross-entropy loss
- Softmax function

**Code:**

- Loss function implementations
- Training loop structure
- Gradient zeroing
- Basic training on simple datasets

**Papers:**

- _Deep Learning_ (Goodfellow) - Optimization chapter

---

### Level 7 — Optimization

**Manual Work:**

- Stochastic Gradient Descent (SGD)
- Momentum derivation
- Adam algorithm understanding

**Code:**

- Optimizer classes from scratch:
  - `SGD`
  - `Momentum`
  - `Adam`
- Learning rate scheduling

**Papers:**

- _Adam: A Method for Stochastic Optimization_ (Kingma & Ba)

---

### Level 8 — Regularization

**Manual Work:**

- Overfitting intuition
- Bias-variance tradeoff
- Regularization techniques

**Code:**

- Dropout implementation
- Weight decay (L2 regularization)
- Early stopping
- Batch normalization basics

**Papers:**

- _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_ (Srivastava et al.)

---

## PHASE III — Attention & Transformers (Levels 9-14)

### Goal: Rebuild modern AI

### Level 9 — Sequence Models

**Manual Work:**

- RNN unrolling
- Vanishing gradient problem
- Backpropagation through time (BPTT)

**Code:**

- Simple RNN implementation
- BPTT training
- Gradient clipping

**Papers:**

- _Recurrent Neural Network Regularization_ (Zaremba et al.)

---

### Level 10 — Attention

**Manual Work:**

- Query, Key, Value (QKV) concept
- Scaled dot-product attention
- Attention weights visualization

**Code:**

- Attention layer implementation (no batching initially)
- Scaled dot-product attention
- Attention mask support

**Papers:**

- _Attention Is All You Need_ (Vaswani et al.) - Core paper

---

### Level 11 — Multi-Head Attention

**Manual Work:**

- Head splitting mathematics
- Projection matrices
- Multi-head attention intuition

**Code:**

- Multi-head attention implementation
- Masking support
- Efficient computation

**Papers:**

- _The Annotated Transformer_ - Implementation guide

---

### Level 12 — Transformer Block

**Code:**

- LayerNorm implementation
- Residual connections
- Full transformer block
- Positional encoding

**Papers:**

- _Deep Residual Learning for Image Recognition_ (He et al.) - Residual connections
- _Attention Is All You Need_ - Full architecture

---

### Level 13 — Language Modeling

**Manual Work:**

- Next-token prediction
- Teacher forcing
- Autoregressive generation

**Code:**

- Mini GPT implementation (character-level)
- Training loop for language modeling
- Text generation

**Papers:**

- _Scaling Laws for Neural Language Models_ (Kaplan et al.) - Understanding scale

---

### Level 14 — Training at Scale (Conceptual)

**Learn:**

- Batch size effects
- Learning rate scheduling
- Gradient clipping and accumulation
- Mixed precision training concepts

**Code:**

- Advanced training utilities
- Learning rate schedulers
- Gradient accumulation

**Papers:**

- _GPipe: Efficient Training of Giant Neural Networks_ - Pipeline parallelism
- _FlashAttention: Fast and Memory-Efficient Exact Attention_ - Optimization techniques

---

## Current Status

### Implemented

- **Level 1 (Partial)**: Basic `Tensor` class with:
  - Shape tracking
  - Element-wise operations (`add`, `sub`, `multiply`)
  - Matrix multiplication (`matmul`)
  - Support for scalars, vectors, and matrices

### In Progress

- Completing Level 1 exercises and tests
- Preparing for Level 2 (Calculus for Learning)

### Repository Scope

This repository will **stop at Phase III (Level 14)**. Phase IV and beyond (alignment, evals, safety, RLHF, interpretability) will be implemented in separate repositories using PyTorch, as they require:

- Large-scale infrastructure
- Established framework capabilities
- Production-grade tooling

The goal of this repository is to build **understanding** up to the transformer level. Beyond that, we leverage established frameworks while applying the deep understanding gained here.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Basic understanding of linear algebra
- Willingness to work through problems manually first

### Installation

```bash
# Clone the repository
git clone https://github.com/glody007/deep-learning-saitama-way
cd deep-learning-saitama-way

# No external dependencies required initially
# Pure Python implementation
```

### Usage

```python
from saitama_dl_framework.tensor import Tensor

# Create tensors
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])

# Operations
c = a.add(b)
d = a.matmul(b)
```

### Running Tests

```bash
python -m pytest tests/
```

---

## Learning Path

1. **Start with Level 1**: Complete the manual exercises in `exercises/01_linear_algebra_computation.md`
2. **Implement the code**: Build the tensor operations from scratch
3. **Test your understanding**: Run the tests and verify correctness
4. **Move to the next level**: Only proceed when you fully understand the current level

---

## Philosophy in Practice

### The Saitama Way Checklist

Before moving to the next level, ask yourself:

- [ ] Can I explain this concept to someone else?
- [ ] Can I implement it from memory?
- [ ] Have I worked through the math manually?
- [ ] Do I understand _why_ it works, not just _how_ to use it?

If you can't check all boxes, you're not ready for the next level. **That's okay.** Mastery takes time.

---

## Contributing

This is a personal learning project, but if you're following a similar path:

1. Focus on understanding over speed
2. Document your learning process
3. Share insights, not just code
4. Help others understand the fundamentals

---

## Resources

### Papers Referenced

- _Automatic Differentiation in Machine Learning_ (Baydin et al.)
- _Attention Is All You Need_ (Vaswani et al.)
- _Adam: A Method for Stochastic Optimization_ (Kingma & Ba)
- _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_ (Srivastava et al.)
- _Deep Residual Learning for Image Recognition_ (He et al.)
- _Scaling Laws for Neural Language Models_ (Kaplan et al.)
- _GPipe: Efficient Training of Giant Neural Networks_ (Huang et al.)
- _FlashAttention: Fast and Memory-Efficient Exact Attention_ (Dao et al.)

### Books

- _Deep Learning_ (Goodfellow, Bengio, Courville)
- _Neural Networks and Deep Learning_ (Nielsen)

---

## License

This project is for educational purposes. Use it to learn, understand, and build your own understanding of deep learning.

---

**Remember**: The goal isn't to build the fastest framework. The goal is to **understand** how deep learning works, from the ground up. The Saitama way: master the basics, and everything else follows.
