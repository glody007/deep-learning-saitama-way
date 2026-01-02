# Week 1  Linear Algebra & Computation

**Goal: Think like a neural network**

Neural networks are not magic.

They are compositions of **linear algebra operations** applied at scale.

This week builds the mental model of a neural network using:

- Scalars
- Vectors
- Matrices

## MANUAL (By Hand — No Code)

### Exercise 1 — Scalars, Vectors, Matrices

Given:

Scalar:
s = 3

Vector:
v = [2, -1, 4]

Matrix:
M =
[ [1, 0, -1],
[2, 3, 4] ]

Tasks:

1. Compute s × v
2. Compute s × M
3. Explain what scaling does to:
    - a vector
    - a matrix

Neural intuition:
Scaling weights amplifies or attenuates neuron influence.

### Exercise 2 — Vector Addition & Subtraction

Given:

a = [3, 1, -2]

b = [1, 4, 0]

Tasks:

1. Compute a + b
2. Compute a − b
3. Explain why vectors must have the same shape
4. In neural networks, what does addition usually represent?

### Exercise 3 — Element-wise Multiplication (⊙)

Given:

x = [2, 3, 4]

g = [1, 0.5, 2]

Tasks:

1. Compute x ⊙ g
2. How is this different from a dot product?
3. Give one neural network operation that uses element-wise multiplication

Neural intuition:
Gating, masking, and attention rely on element-wise operations.

### Exercise 4 — Dot Product (Single Neuron)

Given:

x = [1, 2, 3]

w = [0.5, -1, 2]

Tasks:

1. Compute x · w
2. Rewrite as a weighted sum
3. Explain why this represents one neuron (no activation)

### Exercise 5 — Matrix × Vector (Dense Layer)

Given:

W =
[ [1, 0, -1],
[2, 1, 3] ]

x =
[ 2,
1,
0 ]

Tasks:

1. Compute W × x
2. How many neurons are in this layer?
3. What does each row of W represent?

### Exercise 6 — Matrix Addition & Subtraction

Given:

A =
[ [1, 2],
[3, 4] ]

B =
[ [5, 6],
[7, 8] ]

Tasks:

1. Compute A + B
2. Compute A − B
3. Why must matrix shapes match?
4. Where does this appear in neural networks?

### Exercise 7 — Matrix × Matrix (Stacked Layers)

Given:

A =
[ [1, 2],
[3, 4] ]

B =
[ [2, 0],
[1, 2] ]

Tasks:

1. Compute A × B
2. Why is A × B ≠ B × A?
3. What does stacking matrices mean in deep learning?

### Exercise 8 — Broadcasting (Bias Addition)

Given:

X =
[ [1, 2, 3],
[4, 5, 6] ]

b = [10, 20, 30]

Tasks:

1. Compute X + b manually
2. Explain broadcasting in your own words
3. What does b represent in a neural network?

## CODE (From Scratch — No Libraries)

### Exercise 9 — Tensor Structure

Implement a Tensor with:

- data (list or list of lists)
- shape attribute

Requirements:

- Support scalars, vectors, matrices
- Validate rectangular matrices

### Exercise 10 — Element-wise Operations

Implement:

- add
- sub
- mul (element-wise)

Rules:

- Same shape only
- No broadcasting
- Raise errors on mismatch

Neural intuition:
Activations, gradients, and masks flow element-wise.


### Exercise 11 — Matrix Multiplication

Implement:

- matmul
- add
- sub
- multiplication(element wise)

Rules:

- Validate shapes
- Triple nested loops
- No NumPy

Neural intuition:
This is the full forward pass of a dense layer.


## 🔬 THINKING EXERCISES

Answer in writing:

1. Why are most neural networks combinations of:
    - matmul
    - add
    - element-wise ops?
2. Why is shape safety critical?
3. What would silently break without shape checks?
4. Why are GPUs perfect for these operations?


## MASTERY CHECK

You pass Week 1 if you can:

- Perform dot products instantly
- Explain a dense layer using matrices only
- Implement matmul from memory
- Explain broadcasting intuitively