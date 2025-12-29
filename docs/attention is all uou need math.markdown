The paper

Title: Attention Is All You Need
Published: 2017
Authors:
Ashish Vaswani
Noam Shazeer
Niki Parmar
Jakob Uszkoreit
Llion Jones
Aidan N. Gomez
Łukasz Kaiser
Illia Polosukhin

All were researchers at Google Brain and Google Research.

What problem they solved

Before this paper, language models relied on recurrence or convolution.

Recurrent models processed text one step at a time. This slowed training and made long-range relationships weak.

Convolutional models improved speed but still struggled with long dependencies.

The authors asked a simple question.
What if sequence order did not require recurrence at all?

The mathematical foundation

The entire model is built on linear algebra, probability, and optimization.

1. Vector representations

Each token becomes a vector in a high-dimensional space.

These vectors are learned embeddings. Meaning is represented by position and direction, not symbols.

2. Scaled dot-product attention

For each token, the model computes:

Q = query
K = key
V = value

Attention is calculated as:

Attention(Q, K, V) = softmax((Q · Kᵀ) / √dₖ) V

This does three things:

Measures similarity between tokens

Normalizes influence with softmax

Produces a weighted sum of information

This is pure linear algebra and probability.

3. Multi-head attention

Instead of one attention operation, the model runs several in parallel.

Each head learns a different type of relationship:

syntax

distance

agreement

structure

The outputs are concatenated and projected back into a single space.

This allows the model to view language from multiple mathematical perspectives at once.

4. Positional encoding

Because the model has no inherent sense of order, position is added using sine and cosine functions.

These functions encode position continuously and allow the model to generalize to longer sequences.

No recurrence. No memory cells. Only math.

5. Optimization and learning

Training uses gradient descent.

Loss is computed by comparing predicted tokens to true ones.

Gradients flow backward through the network using the chain rule.

Weights update incrementally until prediction error drops.

Why it mattered

The design allowed full parallelization. Training became faster and more stable.

Sequence length stopped being a hard limit.

Performance scaled with data and compute rather than architectural complexity.

This is why nearly all modern language models use this structure.

The real contribution

The paper did not invent intelligence.

It proved that attention alone can model structure, context, and meaning.

It replaced handcrafted linguistic rules with mathematical alignment.

That insight reshaped machine learning.

And it started a shift where scale, not symbolism, became the primary driver of progres
