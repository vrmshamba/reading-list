# The Calculus Evolution: From Algebra to Causality to Graphs

A mathematical lineage tracing how solving for unknowns evolved from Al-Khwarizmi's algebra to modern AI.

---


## The Mathematical Progression
```
Classical Algebra (Al-Khwarizmi, 820 CE)
    ↓
Classical Calculus (Newton/Leibniz, 1680s)
    ↓
Multivariable Calculus (Euler, 1700s)
    ↓
Vector Calculus (Maxwell, 1860s)
    ↓
Tensor Calculus (Einstein, 1915)
    ↓
[BRANCHING POINT]
    ↓
┌───────────────┴───────────────┐
│                               │
Graph Calculus              Causal Calculus
(Spectral Theory)           (Pearl, 2000s)
1990s-present               
│                               │
└───────────────┬───────────────┘
                ↓
    Modern AI (Integration)
        2020s-present
```

---

## Stage 1: Classical Calculus (The Foundation)

### From Algebra to Calculus

**Al-Khwarizmi gave us:**
$$x^2 + 10x = 39$$

**Newton/Leibniz asked:** "What if $x$ is changing?"

$$\frac{d}{dt}(x^2 + 10x) = \frac{dx}{dt}(2x + 10)$$

**The key insight:** Rate of change is itself a function

**Why this mattered for AI:**
- Gradient descent needs derivatives
- Optimization is finding where $\frac{df}{dx} = 0$
- Backpropagation is chain rule applied repeatedly

---

## Stage 2: Multivariable Calculus (Scaling Up)

### From One Variable to Many

**Single variable:**
$$f(x) = x^2$$
$$\frac{df}{dx} = 2x$$

**Multiple variables:**
$$f(x, y, z) = x^2 + 2y^3 + 3z$$

**Partial derivatives:**
$$\frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = 6y^2, \quad \frac{\partial f}{\partial z} = 3$$

**The gradient (vector of all partial derivatives):**
$$\nabla f = \begin{bmatrix} 2x \\ 6y^2 \\ 3 \end{bmatrix}$$

**Why this mattered for AI:**
- Neural networks have millions of parameters
- Need to compute $\frac{\partial \text{Loss}}{\partial w_i}$ for each weight $w_i$
- Gradient tells you which direction to adjust each parameter

---

## Stage 3: Vector Calculus (Fields and Flows)

### From Points to Fields

**New question:** "How do things flow through space?"

**Operations:**
- **Gradient** ($\nabla f$): How steep is the climb?
- **Divergence** ($\nabla \cdot \mathbf{F}$): Is stuff flowing in or out?
- **Curl** ($\nabla \times \mathbf{F}$): Is it spinning?

**Example: Heat flow**
$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

Where:
- $T$ = temperature at each point
- $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$ (Laplacian)
- Heat flows from hot to cold

**Why this mattered for AI:**
- Convolutional neural networks use discrete versions of these operators
- Information "flows" through network layers
- Attention mechanisms compute "flows" of importance

---

## Stage 4: Tensor Calculus (Einstein's Geometry)

### From Flat Space to Curved Space

**Classical calculus assumes:** Euclidean (flat) geometry

**Einstein needed:** Calculations on curved spacetime

**The tool:** Tensors (generalized matrices)

**Metric tensor** $g_{\mu\nu}$: describes how to measure distance in curved space

**Einstein field equations:**
$$R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}$$

(Calculus on curved surfaces)

**Why this mattered for AI:**
- Neural networks operate in high-dimensional curved spaces
- Parameter space is not flat
- Natural gradient descent uses Riemannian geometry (like Einstein's)

---

## BRANCHING POINT: Two Different Extensions

After tensor calculus, mathematics branched into two directions relevant for AI.

---

## Branch 1: Graph Calculus (Discrete Structures)

### The Problem with Classical Calculus

Classical calculus assumes **continuous** space:
- You can take infinitely small steps
- Derivatives exist everywhere
- Space is smooth

**Real networks are discrete:**
- Social networks: distinct people
- Molecules: distinct atoms
- Road networks: distinct intersections

**You can't take a "derivative" of a graph... or can you?**

---

### Graph Calculus: Discrete Derivatives

#### 1. The Graph Laplacian

For a graph with nodes $V$ and edges $E$:

**Adjacency matrix** $A$:
$$A_{ij} = \begin{cases} 1 & \text{if edge between } i \text{ and } j \\ 0 & \text{otherwise} \end{cases}$$

**Degree matrix** $D$:
$$D_{ii} = \text{number of neighbors of node } i$$

**Graph Laplacian:**
$$L = D - A$$

**Why "Laplacian"?**

In continuous calculus, the Laplacian is:
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

It measures "how different is this point from its neighbors"

**On a graph:**
$$(Lf)_i = \sum_{j \sim i} (f_i - f_j)$$

Same idea! How different is node $i$'s value from its neighbors?

---

#### 2. Graph Derivatives

**Classical derivative:**
$$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Graph derivative** (discrete approximation):
$$\frac{\partial f}{\partial e_{ij}} = f_j - f_i$$

The "derivative" along edge $e_{ij}$ is just the difference in values.

---

#### 3. Graph Gradient

For a function $f$ on nodes, the **graph gradient** is:

$$\nabla_G f = \begin{bmatrix} f_1 - f_2 \\ f_1 - f_3 \\ f_2 - f_4 \\ \vdots \end{bmatrix}$$

(one entry per edge)

---

#### 4. Spectral Graph Theory

**Key insight:** The eigenvectors of $L$ tell you about graph structure

**The eigenvalue equation:**
$$L \mathbf{v} = \lambda \mathbf{v}$$

**What eigenvectors reveal:**
- Smallest eigenvalue $\lambda_1 = 0$ (always)
- Second smallest $\lambda_2$ = "algebraic connectivity" (how well-connected the graph is)
- Higher eigenvalues capture finer structural patterns

**Graph Fourier Transform:**

Just like classical Fourier transforms decompose signals into frequencies, graph Fourier transforms decompose graph signals into "graph frequencies"

$$\hat{f} = U^T f$$

Where $U$ = eigenvectors of $L$

---

#### 5. Graph Convolution

**Classical convolution** (used in image CNNs):
$$(f * g)(x) = \int f(x-y) g(y) dy$$

**Graph convolution:**
$$f *_G g = U \left( (U^T f) \odot (U^T g) \right)$$

Where:
- $U$ = eigenvectors of graph Laplacian
- $\odot$ = element-wise multiplication
- This is **spectral graph convolution**

**Simplified version (spatial graph convolution):**
$$h_v^{(k+1)} = \sigma \left( W^{(k)} \sum_{u \in \mathcal{N}(v)} \frac{h_u^{(k)}}{\sqrt{d_v d_u}} \right)$$

This is what Graph Neural Networks actually use!

---

### Summary: Graph Calculus Operations

| Classical Calculus | Graph Calculus | Meaning |
|-------------------|----------------|---------|
| $\frac{\partial f}{\partial x}$ | $f_j - f_i$ | Rate of change |
| $\nabla f$ | Edge differences | Gradient |
| $\nabla^2 f$ | $Lf = (D-A)f$ | Laplacian |
| Fourier Transform | $U^T f$ | Frequency decomposition |
| Convolution | $U((U^T f) \odot (U^T g))$ | Filtering |

---

## Branch 2: Causal Calculus (Pearl's Framework)

### The Problem with Statistical Calculus

**Classical probability:**
$$P(Y | X) = \frac{P(X, Y)}{P(X)}$$

This tells you: "If we observe $X$, what's the probability of $Y$?"

**But it doesn't tell you:** "If we change $X$, what happens to $Y$?"

**Example:**
- $P(\text{Fire} | \text{Smoke})$ is high
- But making more smoke doesn't cause fires!
- Correlation ≠ Causation

---

### Pearl's do-Calculus: The New Operations

Pearl added a new operator to probability theory: $do(X = x)$

**Three levels of reasoning:**

1. **Seeing** (Association): $P(Y | X = x)$
   - "I observe $X = x$, what's likely for $Y$?"
   - This is standard statistics

2. **Doing** (Intervention): $P(Y | do(X = x))$
   - "I force $X = x$, what happens to $Y$?"
   - This is causal inference

3. **Imagining** (Counterfactual): $P(Y_x | X' = x', Y' = y')$
   - "Given I saw $X' = x'$ and $Y' = y'$, what would $Y$ have been if I had done $X = x$?"
   - This is counterfactual reasoning

---

### The Three Rules of do-Calculus

**Rule 1: Insertion/Deletion of observations**
$$P(y | do(x), z, w) = P(y | do(x), w) \quad \text{if } (Y \perp\!\!\!\perp Z | X, W)_{\overline{X}}$$

**Rule 2: Action/observation exchange**
$$P(y | do(x), do(z), w) = P(y | do(x), z, w) \quad \text{if } (Y \perp\!\!\!\perp Z | X, W)_{\overline{X}, \underline{Z}}$$

**Rule 3: Insertion/deletion of actions**
$$P(y | do(x), do(z), w) = P(y | do(x), w) \quad \text{if } (Y \perp\!\!\!\perp Z | X, W)_{\overline{X}, \overline{Z(W)}}$$

(The notation $\overline{X}$ means "remove incoming arrows to $X$" in causal graph)

---

### Causal Calculus Operations

#### 1. The Backdoor Adjustment

If you want to know the causal effect of $X$ on $Y$, but there are confounders $Z$:

$$P(y | do(x)) = \sum_z P(y | x, z) P(z)$$

**Intuition:** Control for confounders by averaging over their values

---

#### 2. The Front-door Adjustment

If you can't measure confounders, but you know a mediator $M$:

$$P(y | do(x)) = \sum_m P(m | x) \sum_{x'} P(y | x', m) P(x')$$

**Intuition:** Trace the causal path through the mediator

---

#### 3. Instrumental Variables

If you have an instrument $Z$ that affects $X$ but not $Y$ directly:

$$\text{Causal effect} = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, X)}$$

---

### Graphical Representation: Causal DAGs

Causal relationships are represented as **Directed Acyclic Graphs (DAGs)**:
```
      Z (Confounder)
     ↙ ↘
    X → Y
```

**Operations on causal graphs:**
- **Mutilate** ($do(X=x)$): Remove all incoming arrows to $X$
- **Condition** ($P(Y|X=x)$): Keep graph unchanged, just filter data

---

## The Integration: Where Graph Calculus Meets Causal Calculus

### The Problem AI is Solving Now

**We have:**
- **Graph Calculus**: Math for discrete network structures
- **Causal Calculus**: Math for cause-and-effect reasoning

**We need:**
- **Causal reasoning on graph structures**

---

### Causal Graph Neural Networks

**Standard GNN:**
$$h_v^{(k+1)} = \sigma \left( W^{(k)} \text{AGG} \left( \{h_u^{(k)} : u \in \mathcal{N}(v)\} \right) \right)$$

**Problem:** This treats all edges equally (correlation)

**Causal GNN:**
$$h_v^{(k+1)} = \sigma \left( W^{(k)} \sum_{u \in \mathcal{N}(v)} \alpha_{uv} \cdot h_u^{(k)} \right)$$

Where $\alpha_{uv}$ is the **causal strength** from $u$ to $v$, estimated using:

$$\alpha_{uv} = P(h_v | do(h_u)) - P(h_v | do(\neg h_u))$$

---

### The Math Integration

**Combining both frameworks:**

1. **Graph Laplacian** (from graph calculus):
   $$L = D - A$$

2. **Causal weights** (from causal calculus):
   $$W_{ij}^{\text{causal}} = P(Y_j | do(X_i)) - P(Y_j)$$

3. **Causal Graph Laplacian**:
   $$L^{\text{causal}} = D^{\text{causal}} - W^{\text{causal}}$$

Where:
- $D^{\text{causal}}$ = diagonal matrix of causal influence strengths
- $W^{\text{causal}}$ = matrix of pairwise causal effects

---

### Example: Viral Spread with Causality

**Standard GNN approach:**
"User A shares → Users B, C, D might share" (correlation)

**Causal GNN approach:**
"If we $do(\text{User A shares})$, then User B shares with probability 0.8, but User C shares with probability 0.1 (even though both are connected)"

**The formula:**

$$P(\text{Share}_v | do(\text{Share}_u)) = \sigma \left( \sum_{u \in \mathcal{N}(v)} \beta_{uv}^{\text{causal}} \cdot I(\text{Share}_u) \right)$$

Where:
- $\beta_{uv}^{\text{causal}}$ = causal treatment effect of $u$ on $v$
- Estimated from observational data using backdoor adjustment
- Encoded into graph structure

---

## Where You Got Lost (And How to Get Back)

**The confusion point:**

You went from:
1. ✅ Algebra (clear)
2. ✅ Calculus (clear)
3. ✅ Graph Neural Networks (clear)
4. ❓ "Graphical calculus" + "Causal calculus" (blurry)

**The clarification:**

**"Graphical calculus"** is informal term for:
- Graph Laplacian operators
- Spectral graph theory
- Discrete differential geometry

**"Causal calculus"** is formal term for:
- Pearl's do-calculus
- Causal inference rules
- Counterfactual reasoning

**They are separate branches that AI is now merging**

---

## The Clear Lineage
```
1. Al-Khwarizmi (820 CE): Solve for unknown x
   ↓
2. Newton/Leibniz (1680s): What if x changes? → dx/dt
   ↓
3. Euler (1740s): What if many variables change? → ∇f
   ↓
4. Maxwell (1860s): What if things flow? → ∇·F, ∇×F
   ↓
5. Einstein (1915): What if space curves? → Tensor calculus
   ↓
6. SPLIT (1990s-2000s):
   ├─ Graph Theory: What if discrete not continuous? → Graph Laplacian
   └─ Pearl: What if correlation ≠ causation? → do-calculus
   ↓
7. TODAY (2020s): Can we combine them? → Causal GNNs
```

---

## The Key Insight

**The evolution of "solving for unknowns":**

- **Algebra**: Solve $x^2 + 10x = 39$ → Find $x$
- **Calculus**: Minimize $f(x)$ → Find $x$ where $\frac{df}{dx} = 0$
- **Multivariable**: Minimize $f(x_1, ..., x_n)$ → Find where $\nabla f = 0$
- **Graph Calculus**: Minimize $f$ on graph → Find $\mathbf{x}$ where $L\mathbf{x} = 0$
- **Causal**: Find $X$ that causes $Y$ → Identify where $P(Y|do(X)) - P(Y) \neq 0$
- **Causal + Graph**: Find influential nodes that cause cascades → Maximize $\sum_v P(\text{Adopt}_v | do(\text{Seed}_u))$

**It's all still algebra at heart: solving for unknowns.**

The unknowns just got more sophisticated:
- From numbers ($x = 5$)
- To functions ($f(x)$)
- To vectors ($\mathbf{x}$)
- To graph signals ($f_v$ for each node $v$)
- To causal effects ($\beta_{uv}^{\text{causal}}$)

---

## Further Reading

### Graph Calculus
- [Spectral Graph Theory](https://mathweb.ucsd.edu/~fan/research/revised.html) by Fan Chung
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1812.08434)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

### Causal Calculus
- [The Book of Why](http://bayes.cs.ucla.edu/WHY/) by Judea Pearl
- [Causal Inference in Statistics: A Primer](http://bayes.cs.ucla.edu/PRIMER/)
- [DoWhy: Python library for causal inference](https://github.com/py-why/dowhy)

### Integration
- [Causal Graph Neural Networks](https://arxiv.org/abs/2011.02534)
- [Learning Causal Effects on Hypergraphs](https://arxiv.org/abs/2207.04049)

---

## Repository Structure
```
calculus-evolution/
├── README.md (this file)
├── 01-classical-calculus.md
├── 02-graph-calculus.md
├── 03-causal-calculus.md
├── 04-integration.md
├── code/
│   ├── graph_laplacian.py
│   ├── causal_inference.py
│   └── causal_gnn.py
└── notebooks/
    ├── graph_calculus_examples.ipynb
    ├── causal_calculus_examples.ipynb
    └── viral_spread_simulation.ipynb
```
