# Graph Neural Networks: From Foundations to Applications

Two comprehensive guides covering GNN theory and real-world implementation.

---

## GNN Part 1: Foundations and Mathematics

### Introduction: The Shift from Grids to Graphs

In the "Algebraic Phase" of AI, we treated everything like a grid (images) or a sequence (text). But most of the real world—molecules, social networks, the power grid, and even the "knowledge graph" in your brain—is a Graph.

---

### 1. The Geometry: Nodes, Edges, and Topology

In a standard neural network, the input has a fixed size and order. In a graph, there is no "first" node or "left-to-right" flow.

**Core Components:**

- **Nodes ($V$)**: The entities (e.g., atoms in a molecule, people in a network)
- **Edges ($E$)**: The relationships (e.g., chemical bonds, friendships)
- **Adjacency Matrix ($A$)**: A square matrix where $A_{ij} = 1$ if node $i$ and node $j$ are connected, and $0$ otherwise

---

### 2. The Math: Message Passing

The "magic" of GNNs is the Message Passing phase. Instead of looking at the whole graph at once (which is computationally impossible for large networks), each node "talks" to its neighbors to update its own understanding of the world.

This happens in two steps: **Aggregate** and **Update**.

#### The Formula

For a node $v$, its state (or "embedding") at the next layer $k+1$ is calculated as:

$$h_v^{(k+1)} = \sigma \left( W^{(k)} \cdot \text{AGGREGATE} \left( h_v^{(k)}, \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right) \right)$$

#### Breaking down the variables

- $h_v^{(k)}$: The current features of node $v$
- $\mathcal{N}(v)$: The set of neighbor nodes connected to $v$
- $\text{AGGREGATE}$: A function (like Sum, Mean, or Max) that collects info from neighbors. It must be permutation invariant (the order of neighbors shouldn't matter)
- $W^{(k)}$: The learnable weight matrix (the "intelligence" the AI is training)
- $\sigma$: A non-linear activation function (like ReLU)

---

### 3. Why This Solves "Wall 1" (Combinatorial Explosion)

Traditional algebra assumes "smooth landscapes," but real problems like Protein Folding are discrete.

GNNs solve this by encoding the inductive bias of the problem. If you are predicting how a protein folds (AlphaFold), you don't treat every atom as a random point in space; you treat them as nodes in a graph where edges represent chemical bonds and physical proximity. 

This drastically reduces the search space because the AI "knows" atoms influence only those they are connected to or near.

---

### 4. Advanced GNN: The Graph Convolutional Network (GCN)

One of the most popular versions is the GCN. It simplifies the message-passing into a specialized form of matrix multiplication that uses the normalized Laplacian of the graph.

$$Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X W$$

**Components:**

- $\tilde{A}$: The adjacency matrix plus self-connections (so nodes "talk" to themselves too)
- $\tilde{D}$: The degree matrix (how many neighbors each node has), used for normalization so that nodes with 1,000 neighbors don't "overpower" the signal from nodes with 2 neighbors
- $X$: Input feature matrix
- $W$: Learnable weight matrix

---

### 5. Types of Aggregation Functions

Different GNN architectures use different aggregation strategies:

#### Mean Aggregation
$$h_v^{(k+1)} = \sigma \left( W \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(k)} \right)$$

**Advantage:** Stable across different node degrees

#### Sum Aggregation
$$h_v^{(k+1)} = \sigma \left( W \cdot \sum_{u \in \mathcal{N}(v)} h_u^{(k)} \right)$$

**Advantage:** Preserves total information from neighborhood

#### Max Aggregation
$$h_v^{(k+1)} = \sigma \left( W \cdot \max_{u \in \mathcal{N}(v)} h_u^{(k)} \right)$$

**Advantage:** Captures most salient features

#### Attention-Based Aggregation (Graph Attention Networks)
$$h_v^{(k+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u^{(k)} \right)$$

Where $\alpha_{vu}$ are learned attention weights that determine how much node $v$ should attend to node $u$.

---

### 6. Key GNN Architectures

#### Graph Convolutional Networks (GCN)
- Uses normalized adjacency matrix
- Efficient spectral convolution
- Good for semi-supervised node classification

#### GraphSAGE (Graph Sample and Aggregate)
- Samples fixed-size neighborhoods
- Scales to billion-node graphs
- Inductive learning (works on unseen nodes)

$$h_v^{(k)} = \sigma \left( W \cdot \text{CONCAT}(h_v^{(k-1)}, \text{AGG}(\{h_u^{(k-1)}, \forall u \in \mathcal{N}(v)\})) \right)$$

#### Graph Attention Networks (GAT)
- Learns importance weights for neighbors
- Self-attention mechanism
- Handles varying node degrees naturally

#### Message Passing Neural Networks (MPNN)
- General framework encompassing many GNN variants
- Explicit message and update functions
- Flexible for different graph types

---

### 7. Training Graph Neural Networks

#### Node-Level Tasks
Predict properties of individual nodes (e.g., user classification, atom type prediction)

**Loss function:**
$$\mathcal{L} = \sum_{v \in V_{labeled}} \text{CrossEntropy}(y_v, \hat{y}_v)$$

#### Edge-Level Tasks
Predict relationships between nodes (e.g., link prediction, molecular bond prediction)

**Loss function:**
$$\mathcal{L} = \sum_{(u,v) \in E} \text{BCE}(A_{uv}, \sigma(h_u^T h_v))$$

#### Graph-Level Tasks
Predict properties of entire graphs (e.g., molecule toxicity, social network type)

**Approach:** Add readout function to aggregate all node embeddings

$$h_G = \text{READOUT}(\{h_v^{(K)} | v \in G\})$$

Common readout functions: sum, mean, max pooling, or attention-based pooling.

---

### 8. The Power of Multi-Hop Aggregation

Each GNN layer allows information to propagate one hop further:

- **1 layer**: Node knows its immediate neighbors
- **2 layers**: Node knows neighbors of neighbors
- **K layers**: Node has information from K-hop neighborhood

**The receptive field grows exponentially:**
- Layer 1: Average degree $d$ nodes
- Layer 2: $d^2$ nodes
- Layer K: $d^K$ nodes

**Trade-off:** More layers = more information, but also:
- Over-smoothing (all nodes become similar)
- Computational cost increases
- Risk of overfitting

**Typical practice:** 2-4 layers for most tasks

---

### 9. Handling Different Graph Types

#### Directed Graphs
Use separate weight matrices for incoming and outgoing edges:

$$h_v = \sigma(W_{in} \sum_{u \to v} h_u + W_{out} \sum_{v \to u} h_u)$$

#### Weighted Graphs
Incorporate edge weights into aggregation:

$$h_v = \sigma \left( W \sum_{u \in \mathcal{N}(v)} w_{uv} \cdot h_u \right)$$

#### Heterogeneous Graphs
Different node types and edge types require type-specific transformations:

$$h_v^{t} = \sigma \left( \sum_{r \in R} \sum_{u \in \mathcal{N}_r(v)} W_r h_u \right)$$

Where $R$ is the set of edge types (relations).

#### Dynamic/Temporal Graphs
Edges and nodes change over time:

$$h_v(t) = \text{RNN}(h_v(t-1), \text{AGG}(\{h_u(t) : u \in \mathcal{N}(v, t)\}))$$

---

### 10. Comparison: Algebra vs. Graphs

| Feature | Standard Algebra (ML) | Graph Neural Networks |
|---------|----------------------|----------------------|
| Data Structure | Vectors / Tensors (Flat) | Non-Euclidean Graphs (Linked) |
| Relationship | Implicit (learned) | Explicit (defined by edges) |
| Invariance | Translation (CNNs) | Permutation (Order doesn't matter) |
| Scaling | Fixed input size | Any graph size/shape |
| Inductive Bias | Grid structure | Network structure |
| Operation | Convolution on grid | Message passing on graph |

---

### 11. Limitations and Challenges

#### Over-Smoothing
After many layers, node representations become indistinguishable.

**Solutions:**
- Residual connections: $h_v^{(k+1)} = h_v^{(k)} + \Delta h_v^{(k+1)}$
- Layer normalization
- Jumping knowledge connections

#### Scalability
Full-batch training on large graphs is memory-intensive.

**Solutions:**
- Mini-batch sampling (GraphSAGE)
- Cluster-based methods
- Graph sampling strategies

#### Expressiveness
Standard GNNs cannot distinguish certain graph structures (related to Weisfeiler-Leman test).

**Solutions:**
- Higher-order GNNs
- Subgraph-based methods
- Add node/edge features

#### Heterophily
When connected nodes are dissimilar (opposite of homophily).

**Solutions:**
- Attention mechanisms
- Ego/alter embeddings
- Signed GNNs

---

### 12. Implementation Example: Simple GNN Layer
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        """
        x: Node feature matrix (N x in_features)
        adj: Adjacency matrix (N x N)
        """
        # Aggregate: sum of neighbor features
        aggregated = torch.matmul(adj, x)
        
        # Transform: apply linear transformation
        out = self.linear(aggregated)
        
        # Activate: apply non-linearity
        return F.relu(out)

class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GNNLayer(in_features, hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_features, hidden_features))
        
        # Output layer
        self.layers.append(GNNLayer(hidden_features, out_features))
    
    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # No activation on final layer for node classification
        x = self.layers[-1](x, adj)
        return x
```

---

### 13. Key Takeaways

1. **GNNs extend deep learning to graph-structured data** by replacing fixed-size operations with permutation-invariant aggregations.

2. **Message passing is the core mechanism**: nodes iteratively aggregate information from neighbors.

3. **The graph structure provides inductive bias**, drastically reducing search spaces for combinatorial problems.

4. **Different architectures optimize for different properties**: GCN for efficiency, GAT for flexibility, GraphSAGE for scalability.

5. **Multi-hop aggregation allows nodes to gather information from distant parts of the graph**, but too many layers cause over-smoothing.

6. **GNNs work on any graph type**: social networks, molecules, knowledge graphs, transportation networks, citation networks.

---

## Further Reading

### Foundational Papers
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (GCN)
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (GraphSAGE)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (GAT)

### Surveys and Reviews
- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)

### Libraries and Tools
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Deep Graph Library (DGL)](https://www.dgl.ai/)
- [Spektral (Keras/TensorFlow)](https://graphneural.network/)

---


