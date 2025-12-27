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

# GNN Part 2: Social Network Analysis and Real-World Applications

## From Metrics to Flow: Understanding Social Dynamics

When we apply Graph Neural Networks (GNNs) to Social Network Analysis (SNA), we move from simple metrics (like "how many friends do you have?") to a deep, structural understanding of how influence, information, and behavior flow through a population.

In a social network, you aren't just a list of attributes; you are defined by the people you know and how they interact. GNNs capture this by treating the social network as a live "computational map."

---

## 1. Key Tasks in Social GNNs

Social platforms use GNNs to solve three main "puzzles":

### Node Classification (Who are you?)

Predicting a user's interests, political leaning, or even whether an account is a "bot" based on their neighborhood. 

**The mechanism:** If most of your neighbors are identified as "spam bots," the GNN aggregates that "bot-like" signal to your node.

**Mathematical formulation:**
$$P(\text{label}_v | G) = \text{softmax}(W \cdot h_v^{(K)})$$

Where $h_v^{(K)}$ is the final embedding after K layers of message passing.

**Real-world applications:**
- Bot detection on Twitter
- Interest prediction on Pinterest
- Political affiliation inference
- Fake account identification

---

### Link Prediction (Who should you know?)

This powers "People You May Know." Instead of just looking for mutual friends, the GNN looks at the embeddings (mathematical fingerprints) of two users. If their structural roles are similar, it predicts a future link.

**The mechanism:**
$$P(\text{edge}_{uv}) = \sigma(h_u^T h_v)$$

Or more sophisticated:
$$P(\text{edge}_{uv}) = \text{MLP}(\text{CONCAT}(h_u, h_v, |h_u - h_v|, h_u \odot h_v))$$

**Why this works better than traditional methods:**

Traditional approach: Count mutual friends
$$\text{score}(u,v) = |\mathcal{N}(u) \cap \mathcal{N}(v)|$$

GNN approach: Learn structural similarity
$$\text{score}(u,v) = \text{similarity}(\text{role}(u), \text{role}(v))$$

**Example:** Two users might have zero mutual friends but both be "bridge nodes" connecting different communities. GNN can identify this structural similarity.

---

### Community Detection (Where do you belong?)

Identifying cohesive "clusters" or "echo chambers." Unlike traditional clustering, GNNs find communities based on both topology (who follows whom) and content (what they talk about).

**Traditional clustering:**
- K-means: Uses only node features
- Louvain: Uses only graph structure

**GNN approach:** Combines both
$$\text{community}_v = \arg\max_c P(c | h_v^{(K)})$$

Where $h_v^{(K)}$ encodes both node features and graph structure.

**Applications:**
- Echo chamber identification
- Coordinated inauthentic behavior detection
- Interest-based group recommendation
- Tribal polarization analysis

---

## 2. The Math of Influence: Social Embeddings

In SNA, we use the Message Passing formula, but the features ($X$) often represent social data like interests, locations, or text from posts.

After $K$ layers of message passing, the final embedding $z_u$ for a user $u$ contains information from their $K$-hop neighborhood.

$$z_u = \text{COMBINE} \left( h_u^{(K-1)}, \text{AGGREGATE} \left( \{ h_v^{(K-1)} : v \in \mathcal{N}(u) \} \right) \right)$$

### Understanding Hops

**1-hop:** Your direct friends' influence
$$h_v^{(1)} = \sigma(W^{(0)} \sum_{u \in \mathcal{N}(v)} h_u^{(0)})$$

**2-hop:** Your "friends of friends" (the broader "vibe" of your social circle)
$$h_v^{(2)} = \sigma(W^{(1)} \sum_{u \in \mathcal{N}(v)} h_u^{(1)})$$

**K-hop:** The global community structure

**Key insight:** Each layer exponentially expands the receptive field. With 3 layers and average degree 50:
- Layer 0: Your features
- Layer 1: 50 friends
- Layer 2: 2,500 friends of friends
- Layer 3: 125,000 third-degree connections

This is why shallow GNNs (2-4 layers) capture global structure effectively.

---

## 3. Real-World Applications

### A. Viral Marketing & Influence Maximization

Companies use GNNs to identify "Super-Spreaders." These aren't just people with millions of followers; they are nodes that act as bridges between different communities.

#### The Influence Maximization Problem

**Goal:** Select K users to seed a campaign that maximizes total reach.

**Traditional approach:** Select highest-degree nodes (most followers)

**GNN approach:** Select highest-influence nodes (considering network structure)

$$\text{Influence}(u) = \sum_{v \in V} P(\text{activated}_v | \text{seed}_u)$$

Where $P(\text{activated}_v | \text{seed}_u)$ is estimated using:
1. GNN embeddings to predict cascade probability
2. Causal inference to isolate true influence (not just correlation)
3. Simulation of information diffusion

#### Why Bridges Matter More Than Hubs
```
Community A          Bridge Node          Community B
(100 users)              ↕               (100 users)
  ○ ○ ○                  ●                  ○ ○ ○
 ○ ○ ○ ○                                  ○ ○ ○ ○
  ○ ○ ○                                    ○ ○ ○
```

Seeding the bridge node reaches 200 users across two communities.
Seeding a hub within Community A reaches 100 users in one community.

**GNN identifies bridges through:**
- High betweenness centrality in learned embeddings
- Structural hole spanning in latent space
- Cross-community message passing patterns

---

### B. Recommendation Systems (Pinterest, Twitter, LinkedIn)

#### Case Study: PinSage (Pinterest's GNN)

Pinterest developed **PinSage**, a massive GNN that operates on billions of nodes. It treats "Pins" and "Boards" as a bipartite graph.

**Graph structure:**
- Nodes: Pins (images) and Boards (collections)
- Edges: "Pin P is on Board B"

**The insight:** By learning which Pins are often placed on the same Boards by similar users, it recommends content that is structurally related, even if the text descriptions are different.

**Why this beats traditional methods:**

Traditional collaborative filtering:
$$\text{score}(u, p) = \sum_{u' \sim u} \text{similarity}(u, u') \cdot \text{rating}(u', p)$$

PinSage (GNN-based):
$$\text{score}(u, p) = h_u^T h_p$$

Where:
- $h_u$ = user embedding (aggregated from their boards and engagement history)
- $h_p$ = pin embedding (aggregated from boards it appears on and similar pins)

**Results:**
- 150% increase in click-through rate
- Handles 3 billion nodes, 18 billion edges
- Updates in real-time as users interact

---

#### Architecture Details: How PinSage Scales

**Challenge:** Full-batch GNN training on billions of nodes is impossible.

**Solution:** Random walk-based sampling + importance pooling

**Algorithm:**

1. **Sampling:** For each target pin, perform random walks to sample K important neighbors (not all neighbors)

2. **Importance pooling:**
$$h_v = \text{pool}(\{h_u \cdot w_u : u \in \text{Sample}(\mathcal{N}(v))\})$$

Where $w_u$ measures importance of neighbor $u$.

3. **Multi-layer aggregation:**
```
Layer 1: Aggregate from sampled pins
Layer 2: Aggregate from sampled boards  
Layer 3: Combine for final embedding
```

4. **Efficient inference:** Pre-compute and cache embeddings, update incrementally.

**Key innovation:** They don't train on the full graph. They sample relevant subgraphs for each minibatch.

---

### C. Detecting Toxic Communities and Misinformation

GNNs are highly effective at spotting "coordinated inauthentic behavior." Because bots often have highly specific, non-human connection patterns, a GNN recognizes these "topological signatures" and flags them for review.

#### Bot Detection Patterns

**Normal user patterns:**
- Power-law degree distribution (few hubs, many low-degree nodes)
- High clustering coefficient (friends of friends are friends)
- Temporal variation in activity (sleep cycles, work hours)
- Diverse content interests

**Bot network patterns:**
- Abnormal degree uniformity (all bots have similar connection counts)
- Perfect cliques or stars (geometric structures)
- Synchronized activity (all bots active at same times)
- Repetitive content patterns

**GNN detection mechanism:**

$$P(\text{bot}_v) = f(h_v^{(K)}, \text{temporal}(v), \text{content}(v))$$

Where:
- $h_v^{(K)}$ captures structural anomalies
- $\text{temporal}(v)$ captures timing patterns
- $\text{content}(v)$ captures text repetition

**Why GNNs outperform feature-based classifiers:**

Traditional ML:
- Features: account age, tweet frequency, follower ratio
- Problem: Sophisticated bots mimic these statistics

GNN approach:
- Embedding captures: "You're connected to 47 other accounts that are all connected to each other in a perfect star pattern, and you all tweeted the same message within 3 minutes"
- This structural pattern is nearly impossible for bots to avoid while remaining effective

---

#### Coordinated Inauthentic Behavior (CIB) Detection

**Facebook's approach (published research):**

1. Build temporal interaction graph
   - Nodes: accounts
   - Edges: interactions (likes, shares, comments) with timestamps

2. Train GNN to learn embeddings

3. Cluster embeddings to find coordinated groups

4. Flag clusters with:
   - High internal interaction density
   - Low external interaction
   - Synchronized timing patterns
   - Centralized control structure (one node with high out-degree)

**Mathematical formulation:**

$$\text{CIB\_score}(C) = \frac{\text{density}(C)}{\text{external\_edges}(C)} \cdot \text{synchrony}(C) \cdot \text{centralization}(C)$$

Where:
- $\text{density}(C) = \frac{|E_C|}{|V_C|(|V_C|-1)/2}$
- $\text{synchrony}(C) = 1 - \text{variance}(\text{timestamps}(C))$
- $\text{centralization}(C) = \max_v \frac{\text{degree}(v) - \text{avg\_degree}}{\text{max\_possible\_degree}}$

---

## 4. Why This Matters for the "Next Phase"

Traditional social analysis was "static"—it looked at a snapshot. GNNs are often **Temporal (Dynamic Graphs)**, meaning they learn how the network evolves.

### Temporal Graph Networks (TGN)

**Challenge:** Social networks change constantly. New users join, relationships form and break, content goes viral and fades.

**Static GNN assumption:** Graph structure is fixed during training.

**Reality:** Graph at time $t+1$ differs from graph at time $t$.

**Temporal GNN approach:**

$$h_v(t) = \text{GRU}(h_v(t-1), \text{AGG}(\{(h_u(t), e_{uv}(t)) : (u,v) \in E(t)\}))$$

Where:
- $h_v(t)$ = node embedding at time $t$
- $e_{uv}(t)$ = edge features at time $t$ (interaction type, timestamp)
- GRU = recurrent unit that maintains memory across time

**Applications:**
- Predicting future connections before they form
- Detecting emerging communities
- Early warning for viral content
- Tracking influence evolution over time

---

### Dynamic Graph Example: Twitter During Elections

**Observation window:** 3 months before election

**Graph evolution:**
```
Month 1: Sparse cross-party interactions
Month 2: Polarization increases, bridge nodes decrease
Month 3: Nearly complete segregation into echo chambers
```

**Temporal GNN captures:**
- How individual users shift positions (centrists move toward extremes)
- How new edges form preferentially within communities (homophily strengthens)
- How information diffusion changes (cross-party sharing drops)

**Prediction task:** Given trajectory through Month 2, predict:
- Which users will become highly active in Month 3
- Which communities will grow fastest
- Which bridge nodes will remain vs. disconnect

**Why temporal matters:**

Static GNN on Month 3 snapshot: "These communities are polarized"
Temporal GNN on Month 1-3 sequence: "Community A is radicalizing rapidly, Community B is stable, Bridge nodes are under pressure and likely to disconnect"

The second provides actionable insight.

---

## 5. Comparison: Traditional SNA vs. GNN-based SNA

| Feature | Traditional SNA | GNN-based SNA |
|---------|----------------|---------------|
| User Data | Demographics only | Demographics + Social Context |
| Connections | Binary (Friend/Not Friend) | Weighted (Strength of influence) |
| Scaling | Struggles with millions of nodes | Handles billions (via sampling like GraphSAGE) |
| Logic | Heuristic (e.g., "Common Neighbors") | Learned (Optimizes for a specific task) |
| Temporal | Static snapshots | Dynamic evolution |
| Features | Hand-crafted (centrality, clustering) | Learned embeddings |
| Prediction | Regression on features | End-to-end neural prediction |
| Interpretability | High (clear metrics) | Lower (black box embeddings) |
| Accuracy | Moderate | State-of-the-art |

---

## 6. Advanced Topics

### Signed Networks (Friend/Foe Relationships)

Social networks aren't just positive connections. There are also:
- Dislikes, blocks, unfollows
- Upvotes vs. downvotes (Reddit)
- Trust vs. distrust

**Signed GNN approach:**

$$h_v = \sigma \left( W^+ \sum_{u \in \mathcal{N}^+(v)} h_u + W^- \sum_{u \in \mathcal{N}^-(v)} h_u \right)$$

Where:
- $\mathcal{N}^+(v)$ = positive neighbors (friends)
- $\mathcal{N}^-(v)$ = negative neighbors (foes)
- $W^+, W^-$ = separate weight matrices

**Application:** Predicting social balance theory
- "Friend of friend = friend"
- "Enemy of enemy = friend"  
- "Friend of enemy = enemy"

---

### Attributed Networks (Rich Node Features)

Social network nodes aren't just IDs. They have:
- Profile information (age, location, occupation)
- Behavioral data (posting frequency, engagement patterns)
- Content data (text of posts, images shared)

**GNN approach:** Initialize node features with these attributes

$$h_v^{(0)} = \text{CONCAT}(\text{profile}(v), \text{behavior}(v), \text{content\_embedding}(v))$$

Then aggregate through layers:

$$h_v^{(k+1)} = \sigma \left( W^{(k)} \left[ h_v^{(k)}, \text{AGG}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}) \right] \right)$$

**Key insight:** Network structure + node attributes = powerful predictions

---

### Multi-Relational Networks

Social platforms have multiple relationship types:
- Twitter: follow, retweet, mention, reply
- LinkedIn: connection, endorsement, recommendation
- Facebook: friend, family, colleague, in same group

**Heterogeneous GNN approach:**

$$h_v = \sigma \left( \sum_{r \in R} W_r \sum_{u \in \mathcal{N}_r(v)} h_u \right)$$

Where $R$ = set of relation types

**Why this matters:** Different relations carry different semantics
- Follow ≠ strong connection (can be one-way, passive)
- Retweet = active endorsement (stronger signal)
- Reply = engagement (bidirectional interaction)

Heterogeneous GNN learns these distinctions automatically.

---

## 7. Implementation: Social Network GNN
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

class SocialGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=3):
        super(SocialGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, num_classes))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# Example: Bot detection on social network
def train_bot_detector(data):
    model = SocialGNN(
        num_features=data.num_features,
        hidden_dim=128,
        num_classes=2,  # bot vs. human
        num_layers=3
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(
