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
