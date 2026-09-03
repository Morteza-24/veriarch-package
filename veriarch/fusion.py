"""Structural/semantic soft-clustering fusion.

Reimplements Mo2oM's NOCD-based branches (a lightweight Bernoulli-Poisson
GNN, following Shchur & Gunnemann 2019) and the weighted fusion of
Eq. 1 in methodology.tex, producing the reference prior M that the
Critic later checks the architect agent's proposals against.
"""

import numpy as np


def build_adjacency_from_similarity(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """Sparsify a dense similarity matrix into a symmetric k-NN adjacency,
    used to turn semantic-embedding cosine similarity into a graph the GNN
    can message-pass over."""
    y = sim.shape[0]
    adj = np.zeros_like(sim)
    for i in range(y):
        neighbors = np.argsort(-sim[i])[: k + 1]
        for j in neighbors:
            if j != i:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    adj_hat = adj + np.eye(adj.shape[0])
    deg = adj_hat.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    d = np.diag(deg_inv_sqrt)
    return d @ adj_hat @ d


def train_nocd(
    adj: np.ndarray,
    features: np.ndarray,
    num_services: int,
    hidden_dim: int = 128,
    epochs: int = 200,
    lr: float = 1e-2,
) -> np.ndarray:
    """Train a 2-layer GCN with a Bernoulli-Poisson edge objective to
    produce a soft, overlapping community-membership matrix in
    [0, 1]^{Y x num_services}. Called once for the structural graph and
    once for the semantic k-NN graph."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    y = adj.shape[0]
    a_norm = torch.tensor(_normalize_adj(adj), dtype=torch.float32)
    x = torch.tensor(features, dtype=torch.float32)
    a = torch.tensor(adj, dtype=torch.float32)

    class Gcn(nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()
            self.w1 = nn.Linear(in_dim, hidden)
            self.w2 = nn.Linear(hidden, out_dim)

        def forward(self, feats, adjacency_norm):
            h = functional.relu(adjacency_norm @ self.w1(feats))
            z = adjacency_norm @ self.w2(h)
            return functional.relu(z)

    model = Gcn(x.shape[1], hidden_dim, num_services)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edge_index = a.nonzero(as_tuple=False)
    all_pairs = torch.combinations(torch.arange(y), r=2)

    for _ in range(epochs):
        optimizer.zero_grad()
        z = model(x, a_norm)

        pu, pv = z[edge_index[:, 0]], z[edge_index[:, 1]]
        pos_dot = (pu * pv).sum(dim=1).clamp(min=1e-8)
        pos_loss = -torch.log1p(-torch.exp(-pos_dot) + 1e-8).mean()

        neg_dot = (z[all_pairs[:, 0]] * z[all_pairs[:, 1]]).sum(dim=1)
        neg_loss = neg_dot.mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        z = model(x, a_norm).numpy()
    z = z / (z.max() + 1e-8)
    return np.clip(z, 0.0, 1.0)


def fuse_membership(m_str: np.ndarray, m_sem: np.ndarray, alpha: float) -> np.ndarray:
    """M = alpha * M^sem + (1 - alpha) * M^str   (Eq. 1)."""
    return alpha * m_sem + (1 - alpha) * m_str
