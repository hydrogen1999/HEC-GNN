"""
layers.py -- GIN, GINE, MPNN layers, and scatter operations.

Self-contained: NO PyTorch Geometric dependency.
All message passing is implemented via scatter operations on edge_index tensors.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Scatter operations
# ============================================================================

def scatter_add(src: torch.Tensor, index: torch.LongTensor, dim_size: int) -> torch.Tensor:
    """out[index[i]] += src[i]  along dim 0."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.LongTensor, dim_size: int) -> torch.Tensor:
    """out[index[i]] = mean(src[i], ...)."""
    s = scatter_add(src, index, dim_size)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), ones)
    count = count.clamp(min=1.0)
    return s / count


def scatter_max(src: torch.Tensor, index: torch.LongTensor, dim_size: int) -> torch.Tensor:
    """out[index[i]] = max(src[i], ...)."""
    out = torch.full((dim_size, src.size(1)), float("-inf"), device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand_as(src)
    out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    out = out.masked_fill(out == float("-inf"), 0.0)
    return out


def scatter_min(src: torch.Tensor, index: torch.LongTensor, dim_size: int) -> torch.Tensor:
    """out[index[i]] = min(src[i], ...)."""
    out = torch.full((dim_size, src.size(1)), float("inf"), device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand_as(src)
    out.scatter_reduce_(0, idx, src, reduce="amin", include_self=True)
    out = out.masked_fill(out == float("inf"), 0.0)
    return out


# ============================================================================
# GIN Layer (no edge features -- for intra-chain Stage 1)
# ============================================================================

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer (Xu et al., 2019).

    h_i' = MLP((1 + eps) * h_i + sum_{j in N(i)} h_j)

    No edge features needed because all intra-chain edges are identical
    ferromagnetic couplings.
    """

    def __init__(self, hidden_dim: int, eps_init: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] node features.
            edge_index: [2, E] edges (src, dst).
        """
        src, dst = edge_index[0], edge_index[1]
        # Aggregate neighbor features
        msg = x[src]  # [E, D]
        agg = scatter_add(msg, dst, dim_size=x.size(0))  # [N, D]
        # Update
        out = self.mlp((1 + self.eps) * x + agg)
        return self.norm(out + x)  # residual + LayerNorm


# ============================================================================
# GINE Layer (with edge features -- for Stage 3 logical graph)
# ============================================================================

class GINELayer(nn.Module):
    """GINEConv: edge-aware GIN (Hu et al., 2020).

    Standard additive formulation:
      msg_{j->i} = ReLU(h_j + W_e * e_{ij})
      h_i' = MLP((1 + eps) * h_i + sum_j msg_{j->i})
    with residual connection and LayerNorm for training stability.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, eps_init: float = 0.0):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        # Standard GINEConv: additive edge embedding
        msg = torch.relu(x[src] + self.edge_proj(edge_attr))
        agg = scatter_add(msg, dst, dim_size=x.size(0))
        out = self.mlp((1 + self.eps) * x + agg)
        return self.norm(out + x)


# ============================================================================
# MPNN Layer (for Stage 2 qubit pairing recovery)
# ============================================================================

class MPNNLayer(nn.Module):
    """Message Passing Neural Network layer for Stage 2.

    Processes inter-chain edges only. Uses edge features to modulate messages.

    msg_{j->i} = MLP_msg([h_j || e_{ij}])
    h_i' = MLP_upd(h_i + sum_j msg_{j->i})
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] node features.
            edge_index: [2, E] inter-chain edges.
            edge_attr: [E, edge_dim] inter-chain edge features.
        """
        src, dst = edge_index[0], edge_index[1]
        # Messages
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        agg = scatter_add(msg, dst, dim_size=x.size(0))
        # Update
        out = self.upd_mlp(x + agg)
        return self.norm(out + x)
