"""
agnn.py -- Anisotropic Graph Neural Network for chain strength prediction.

Anisotropic GNN: message passing where edge features multiplicatively gate
the node messages, creating direction-dependent (anisotropic) information flow.
Unlike isotropic GNNs (GCN, GIN) where all neighbors contribute equally or
additively, AGNN uses edge-conditioned filters that modulate each message
based on the edge's physical properties (coupling strength, direction, type).

Key difference from GIN-E (additive): msg = ReLU(h_j + W_e * e_ij)
AGNN (multiplicative/gating):         msg = sigma(W_gate * e_ij) * W_val(h_j)

This is critical for chain strength prediction because:
  - Intra-chain edges (ferromagnetic, uniform) vs inter-chain edges (problem-dependent)
    have fundamentally different physics — anisotropic filtering captures this
  - Coupling magnitudes directionally modulate how strongly information propagates
  - Hardware topology imposes directional asymmetry (Pegasus odd/even couplers)

Two variants:
  1. FlatAGNN:   Flat anisotropic GNN on full embedded graph
  2. HEC-AGNN:   Hierarchical architecture with anisotropic layers at each stage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.layers import scatter_add, scatter_mean, scatter_max, scatter_min


# ============================================================================
# Anisotropic Layers
# ============================================================================

class AnisoLayer(nn.Module):
    """Anisotropic message passing layer (no edge features).

    For intra-chain edges where all edges are identical ferromagnetic couplings,
    we still apply learnable anisotropic gating via relative node features:

        gate_ij = sigma(W_gate * [h_i - h_j])
        msg_ij  = gate_ij * W_val(h_j)
        h_i'    = MLP((1+eps)*h_i + sum_j msg_ij) + h_i

    The gate learns which neighbor directions carry useful structural info
    (e.g., chain endpoints vs interior qubits).
    """

    def __init__(self, hidden_dim: int, eps_init: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        # Direction-dependent gate from relative features
        diff = x[dst] - x[src]  # relative: target - source
        g = self.gate(diff)     # [E, D] in (0, 1)

        # Gated message
        val = self.value(x[src])  # [E, D]
        msg = g * val             # anisotropic: direction modulates content

        agg = scatter_add(msg, dst, dim_size=N)
        out = self.mlp((1 + self.eps) * x + agg)
        return self.norm(out + x)


class AnisoEdgeLayer(nn.Module):
    """Anisotropic message passing with edge features.

    Edge features produce a per-dimension gating vector that multiplicatively
    modulates node messages. This is the core anisotropic mechanism:

        gate_ij = sigma(W_gate * e_ij)          -- edge → D-dim gate
        msg_ij  = gate_ij * (h_j + W_e * e_ij)  -- gated message
        h_i'    = MLP((1+eps)*h_i + sum_j msg_ij) + h_i

    The multiplicative gating means edge properties (coupling strength,
    type, direction) control HOW MUCH of each feature dimension passes
    through, not just what gets added. This creates truly anisotropic
    information flow.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, eps_init: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        # Edge → gating vector (D-dimensional sigmoid)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Edge → additive embedding (like GIN-E)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        # Anisotropic gate from edge features
        gate = self.edge_gate(edge_attr)           # [E, D] in (0, 1)
        # Content: node + edge embedding
        content = x[src] + self.edge_proj(edge_attr)  # [E, D]
        # Gated message: edge controls per-dimension flow
        msg = gate * F.relu(content)               # [E, D]

        agg = scatter_add(msg, dst, dim_size=N)
        out = self.mlp((1 + self.eps) * x + agg)
        return self.norm(out + x)


# ============================================================================
# Pooling utilities
# ============================================================================

def scatter_pool_4way(h, batch_idx, n_out):
    """sum || mean || max || min → 4H."""
    s = scatter_add(h, batch_idx, n_out)
    m = scatter_mean(h, batch_idx, n_out)
    mx = scatter_max(h, batch_idx, n_out)
    mn = scatter_min(h, batch_idx, n_out)
    return torch.cat([s, m, mx, mn], dim=-1)


def scatter_pool_3way(h, batch_idx, n_out):
    """sum || mean || max → 3H."""
    s = scatter_add(h, batch_idx, n_out)
    m = scatter_mean(h, batch_idx, n_out)
    mx = scatter_max(h, batch_idx, n_out)
    return torch.cat([s, m, mx], dim=-1)


# ============================================================================
# FlatAGNN: single-level anisotropic GNN on full embedded graph
# ============================================================================

class FlatAGNN(nn.Module):
    """Flat Anisotropic GNN on full embedded graph.

    Like FlatGNN but with anisotropic (edge-gated) message passing.
    Edge features [type_indicator, J/RMS, |J|/RMS] control per-dimension
    gating, so chain edges and inter-chain edges propagate information
    through different feature channels.
    """

    def __init__(self, input_dim=7, hidden_dim=128, edge_dim=3,
                 num_layers=6, K=20, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            AnisoEdgeLayer(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        graph_dim = 3 * hidden_dim
        pool_dim = 4 * hidden_dim
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, K),
        )
        self.cbr_head = nn.Sequential(
            nn.Linear(pool_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rms_head = nn.Sequential(
            nn.Linear(graph_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        x = batch["x"]
        chain_ei = batch["chain_edge_index"]
        inter_ei = batch["inter_edge_index"]

        # Merge edges with type-preserving features
        ei_list, ef_list = [], []
        if chain_ei.numel() > 0:
            n_chain = chain_ei.size(1)
            cf = torch.zeros(n_chain, 3, device=x.device)
            cf[:, 0] = 1.0  # type indicator for chain edges
            ei_list.append(chain_ei)
            ef_list.append(cf)
        if inter_ei.numel() > 0:
            ei_list.append(inter_ei)
            ef_list.append(batch["inter_edge_attr"])

        if ei_list:
            all_edges = torch.cat(ei_list, dim=1)
            all_attrs = torch.cat(ef_list, dim=0)
        else:
            all_edges = torch.zeros(2, 0, dtype=torch.long, device=x.device)
            all_attrs = torch.zeros(0, 3, device=x.device)

        chain_batch = batch["chain_batch"]
        graph_batch = batch["graph_batch"]
        n_graphs = batch["batch_size"]
        n_chains = graph_batch.size(0)

        h = F.relu(self.encoder(x))
        for layer in self.layers:
            h = layer(h, all_edges, all_attrs)

        # Chain-level pool
        chain_repr = scatter_pool_4way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)

        # Graph-level pool
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = scatter_pool_3way(chain_h, graph_batch, n_graphs)
        energy_pred = self.head(graph_repr)
        rms_pred = self.rms_head(graph_repr).squeeze(-1)

        return energy_pred, cbr_pred, rms_pred


# ============================================================================
# HEC-AGNN: Hierarchical anisotropic GNN
# ============================================================================

class HECAGNN(nn.Module):
    """HEC-AGNN: Hierarchical Anisotropic GNN.

    Same 3-stage hierarchy as HEC-GNN but with anisotropic layers:
      Stage 1: AnisoLayer (intra-chain, direction-dependent gating)
      Stage 2: MPNN (qubit pairing recovery, reused from HEC-GNN)
      Stage 3: AnisoEdgeLayer (logical graph, edge-gated message passing)

    The anisotropic gating at Stage 1 lets the model learn which chain
    positions (endpoint vs interior) and which neighbor directions matter
    for vulnerability detection. At Stage 3, logical edge features
    (coupling strength, field magnitude) gate information flow between
    logical variables.
    """

    def __init__(self, input_dim=7, hidden_dim=128, L1=3, L3=3, K=20,
                 dropout=0.1, eps_init=0.0):
        super().__init__()
        from src.models.layers import MPNNLayer

        H = hidden_dim

        # Stage 1: Anisotropic intra-chain
        self.stage1_encoder = nn.Linear(input_dim, H)
        self.stage1_layers = nn.ModuleList([
            AnisoLayer(H, eps_init) for _ in range(L1)
        ])
        stage1_out = 4 * H  # 4-way pool

        # Stage 2: MPNN qubit pairing (same as HEC-GNN)
        self.qubit_proj = nn.Linear(stage1_out + input_dim, stage1_out)
        self.mpnn = MPNNLayer(stage1_out, 3)
        stage2_out = 4 * stage1_out
        self.stage2_proj = nn.Sequential(
            nn.Linear(stage2_out, stage1_out), nn.ReLU())

        # Stage 3: Anisotropic on logical graph
        self.stage3_encoder = nn.Linear(stage1_out + 1, H)  # +1 for log(chain_len)
        self.stage3_layers = nn.ModuleList([
            AnisoEdgeLayer(H, 5, eps_init) for _ in range(L3)  # 5-dim logical edge features
        ])
        graph_dim = 3 * H

        # Output heads
        self.energy_head = nn.Sequential(
            nn.Linear(graph_dim, H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, K))
        self.cbr_head = nn.Sequential(
            nn.Linear(stage1_out, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        self.rms_head = nn.Sequential(
            nn.Linear(graph_dim, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))

    def forward(self, batch):
        x = batch["x"]
        chain_ei = batch["chain_edge_index"]
        inter_ei = batch["inter_edge_index"]
        inter_ea = batch["inter_edge_attr"]
        chain_batch = batch["chain_batch"]
        logical_ei = batch["logical_edge_index"]
        logical_ea = batch["logical_edge_attr"]
        graph_batch = batch["graph_batch"]
        chain_lengths = batch["chain_lengths"]

        n_chains = chain_batch.max().item() + 1 if chain_batch.numel() > 0 else 1
        n_graphs = graph_batch.max().item() + 1 if graph_batch.numel() > 0 else 1

        # Stage 1: Anisotropic intra-chain encoding
        h = self.stage1_encoder(x)
        for layer in self.stage1_layers:
            h = layer(h, chain_ei)
        c1 = scatter_pool_4way(h, chain_batch, n_chains)  # [N_chains, 4H]
        cbr_pred = self.cbr_head(c1).squeeze(-1)

        # Stage 2: Qubit pairing recovery
        c_bc = c1[chain_batch]  # broadcast chain repr to qubits
        h_q = F.relu(self.qubit_proj(torch.cat([c_bc, x], dim=-1)))
        if inter_ei.numel() > 0:
            h_q = self.mpnn(h_q, inter_ei, inter_ea)
        c2_raw = scatter_pool_4way(h_q, chain_batch, n_chains)
        c2 = self.stage2_proj(c2_raw) + c1  # residual

        # Stage 3: Anisotropic on logical graph
        log_cl = torch.log(chain_lengths.float().clamp(min=1.0)).unsqueeze(-1)
        h3 = self.stage3_encoder(torch.cat([c2, log_cl], dim=-1))
        for layer in self.stage3_layers:
            h3 = layer(h3, logical_ei, logical_ea)
        z = scatter_pool_3way(h3, graph_batch, n_graphs)  # [B, 3H]

        energy_pred = self.energy_head(z)
        rms_pred = self.rms_head(z).squeeze(-1)

        return energy_pred, cbr_pred, rms_pred
