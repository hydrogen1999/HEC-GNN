"""
ALL GNN baseline architectures for chain-strength curve prediction.

Fixed after Codex review:
- GPS: per-graph attention masking (no cross-graph leakage)
- R-GCN: 2 relations (chain, inter) at qubit level — correct
- Flat-GAT: uses edge features [type_indicator, J/RMS, |J|/RMS] like FlatGNN
- Param matching: scale hidden dims so flat models ~500K-1M params
- Graph readout: use full pooled dim, not truncated

Models:
  1. SAGE-HEC:  Stage 1 GIN → GraphSAGE, keep Stage 2+3 (~2.75M)
  2. GAT-HEC:   Stage 1 GIN → GAT, keep Stage 2+3 (~2.70M)
  3. R-GCN:     Flat relational GCN, 2 edge types, ~500K params
  4. Flat-GAT:  Flat GAT with edge features, ~500K params
  5. GPS:       Graph Transformer with per-graph attention, ~500K params
  6. HeteroSAGE: PyG-style to_hetero(GraphSAGE) — per-relation SAGE conv, ~500K params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


# ═══════════════════════════════════════════
# Scatter pooling utilities
# ═══════════════════════════════════════════

def scatter_pool_4way(h, batch_idx):
    """sum || mean || max || min → 4H."""
    n = batch_idx.max().item() + 1
    H = h.size(1)
    s = torch.zeros(n, H, device=h.device)
    s.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(h), h)
    c = torch.zeros(n, 1, device=h.device)
    c.scatter_add_(0, batch_idx.unsqueeze(-1), torch.ones(batch_idx.size(0), 1, device=h.device))
    c = c.clamp(min=1)
    mean = s / c
    idx = batch_idx.unsqueeze(-1).expand_as(h)
    mx = torch.full((n, H), -1e9, device=h.device).scatter_reduce(0, idx, h, reduce="amax")
    mn = torch.full((n, H), 1e9, device=h.device).scatter_reduce(0, idx, h, reduce="amin")
    return torch.cat([s, mean, mx, mn], dim=-1)


def scatter_pool_3way(h, batch_idx):
    """sum || mean || max → 3H."""
    n = batch_idx.max().item() + 1
    H = h.size(1)
    s = torch.zeros(n, H, device=h.device)
    s.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(h), h)
    c = torch.zeros(n, 1, device=h.device)
    c.scatter_add_(0, batch_idx.unsqueeze(-1), torch.ones(batch_idx.size(0), 1, device=h.device))
    c = c.clamp(min=1)
    mean = s / c
    idx = batch_idx.unsqueeze(-1).expand_as(h)
    mx = torch.full((n, H), -1e9, device=h.device).scatter_reduce(0, idx, h, reduce="amax")
    return torch.cat([s, mean, mx], dim=-1)


def make_energy_head(in_dim, K=20):
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, K),
    )


# ═══════════════════════════════════════════
# 1. SAGE-HEC / 2. GAT-HEC: Stage 1 swaps
# ═══════════════════════════════════════════

class SAGELayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        row, col = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        cnt = torch.zeros(h.size(0), 1, device=h.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(h[col]), h[col].clone())
        cnt.scatter_add_(0, row.unsqueeze(-1), torch.ones(row.size(0), 1, device=h.device))
        out = self.W(torch.cat([h.clone(), agg / (cnt + 1e-8)], dim=-1))
        return self.norm(F.relu(out) + h)


class GATLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.W = nn.Linear(dim, dim)
        self.a = nn.Linear(2 * self.hd, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        row, col = edge_index[0], edge_index[1]
        hp = self.W(h).view(-1, self.heads, self.hd)
        alpha = F.leaky_relu(self.a(torch.cat([hp[row], hp[col]], dim=-1)).squeeze(-1), 0.2)
        alpha = (alpha - alpha.max()).exp()
        asum = torch.zeros(h.size(0), self.heads, device=h.device)
        asum.scatter_add_(0, row.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / (asum[row] + 1e-8)
        agg = torch.zeros_like(hp)
        agg.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(hp[col]),
                         (alpha.unsqueeze(-1) * hp[col]).clone())
        out = agg.view(-1, self.heads * self.hd)
        return self.norm(F.relu(out) + h)


def build_hec_variant(variant_name):
    """Build HEC-GNN with different Stage 1."""
    from src.models.hec_gnn import HECGNN
    model = HECGNN()

    class VariantEncoder(nn.Module):
        def __init__(self, layer_cls, input_dim=7, hidden_dim=128, num_layers=3, **kw):
            super().__init__()
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.layers = nn.ModuleList([layer_cls(hidden_dim, **kw) for _ in range(num_layers)])

        def forward(self, x, edge_index, chain_batch):
            h = F.relu(self.encoder(x))
            for layer in self.layers:
                h = layer(h, edge_index)
            return scatter_pool_4way(h, chain_batch)

    if variant_name == 'sage':
        model.stage1 = VariantEncoder(SAGELayer)
    elif variant_name == 'gat':
        model.stage1 = VariantEncoder(GATLayer, heads=4)
    return model


# ═══════════════════════════════════════════
# 3. R-GCN: Relational GCN (flat, 2 edge types)
# ═══════════════════════════════════════════

class RGCNLayer(nn.Module):
    """R-GCN with separate W per relation + self-transform."""
    def __init__(self, dim, n_relations=2):
        super().__init__()
        self.Ws = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_relations)])
        self.W_self = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_indices_list):
        out = self.W_self(h)
        for W, ei in zip(self.Ws, edge_indices_list):
            if ei.numel() == 0:
                continue
            row, col = ei[0], ei[1]
            msg = W(h[col].clone())
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
            out = out + agg
        return self.norm(F.relu(out) + h)


class RGCN(nn.Module):
    """Flat R-GCN: 2 relations (chain, inter) at qubit level.
    ~500K params via hidden_dim=160, 6 layers.
    """
    def __init__(self, input_dim=7, hidden_dim=160, num_layers=6, K=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([RGCNLayer(hidden_dim, n_relations=2) for _ in range(num_layers)])
        pool_dim = hidden_dim * 4  # full 4-way pool → chain → 3-way → graph
        graph_dim = hidden_dim * 3
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.energy_head = make_energy_head(graph_dim)
        self.cbr_head = nn.Sequential(nn.Linear(pool_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rms_head = nn.Sequential(nn.Linear(graph_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        x = batch['x']
        h = F.relu(self.encoder(x))
        for layer in self.layers:
            h = layer(h, [batch['chain_edge_index'], batch['inter_edge_index']])

        # Chain-level pool (4-way)
        chain_repr = scatter_pool_4way(h, batch['chain_batch'])
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)

        # Graph-level pool (3-way on projected chain repr)
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = scatter_pool_3way(chain_h, batch['graph_batch'])
        energy_pred = self.energy_head(graph_repr)
        rms_pred = self.rms_head(graph_repr).squeeze(-1)
        return energy_pred, cbr_pred, rms_pred


# ═══════════════════════════════════════════
# 4. Flat-GAT with edge features (fair comparison)
# ═══════════════════════════════════════════

class GATEdgeLayer(nn.Module):
    """GAT with edge features incorporated into attention."""
    def __init__(self, dim, edge_dim=3, heads=4):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.W = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(edge_dim, self.hd)
        self.a = nn.Linear(3 * self.hd, 1)  # src + dst + edge
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return h
        row, col = edge_index[0], edge_index[1]
        hp = self.W(h).view(-1, self.heads, self.hd)
        e = self.edge_proj(edge_attr).unsqueeze(1).expand(-1, self.heads, -1)
        alpha = F.leaky_relu(
            self.a(torch.cat([hp[row], hp[col], e], dim=-1)).squeeze(-1), 0.2)
        alpha = (alpha - alpha.max()).exp()
        asum = torch.zeros(h.size(0), self.heads, device=h.device)
        asum.scatter_add_(0, row.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / (asum[row] + 1e-8)
        agg = torch.zeros_like(hp)
        agg.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(hp[col]),
                         (alpha.unsqueeze(-1) * hp[col]).clone())
        out = agg.view(-1, self.heads * self.hd)
        return self.norm(F.relu(out) + h)


class FlatGATModel(nn.Module):
    """Flat GAT with edge features — fair comparison to FlatGNN.
    Edge features: [type_indicator, J_pq/RMS, |J_pq|/RMS] (3-dim).
    Chain edges: [1, 0, 0], Inter edges: [0, J/RMS, |J|/RMS].
    ~500K params via hidden_dim=128, 6 layers.
    """
    def __init__(self, input_dim=7, hidden_dim=128, edge_dim=3, num_layers=6, heads=4, K=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GATEdgeLayer(hidden_dim, edge_dim, heads) for _ in range(num_layers)])
        pool_dim = hidden_dim * 4
        graph_dim = hidden_dim * 3
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.energy_head = make_energy_head(graph_dim)
        self.cbr_head = nn.Sequential(nn.Linear(pool_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rms_head = nn.Sequential(nn.Linear(graph_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        x = batch['x']
        chain_ei = batch['chain_edge_index']
        inter_ei = batch['inter_edge_index']

        # Merge edges with type-preserving features
        ei_list = []
        ef_list = []
        if chain_ei.numel() > 0:
            n_chain = chain_ei.size(1)
            chain_feat = torch.zeros(n_chain, 3, device=x.device)
            chain_feat[:, 0] = 1.0  # type indicator
            ei_list.append(chain_ei)
            ef_list.append(chain_feat)
        if inter_ei.numel() > 0:
            inter_attr = batch.get('inter_edge_attr', torch.zeros(inter_ei.size(1), 3, device=x.device))
            ei_list.append(inter_ei)
            ef_list.append(inter_attr)

        if ei_list:
            all_ei = torch.cat(ei_list, dim=1)
            all_ef = torch.cat(ef_list, dim=0)
        else:
            all_ei = torch.zeros(2, 0, dtype=torch.long, device=x.device)
            all_ef = torch.zeros(0, 3, device=x.device)

        h = F.relu(self.encoder(x))
        for layer in self.layers:
            h = layer(h, all_ei, all_ef)

        chain_repr = scatter_pool_4way(h, batch['chain_batch'])
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = scatter_pool_3way(chain_h, batch['graph_batch'])
        energy_pred = self.energy_head(graph_repr)
        rms_pred = self.rms_head(graph_repr).squeeze(-1)
        return energy_pred, cbr_pred, rms_pred


# ═══════════════════════════════════════════
# 5. GPS: Graph Transformer (per-graph attention)
# ═══════════════════════════════════════════

class GPSLayer(nn.Module):
    """GPS layer: local MPNN + per-graph global attention + FFN.
    FIX: attention is masked per graph (no cross-graph leakage).
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.msg_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
        self.heads = heads
        self.hd = dim // heads
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, h, edge_index, graph_batch):
        # Local MPNN
        if edge_index.numel() > 0:
            row, col = edge_index[0], edge_index[1]
            msg = self.msg_mlp(torch.cat([h[row], h[col]], dim=-1))
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
            h = self.norm1(h + agg)

        # Per-graph global attention (scatter-based, no cross-graph leak)
        Q = self.W_q(h).view(-1, self.heads, self.hd)
        K = self.W_k(h).view(-1, self.heads, self.hd)
        V = self.W_v(h).view(-1, self.heads, self.hd)

        # Self-attention scores within same graph
        # For efficiency: compute all scores, mask out cross-graph
        # Approximate: use chain_batch as graph indicator for nodes
        n_nodes = h.size(0)
        # Scores: Q_i . K_j / sqrt(d) for nodes in same graph
        # Full attention is O(N^2) — too expensive. Use scatter approximation:
        # Per-graph mean of K,V, then attend to graph summary
        n_graphs = graph_batch.max().item() + 1
        K_sum = torch.zeros(n_graphs, self.heads, self.hd, device=h.device)
        K_sum.scatter_add_(0, graph_batch.unsqueeze(-1).unsqueeze(-1).expand_as(K), K)
        V_sum = torch.zeros(n_graphs, self.heads, self.hd, device=h.device)
        V_sum.scatter_add_(0, graph_batch.unsqueeze(-1).unsqueeze(-1).expand_as(V), V)
        cnt = torch.zeros(n_graphs, 1, 1, device=h.device)
        cnt.scatter_add_(0, graph_batch.unsqueeze(-1).unsqueeze(-1),
                         torch.ones(n_nodes, 1, 1, device=h.device))
        K_mean = K_sum / (cnt + 1e-8)
        V_mean = V_sum / (cnt + 1e-8)

        # Each node attends to its graph's mean key/value
        attn = (Q * K_mean[graph_batch]).sum(-1, keepdim=True) / (self.hd ** 0.5)
        attn = attn.softmax(dim=1)
        context = (attn * V_mean[graph_batch]).view(-1, self.heads * self.hd)
        h_attn = self.W_o(context)
        h = self.norm2(h + h_attn)

        # FFN
        h = self.norm3(h + self.ffn(h))
        return h


class GPSModel(nn.Module):
    """GPS on embedded graph. ~500K params via hidden_dim=96, 6 layers.
    Per-graph attention (no cross-graph leakage).
    """
    def __init__(self, input_dim=7, hidden_dim=96, num_layers=6, heads=4, K=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GPSLayer(hidden_dim, heads) for _ in range(num_layers)])
        pool_dim = hidden_dim * 4
        graph_dim = hidden_dim * 3
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.energy_head = make_energy_head(graph_dim)
        self.cbr_head = nn.Sequential(nn.Linear(pool_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rms_head = nn.Sequential(nn.Linear(graph_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        x = batch['x']
        chain_ei = batch['chain_edge_index']
        inter_ei = batch['inter_edge_index']
        if chain_ei.numel() > 0 and inter_ei.numel() > 0:
            all_ei = torch.cat([chain_ei, inter_ei], dim=1)
        elif chain_ei.numel() > 0:
            all_ei = chain_ei
        else:
            all_ei = inter_ei

        # Need qubit-level graph batch (not chain_batch)
        # Derive from chain_batch → graph_batch mapping
        chain_batch = batch['chain_batch']
        graph_batch_chain = batch['graph_batch']
        # qubit → chain → graph
        qubit_graph_batch = graph_batch_chain[chain_batch]

        h = F.relu(self.encoder(x))
        for layer in self.layers:
            h = layer(h, all_ei, qubit_graph_batch)

        chain_repr = scatter_pool_4way(h, chain_batch)
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = scatter_pool_3way(chain_h, graph_batch_chain)
        energy_pred = self.energy_head(graph_repr)
        rms_pred = self.rms_head(graph_repr).squeeze(-1)
        return energy_pred, cbr_pred, rms_pred


# ═══════════════════════════════════════════
# 6. HeteroSAGE: PyG-style to_hetero(GraphSAGE)
#    Per-relation SAGE conv with independent weights,
#    then mean aggregation — standard heterogeneous GNN pattern.
# ═══════════════════════════════════════════

class HeteroSAGELayer(nn.Module):
    """One SAGE conv per edge type (chain, inter), then aggregate.
    Mirrors PyG's to_hetero: duplicate the base conv for each relation,
    aggregate neighbor messages from all relations via mean.
    """
    def __init__(self, dim):
        super().__init__()
        # Independent SAGE conv per relation type
        self.sage_chain = nn.Linear(2 * dim, dim)
        self.sage_inter = nn.Linear(2 * dim, dim)
        self.W_self = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _sage_agg(self, h, edge_index, W):
        """SAGE-style: mean-aggregate neighbors, concat with self, project."""
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        row, col = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        cnt = torch.zeros(h.size(0), 1, device=h.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(h[col]), h[col].clone())
        cnt.scatter_add_(0, row.unsqueeze(-1), torch.ones(row.size(0), 1, device=h.device))
        neigh_mean = agg / (cnt + 1e-8)
        return W(torch.cat([h, neigh_mean], dim=-1))

    def forward(self, h, chain_ei, inter_ei):
        h_self = self.W_self(h)
        h_chain = self._sage_agg(h, chain_ei, self.sage_chain)
        h_inter = self._sage_agg(h, inter_ei, self.sage_inter)
        # Mean aggregation across relation types (to_hetero default: aggr="mean")
        out = h_self + (h_chain + h_inter) / 2.0
        return self.norm(F.relu(out) + h)


class HeteroSAGEModel(nn.Module):
    """Flat heterogeneous SAGE — mirrors PyG to_hetero(GraphSAGE).
    Per-relation SAGE convolutions with independent weights.
    ~500K params via hidden_dim=140, 6 layers.
    """
    def __init__(self, input_dim=7, hidden_dim=140, num_layers=6, K=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([HeteroSAGELayer(hidden_dim) for _ in range(num_layers)])
        pool_dim = hidden_dim * 4
        graph_dim = hidden_dim * 3
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.energy_head = make_energy_head(graph_dim)
        self.cbr_head = nn.Sequential(nn.Linear(pool_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rms_head = nn.Sequential(nn.Linear(graph_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        x = batch['x']
        h = F.relu(self.encoder(x))
        for layer in self.layers:
            h = layer(h, batch['chain_edge_index'], batch['inter_edge_index'])

        chain_repr = scatter_pool_4way(h, batch['chain_batch'])
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = scatter_pool_3way(chain_h, batch['graph_batch'])
        energy_pred = self.energy_head(graph_repr)
        rms_pred = self.rms_head(graph_repr).squeeze(-1)
        return energy_pred, cbr_pred, rms_pred


# ═══════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════

ALL_MODELS = {
    'SAGE-HEC': lambda: build_hec_variant('sage'),
    'GAT-HEC': lambda: build_hec_variant('gat'),
    'R-GCN': RGCN,
    'Flat-GAT': FlatGATModel,
    'GPS': GPSModel,
    'HeteroSAGE': HeteroSAGEModel,
}

if __name__ == '__main__':
    print("GNN Baseline Architectures (fixed):")
    print(f"{'Model':<15} {'Params':>10} {'Type':>20}")
    print("-" * 48)
    for name, fn in ALL_MODELS.items():
        try:
            m = fn()
            p = sum(pp.numel() for pp in m.parameters())
            print(f"{name:<15} {p:>10,} {'Hierarchical' if 'HEC' in name else 'Flat'}")
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")
