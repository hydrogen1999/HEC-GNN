"""
GNN architecture baselines for chain-strength curve prediction.
Each replaces Stage 1 (intra-chain encoder) of HEC-GNN with a different GNN layer,
keeping Stage 2 (MPNN) and Stage 3 (GINEConv) identical.

Baselines:
  1. GCN-HEC: GCN (degree-normalized) for intra-chain
  2. GAT-HEC: GAT (attention) for intra-chain
  3. SAGE-HEC: GraphSAGE (mean-aggregation) for intra-chain
  4. HeteroGNN: PyG HeteroConv with different operator per edge type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ── Stage 1 alternatives ──

class GCNIntraChain(nn.Module):
    """GCN for intra-chain: h_i' = σ(W · mean_neighbor(h_j) + b)

    Physics concern: degree normalization DOWN-WEIGHTS chain endpoints,
    but endpoints are most vulnerable to defection → wrong inductive bias.
    """
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'W': nn.Linear(hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
            }))

    def forward(self, x, edge_index, chain_batch):
        h = F.relu(self.encoder(x))
        row, col = edge_index
        for layer in self.layers:
            # Degree-normalized aggregation
            deg = torch.zeros(h.size(0), device=h.device)
            deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            deg_inv = (deg + 1e-8).pow(-1)
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, row.unsqueeze(-1).expand_as(h[col]), h[col])
            agg = agg * deg_inv.unsqueeze(-1)
            out = layer['W'](agg)
            h = layer['norm'](F.relu(out) + h)

        # 4-way pooling per chain
        from models.hec_gnn import _scatter_pool
        c = _scatter_pool(h, chain_batch)
        return c


class GATIntraChain(nn.Module):
    """GAT for intra-chain: attention-weighted neighbor aggregation.

    Physics concern: all intra-chain edges carry identical J_c coupling,
    so there is no edge heterogeneity to attend to → attention is wasteful.
    """
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'W': nn.Linear(hidden_dim, hidden_dim),
                'a_l': nn.Linear(self.head_dim, 1),
                'a_r': nn.Linear(self.head_dim, 1),
                'norm': nn.LayerNorm(hidden_dim),
            }))

    def forward(self, x, edge_index, chain_batch):
        h = F.relu(self.encoder(x))
        row, col = edge_index
        for layer in self.layers:
            h_proj = layer['W'](h).view(-1, self.heads, self.head_dim)
            # Attention
            alpha_l = layer['a_l'](h_proj[row]).squeeze(-1)
            alpha_r = layer['a_r'](h_proj[col]).squeeze(-1)
            alpha = F.leaky_relu(alpha_l + alpha_r, 0.2)
            alpha = alpha - alpha.max()
            alpha = alpha.exp()
            alpha_sum = torch.zeros(h.size(0), self.heads, device=h.device)
            alpha_sum.scatter_add_(0, row.unsqueeze(-1).expand_as(alpha), alpha)
            alpha = alpha / (alpha_sum[row] + 1e-8)
            # Aggregate
            agg = torch.zeros_like(h_proj)
            agg.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(h_proj[col]),
                             alpha.unsqueeze(-1) * h_proj[col])
            out = agg.view(-1, self.heads * self.head_dim)
            h = layer['norm'](F.relu(out) + h)

        from models.hec_gnn import _scatter_pool
        c = _scatter_pool(h, chain_batch)
        return c


class SAGEIntraChain(nn.Module):
    """GraphSAGE for intra-chain: concat(self, mean_neighbor) → MLP.

    Physics concern: mean aggregation weaker than GIN's sum for
    distinguishing multisets (qubit configurations within chains).
    """
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'W': nn.Linear(2 * hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
            }))

    def forward(self, x, edge_index, chain_batch):
        h = F.relu(self.encoder(x))
        row, col = edge_index
        for layer in self.layers:
            agg = torch.zeros_like(h)
            count = torch.zeros(h.size(0), 1, device=h.device)
            agg.scatter_add_(0, row.unsqueeze(-1).expand_as(h[col]), h[col])
            count.scatter_add_(0, row.unsqueeze(-1),
                               torch.ones(row.size(0), 1, device=h.device))
            agg = agg / (count + 1e-8)
            out = layer['W'](torch.cat([h, agg], dim=-1))
            h = layer['norm'](F.relu(out) + h)

        from models.hec_gnn import _scatter_pool
        c = _scatter_pool(h, chain_batch)
        return c


# ── Full model wrappers ──

def build_variant_model(stage1_type='gin', **kwargs):
    """Build HEC-GNN variant with different Stage 1.

    Args:
        stage1_type: 'gin' (default), 'gcn', 'gat', 'sage'
    Returns:
        nn.Module with same interface as HECGNN
    """
    from models.hec_gnn import HECGNN

    model = HECGNN(**kwargs)

    if stage1_type == 'gin':
        return model  # default
    elif stage1_type == 'gcn':
        model.stage1 = GCNIntraChain()
    elif stage1_type == 'gat':
        model.stage1 = GATIntraChain()
    elif stage1_type == 'sage':
        model.stage1 = SAGEIntraChain()
    else:
        raise ValueError(f"Unknown stage1_type: {stage1_type}")

    return model


# ── Architecture summary ──
if __name__ == '__main__':
    for name, cls in [('GCN', GCNIntraChain), ('GAT', GATIntraChain), ('SAGE', SAGEIntraChain)]:
        m = cls()
        params = sum(p.numel() for p in m.parameters())
        print(f"{name} Stage 1: {params:,} params")

    from models.hec_gnn import HECGNN
    for variant in ['gin', 'gcn', 'gat', 'sage']:
        m = build_variant_model(variant)
        params = sum(p.numel() for p in m.parameters())
        print(f"HEC-{variant.upper()}: {params:,} total params")
