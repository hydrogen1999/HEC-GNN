"""
extended.py -- Extended GNN architectures for comprehensive comparison.

Architectures added (all self-contained, no PyG dependency):

Layer-level:
  1. GCNLayer       — Graph Convolutional Network (Kipf & Welling, 2017)
  2. GINPureLayer   — GIN without edge features (Xu et al., 2019)
  3. SAGEFlatLayer  — GraphSAGE flat (Hamilton et al., 2017)
  4. GATv2Layer     — GATv2 with dynamic attention (Brody et al., 2022)
  5. PNALayer       — Principal Neighbourhood Aggregation (Corso et al., 2020)
  6. GatedGCNLayer  — Gated Graph ConvNet (Bresson & Laurent, 2017)
  7. EdgeConvLayer  — Dynamic Edge Convolution / DGCNN (Wang et al., 2019)
  8. APPNPLayer     — Approximate PPR propagation (Gasteiger et al., 2019)

Model-level (flat):
  9. FlatGCN, FlatGIN, FlatSAGE, FlatGATv2
  10. FlatPNA, FlatGatedGCN, FlatEdgeConv
  11. APPNPModel
  12. DeepSetsModel  — No message passing baseline
  13. XGBoostModel   — Non-neural baseline (wrapper)

Hierarchical (HEC-*):
  14. HEC-PNA, HEC-GatedGCN, HEC-GATv2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.layers import scatter_add, scatter_mean, scatter_max, scatter_min


# ============================================================================
# Pooling utilities (shared)
# ============================================================================

def pool_4way(h, idx, n):
    return torch.cat([scatter_add(h, idx, n), scatter_mean(h, idx, n),
                      scatter_max(h, idx, n), scatter_min(h, idx, n)], dim=-1)

def pool_3way(h, idx, n):
    return torch.cat([scatter_add(h, idx, n), scatter_mean(h, idx, n),
                      scatter_max(h, idx, n)], dim=-1)

def _merge_edges(batch):
    """Merge chain + inter edges with type features."""
    x = batch["x"]
    chain_ei = batch["chain_edge_index"]
    inter_ei = batch["inter_edge_index"]
    ei_list, ef_list = [], []
    if chain_ei.numel() > 0:
        cf = torch.zeros(chain_ei.size(1), 3, device=x.device)
        cf[:, 0] = 1.0
        ei_list.append(chain_ei); ef_list.append(cf)
    if inter_ei.numel() > 0:
        ei_list.append(inter_ei); ef_list.append(batch["inter_edge_attr"])
    if ei_list:
        return torch.cat(ei_list, dim=1), torch.cat(ef_list, dim=0)
    return torch.zeros(2, 0, dtype=torch.long, device=x.device), torch.zeros(0, 3, device=x.device)


def _make_heads(pool_dim, graph_dim, K=20):
    """Standard energy/cbr/rms heads."""
    energy = nn.Sequential(nn.Linear(graph_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                           nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, K))
    cbr = nn.Sequential(nn.Linear(pool_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    rms = nn.Sequential(nn.Linear(graph_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    return energy, cbr, rms


# ============================================================================
# 1. GCN Layer
# ============================================================================

class GCNLayer(nn.Module):
    """GCN: h_i' = sigma(W * mean(h_j for j in N(i) ∪ {i}))"""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        # Add self-loops
        self_idx = torch.arange(N, device=h.device)
        all_src = torch.cat([src, self_idx])
        all_dst = torch.cat([dst, self_idx])
        # Degree normalization
        deg = torch.zeros(N, device=h.device)
        deg.scatter_add_(0, all_dst, torch.ones(all_dst.size(0), device=h.device))
        deg_inv_sqrt = (deg.clamp(min=1) ** -0.5)
        norm_coeff = deg_inv_sqrt[all_src] * deg_inv_sqrt[all_dst]
        msg = norm_coeff.unsqueeze(-1) * h[all_src]
        agg = scatter_add(msg, all_dst, N)
        return self.norm(F.relu(self.W(agg)) + h)


# ============================================================================
# 2. GIN Pure (no edge features)
# ============================================================================

class GINPureLayer(nn.Module):
    """GIN: h_i' = MLP((1+eps)*h_i + sum h_j)"""
    def __init__(self, dim, eps_init=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        agg = scatter_add(h[src], dst, h.size(0))
        return self.norm(self.mlp((1 + self.eps) * h + agg) + h)


# ============================================================================
# 3. GraphSAGE Flat
# ============================================================================

class SAGEFlatLayer(nn.Module):
    """SAGE: h_i' = W * [h_i || mean(h_j)]"""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        agg = scatter_mean(h[src], dst, N)
        return self.norm(F.relu(self.W(torch.cat([h, agg], dim=-1))) + h)


# ============================================================================
# 4. GATv2 — Dynamic attention (Brody et al., 2022)
# ============================================================================

class GATv2Layer(nn.Module):
    """GATv2: alpha = a^T LeakyReLU(W[h_i || h_j]) — dynamic, not static."""
    def __init__(self, dim, heads=4, edge_dim=3):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.W_src = nn.Linear(dim, dim)
        self.W_dst = nn.Linear(dim, dim)
        self.W_edge = nn.Linear(edge_dim, dim) if edge_dim > 0 else None
        self.a = nn.Linear(self.hd, 1)
        self.W_o = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index, edge_attr=None):
        if edge_index.numel() == 0:
            return h
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        hs = self.W_src(h).view(N, self.heads, self.hd)
        hd = self.W_dst(h).view(N, self.heads, self.hd)
        # Dynamic: apply nonlinearity AFTER concat (key difference from GATv1)
        msg = hs[src] + hd[dst]
        if self.W_edge is not None and edge_attr is not None:
            msg = msg + self.W_edge(edge_attr).view(-1, self.heads, self.hd)
        msg = F.leaky_relu(msg, 0.2)
        alpha = self.a(msg).squeeze(-1)  # [E, heads]
        # Softmax per dst
        alpha_max = torch.full((N, self.heads), -1e9, device=h.device)
        alpha_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(alpha), alpha, reduce="amax")
        alpha = (alpha - alpha_max[dst]).exp()
        alpha_sum = torch.zeros(N, self.heads, device=h.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[dst] + 1e-8)
        # Weighted agg
        val = hs[src] * alpha.unsqueeze(-1)
        agg = torch.zeros(N, self.heads, self.hd, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(val), val)
        out = self.W_o(agg.view(N, -1))
        return self.norm(F.relu(out) + h)


# ============================================================================
# 5. PNA — Principal Neighbourhood Aggregation (Corso et al., 2020)
# ============================================================================

class PNALayer(nn.Module):
    """PNA: 4 aggregators (sum, mean, max, min) × 3 scalers (identity, amplification, attenuation).

    Recommended config (Corso et al.): hidden=128, layers=4, edge_dim>0.
    The multi-aggregator approach is proven strongest on molecular graph regression.
    """
    def __init__(self, dim, edge_dim=3, avg_degree=5.0):
        super().__init__()
        self.avg_degree = avg_degree
        # 4 aggregators × 3 scalers = 12 channels per input dim
        self.pre_mlp = nn.Linear(dim + edge_dim, dim)
        self.post_mlp = nn.Sequential(
            nn.Linear(dim * 12, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index, edge_attr=None):
        if edge_index.numel() == 0:
            return h
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        # Message with edge features
        if edge_attr is not None:
            msg = F.relu(self.pre_mlp(torch.cat([h[src], edge_attr], dim=-1)))
        else:
            msg = h[src]
        # 4 aggregators
        a_sum = scatter_add(msg, dst, N)
        a_mean = scatter_mean(msg, dst, N)
        a_max = scatter_max(msg, dst, N)
        a_min = scatter_min(msg, dst, N)
        # 3 scalers per aggregator: identity, amplification (log(d+1)), attenuation (1/log(d+1))
        deg = torch.zeros(N, device=h.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=h.device))
        log_deg = torch.log(deg.clamp(min=1) + 1).unsqueeze(-1)
        inv_log_deg = 1.0 / log_deg.clamp(min=0.1)
        scaled = []
        for agg in [a_sum, a_mean, a_max, a_min]:
            scaled.extend([agg, agg * log_deg, agg * inv_log_deg])
        combined = torch.cat(scaled, dim=-1)  # [N, 12*dim]
        out = self.post_mlp(combined)
        return self.norm(out + h)


# ============================================================================
# 6. GatedGCN (Bresson & Laurent, 2017)
# ============================================================================

class GatedGCNLayer(nn.Module):
    """GatedGCN: edge gates control message flow.

    gate_ij = sigma(A*h_i + B*h_j + C*e_ij)
    msg_ij = gate_ij * (D*h_j)
    e_ij' = gate_ij (edge features updated too)

    Proven strong on LRGB (Long Range Graph Benchmark).
    Recommended: hidden=128, layers=4-6, with edge features.
    """
    def __init__(self, dim, edge_dim=3):
        super().__init__()
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(edge_dim, dim)
        self.D = nn.Linear(dim, dim)
        self.E = nn.Linear(dim, edge_dim)  # Update edge features
        self.norm_h = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(edge_dim)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return h, edge_attr
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        # Gate
        gate = torch.sigmoid(self.A(h[dst]) + self.B(h[src]) + self.C(edge_attr))
        # Message
        msg = gate * self.D(h[src])
        agg = scatter_add(msg, dst, N)
        # Normalize by sum of gates
        gate_sum = scatter_add(gate, dst, N).clamp(min=1e-6)
        agg = agg / gate_sum
        h_new = self.norm_h(F.relu(agg) + h)
        # Update edge features
        e_new = self.norm_e(self.E(gate) + edge_attr)
        return h_new, e_new


# ============================================================================
# 7. EdgeConv / DGCNN (Wang et al., 2019)
# ============================================================================

class EdgeConvLayer(nn.Module):
    """EdgeConv: msg_ij = MLP(h_i || h_j - h_i) — captures local geometry.

    Good for physical systems where relative differences matter.
    Recommended: hidden=128, layers=4-5.
    """
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        diff = h[src] - h[dst]  # relative features
        msg = self.mlp(torch.cat([h[dst], diff], dim=-1))
        agg = scatter_max(msg, dst, N)
        return self.norm(agg + h)


# ============================================================================
# 8. APPNP (Gasteiger et al., 2019)
# ============================================================================

class APPNPPropagation(nn.Module):
    """APPNP: Separate feature transform from propagation.

    h^(0) = MLP(x)
    h^(k) = (1-alpha) * A_norm * h^(k-1) + alpha * h^(0)

    Key: alpha (teleport) ∈ [0.1, 0.2] typically.
    Recommended: K_hops=10, alpha=0.1, hidden=128.
    Very parameter-efficient: only MLP params, propagation is parameter-free.
    """
    def __init__(self, dim, K_hops=10, alpha=0.1):
        super().__init__()
        self.K_hops = K_hops
        self.alpha = alpha

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        N = h.size(0)
        h0 = h
        src, dst = edge_index[0], edge_index[1]
        # Compute degree for normalization
        deg = torch.zeros(N, device=h.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=h.device))
        deg_inv = (deg.clamp(min=1) ** -1.0)

        for _ in range(self.K_hops):
            msg = h[src] * deg_inv[src].unsqueeze(-1)
            agg = scatter_add(msg, dst, N)
            h = (1 - self.alpha) * agg + self.alpha * h0
        return h


# ============================================================================
# DeepSets — No message passing (ablation baseline)
# ============================================================================

class DeepSetsModel(nn.Module):
    """DeepSets with chain-aware pooling (knows chain assignment, no MP).

    Uses chain_batch for hierarchical pooling but NO edge-based message passing.
    This tests: "is MP needed if you already know the chain structure?"
    NOTE: This is NOT a true DeepSets baseline because it uses chain_batch.
    Use TrueDeepSetsModel for a proper no-structure baseline.
    """
    def __init__(self, input_dim=7, hidden_dim=128, K=20, dropout=0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        pool_dim = 4 * hidden_dim
        graph_dim = 3 * hidden_dim
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.rho = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, K))
        self.cbr_head = nn.Sequential(nn.Linear(pool_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rms_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        x = batch["x"]
        chain_batch = batch["chain_batch"]
        graph_batch = batch["graph_batch"]
        n_chains = graph_batch.size(0)
        n_graphs = batch["batch_size"]

        h = self.phi(x)
        chain_repr = pool_4way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = pool_3way(chain_h, graph_batch, n_graphs)
        z = self.rho(graph_repr)
        return self.energy_head(z), cbr_pred, self.rms_head(z).squeeze(-1)


class TrueDeepSetsModel(nn.Module):
    """True DeepSets: flat pool over ALL qubits, NO chain structure, NO MP.

    phi(x_i) → flat sum/mean/max over entire graph → rho(z) → output.
    Does NOT use chain_batch at all — treats all qubits as an unstructured set.
    This is the proper lower bound: if this works well, node features alone suffice.
    """
    def __init__(self, input_dim=7, hidden_dim=128, K=20, dropout=0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        graph_dim = 3 * hidden_dim  # sum + mean + max (flat over all qubits)
        self.rho = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.energy_head = nn.Linear(hidden_dim // 2, K)
        self.rms_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch):
        x = batch["x"]
        chain_batch = batch["chain_batch"]
        graph_batch = batch["graph_batch"]
        n_graphs = batch["batch_size"]
        n_chains = graph_batch.size(0)

        # Per-qubit MLP (no MP)
        h = self.phi(x)

        # Flat pool directly over ALL qubits → graph (skip chain level entirely)
        qubit_graph_batch = graph_batch[chain_batch]
        z_sum = scatter_add(h, qubit_graph_batch, n_graphs)
        z_mean = scatter_mean(h, qubit_graph_batch, n_graphs)
        z_max = scatter_max(h, qubit_graph_batch, n_graphs)
        z = torch.cat([z_sum, z_mean, z_max], dim=-1)

        out = self.rho(z)
        energy_pred = self.energy_head(out)
        rms_pred = self.rms_head(out).squeeze(-1)
        cbr_pred = torch.zeros(n_chains, device=x.device)  # no chain-level prediction possible

        return energy_pred, cbr_pred, rms_pred


# ============================================================================
# Flat model factory — generic wrapper for any layer type
# ============================================================================

class FlatGNNGeneric(nn.Module):
    """Generic flat GNN: encoder -> L layers -> pool -> heads.

    Works with any layer that takes (h, edge_index) or (h, edge_index, edge_attr).
    """
    def __init__(self, layer_fn, input_dim=7, hidden_dim=128, num_layers=6,
                 edge_dim=3, K=20, uses_edge_attr=False, updates_edges=False):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([layer_fn() for _ in range(num_layers)])
        self.uses_edge_attr = uses_edge_attr
        self.updates_edges = updates_edges
        pool_dim = 4 * hidden_dim
        graph_dim = 3 * hidden_dim
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.energy_head, self.cbr_head, self.rms_head = _make_heads(pool_dim, graph_dim, K)

    def forward(self, batch):
        x = batch["x"]
        all_ei, all_ea = _merge_edges(batch)
        chain_batch = batch["chain_batch"]
        graph_batch = batch["graph_batch"]
        n_chains = graph_batch.size(0)
        n_graphs = batch["batch_size"]

        h = F.relu(self.encoder(x))
        for layer in self.layers:
            if self.updates_edges:
                h, all_ea = layer(h, all_ei, all_ea)
            elif self.uses_edge_attr:
                h = layer(h, all_ei, all_ea)
            else:
                h = layer(h, all_ei)

        chain_repr = pool_4way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = pool_3way(chain_h, graph_batch, n_graphs)
        energy_pred = self.energy_head(graph_repr)
        rms_pred = self.rms_head(graph_repr).squeeze(-1)
        return energy_pred, cbr_pred, rms_pred


class APPNPModel(nn.Module):
    """APPNP: MLP feature transform + parameter-free propagation.

    Very parameter-efficient. Good when node features are informative
    and you mainly need to smooth them over the graph.
    """
    def __init__(self, input_dim=7, hidden_dim=128, K_hops=10,
                 alpha=0.1, K=20, dropout=0.1):
        super().__init__()
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.propagate = APPNPPropagation(hidden_dim, K_hops, alpha)
        pool_dim = 4 * hidden_dim
        graph_dim = 3 * hidden_dim
        self.chain_proj = nn.Linear(pool_dim, hidden_dim)
        self.energy_head, self.cbr_head, self.rms_head = _make_heads(pool_dim, graph_dim, K)

    def forward(self, batch):
        x = batch["x"]
        all_ei, _ = _merge_edges(batch)
        chain_batch = batch["chain_batch"]
        graph_batch = batch["graph_batch"]
        n_chains = graph_batch.size(0)
        n_graphs = batch["batch_size"]

        h = self.feature_mlp(x)
        h = self.propagate(h, all_ei)

        chain_repr = pool_4way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)
        chain_h = F.relu(self.chain_proj(chain_repr))
        graph_repr = pool_3way(chain_h, graph_batch, n_graphs)
        energy_pred = self.energy_head(graph_repr)
        rms_pred = self.rms_head(graph_repr).squeeze(-1)
        return energy_pred, cbr_pred, rms_pred


# ============================================================================
# Hierarchical variants — swap Stage 1 of HEC-GNN
# ============================================================================

class HECVariant(nn.Module):
    """Generic HEC-* model: custom Stage 1 + MPNN Stage 2 + GINE Stage 3.

    Enables testing different intra-chain encoders while keeping the
    hierarchical structure and inter-chain/logical stages the same.
    """
    def __init__(self, stage1_layer_fn, input_dim=7, hidden_dim=128,
                 L1=3, L3=3, K=20, dropout=0.1,
                 stage1_uses_edge_attr=False, stage1_edge_dim=3):
        super().__init__()
        from src.models.layers import MPNNLayer, GINELayer

        H = hidden_dim
        # Stage 1
        self.stage1_encoder = nn.Linear(input_dim, H)
        self.stage1_layers = nn.ModuleList([stage1_layer_fn() for _ in range(L1)])
        self.stage1_uses_edge_attr = stage1_uses_edge_attr
        stage1_out = 4 * H

        # Stage 2: MPNN
        self.qubit_proj = nn.Linear(stage1_out + input_dim, stage1_out)
        self.mpnn = MPNNLayer(stage1_out, 3)
        self.stage2_proj = nn.Sequential(nn.Linear(4 * stage1_out, stage1_out), nn.ReLU())

        # Stage 3: GINE
        self.stage3_encoder = nn.Linear(stage1_out + 1, H)
        self.stage3_layers = nn.ModuleList([GINELayer(H, 5) for _ in range(L3)])
        self.stage3_updates_edges = False
        graph_dim = 3 * H

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

        # Stage 1
        h = self.stage1_encoder(x)
        for layer in self.stage1_layers:
            h = layer(h, chain_ei)
        c1 = pool_4way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(c1).squeeze(-1)

        # Stage 2
        c_bc = c1[chain_batch]
        h_q = F.relu(self.qubit_proj(torch.cat([c_bc, x], dim=-1)))
        if inter_ei.numel() > 0:
            h_q = self.mpnn(h_q, inter_ei, inter_ea)
        c2 = self.stage2_proj(pool_4way(h_q, chain_batch, n_chains)) + c1

        # Stage 3
        log_cl = torch.log(chain_lengths.float().clamp(min=1.0)).unsqueeze(-1)
        h3 = self.stage3_encoder(torch.cat([c2, log_cl], dim=-1))
        ea = logical_ea
        for layer in self.stage3_layers:
            if self.stage3_updates_edges:
                h3, ea = layer(h3, logical_ei, ea)
            else:
                h3 = layer(h3, logical_ei, ea)
        z = pool_3way(h3, graph_batch, n_graphs)

        return self.energy_head(z), cbr_pred, self.rms_head(z).squeeze(-1)


class HECGatedGCNTrue(nn.Module):
    """True HEC-GatedGCN: EdgeConv at Stage 1, GatedGCN at Stage 3.

    This is the CORRECT implementation:
    - Stage 1: EdgeConv (direction-sensitive, no edge features needed)
    - Stage 2: MPNN (qubit pairing, shared)
    - Stage 3: GatedGCN (edge-gated with edge updates on logical graph)

    The previous hec_gatedgcn was identical to hec_edgeconv (both used
    GINEConv at Stage 3). This version uses actual GatedGCN layers at
    Stage 3 where rich edge features (5-dim) are available.
    """
    def __init__(self, input_dim=7, hidden_dim=128, L1=3, L3=3, K=20,
                 dropout=0.1, logical_edge_dim=5):
        super().__init__()
        from src.models.layers import MPNNLayer

        H = hidden_dim
        # Stage 1: EdgeConv (direction-sensitive)
        self.stage1_encoder = nn.Linear(input_dim, H)
        self.stage1_layers = nn.ModuleList([EdgeConvLayer(H) for _ in range(L1)])
        stage1_out = 4 * H

        # Stage 2: MPNN
        self.qubit_proj = nn.Linear(stage1_out + input_dim, stage1_out)
        self.mpnn = MPNNLayer(stage1_out, 3)
        self.stage2_proj = nn.Sequential(nn.Linear(4 * stage1_out, stage1_out), nn.ReLU())

        # Stage 3: TRUE GatedGCN (edge-gated with edge updates)
        self.stage3_encoder = nn.Linear(stage1_out + 1, H)
        self.stage3_edge_encoder = nn.Linear(logical_edge_dim, H)  # project edge features to H
        self.stage3_layers = nn.ModuleList([
            GatedGCNLayer(H, edge_dim=H) for _ in range(L3)  # GatedGCN needs edge_dim = H
        ])
        graph_dim = 3 * H

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

        # Stage 1: EdgeConv
        h = self.stage1_encoder(x)
        for layer in self.stage1_layers:
            h = layer(h, chain_ei)
        c1 = pool_4way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(c1).squeeze(-1)

        # Stage 2: MPNN
        c_bc = c1[chain_batch]
        h_q = F.relu(self.qubit_proj(torch.cat([c_bc, x], dim=-1)))
        if inter_ei.numel() > 0:
            h_q = self.mpnn(h_q, inter_ei, inter_ea)
        c2 = self.stage2_proj(pool_4way(h_q, chain_batch, n_chains)) + c1

        # Stage 3: GatedGCN (with edge updates!)
        log_cl = torch.log(chain_lengths.float().clamp(min=1.0)).unsqueeze(-1)
        h3 = self.stage3_encoder(torch.cat([c2, log_cl], dim=-1))
        ea = self.stage3_edge_encoder(logical_ea)  # project 5-dim → H-dim
        for layer in self.stage3_layers:
            h3, ea = layer(h3, logical_ei, ea)  # GatedGCN updates both h and e
        z = pool_3way(h3, graph_batch, n_graphs)

        return self.energy_head(z), cbr_pred, self.rms_head(z).squeeze(-1)
