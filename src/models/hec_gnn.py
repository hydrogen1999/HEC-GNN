"""
hec_gnn.py -- Hierarchical Energy-Curve GNN (HEC-GNN).

The core model from the paper. Three-stage hierarchical backbone:
  Stage 1: Intra-chain GIN  (qubit -> chain)
  Stage 2: Qubit pairing recovery  (single MPNN on inter-chain edges)
  Stage 3: Global logical integration (GINE on logical graph)

Output heads:
  - Energy curve: MLP -> [K] predicted E(r_k)
  - CBR auxiliary: per-chain break probability (BCE)
  - RMS auxiliary: global RMS(J) prediction (L1)
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import (
    GINLayer, GINELayer, MPNNLayer,
    scatter_add, scatter_mean, scatter_max, scatter_min,
)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
K = 20                  # grid points
NODE_DIM = 7            # [h/rms, |h|/rms, deg_C, deg_x, |C_i|, delta/rms, 1_{singleton}]
INTER_EDGE_DIM = 3      # [0, J_pq/rms, |J_pq|/rms]
LOGICAL_EDGE_DIM = 5    # [J_ij/rms, |J_ij|/rms, |E_x^ij|, mu/rms, sigma/rms]


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
class ModelConfig:
    def __init__(self, node_dim=NODE_DIM, hidden_dim=128, L1=3, L3=3,
                 K=K, dropout=0.1, eps_init=0.0):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.L1 = L1
        self.L3 = L3
        self.K = K
        self.dropout = dropout
        self.eps_init = eps_init


# ============================================================================
# Stage 1: Intra-Chain Structural Encoding
# ============================================================================

class IntraChainEncoder(nn.Module):
    """Stage 1: Process each chain independently with GIN.

    - GIN layers (no edge features -- all chain edges are identical ferromagnetic).
    - 4-aggregation pooling: sum || mean || max || min.
      The min operator isolates weakest-link vulnerabilities.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 eps_init: float = 0.0):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, eps_init=eps_init) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, chain_edge_index: torch.Tensor,
                chain_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N_qubits, node_dim] raw qubit features.
            chain_edge_index: [2, E_chain] intra-chain edges.
            chain_batch: [N_qubits] -> chain index.

        Returns:
            c1: [N_chains, 4*hidden_dim] chain representations.
        """
        n_chains = chain_batch.max().item() + 1 if chain_batch.numel() > 0 else 1

        h = self.encoder(x)  # [N_qubits, hidden_dim]

        # GIN message passing (intra-chain only)
        for layer in self.layers:
            h = layer(h, chain_edge_index)

        # 4-aggregation pooling per chain: sum || mean || max || min
        c_sum = scatter_add(h, chain_batch, n_chains)
        c_mean = scatter_mean(h, chain_batch, n_chains)
        c_max = scatter_max(h, chain_batch, n_chains)
        c_min = scatter_min(h, chain_batch, n_chains)

        return torch.cat([c_sum, c_mean, c_max, c_min], dim=-1)  # [N_chains, 4H]


# ============================================================================
# Stage 2: Qubit Pairing Recovery
# ============================================================================

class QubitPairingLayer(nn.Module):
    """Stage 2: Recover qubit-level pairing across chain boundaries.

    A strict chain-level pooling discards spatial alignment -- which
    vulnerable qubit is coupled to a heavy external load.  This stage
    recovers that information via a single MPNN layer on inter-chain edges.

    1. Reconstruct qubit states: h_q' = [c_i || x_q]
    2. Single MPNN layer across inter-chain edges only
    3. Re-pool to refined chain representations c_i^(2)
    """

    def __init__(self, chain_repr_dim: int, node_dim: int, inter_edge_dim: int = 3):
        super().__init__()
        self.qubit_proj = nn.Linear(chain_repr_dim + node_dim, chain_repr_dim)
        self.mpnn = MPNNLayer(chain_repr_dim, inter_edge_dim)

    def forward(self, c1: torch.Tensor, x: torch.Tensor,
                inter_edge_index: torch.Tensor, inter_edge_attr: torch.Tensor,
                chain_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c1: [N_chains, 4H] chain representations from Stage 1.
            x: [N_qubits, node_dim] original qubit features.
            inter_edge_index: [2, E_inter] inter-chain edges at qubit level.
            inter_edge_attr: [E_inter, 3] inter-chain edge features.
            chain_batch: [N_qubits] -> chain index.

        Returns:
            c2: [N_chains, 4H] refined chain representations.
        """
        n_qubits = x.size(0)
        n_chains = chain_batch.max().item() + 1 if chain_batch.numel() > 0 else 1
        chain_repr_dim = c1.size(1)

        # Reconstruct qubit-level states: broadcast chain repr to each qubit
        # h_q' = [c_{chain(q)} || x_q]
        c_broadcast = c1[chain_batch]  # [N_qubits, 4H]
        h_q = torch.cat([c_broadcast, x], dim=-1)  # [N_qubits, 4H + node_dim]
        h_q = self.qubit_proj(h_q)  # [N_qubits, 4H]

        # Single MPNN layer on inter-chain edges only
        if inter_edge_index.numel() > 0:
            h_q = self.mpnn(h_q, inter_edge_index, inter_edge_attr)

        # Re-pool to chain level (same 4-aggregation)
        c_sum = scatter_add(h_q, chain_batch, n_chains)
        c_mean = scatter_mean(h_q, chain_batch, n_chains)
        c_max = scatter_max(h_q, chain_batch, n_chains)
        c_min = scatter_min(h_q, chain_batch, n_chains)

        c2 = torch.cat([c_sum, c_mean, c_max, c_min], dim=-1)  # [N_chains, 4*(4H)]

        # Project back to same dim as c1 for residual-like behavior
        # Actually, we want the chain_repr_dim to stay at 4H going into Stage 3.
        # So we use a projection here.
        return c2


# ============================================================================
# Stage 3: Global Logical Integration
# ============================================================================

class LogicalIntegrator(nn.Module):
    """Stage 3: GINEConv on the logical graph.

    1. Augment chain representations with log(|C_i|)
    2. L3 layers of edge-aware GINE over logical edges
    3. Global readout: sum + mean + max pooling over V_L
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3,
                 edge_dim: int = 5, eps_init: float = 0.0):
        super().__init__()
        # +1 for log(chain_length) augmentation
        self.encoder = nn.Linear(in_dim + 1, hidden_dim)
        self.layers = nn.ModuleList([
            GINELayer(hidden_dim, edge_dim, eps_init=eps_init)
            for _ in range(num_layers)
        ])

    def forward(self, c: torch.Tensor, chain_lengths: torch.Tensor,
                logical_edge_index: torch.Tensor, logical_edge_attr: torch.Tensor,
                graph_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: [N_chains, in_dim] refined chain representations.
            chain_lengths: [N_chains] length of each chain.
            logical_edge_index: [2, E_logical] logical graph edges.
            logical_edge_attr: [E_logical, 5] logical edge features.
            graph_batch: [N_chains] -> graph index.

        Returns:
            z: [B, 3*hidden_dim] global graph representation.
        """
        n_chains = c.size(0)
        n_graphs = graph_batch.max().item() + 1 if graph_batch.numel() > 0 else 1

        # Augment with log(chain_length)
        log_cl = torch.log(chain_lengths.float().clamp(min=1.0)).unsqueeze(-1)  # [N_chains, 1]
        h = torch.cat([c, log_cl], dim=-1)  # [N_chains, in_dim + 1]
        h = self.encoder(h)  # [N_chains, hidden_dim]

        # GINE message passing on logical graph
        for layer in self.layers:
            h = layer(h, logical_edge_index, logical_edge_attr)

        # Global pooling: sum + mean + max
        z_sum = scatter_add(h, graph_batch, n_graphs)
        z_mean = scatter_mean(h, graph_batch, n_graphs)
        z_max = scatter_max(h, graph_batch, n_graphs)

        return torch.cat([z_sum, z_mean, z_max], dim=-1)  # [B, 3H]


# ============================================================================
# Full HEC-GNN Model
# ============================================================================

class HECGNN(nn.Module):
    """Hierarchical Energy-Curve GNN.

    G_E -> Stage1 (intra-chain GIN)
        -> Stage2 (qubit pairing MPNN)
        -> Stage3 (logical GINE)
        -> MLP -> [E(r_1), ..., E(r_K)]

    Auxiliary heads:
        - CBR: per-chain break probability (from Stage 1 output)
        - RMS: global RMS(J) prediction (from graph representation)
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.config = config
        H = config.hidden_dim

        # Stage 1: Intra-chain encoding
        self.stage1 = IntraChainEncoder(
            input_dim=config.node_dim,
            hidden_dim=H,
            num_layers=config.L1,
            eps_init=config.eps_init,
        )
        stage1_out_dim = 4 * H  # sum + mean + max + min

        # Stage 2: Qubit pairing recovery
        self.stage2 = QubitPairingLayer(
            chain_repr_dim=stage1_out_dim,
            node_dim=config.node_dim,
            inter_edge_dim=3,
        )
        # After Stage 2 re-pooling with 4 aggregations on projected qubit features
        # The qubit features are projected to stage1_out_dim, then 4-agg pooled
        stage2_out_dim = 4 * stage1_out_dim  # 4 * 4H = 16H

        # Projection to manageable dim before Stage 3
        self.stage2_proj = nn.Sequential(
            nn.Linear(stage2_out_dim, stage1_out_dim),
            nn.ReLU(),
        )
        stage3_in_dim = stage1_out_dim  # 4H

        # Stage 3: Global logical integration
        self.stage3 = LogicalIntegrator(
            in_dim=stage3_in_dim,
            hidden_dim=H,
            num_layers=config.L3,
            edge_dim=LOGICAL_EDGE_DIM,
            eps_init=config.eps_init,
        )
        graph_repr_dim = 3 * H  # sum + mean + max

        # Energy curve prediction head: z -> [K]
        self.energy_head = nn.Sequential(
            nn.Linear(graph_repr_dim, H),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, config.K),
        )

        # CBR auxiliary head: per-chain break probability
        # Operates on Stage 1 chain representations
        self.cbr_head = nn.Sequential(
            nn.Linear(stage1_out_dim, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

        # RMS auxiliary head: global RMS(J) prediction
        self.rms_head = nn.Sequential(
            nn.Linear(graph_repr_dim, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the 3-stage hierarchy.

        Args:
            batch: collated batch dict with keys:
                x, chain_edge_index, inter_edge_index, inter_edge_attr,
                chain_batch, logical_edge_index, logical_edge_attr,
                graph_batch, chain_lengths.

        Returns:
            energy_pred: [B, K] predicted energy curve.
            cbr_pred: [N_chains] predicted per-chain break probability (logits).
            rms_pred: [B] predicted RMS(J).
        """
        x = batch["x"]
        chain_edge_index = batch["chain_edge_index"]
        inter_edge_index = batch["inter_edge_index"]
        inter_edge_attr = batch["inter_edge_attr"]
        chain_batch = batch["chain_batch"]
        logical_edge_index = batch["logical_edge_index"]
        logical_edge_attr = batch["logical_edge_attr"]
        graph_batch = batch["graph_batch"]
        chain_lengths = batch["chain_lengths"]

        # Stage 1: Intra-chain encoding
        c1 = self.stage1(x, chain_edge_index, chain_batch)  # [N_chains, 4H]

        # CBR auxiliary prediction (from Stage 1 representations)
        cbr_pred = self.cbr_head(c1).squeeze(-1)  # [N_chains]

        # Stage 2: Qubit pairing recovery
        c2_raw = self.stage2(c1, x, inter_edge_index, inter_edge_attr, chain_batch)
        c2 = self.stage2_proj(c2_raw)  # [N_chains, 4H]

        # Residual connection from Stage 1
        c2 = c2 + c1

        # Stage 3: Global logical integration
        z = self.stage3(c2, chain_lengths, logical_edge_index, logical_edge_attr,
                        graph_batch)  # [B, 3H]

        # Energy curve prediction
        energy_pred = self.energy_head(z)  # [B, K]

        # RMS auxiliary prediction
        rms_pred = self.rms_head(z).squeeze(-1)  # [B]

        return energy_pred, cbr_pred, rms_pred

    def predict_r_star(self, batch: Dict[str, torch.Tensor],
                       grid: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict optimal r* with parabolic interpolation.

        Args:
            batch: collated batch dict.
            grid: [K] tensor of grid points. If None, uses default GRID.

        Returns:
            r_star: [B] predicted optimal normalized chain strength.
            energy_pred: [B, K] predicted energy curve.
        """
        if grid is None:
            grid = torch.tensor(
                np.logspace(math.log10(0.02), math.log10(5.0), K).astype(np.float32),
                device=batch["x"].device)


        energy_pred, _, _ = self.forward(batch)
        r_star = parabolic_argmin(energy_pred, grid)
        return r_star, energy_pred


def parabolic_argmin(energy_pred: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Recover sub-grid r* via parabolic interpolation.

    For each instance, find the grid argmin, then fit a parabola through
    the minimum and its two neighbors to get a continuous r* estimate.

    Args:
        energy_pred: [B, K] predicted energy values.
        grid: [K] grid points.

    Returns:
        r_star: [B] interpolated optimal r values.
    """
    B, K_val = energy_pred.shape
    device = energy_pred.device

    # Discrete argmin
    k_star = energy_pred.argmin(dim=1)  # [B]

    r_star = torch.zeros(B, device=device)
    for i in range(B):
        k = k_star[i].item()
        if k == 0 or k == K_val - 1:
            # Boundary: no interpolation possible
            r_star[i] = grid[k]
        else:
            # Parabolic interpolation using 3 points
            r_l, r_m, r_r = grid[k - 1].item(), grid[k].item(), grid[k + 1].item()
            e_l = energy_pred[i, k - 1].item()
            e_m = energy_pred[i, k].item()
            e_r = energy_pred[i, k + 1].item()

            # Fit parabola: minimize a*r^2 + b*r + c through 3 points
            # Vertex at r = -b / (2a)
            denom = 2.0 * ((r_l - r_m) * (e_l - e_r) - (r_l - r_r) * (e_l - e_m))
            if abs(denom) < 1e-12:
                r_star[i] = grid[k]
            else:
                numer = ((r_l - r_m) ** 2 * (e_l - e_r) -
                         (r_l - r_r) ** 2 * (e_l - e_m))
                r_opt = r_l - numer / denom
                # Clamp to the interval [r_l, r_r]
                r_opt = max(r_l, min(r_r, r_opt))
                r_star[i] = r_opt

    return r_star
