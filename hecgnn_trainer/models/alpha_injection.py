"""Hardware-rescaling alpha-vector injection.

The prediction head consumes z concatenated with
alpha = [alpha(r_1 * RMS(J)), ..., alpha(r_K * RMS(J))], where alpha is the
D-Wave Advantage `auto_scale` factor that the hardware would apply on
submission. Two formulas are supported:

  * "hardware" (default): the full D-Wave auto_scale formula
        S(r_k) = max{ max(h)/4, -min(h)/4, max(J)/1, -min(J)/2,
                       (r_k * RMS(J)) / 2, 1 }
        alpha(r_k) = 1 / S(r_k)
  * "rms": the simpler 1/max(1, r) ablation formula (no h, J terms).

This module exposes:
  * compute_alpha_vector(batch, K, mode) -- canonical alpha computation.
  * AlphaInjectionWrapper -- attaches the alpha vector at any base model's
    energy/curve head, replacing it with an MLP that takes [z || alpha].
  * FlatGNNAlpha / FlatGNNLargeAlpha -- standalone FlatGNN variants used by
    the alpha-vs-hierarchy ablation table.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.layers import scatter_add, scatter_mean, scatter_max
from src.models.baselines import GRID, K

from hecgnn_trainer.models._wrapper_utils import (
    CaptureZ, detect_head_input_dim, install_capture_head,
)


def compute_alpha_vector(batch, K=20, grid=None, mode="hardware"):
    """Compute the per-grid rescaling vector alpha for each graph in `batch`.

    Args:
        batch: collated batch dict produced by the dataset's collate_fn.
        K: number of grid points.
        grid: `[K]` array of r values. Defaults to `logspace(0.02, 5.0, K)`.
        mode: `"hardware"` for the D-Wave auto_scale formula (default), or
            `"rms"` for the simpler `1/max(1, r)` ablation formula.

    Returns:
        `alpha` tensor of shape `[B, K]`.

    The hardware formula computes
        S(r_k) = max{ max(h)/H_MAX, -min(h)/H_MAX,
                       max(J)/J_EXT_MAX, -min(J)/J_EXT_MIN,
                       (r_k * RMS(J)) / J_EXT_MIN, 1 },
        alpha(r_k) = 1 / S(r_k)
    with `H_MAX=4`, `J_EXT_MAX=1`, `J_EXT_MIN=2` (D-Wave Advantage Pegasus).
    See D-Wave Ocean docs and arXiv:2503.08303 for the underlying physics.
    """
    if grid is None:
        grid = np.logspace(math.log10(0.02), math.log10(5.0), K).astype(np.float32)

    B = batch['batch_size']
    device = batch['x'].device
    rms = batch['rms_targets']
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    alpha = torch.zeros(B, K, device=device)

    if mode == "rms":
        for gi in range(B):
            alpha[gi] = 1.0 / torch.clamp(grid_tensor, min=0.1)
        alpha = alpha / (alpha.max(dim=-1, keepdim=True).values + 1e-8)

    elif mode == "hardware":
        # D-Wave Advantage Pegasus hardware limits.
        H_MAX = 4.0
        J_EXT_MAX = 1.0
        J_EXT_MIN = 2.0

        for gi in range(B):
            jc_grid = grid_tensor * rms[gi]

            # Qubit feature 1 stores |h|/RMS; unnormalize to recover |h_phys|.
            chain_mask = (batch['graph_batch'][batch['chain_batch']] == gi)
            qf = batch['x'][chain_mask]
            if qf.size(0) > 0:
                h_max_abs = (qf[:, 1] * rms[gi]).max().item()
                s_h = h_max_abs / H_MAX

                lei = batch['logical_edge_index']
                if lei.numel() > 0:
                    le_mask = (batch['graph_batch'][lei[0]] == gi)
                    if le_mask.any():
                        le_attr = batch['logical_edge_attr'][le_mask]
                        j_raw = (le_attr[:, 0] * rms[gi]).abs()
                        j_max = j_raw.max().item() if j_raw.numel() > 0 else 0.0
                        s_j_pos = j_max / J_EXT_MAX
                        s_j_neg = j_max / J_EXT_MIN
                    else:
                        s_j_pos, s_j_neg = 0.0, 0.0
                else:
                    s_j_pos, s_j_neg = 0.0, 0.0
            else:
                s_h, s_j_pos, s_j_neg = 0.0, 0.0, 0.0

            s_chain = jc_grid / J_EXT_MIN
            s_problem = max(s_h, s_j_pos, s_j_neg)
            scaling = torch.clamp(
                torch.max(s_chain, torch.tensor(s_problem, device=device)),
                min=1.0,
            )
            alpha[gi] = 1.0 / scaling

    else:
        raise ValueError(f"Unknown alpha mode: {mode}. Use 'rms' or 'hardware'.")

    return alpha


class AlphaInjectionHead(nn.Module):
    """Energy prediction head with α-vector injection.

    MLP_curve(z || α) where α ∈ R^K is concatenated with graph repr z.
    """

    def __init__(self, graph_dim, hidden_dim, K=20, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(graph_dim + K, hidden_dim),  # +K for alpha concat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, K),
        )

    def forward(self, z, alpha):
        """
        Args:
            z: [B, graph_dim] graph representation
            alpha: [B, K] rescaling vector
        """
        return self.head(torch.cat([z, alpha], dim=-1))


class FlatGNNAlpha(nn.Module):
    """FlatGNN with alpha-vector injection at the head.

    Same as FlatGNN but `energy_head` takes `[z || alpha]` instead of just z.
    Used by the alpha-vs-hierarchy ablation table.

    Args:
        alpha_mode: `"rms"` for the simpler `1/max(1, r)` formula or
            `"hardware"` for the D-Wave auto_scale formula.
    """

    def __init__(self, node_dim=7, edge_dim=3, hidden_dim=128,
                 num_layers=6, K=20, dropout=0.1, alpha_mode="rms"):
        super().__init__()
        from src.models.layers import GINELayer

        self.encoder = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GINELayer(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        graph_dim = 3 * hidden_dim
        self.K = K
        self.alpha_mode = alpha_mode

        # α-injected head (key difference from standard FlatGNN)
        self.energy_head = AlphaInjectionHead(graph_dim, hidden_dim, K, dropout)

        # Standard auxiliary heads
        self.cbr_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.rms_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim // 2),
            nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, batch):
        x = batch["x"]
        chain_ei = batch["chain_edge_index"]
        inter_ei = batch["inter_edge_index"]

        # Merge edges
        ei_list, ef_list = [], []
        if chain_ei.numel() > 0:
            cf = torch.zeros(chain_ei.size(1), 3, device=x.device)
            cf[:, 0] = 1.0
            ei_list.append(chain_ei); ef_list.append(cf)
        if inter_ei.numel() > 0:
            ei_list.append(inter_ei); ef_list.append(batch["inter_edge_attr"])
        if ei_list:
            all_edges = torch.cat(ei_list, dim=1)
            all_attrs = torch.cat(ef_list, dim=0)
        else:
            all_edges = torch.zeros(2, 0, dtype=torch.long, device=x.device)
            all_attrs = torch.zeros(0, 3, device=x.device)

        chain_batch = batch["chain_batch"]
        graph_batch = batch["graph_batch"]
        qubit_graph_batch = graph_batch[chain_batch]
        n_graphs = batch["batch_size"]
        n_chains = graph_batch.size(0)

        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, all_edges, all_attrs)

        # Chain-level pool (4-way: sum + mean + max + min)
        from src.models.layers import scatter_min
        chain_repr = torch.cat([
            scatter_add(h, chain_batch, n_chains),
            scatter_mean(h, chain_batch, n_chains),
            scatter_max(h, chain_batch, n_chains),
            scatter_min(h, chain_batch, n_chains),
        ], dim=-1)
        cbr_pred = self.cbr_head(chain_repr).squeeze(-1)

        # Graph-level pool (3-way: sum + mean + max)
        z_sum = scatter_add(h, qubit_graph_batch, n_graphs)
        z_mean = scatter_mean(h, qubit_graph_batch, n_graphs)
        z_max = scatter_max(h, qubit_graph_batch, n_graphs)
        z = torch.cat([z_sum, z_mean, z_max], dim=-1)

        # Inject the alpha vector at the head.
        alpha = compute_alpha_vector(batch, self.K, mode=self.alpha_mode)
        energy_pred = self.energy_head(z, alpha)

        rms_pred = self.rms_head(z).squeeze(-1)

        return energy_pred, cbr_pred, rms_pred


class FlatGNNLargeAlpha(FlatGNNAlpha):
    """FlatGNN-Large with α-injection. Parameter-matched ablation against HEC-GNN."""

    def __init__(self, node_dim=7, edge_dim=3, hidden_dim=192,
                 num_layers=8, K=20, dropout=0.1, alpha_mode="rms"):
        super().__init__(node_dim, edge_dim, hidden_dim, num_layers, K, dropout, alpha_mode)


class AlphaInjectionWrapper(nn.Module):
    """Wrap any base model so its energy head consumes the alpha vector.

    Pipeline:
      1. Detect the base model's energy-head input dim (`graph_dim`).
      2. Replace the head with a CaptureZ stub so the graph representation z
         can be intercepted on every forward pass.
      3. Build a new head whose input is `[z || alpha]` with dim
         `graph_dim + K`.
      4. At forward time, run the base model to obtain `(cbr_pred, rms_pred)`
         and `z`, compute the alpha vector via `compute_alpha_vector`,
         concatenate, and emit a new energy curve. `cbr_pred` and `rms_pred`
         pass through unchanged.

    Args:
        base_model: any registered model whose forward returns
            `(energy_pred [B, K], cbr_pred, rms_pred)`.
        K: grid size (number of chain-strength ratios).
        hidden_dim: hidden width of the new head MLP. Inferred from
            `graph_dim` if `None`.
        dropout: dropout in the new head.
        alpha_mode: `"hardware"` (D-Wave auto_scale, default), `"rms"`
            (simpler `1/max(1, r)` formula, for ablation), or `"none"`
            (passthrough; behaves like the base model).
    """

    def __init__(self, base_model, K=20, hidden_dim=None, dropout=0.1,
                 alpha_mode="hardware"):
        super().__init__()
        self.base = base_model
        self.K = K
        self.alpha_mode = alpha_mode

        if alpha_mode == "none":
            return

        graph_dim = detect_head_input_dim(base_model, default=3 * 128)
        self._graph_dim = graph_dim
        H = hidden_dim or max(graph_dim // 3, 64)

        self._capture = CaptureZ(K)
        install_capture_head(self.base, self._capture)

        self.energy_head = nn.Sequential(
            nn.Linear(graph_dim + K, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, K),
        )

    def forward(self, batch):
        if self.alpha_mode == "none":
            return self.base(batch)
        _, cbr_pred, rms_pred = self.base(batch)
        z = self._capture.z
        alpha = compute_alpha_vector(batch, self.K, mode=self.alpha_mode)
        z_aug = torch.cat([z, alpha], dim=-1)
        energy_pred = self.energy_head(z_aug)
        return energy_pred, cbr_pred, rms_pred


def wrap_with_alpha(base_model, K=20, dropout=0.1, alpha_mode="hardware"):
    """Wrap a base model with `AlphaInjectionWrapper` unless disabled.

    Returns `base_model` unchanged when `alpha_mode == "none"`. Otherwise
    returns an `AlphaInjectionWrapper(base_model, K, dropout, alpha_mode)`.
    Used by `hecgnn_trainer.registry.build_model` to apply the default
    alpha-injection to every architecture in the registry.
    """
    if alpha_mode == "none":
        return base_model
    return AlphaInjectionWrapper(base_model, K=K, dropout=dropout,
                                  alpha_mode=alpha_mode)
