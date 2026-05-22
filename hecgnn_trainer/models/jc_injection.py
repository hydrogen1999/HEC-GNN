"""Models that consume J_c (or alpha derived from J_c) explicitly at the head.

Two variants are exposed:

  * `JCDirectHEC`: HEC-GNN whose energy head receives the raw J_c vector
    `J_c(r_k) = r_k * RMS(J)` together with `RMS(J)` itself, so the model
    is given the physical scale directly rather than inferring it from
    RMS-normalized features.
  * `HardwareNormHEC`: HEC-GNN whose head receives the D-Wave auto_scale
    alpha vector. The underlying `compute_alpha_vector` formula matches the
    rescaling that the QPU applies on submission.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.hec_gnn import HECGNN, ModelConfig as HECModelConfig
from src.models.layers import scatter_add, scatter_mean, scatter_max

K = 20
GRID = np.logspace(math.log10(0.02), math.log10(5.0), K).astype(np.float32)


class JCDirectHEC(nn.Module):
    """HEC-GNN with direct J_C injection at the prediction head.

    Instead of the model learning what J_C values mean from normalized features,
    we explicitly provide J_C(r_k) = r_k * RMS(J) for each grid point k
    as additional input to the energy head.

    head input: [z (graph repr) || J_C_vector (K values) || RMS(J)]
    The model gets BOTH the learned graph features AND the actual physical scale.
    """

    def __init__(self, hidden_dim=128, L1=3, L3=3, K=20,
                 dropout=0.1, eps_init=0.0):
        super().__init__()
        H = hidden_dim
        hcfg = HECModelConfig(hidden_dim=H, L1=L1, L3=L3, K=K,
                               dropout=dropout, eps_init=eps_init)
        self.backbone = HECGNN(hcfg)

        # Energy head: [z (3H) || J_c vector (K) || RMS(J) (1)] -> K-dim curve.
        graph_dim = 3 * H
        self.backbone.energy_head = nn.Sequential(
            nn.Linear(graph_dim + K + 1, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, K),
        )
        self.K = K
        self.register_buffer('grid', torch.tensor(GRID[:K], dtype=torch.float32))

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

        bb = self.backbone
        c1 = bb.stage1(x, chain_ei, chain_batch)
        cbr_pred = bb.cbr_head(c1).squeeze(-1)

        c2_raw = bb.stage2(c1, x, inter_ei, inter_ea, chain_batch)
        c2 = bb.stage2_proj(c2_raw) + c1

        z = bb.stage3(c2, chain_lengths, logical_ei, logical_ea, graph_batch)

        rms = batch['rms_targets']
        jc_vector = self.grid.unsqueeze(0) * rms.unsqueeze(1)
        z_augmented = torch.cat([z, jc_vector, rms.unsqueeze(1)], dim=-1)
        energy_pred = bb.energy_head(z_augmented)

        rms_pred = bb.rms_head(z).squeeze(-1)
        return energy_pred, cbr_pred, rms_pred


class HardwareNormHEC(nn.Module):
    """HEC-GNN with hardware-aware normalization.

    Instead of RMS(J) normalization, use the QPU's actual rescaling:
      α(J_c) = max(|h_max|, |J_max|, J_c) / hardware_range

    The grid is in α-space, and features are normalized by the
    hardware's energy scale at each grid point.

    This matches what the QPU actually sees — the model learns
    in the hardware's coordinate system.
    """

    def __init__(self, hidden_dim=128, L1=3, L3=3, K=20,
                 dropout=0.1, eps_init=0.0):
        super().__init__()
        hcfg = HECModelConfig(hidden_dim=hidden_dim, L1=L1, L3=L3, K=K,
                               dropout=dropout, eps_init=eps_init)
        self.backbone = HECGNN(hcfg)

        H = hidden_dim
        graph_dim = 3 * H
        # Energy head takes z + α_vector (K values)
        self.backbone.energy_head = nn.Sequential(
            nn.Linear(graph_dim + K, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, K),
        )
        self.K = K
        self.register_buffer('grid', torch.tensor(GRID[:K], dtype=torch.float32))

    def _compute_alpha(self, batch):
        """Return the D-Wave auto_scale alpha vector for this batch.

        Delegates to `hecgnn_trainer.models.alpha_injection.compute_alpha_vector`
        with `mode="hardware"`, so the full formula

            S(r_k) = max{ max(h)/4, -min(h)/4, max(J)/1, -min(J)/2,
                          (r_k * RMS(J)) / 2, 1 },
            alpha(r_k) = 1 / S(r_k)

        is used, including the problem h/J contributions in addition to the
        chain-coupler term.
        """
        from hecgnn_trainer.models.alpha_injection import compute_alpha_vector
        return compute_alpha_vector(batch, self.K, mode="hardware")

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

        bb = self.backbone
        c1 = bb.stage1(x, chain_ei, chain_batch)
        cbr_pred = bb.cbr_head(c1).squeeze(-1)
        c2_raw = bb.stage2(c1, x, inter_ei, inter_ea, chain_batch)
        c2 = bb.stage2_proj(c2_raw) + c1
        z = bb.stage3(c2, chain_lengths, logical_ei, logical_ea, graph_batch)

        # Hardware rescaling vector
        alpha = self._compute_alpha(batch)
        z_aug = torch.cat([z, alpha], dim=-1)  # [B, 3H + K]
        energy_pred = bb.energy_head(z_aug)
        rms_pred = bb.rms_head(z).squeeze(-1)

        return energy_pred, cbr_pred, rms_pred
