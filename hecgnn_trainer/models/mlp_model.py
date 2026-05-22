"""
mlp_model.py -- MLP baseline for chain strength curve prediction.

Two variants:
  1. MLPCurve:   18 hand-crafted features -> K energy values (curve output)
  2. MLPScalar:  18 features -> scalar r* (regresses r* directly)

Uses the same 18-feature extraction as LinearRegBaseline.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


K = 20
R_MIN = 0.02
R_MAX = 5.0
GRID = np.logspace(math.log10(R_MIN), math.log10(R_MAX), K).astype(np.float32)


def extract_features_batch(batch):
    """Extract 18 hand-crafted features from a collated batch.

    Returns [B, 18] tensor. Uses batch-level statistics computed from
    the graph structure tensors.
    """
    B = batch['batch_size']
    device = batch['x'].device

    features = torch.zeros(B, 18, device=device)
    chain_batch = batch['chain_batch']
    graph_batch = batch['graph_batch']
    x = batch['x']
    rms = batch['rms_targets']

    for gi in range(B):
        # Chain-level stats
        chain_mask = (graph_batch == gi)
        chain_lens = batch['chain_lengths'][chain_mask].float()
        n_chains = chain_lens.size(0)
        total_qubits = chain_lens.sum().item()

        # Qubit features for this graph
        qubit_mask = (graph_batch[chain_batch] == gi)
        qf = x[qubit_mask]  # [n_q, 7]

        # Logical edges for this graph
        lei = batch['logical_edge_index']
        if lei.numel() > 0:
            le_mask = chain_mask[lei[0]] & chain_mask[lei[1]]
            n_logical_edges = le_mask.sum().item()
            if n_logical_edges > 0:
                le_attr = batch['logical_edge_attr'][le_mask]
                j_vals = le_attr[:, 1].abs()  # |J_ij|/rms
            else:
                j_vals = torch.zeros(1, device=device)
        else:
            n_logical_edges = 0
            j_vals = torch.zeros(1, device=device)

        density = 2 * n_logical_edges / max(n_chains * (n_chains - 1), 1)

        # h_logical approximation from qubit features
        h_vals = qf[:, 0].abs() if qf.size(0) > 0 else torch.zeros(1, device=device)

        # UTC approximation from features
        deg_c = qf[:, 2] if qf.size(0) > 0 else torch.zeros(1, device=device)
        delta_prob = qf[:, 5] if qf.size(0) > 0 else torch.zeros(1, device=device)
        utc_vals = delta_prob / (deg_c + 1e-8)
        utc_val = utc_vals.max().item() if utc_vals.numel() > 0 else 0.0

        features[gi] = torch.tensor([
            chain_lens.mean().item(), chain_lens.max().item(),
            chain_lens.min().item(), chain_lens.std().item() if n_chains > 1 else 0.0,
            j_vals.mean().item(), j_vals.max().item(),
            j_vals.min().item(), j_vals.std().item() if j_vals.size(0) > 1 else 0.0,
            float(n_chains), float(n_logical_edges), density, rms[gi].item(),
            h_vals.max().item(), h_vals.mean().item(),
            utc_val,
            deg_c.max().item(), deg_c.mean().item(),
            total_qubits,
        ], device=device)

    return features


class MLPCurve(nn.Module):
    """MLP: hand-crafted features -> K energy curve values."""

    def __init__(self, feature_dim=18, hidden_layers=None, K=20, dropout=0.1):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        layers = []
        in_dim = feature_dim
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, K))
        self.net = nn.Sequential(*layers)

    def forward(self, batch):
        feats = extract_features_batch(batch)
        energy_pred = self.net(feats)
        B = batch['batch_size']
        n_chains = batch['graph_batch'].size(0)
        cbr_pred = torch.zeros(n_chains, device=energy_pred.device)
        rms_pred = torch.zeros(B, device=energy_pred.device)
        return energy_pred, cbr_pred, rms_pred


class MLPScalar(nn.Module):
    """MLP: hand-crafted features -> scalar r* prediction.

    For compatibility with the curve-based evaluation pipeline, converts
    scalar r* to a synthetic curve (Gaussian centered at r*).
    """

    def __init__(self, feature_dim=18, hidden_layers=None, K=20, dropout=0.1):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        layers = []
        in_dim = feature_dim
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.K = K
        self.register_buffer('grid', torch.tensor(GRID, dtype=torch.float32))

    def forward(self, batch):
        feats = extract_features_batch(batch)
        r_pred = self.net(feats).squeeze(-1)  # [B]
        # Convert to synthetic curve: lower is better at r_pred
        log_grid = torch.log(self.grid)
        log_r = torch.log(r_pred.clamp(min=1e-4)).unsqueeze(-1)  # [B, 1]
        energy_pred = (log_grid.unsqueeze(0) - log_r) ** 2  # [B, K]

        B = batch['batch_size']
        n_chains = batch['graph_batch'].size(0)
        cbr_pred = torch.zeros(n_chains, device=energy_pred.device)
        rms_pred = torch.zeros(B, device=energy_pred.device)
        return energy_pred, cbr_pred, rms_pred
