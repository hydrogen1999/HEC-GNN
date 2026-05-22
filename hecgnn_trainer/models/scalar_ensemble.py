"""
scalar_ensemble.py -- Scalar prediction ×K ensemble vs curve prediction.

Generic wrapper: wraps ANY registered model, replaces curve head with K scalar heads.
Works with all HEC variants (HEC-GNN, HEC-GatedGCN, HEC-AGNN, etc.) and flat models.

Two approaches:
  1. ScalarEnsembleWrapper: K independent scalar heads on top of any backbone
  2. ScalarDropoutWrapper: single head + MC dropout at inference
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hecgnn_trainer.models._wrapper_utils import (
    CaptureZ, detect_head_input_dim, install_capture_head,
)

K = 20
GRID = np.logspace(math.log10(0.02), math.log10(5.0), K).astype(np.float32)


class ScalarEnsembleWrapper(nn.Module):
    """Wrap ANY model: replace curve output with K independent scalar heads.

    Takes a base model that outputs (energy_pred [B,K], cbr_pred, rms_pred),
    intercepts the graph representation z, and replaces the energy head
    with K scalar heads that each predict r* independently.

    Training: L1(r_pred_k, r_true) for each head k.
    Inference: median of K predictions.
    """

    def __init__(self, base_model, K=20, dropout=0.1, alpha_mode="hardware"):
        """
        Args:
            base_model: any registered model
            K: number of scalar heads / grid points
            dropout: dropout in scalar heads
            alpha_mode: "hardware" (D-Wave auto_scale, default), "rms"
                        (simpler 1/max(1, r) formula, ablation only), or
                        "none"/None to disable alpha-injection entirely.
                        When set, the alpha vector is concatenated to z
                        before the K scalar heads.
        """
        super().__init__()
        self.base = base_model
        self.K = K
        if alpha_mode in (None, "none", False):
            alpha_mode = None
        self.alpha_mode = alpha_mode

        graph_dim = detect_head_input_dim(base_model, default=384)
        self._capture = CaptureZ(K)
        install_capture_head(self.base, self._capture)

        head_input_dim = graph_dim + K if alpha_mode else graph_dim
        H = graph_dim // 3
        self.scalar_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, max(H // 2, 32)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(H // 2, 32), 1),
            ) for _ in range(K)
        ])

        self.register_buffer('grid', torch.tensor(GRID[:K], dtype=torch.float32))
        self._graph_dim = graph_dim

    def forward(self, batch):
        _, cbr_pred, rms_pred = self.base(batch)
        z = self._capture.z

        if self.alpha_mode:
            from hecgnn_trainer.models.alpha_injection import compute_alpha_vector
            alpha = compute_alpha_vector(batch, self.K, mode=self.alpha_mode)
            z_input = torch.cat([z, alpha], dim=-1)
        else:
            z_input = z

        r_preds = torch.cat([head(z_input) for head in self.scalar_heads], dim=-1)
        self._r_preds = r_preds

        r_median = r_preds.median(dim=-1, keepdim=True).values.clamp(min=1e-4)
        log_grid = torch.log(self.grid.clamp(min=1e-4)).unsqueeze(0)
        log_r = torch.log(r_median)
        energy_pred = (log_grid - log_r) ** 2

        return energy_pred, cbr_pred, rms_pred

    def compute_scalar_loss(self, batch):
        r_true = batch['r_star']
        return F.l1_loss(self._r_preds, r_true.unsqueeze(-1).expand_as(self._r_preds))


class ScalarDropoutWrapper(nn.Module):
    """Wrap any base model: single scalar head + MC dropout at inference.

    Accepts the same `alpha_mode` argument as `ScalarEnsembleWrapper`. When
    set, the D-Wave alpha vector is concatenated to z before the scalar head.
    """

    def __init__(self, base_model, K=20, dropout=0.2, alpha_mode="hardware"):
        super().__init__()
        self.base = base_model
        self.K = K
        self.mc_samples = K
        if alpha_mode in (None, "none", False):
            alpha_mode = None
        self.alpha_mode = alpha_mode

        graph_dim = detect_head_input_dim(base_model, default=384)
        self._capture = CaptureZ(K)
        install_capture_head(self.base, self._capture)

        head_input_dim = graph_dim + K if self.alpha_mode else graph_dim
        H = graph_dim // 3
        self.scalar_head = nn.Sequential(
            nn.Linear(head_input_dim, max(H, 64)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(H, 64), max(H // 2, 32)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(H // 2, 32), 1),
        )
        self.register_buffer('grid', torch.tensor(GRID[:K], dtype=torch.float32))

    def forward(self, batch):
        _, cbr_pred, rms_pred = self.base(batch)
        z = self._capture.z

        if self.alpha_mode:
            from hecgnn_trainer.models.alpha_injection import compute_alpha_vector
            alpha = compute_alpha_vector(batch, self.K, mode=self.alpha_mode)
            z_input = torch.cat([z, alpha], dim=-1)
        else:
            z_input = z

        if self.training:
            r_pred = self.scalar_head(z_input)
            self._r_preds = r_pred.expand(-1, self.K)
        else:
            samples = [self.scalar_head(z_input) for _ in range(self.mc_samples)]
            self._r_preds = torch.cat(samples, dim=-1)

        r_median = self._r_preds.median(dim=-1, keepdim=True).values.clamp(min=1e-4)
        log_grid = torch.log(self.grid.clamp(min=1e-4)).unsqueeze(0)
        energy_pred = (log_grid - torch.log(r_median)) ** 2

        return energy_pred, cbr_pred, rms_pred

    def compute_scalar_loss(self, batch):
        r_true = batch['r_star']
        if self.training:
            return F.l1_loss(self._r_preds[:, 0], r_true)
        return F.l1_loss(self._r_preds.median(dim=-1).values, r_true)
