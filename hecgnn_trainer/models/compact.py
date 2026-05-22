"""
compact.py -- Parameter-efficient HEC-GNN variants + optimization techniques.

The standard HEC-GNN has 2.75M params, dominated by:
  - Stage 2 projection: 4-way pool (4×128=512) → MPNN(512,3) → re-pool 4×512=2048 → proj 2048→512
  - This creates a 2048×512 = 1M param bottleneck

Compact techniques:
  1. CompactHEC:     2-way pool (sum+mean), bottleneck projection → ~500K params
  2. SharedHEC:      Weight-shared GIN layers (same weights reused L times) → ~800K
  3. TinyHEC:        H=64, 2-way pool, shared layers → ~150K
  4. BottleneckHEC:  Full 4-way pool but low-rank projection → ~1M

Knowledge Distillation:
  5. DistillLoss:    KL divergence on energy curves + feature matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.layers import (
    GINLayer, GINELayer, MPNNLayer,
    scatter_add, scatter_mean, scatter_max, scatter_min,
)


# ============================================================================
# Pooling variants
# ============================================================================

def pool_2way(h, idx, n):
    """sum || mean → 2H (drops max/min, saves 50% pool params)."""
    return torch.cat([scatter_add(h, idx, n), scatter_mean(h, idx, n)], dim=-1)

def pool_3way(h, idx, n):
    return torch.cat([scatter_add(h, idx, n), scatter_mean(h, idx, n),
                      scatter_max(h, idx, n)], dim=-1)

def pool_4way(h, idx, n):
    return torch.cat([scatter_add(h, idx, n), scatter_mean(h, idx, n),
                      scatter_max(h, idx, n), scatter_min(h, idx, n)], dim=-1)


# ============================================================================
# 1. CompactHEC — 2-way pooling + bottleneck (~500K params)
# ============================================================================

class CompactHEC(nn.Module):
    """Compact HEC-GNN with 2-way pooling and bottleneck projections.

    Key savings vs standard HEC-GNN (2.75M → ~500K):
      - 2-way pool (sum+mean) instead of 4-way: 2H instead of 4H
      - Bottleneck dim at Stage 2: project to H before re-pool
      - Smaller MLP heads

    If this matches standard HEC-GNN performance, it proves the
    hierarchical inductive bias matters more than parameter count.
    """

    def __init__(self, input_dim=7, hidden_dim=128, L1=3, L3=3, K=20,
                 dropout=0.1, eps_init=0.0):
        super().__init__()
        H = hidden_dim
        pool_mult = 2  # 2-way pool

        # Stage 1: GIN (same as standard)
        self.stage1_encoder = nn.Linear(input_dim, H)
        self.stage1_layers = nn.ModuleList([
            GINLayer(H, eps_init) for _ in range(L1)])
        s1_out = pool_mult * H  # 2H

        # Stage 2: Lightweight qubit pairing
        self.qubit_proj = nn.Linear(s1_out + input_dim, H)  # Bottleneck: project to H, not s1_out
        self.mpnn = MPNNLayer(H, 3)
        # Re-pool 2-way on H → 2H, then project to s1_out
        self.stage2_proj = nn.Sequential(nn.Linear(pool_mult * H, s1_out), nn.ReLU())

        # Stage 3: GINE (same structure, takes s1_out + 1)
        self.stage3_encoder = nn.Linear(s1_out + 1, H)
        self.stage3_layers = nn.ModuleList([
            GINELayer(H, 5, eps_init) for _ in range(L3)])
        graph_dim = 3 * H  # 3-way for graph readout

        # Compact heads
        self.energy_head = nn.Sequential(
            nn.Linear(graph_dim, H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, K))
        self.cbr_head = nn.Sequential(nn.Linear(s1_out, 1))
        self.rms_head = nn.Sequential(nn.Linear(graph_dim, 1))

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
        c1 = pool_2way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(c1).squeeze(-1)

        # Stage 2: Bottleneck
        c_bc = c1[chain_batch]
        h_q = F.relu(self.qubit_proj(torch.cat([c_bc, x], dim=-1)))  # → H (not s1_out)
        if inter_ei.numel() > 0:
            h_q = self.mpnn(h_q, inter_ei, inter_ea)
        c2_raw = pool_2way(h_q, chain_batch, n_chains)
        c2 = self.stage2_proj(c2_raw) + c1

        # Stage 3
        log_cl = torch.log(chain_lengths.float().clamp(min=1.0)).unsqueeze(-1)
        h3 = self.stage3_encoder(torch.cat([c2, log_cl], dim=-1))
        for layer in self.stage3_layers:
            h3 = layer(h3, logical_ei, logical_ea)
        z = pool_3way(h3, graph_batch, n_graphs)

        return self.energy_head(z), cbr_pred, self.rms_head(z).squeeze(-1)


# ============================================================================
# 2. SharedHEC — Weight-shared GIN layers (~800K params)
# ============================================================================

class SharedHEC(nn.Module):
    """HEC-GNN with weight-shared GIN layers.

    Instead of L1 separate GIN layers, use 1 GIN layer applied L1 times.
    Same for Stage 3. Reduces params by ~60% with minimal accuracy loss
    (weight tying is proven effective in Transformers / Universal Transformers).
    """

    def __init__(self, input_dim=7, hidden_dim=128, L1=3, L3=3, K=20,
                 dropout=0.1, eps_init=0.0):
        super().__init__()
        H = hidden_dim

        # Stage 1: SINGLE shared GIN layer, applied L1 times
        self.stage1_encoder = nn.Linear(input_dim, H)
        self.stage1_shared = GINLayer(H, eps_init)
        self.L1 = L1
        s1_out = 4 * H

        # Stage 2
        self.qubit_proj = nn.Linear(s1_out + input_dim, s1_out)
        self.mpnn = MPNNLayer(s1_out, 3)
        self.stage2_proj = nn.Sequential(nn.Linear(4 * s1_out, s1_out), nn.ReLU())

        # Stage 3: SINGLE shared GINE layer, applied L3 times
        self.stage3_encoder = nn.Linear(s1_out + 1, H)
        self.stage3_shared = GINELayer(H, 5, eps_init)
        self.L3 = L3
        graph_dim = 3 * H

        self.energy_head = nn.Sequential(
            nn.Linear(graph_dim, H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, K))
        self.cbr_head = nn.Sequential(
            nn.Linear(s1_out, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
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

        # Stage 1: shared layer applied L1 times
        h = self.stage1_encoder(x)
        for _ in range(self.L1):
            h = self.stage1_shared(h, chain_ei)
        c1 = pool_4way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(c1).squeeze(-1)

        # Stage 2
        c_bc = c1[chain_batch]
        h_q = F.relu(self.qubit_proj(torch.cat([c_bc, x], dim=-1)))
        if inter_ei.numel() > 0:
            h_q = self.mpnn(h_q, inter_ei, inter_ea)
        c2 = self.stage2_proj(pool_4way(h_q, chain_batch, n_chains)) + c1

        # Stage 3: shared layer applied L3 times
        log_cl = torch.log(chain_lengths.float().clamp(min=1.0)).unsqueeze(-1)
        h3 = self.stage3_encoder(torch.cat([c2, log_cl], dim=-1))
        for _ in range(self.L3):
            h3 = self.stage3_shared(h3, logical_ei, logical_ea)
        z = pool_3way(h3, graph_batch, n_graphs)

        return self.energy_head(z), cbr_pred, self.rms_head(z).squeeze(-1)


# ============================================================================
# 3. TinyHEC — Minimal hierarchical model (~130K params)
# ============================================================================

class TinyHEC(nn.Module):
    """Smallest possible hierarchical model.

    H=64, 2-way pool, shared layers, compact heads.
    If this outperforms flat models with similar params (~130K vs 160K FlatGNN),
    it's definitive proof that hierarchy >> capacity.
    """

    def __init__(self, input_dim=7, hidden_dim=64, L1=2, L3=2, K=20,
                 dropout=0.1, eps_init=0.0):
        super().__init__()
        H = hidden_dim

        self.stage1_encoder = nn.Linear(input_dim, H)
        self.stage1_shared = GINLayer(H, eps_init)
        self.L1 = L1
        s1_out = 2 * H  # 2-way pool

        self.qubit_proj = nn.Linear(s1_out + input_dim, H)
        self.mpnn = MPNNLayer(H, 3)
        self.stage2_proj = nn.Sequential(nn.Linear(2 * H, s1_out), nn.ReLU())

        self.stage3_encoder = nn.Linear(s1_out + 1, H)
        self.stage3_shared = GINELayer(H, 5, eps_init)
        self.L3 = L3
        graph_dim = 3 * H

        self.energy_head = nn.Sequential(
            nn.Linear(graph_dim, H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, K))
        self.cbr_head = nn.Linear(s1_out, 1)
        self.rms_head = nn.Linear(graph_dim, 1)

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

        h = self.stage1_encoder(x)
        for _ in range(self.L1):
            h = self.stage1_shared(h, chain_ei)
        c1 = pool_2way(h, chain_batch, n_chains)
        cbr_pred = self.cbr_head(c1).squeeze(-1)

        c_bc = c1[chain_batch]
        h_q = F.relu(self.qubit_proj(torch.cat([c_bc, x], dim=-1)))
        if inter_ei.numel() > 0:
            h_q = self.mpnn(h_q, inter_ei, inter_ea)
        c2 = self.stage2_proj(pool_2way(h_q, chain_batch, n_chains)) + c1

        log_cl = torch.log(chain_lengths.float().clamp(min=1.0)).unsqueeze(-1)
        h3 = self.stage3_encoder(torch.cat([c2, log_cl], dim=-1))
        for _ in range(self.L3):
            h3 = self.stage3_shared(h3, logical_ei, logical_ea)
        z = pool_3way(h3, graph_batch, n_graphs)

        return self.energy_head(z), cbr_pred, self.rms_head(z).squeeze(-1)


# ============================================================================
# 4. Knowledge Distillation — Multi-Method Framework
# ============================================================================
#
# Taxonomy (Gou et al., IJCV 2021 "Knowledge Distillation: A Survey"):
#
# A. RESPONSE-BASED: Transfer output-level knowledge
#    - Soft Labels (Hinton 2015): KL(softmax(z_s/T) || softmax(z_t/T))
#    - Score Matching: MSE(z_s, z_t) on raw logits/scores
#    - Rank Distillation: preserve teacher's ranking over grid points
#
# B. FEATURE-BASED: Transfer intermediate representations
#    - FitNet (Romero 2015): MSE(W*f_s, f_t) per layer with adaptor
#    - Attention Transfer (Zagoruyko 2017): MSE(A_s, A_t) on attention maps
#    - For HEC-GNN: match Stage 1/2/3 representations
#
# C. RELATION-BASED: Transfer inter-sample relations
#    - RKD Distance (Park 2019): preserve pairwise distances in batch
#    - RKD Angle (Park 2019): preserve angular relations in batch
#    - CRD (Tian 2020): contrastive representation distillation
#
# ============================================================================


# --- A1. Response-Based: Soft Labels (Hinton et al., 2015) ---

class SoftLabelKD(nn.Module):
    """Classic Hinton-style KD with temperature-softened KL divergence.

    For energy curves, we convert to distributions via softmax(-E/T)
    (lower energy = higher probability, matching physical interpretation).

    Ref: Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
    """
    def __init__(self, temperature=2.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_pred, teacher_pred):
        T = self.T
        t_soft = F.softmax(-teacher_pred.detach() / T, dim=-1)
        s_log_soft = F.log_softmax(-student_pred / T, dim=-1)
        return F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (T * T)


# --- A2. Response-Based: Score Matching (raw output MSE) ---

class ScoreMatchingKD(nn.Module):
    """Direct MSE on raw output scores (no temperature softening).

    Simpler than soft labels but often competitive. Transfers the exact
    shape of the teacher's energy curve, not just its ranking.

    Ref: Ba & Caruana "Do Deep Nets Really Need to be Deep?" (2014)
    """
    def forward(self, student_pred, teacher_pred):
        return F.mse_loss(student_pred, teacher_pred.detach())


# --- A3. Response-Based: Rank Distillation ---

class RankKD(nn.Module):
    """Preserve teacher's ranking over grid points.

    Instead of matching exact values, ensure the student preserves
    which grid points the teacher ranks as lowest energy.
    Uses a margin-based ranking loss on pairwise comparisons.

    Relevant for chain strength: the argmin matters more than exact curve shape.
    """
    def __init__(self, margin=0.1, top_k=5):
        super().__init__()
        self.margin = margin
        self.top_k = top_k

    def forward(self, student_pred, teacher_pred):
        B, K = student_pred.shape
        # Teacher's ranking
        t_rank = teacher_pred.detach().argsort(dim=-1)  # ascending: best first
        t_top = t_rank[:, :self.top_k]   # top-k lowest energy indices
        t_rest = t_rank[:, self.top_k:]  # remaining indices

        loss = torch.tensor(0.0, device=student_pred.device)
        count = 0
        for b in range(B):
            s_top = student_pred[b, t_top[b]]      # student scores at teacher's top-k
            s_rest = student_pred[b, t_rest[b]]     # student scores at rest
            # Each top should be lower than each rest (margin)
            diff = s_top.unsqueeze(-1) - s_rest.unsqueeze(0) + self.margin  # [top_k, K-top_k]
            loss = loss + F.relu(diff).mean()
            count += 1
        return loss / max(count, 1)


# --- B1. Feature-Based: FitNet (Romero et al., 2015) ---

class FitNetKD(nn.Module):
    """FitNet-style feature matching with learnable adaptors.

    Matches intermediate representations at selected hint/guided layers.
    For HEC-GNN: match Stage 1 chain repr, Stage 2 refined repr, Stage 3 graph repr.

    L_hint = sum_l MSE(W_l * f_s^l, f_t^l)

    Ref: Romero et al. "FitNets: Hints for Thin Deep Nets" (2015)
    """
    def __init__(self, hint_pairs):
        """
        Args:
            hint_pairs: list of (student_dim, teacher_dim) for each hint layer
        """
        super().__init__()
        self.adaptors = nn.ModuleList([
            nn.Linear(s_dim, t_dim) for s_dim, t_dim in hint_pairs
        ])

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: list of student intermediate tensors
            teacher_feats: list of teacher intermediate tensors (detached)
        """
        if not student_feats or not teacher_feats:
            return torch.tensor(0.0)
        device = student_feats[0].device
        loss = torch.tensor(0.0, device=device)
        for adaptor, sf, tf in zip(self.adaptors, student_feats, teacher_feats):
            sf_proj = adaptor.to(device)(sf)
            loss = loss + F.mse_loss(sf_proj, tf.detach())
        return loss / len(self.adaptors)


# --- B2. Feature-Based: Attention Transfer (Zagoruyko & Komodakis, 2017) ---

class AttentionTransferKD(nn.Module):
    """Transfer attention maps between teacher and student.

    Attention map A = sum_c (F_c^2) over channels (spatial attention).
    L_AT = sum_l MSE(A_s^l / ||A_s^l||, A_t^l / ||A_t^l||)

    For GNNs: attention = per-node importance (sum of squared features).

    Ref: Zagoruyko & Komodakis "Paying More Attention to Attention" (2017)
    """
    @staticmethod
    def attention_map(features):
        """Compute attention: per-node importance from feature magnitudes."""
        return (features ** 2).sum(dim=-1)  # [N] or [N_chains]

    def forward(self, student_feats, teacher_feats):
        loss = torch.tensor(0.0, device=student_feats[0].device)
        for sf, tf in zip(student_feats, teacher_feats):
            a_s = self.attention_map(sf)
            a_t = self.attention_map(tf.detach())
            # Normalize
            a_s = a_s / (a_s.norm() + 1e-8)
            a_t = a_t / (a_t.norm() + 1e-8)
            loss = loss + F.mse_loss(a_s, a_t)
        return loss / len(student_feats)


# --- C1. Relation-Based: RKD Distance (Park et al., 2019) ---

class RKDDistanceKD(nn.Module):
    """Relational KD — Distance: preserve pairwise distances in batch.

    mu_ij^t = ||z_i^t - z_j^t|| / mean(||z_i^t - z_j^t||)
    L_RKD_D = Huber(mu_ij^s - mu_ij^t) over all pairs (i,j)

    Captures the geometric structure of the representation space.

    Ref: Park et al. "Relational Knowledge Distillation" (CVPR 2019)
    """
    def forward(self, student_repr, teacher_repr):
        """
        Args:
            student_repr: [B, D_s] graph-level representations
            teacher_repr: [B, D_t] graph-level representations (detached)
        """
        t = teacher_repr.detach()
        # Pairwise distances
        d_s = torch.cdist(student_repr, student_repr, p=2)  # [B, B]
        d_t = torch.cdist(t, t, p=2)  # [B, B]
        # Normalize by mean distance
        mu_s = d_s / (d_s.mean() + 1e-8)
        mu_t = d_t / (d_t.mean() + 1e-8)
        return F.smooth_l1_loss(mu_s, mu_t)


# --- C2. Relation-Based: RKD Angle (Park et al., 2019) ---

class RKDAngleKD(nn.Module):
    """Relational KD — Angle: preserve angular relations in batch.

    cos_ijk = <(z_i-z_j), (z_k-z_j)> / (||z_i-z_j|| * ||z_k-z_j||)
    L_RKD_A = Huber(cos_ijk^s - cos_ijk^t) over all triplets

    Captures higher-order geometric relations than distance alone.

    Ref: Park et al. "Relational Knowledge Distillation" (CVPR 2019)
    """
    def forward(self, student_repr, teacher_repr):
        t = teacher_repr.detach()
        B = student_repr.size(0)
        if B < 3:
            return torch.tensor(0.0, device=student_repr.device)

        loss = torch.tensor(0.0, device=student_repr.device)
        count = 0
        # Sample triplets (all consecutive triples for efficiency)
        for j in range(1, B - 1):
            for i, k in [(j - 1, j + 1)]:
                # Student angles
                v1_s = F.normalize(student_repr[i] - student_repr[j], dim=-1)
                v2_s = F.normalize(student_repr[k] - student_repr[j], dim=-1)
                cos_s = (v1_s * v2_s).sum()
                # Teacher angles
                v1_t = F.normalize(t[i] - t[j], dim=-1)
                v2_t = F.normalize(t[k] - t[j], dim=-1)
                cos_t = (v1_t * v2_t).sum()
                loss = loss + F.smooth_l1_loss(cos_s, cos_t)
                count += 1
        return loss / max(count, 1)


# --- B3. Feature-Based: NST / Neuron Selectivity Transfer (Huang & Wang, 2017) ---

class NSTKD(nn.Module):
    """Neuron Selectivity Transfer: match activation distribution via MMD.

    Aligns the distribution of neuron activations between teacher and student
    using Maximum Mean Discrepancy (MMD) with a polynomial kernel.

    Ref: Huang & Wang "Like What You Like: Knowledge Distill via
         Neuron Selectivity Transfer" (2017)
    Impl ref: github.com/AberHu/Knowledge-Distillation-Zoo/kd_losses/nst.py
    """
    def forward(self, student_feats, teacher_feats):
        if not student_feats or not teacher_feats:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=student_feats[0].device)
        for sf, tf in zip(student_feats, teacher_feats):
            sf_n = F.normalize(sf, p=2, dim=-1)
            tf_n = F.normalize(tf.detach(), p=2, dim=-1)
            loss = loss + self._poly_mmd(sf_n, tf_n)
        return loss / max(len(student_feats), 1)

    @staticmethod
    def _poly_mmd(x, y):
        """Polynomial kernel MMD^2."""
        xx = (x @ x.t() + 1).pow(2).mean()
        yy = (y @ y.t() + 1).pow(2).mean()
        xy = (x @ y.t() + 1).pow(2).mean()
        return xx + yy - 2 * xy


# --- B4. Feature-Based: PKT / Probabilistic Knowledge Transfer (Passalis 2018) ---

class PKTKD(nn.Module):
    """Probabilistic Knowledge Transfer: match probability distributions.

    Converts feature representations to probability distributions via
    cosine similarity kernel, then minimizes KL divergence.

    Ref: Passalis & Tefas "Learning Deep Representations with
         Probabilistic Knowledge Transfer" (ECCV 2018)
    Impl ref: github.com/AberHu/Knowledge-Distillation-Zoo/kd_losses/pkt.py
    """
    def forward(self, student_repr, teacher_repr):
        """Both [B, D] graph-level representations."""
        t = teacher_repr.detach()
        # Cosine similarity kernel → probability distribution per sample
        s_sim = F.cosine_similarity(student_repr.unsqueeze(1),
                                     student_repr.unsqueeze(0), dim=-1)  # [B, B]
        t_sim = F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(0), dim=-1)
        # Softmax to get distributions
        s_prob = F.softmax(s_sim, dim=-1)
        t_prob = F.softmax(t_sim, dim=-1)
        return F.kl_div(s_prob.log(), t_prob, reduction='batchmean')


# --- D1. GNN-Specific: Correlation Congruence (Peng et al., 2019) ---

class CorrelationCongruenceKD(nn.Module):
    """Correlation Congruence: match correlation matrices of representations.

    Preserves the correlation structure between feature dimensions,
    which captures how different aspects of chain structure co-vary.

    Ref: Peng et al. "Correlation Congruence for Knowledge Distillation" (ICCV 2019)
    Impl ref: github.com/AberHu/Knowledge-Distillation-Zoo/kd_losses/cc.py
    """
    def forward(self, student_repr, teacher_repr):
        t = teacher_repr.detach()
        # Center features
        s_centered = student_repr - student_repr.mean(dim=0, keepdim=True)
        t_centered = t - t.mean(dim=0, keepdim=True)
        # Correlation matrices
        s_corr = s_centered.t() @ s_centered / max(student_repr.size(0) - 1, 1)
        t_corr = t_centered.t() @ t_centered / max(t.size(0) - 1, 1)
        # Match via Frobenius norm
        return F.mse_loss(s_corr, t_corr)


# --- D2. GNN-Specific: Stage-wise Feature Distillation ---

class StageWiseKD(nn.Module):
    """HEC-GNN stage-wise distillation: match each hierarchical stage.

    Specialized for HEC-GNN: teacher and student both have 3 stages.
    Match chain repr (Stage 1), refined repr (Stage 2), graph repr (Stage 3).
    Uses projection heads when dimensions differ.

    Inspired by: NeurIPS 2023 "Accelerating Molecular GNNs via KD"
    which distills hidden representations across GNN interaction blocks.
    """
    def __init__(self, student_dims, teacher_dims):
        """
        Args:
            student_dims: [s1_dim, s2_dim, s3_dim] per-stage dims
            teacher_dims: [t1_dim, t2_dim, t3_dim] per-stage dims
        """
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(s_d, t_d) if s_d != t_d else nn.Identity()
            for s_d, t_d in zip(student_dims, teacher_dims)
        ])

    def forward(self, student_stages, teacher_stages):
        """
        Args:
            student_stages: [c1_s, c2_s, z_s] per-stage representations
            teacher_stages: [c1_t, c2_t, z_t] per-stage representations
        """
        device = student_stages[0].device
        loss = torch.tensor(0.0, device=device)
        for proj, sf, tf in zip(self.projectors, student_stages, teacher_stages):
            sf_proj = proj.to(device)(sf)
            loss = loss + F.mse_loss(sf_proj, tf.detach())
        return loss / len(self.projectors)


# --- Combined Distillation Loss ---

class MultiMethodKD(nn.Module):
    """Combined distillation loss supporting all methods simultaneously.

    L = w_task * L_task
      + w_soft * L_soft_label
      + w_score * L_score_match
      + w_rank * L_rank
      + w_fitnet * L_fitnet
      + w_attn * L_attention_transfer
      + w_rkd_d * L_rkd_distance
      + w_rkd_a * L_rkd_angle

    Any weight=0 disables that component.
    Default: response-based only (soft labels + score matching).
    """

    def __init__(self,
                 w_task=0.5,
                 w_soft=0.25, temperature=2.0,
                 w_score=0.25,
                 w_rank=0.0, rank_top_k=5,
                 w_fitnet=0.0, hint_pairs=None,
                 w_attn=0.0,
                 w_rkd_d=0.0,
                 w_rkd_a=0.0):
        super().__init__()
        self.w_task = w_task
        self.w_soft = w_soft
        self.w_score = w_score
        self.w_rank = w_rank
        self.w_fitnet = w_fitnet
        self.w_attn = w_attn
        self.w_rkd_d = w_rkd_d
        self.w_rkd_a = w_rkd_a

        if w_soft > 0:
            self.soft_kd = SoftLabelKD(temperature)
        if w_score > 0:
            self.score_kd = ScoreMatchingKD()
        if w_rank > 0:
            self.rank_kd = RankKD(top_k=rank_top_k)
        if w_fitnet > 0 and hint_pairs:
            self.fitnet_kd = FitNetKD(hint_pairs)
        if w_attn > 0:
            self.attn_kd = AttentionTransferKD()
        if w_rkd_d > 0:
            self.rkd_d_kd = RKDDistanceKD()
        if w_rkd_a > 0:
            self.rkd_a_kd = RKDAngleKD()

    def forward(self, student_pred, teacher_pred, target,
                student_feats=None, teacher_feats=None,
                student_repr=None, teacher_repr=None):
        """
        Args:
            student_pred: [B, K] student energy curve
            teacher_pred: [B, K] teacher energy curve
            target: [B, K] ground truth energy curve
            student_feats: list of intermediate features (for FitNet/AT)
            teacher_feats: list of intermediate features (for FitNet/AT)
            student_repr: [B, D] graph representation (for RKD)
            teacher_repr: [B, D] graph representation (for RKD)
        """
        losses = {}
        total = torch.tensor(0.0, device=student_pred.device)

        # Task loss
        L_task = F.l1_loss(student_pred, target)
        losses['task'] = L_task.item()
        total = total + self.w_task * L_task

        # A1. Soft labels
        if self.w_soft > 0:
            L_soft = self.soft_kd(student_pred, teacher_pred)
            losses['soft'] = L_soft.item()
            total = total + self.w_soft * L_soft

        # A2. Score matching
        if self.w_score > 0:
            L_score = self.score_kd(student_pred, teacher_pred)
            losses['score'] = L_score.item()
            total = total + self.w_score * L_score

        # A3. Rank distillation
        if self.w_rank > 0:
            L_rank = self.rank_kd(student_pred, teacher_pred)
            losses['rank'] = L_rank.item()
            total = total + self.w_rank * L_rank

        # B1. FitNet
        if self.w_fitnet > 0 and student_feats and teacher_feats:
            L_fit = self.fitnet_kd(student_feats, teacher_feats)
            losses['fitnet'] = L_fit.item()
            total = total + self.w_fitnet * L_fit

        # B2. Attention transfer
        if self.w_attn > 0 and student_feats and teacher_feats:
            L_attn = self.attn_kd(student_feats, teacher_feats)
            losses['attn'] = L_attn.item()
            total = total + self.w_attn * L_attn

        # C1. RKD Distance
        if self.w_rkd_d > 0 and student_repr is not None and teacher_repr is not None:
            L_rkd_d = self.rkd_d_kd(student_repr, teacher_repr)
            losses['rkd_d'] = L_rkd_d.item()
            total = total + self.w_rkd_d * L_rkd_d

        # C2. RKD Angle
        if self.w_rkd_a > 0 and student_repr is not None and teacher_repr is not None:
            L_rkd_a = self.rkd_a_kd(student_repr, teacher_repr)
            losses['rkd_a'] = L_rkd_a.item()
            total = total + self.w_rkd_a * L_rkd_a

        losses['total'] = total.item()
        return total, losses


# --- Backward-compatible wrapper ---

class DistillationLoss(MultiMethodKD):
    """Backward-compatible: defaults to soft labels + score matching."""
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__(
            w_task=1.0 - alpha,
            w_soft=alpha * 0.5,
            w_score=alpha * 0.5,
            temperature=temperature,
        )

    def forward(self, student_pred, teacher_pred, target):
        total, losses = super().forward(student_pred, teacher_pred, target)
        return total, {'task': losses['task'], 'distill': losses.get('soft', 0) + losses.get('score', 0)}


# ============================================================================
# 5. Pruning utilities
# ============================================================================

def compute_sparsity(model):
    """Compute fraction of zero weights."""
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.data.abs() < 1e-8).sum().item()
    return zeros / max(total, 1)


def magnitude_prune(model, fraction=0.3):
    """Prune lowest-magnitude weights globally.

    Sets the smallest `fraction` of weights to zero using a global threshold.
    Returns pruned model (in-place) and the threshold used.
    """
    all_weights = []
    for name, p in model.named_parameters():
        if 'weight' in name and p.dim() >= 2:
            all_weights.append(p.data.abs().flatten())

    if not all_weights:
        return model, 0.0

    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, fraction).item()

    for name, p in model.named_parameters():
        if 'weight' in name and p.dim() >= 2:
            mask = p.data.abs() >= threshold
            p.data *= mask.float()

    return model, threshold


def count_nonzero_params(model):
    """Count non-zero parameters (effective model size after pruning)."""
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p.data.abs() > 1e-8).sum().item()
    return nonzero, total
