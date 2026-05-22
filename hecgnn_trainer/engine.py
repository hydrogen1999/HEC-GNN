"""
engine.py -- Unified training engine for all architectures.

Model-agnostic: builds model from registry, runs standard training loop
with configurable optimizer, scheduler, loss, and evaluation.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import ChainStrengthDataset, collate_batch, make_dataloaders, GRID, K
from src.models.hec_gnn import parabolic_argmin
from hecgnn_trainer.config import ExperimentConfig
from hecgnn_trainer.registry import build_model, model_info


# ---------------------------------------------------------------
# Loss
# ---------------------------------------------------------------

def compute_loss(energy_pred, cbr_pred, rms_pred, batch,
                 beta_cbr=0.1, beta_rms=0.05,
                 loss_mode="standard"):
    """Compute training loss.

    loss_mode:
      "standard"     — L_energy + beta_cbr*L_cbr + beta_rms*L_rms (default)
      "auxiliary_only" — ONLY L_cbr + L_rms, NO energy curve loss.
                        Tests if break patterns alone encode enough info.
      "energy_only"  — ONLY L_energy, no auxiliary losses.
      "cbr_only"     — ONLY L_cbr (chain break prediction).
    """
    device = energy_pred.device

    L_energy = F.l1_loss(energy_pred, batch['energy_curve'])

    L_cbr = torch.tensor(0.0, device=device)
    if cbr_pred.numel() > 0 and batch['cbr_targets'].numel() > 0:
        cbr_targets = batch['cbr_targets'].clamp(0, 1)
        L_cbr = F.binary_cross_entropy_with_logits(cbr_pred, cbr_targets)

    L_rms = F.l1_loss(rms_pred, batch['rms_targets'])

    if loss_mode == "auxiliary_only":
        total = L_cbr + 0.5 * L_rms
    elif loss_mode == "energy_only":
        total = L_energy
    elif loss_mode == "cbr_only":
        total = L_cbr
    elif loss_mode == "scalar_ensemble":
        # For ScalarEnsembleHEC / ScalarDropoutHEC: use scalar loss on r*
        # model._r_preds is set during forward pass
        total = L_energy  # fallback to curve loss (will be overridden in training loop)
    else:  # standard
        total = L_energy + beta_cbr * L_cbr + beta_rms * L_rms

    return total, {
        'total': total.item(),
        'energy': L_energy.item(),
        'cbr': L_cbr.item(),
        'rms': L_rms.item(),
    }


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def evaluate_model(model, loader, device):
    """Evaluate model with comprehensive metrics.

    Returns dict with:
      Primary (chain-strength quality):
        - delta_e_5pct/2pct/1pct: fraction with δE ≤ 5%/2%/1%
        - gap_pct: mean energy gap % to oracle
        - mae_r: MAE of predicted r*
        - curve_mae: L1 on full K-point curve

      Chain-break metrics:
        - cbr_at_pred: chain-break rate at predicted r*
        - cbr_at_oracle: chain-break rate at oracle r*
        - cbr_reduction: how much predicted r* reduces breaks vs worst

      Quality metrics:
        - gap_best_energy: mean |E(r_pred) - E(r_oracle)| absolute
        - top3_accuracy: is oracle argmin in model's top-3 lowest?
        - rank_correlation: Spearman rho of predicted vs true r*
    """
    grid_tensor = torch.tensor(GRID, dtype=torch.float32, device=device)
    model.eval()

    all_pred_r, all_true_r = [], []
    all_energy_gap, all_curve_mae = [], []
    all_gap_abs = []
    all_cbr_pred, all_cbr_oracle = [], []
    top3_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            energy_pred, _, _ = model(batch)
            B = energy_pred.size(0)
            r_pred = parabolic_argmin(energy_pred, grid_tensor)
            r_true = batch['r_star']

            all_pred_r.extend(r_pred.cpu().tolist())
            all_true_r.extend(r_true.cpu().tolist())

            curve_mae = F.l1_loss(energy_pred, batch['energy_curve'], reduction='none')
            all_curve_mae.extend(curve_mae.mean(dim=1).cpu().tolist())

            for i in range(B):
                ec_true = batch['energy_curve'][i].cpu().numpy()
                ec_pred = energy_pred[i].cpu().numpy()
                true_idx = int(np.argmin(ec_true))
                pred_idx = int(np.argmin(ec_pred))
                e_true = ec_true[true_idx]
                e_at_pred = ec_true[pred_idx]

                # δE gap (relative)
                gap = abs(e_at_pred - e_true) / max(abs(e_true), 1e-8)
                all_energy_gap.append(gap)

                # Absolute energy gap
                all_gap_abs.append(abs(e_at_pred - e_true))

                # Top-3 accuracy
                top3_pred = np.argsort(ec_pred)[:3]
                if true_idx in top3_pred:
                    top3_correct += 1

                # Chain-break rate at predicted vs oracle r*
                if 'break_curve' in batch and batch['break_curve'].numel() > 0:
                    bc = batch['break_curve'][i].cpu().numpy()
                    all_cbr_pred.append(float(bc[pred_idx]))
                    all_cbr_oracle.append(float(bc[true_idx]))

                n_total += 1

    pred_r = np.array(all_pred_r)
    true_r = np.array(all_true_r)

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(pred_r, true_r)
        spearman = float(rho) if not np.isnan(rho) else 0.0
    except ImportError:
        spearman = 0.0

    metrics = {
        # Primary: chain-strength quality
        'mae_r': float(np.mean(np.abs(pred_r - true_r))),
        'curve_mae': float(np.mean(all_curve_mae)),
        'delta_e_mean': float(np.mean(all_energy_gap)),
        'delta_e_5pct': float(np.mean([1 if g <= 0.05 else 0 for g in all_energy_gap])),
        'delta_e_2pct': float(np.mean([1 if g <= 0.02 else 0 for g in all_energy_gap])),
        'delta_e_1pct': float(np.mean([1 if g <= 0.01 else 0 for g in all_energy_gap])),
        'gap_pct': float(np.mean(all_energy_gap) * 100),
        'gap_best_energy': float(np.mean(all_gap_abs)),

        # Chain-break metrics
        'cbr_at_pred': float(np.mean(all_cbr_pred)) if all_cbr_pred else None,
        'cbr_at_oracle': float(np.mean(all_cbr_oracle)) if all_cbr_oracle else None,

        # Quality metrics
        'top3_accuracy': top3_correct / max(n_total, 1),
        'spearman_rho': spearman,
        'n_instances': n_total,
    }

    return metrics


# ---------------------------------------------------------------
# Optimizer & Scheduler builders
# ---------------------------------------------------------------

def build_optimizer(model, cfg: ExperimentConfig):
    tc = cfg.train
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


def build_scheduler(optimizer, cfg: ExperimentConfig):
    tc = cfg.train
    if tc.scheduler == "cosine_warmup":
        def lr_lambda(epoch):
            if epoch < tc.warmup_epochs:
                return (epoch + 1) / tc.warmup_epochs
            progress = (epoch - tc.warmup_epochs) / max(tc.epochs - tc.warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif tc.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=tc.step_size, gamma=tc.step_gamma)
    elif tc.scheduler == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)
    raise ValueError(f"Unknown scheduler: {tc.scheduler}")


# ---------------------------------------------------------------
# Training engine
# ---------------------------------------------------------------

class TrainingEngine:
    """Unified training engine for any registered model."""

    def __init__(self, cfg: ExperimentConfig, callback: Optional[Callable] = None):
        self.cfg = cfg
        self.callback = callback  # fn(event_type, data_dict) for dashboard updates
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _notify(self, event: str, data: dict):
        if self.callback:
            self.callback(event, data)

    def train_one_seed(self, seed: int) -> dict:
        """Train model for one seed. Returns results dict."""
        cfg = self.cfg
        tc = cfg.train
        device = self.device

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data
        dc = cfg.data
        train_path = os.path.join(dc.data_dir, dc.train_file)
        val_path = os.path.join(dc.data_dir, dc.val_file)
        test_path = os.path.join(dc.data_dir, dc.test_file)
        train_loader, val_loader, test_loader = make_dataloaders(
            train_path, val_path, test_path,
            batch_size=tc.batch_size, num_workers=dc.num_workers)

        # Model
        model = build_model(cfg.model).to(device)
        info = model_info(model)

        self._notify("train_start", {
            "name": cfg.name, "seed": seed, "arch": cfg.model.arch,
            "n_params": info["n_params"],
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
        })

        # Optimizer & scheduler
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        best_val_mae = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state = None
        history = []
        t0 = time.time()

        for epoch in range(tc.epochs):
            model.train()
            epoch_losses = []

            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                energy_pred, cbr_pred, rms_pred = model(batch)

                # Scalar ensemble models use their own loss
                if tc.loss_mode == "scalar_ensemble" and hasattr(model, 'compute_scalar_loss'):
                    L_scalar = model.compute_scalar_loss(batch)
                    L_cbr = torch.tensor(0.0, device=device)
                    if cbr_pred.numel() > 0 and batch['cbr_targets'].numel() > 0:
                        L_cbr = F.binary_cross_entropy_with_logits(
                            cbr_pred, batch['cbr_targets'].clamp(0, 1))
                    loss = L_scalar + tc.beta_cbr * L_cbr
                    loss_dict = {'total': loss.item(), 'energy': L_scalar.item(),
                                 'cbr': L_cbr.item(), 'rms': 0.0}
                else:
                    loss, loss_dict = compute_loss(
                        energy_pred, cbr_pred, rms_pred, batch,
                        beta_cbr=tc.beta_cbr, beta_rms=tc.beta_rms,
                        loss_mode=tc.loss_mode)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()
                epoch_losses.append(loss_dict)

            scheduler.step()

            # Validation
            val_metrics = evaluate_model(model, val_loader, device)
            train_loss = float(np.mean([d['energy'] for d in epoch_losses]))

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mae_r': val_metrics['mae_r'],
                'val_delta_e_5pct': val_metrics['delta_e_5pct'],
                'lr': optimizer.param_groups[0]['lr'],
            })

            self._notify("epoch_end", {
                "name": cfg.name, "seed": seed, "epoch": epoch,
                "epochs": tc.epochs, "train_loss": train_loss,
                "val_mae": val_metrics['mae_r'],
                "val_de5": val_metrics['delta_e_5pct'],
                "best_mae": best_val_mae,
                "lr": optimizer.param_groups[0]['lr'],
                "elapsed": time.time() - t0,
            })

            if val_metrics['mae_r'] < best_val_mae:
                best_val_mae = val_metrics['mae_r']
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= tc.patience:
                break

        # Test evaluation
        model.load_state_dict(best_state)
        model.to(device)
        test_metrics = evaluate_model(model, test_loader, device)
        elapsed = time.time() - t0

        # Save
        output_dir = os.path.join(cfg.output_dir, cfg.name)
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{cfg.model.arch}_seed{seed}.pt')
        torch.save(best_state, model_path)

        results = {
            'experiment': cfg.name,
            'arch': cfg.model.arch,
            'seed': seed,
            'n_params': info['n_params'],
            'best_val_mae_r': best_val_mae,
            'best_epoch': best_epoch + 1,
            'elapsed_sec': elapsed,
            'test_metrics': test_metrics,
            'config': {
                'hidden_dim': cfg.model.hidden_dim,
                'num_layers': cfg.model.num_layers,
                'num_layers_stage3': cfg.model.num_layers_stage3,
                'lr': tc.lr, 'batch_size': tc.batch_size,
                'epochs': tc.epochs, 'patience': tc.patience,
                'optimizer': tc.optimizer, 'scheduler': tc.scheduler,
            },
        }

        results_path = os.path.join(output_dir, f'{cfg.model.arch}_seed{seed}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        self._notify("train_done", {
            "name": cfg.name, "seed": seed,
            "test_metrics": test_metrics,
            "elapsed": elapsed,
            "best_epoch": best_epoch + 1,
        })

        return results

    def run(self) -> List[dict]:
        """Train across all seeds. Returns list of results."""
        all_results = []
        for seed in self.cfg.train.seeds:
            results = self.train_one_seed(seed)
            all_results.append(results)

        # Save aggregated results
        if len(all_results) > 1:
            output_dir = os.path.join(self.cfg.output_dir, self.cfg.name)
            os.makedirs(output_dir, exist_ok=True)
            agg_path = os.path.join(output_dir, f'{self.cfg.model.arch}_aggregated.json')
            with open(agg_path, 'w') as f:
                json.dump(all_results, f, indent=2)

        return all_results


# ---------------------------------------------------------------
# Knowledge Distillation Engine
# ---------------------------------------------------------------

class DistillEngine:
    """Train a student model with knowledge distillation from a teacher.

    Usage:
        teacher = load_trained_hec_gnn()
        student_cfg = ExperimentConfig(model=ModelConfig(arch="compact_hec"))
        engine = DistillEngine(student_cfg, teacher, alpha=0.5, temperature=2.0)
        results = engine.run()
    """

    def __init__(self, student_cfg: ExperimentConfig, teacher: torch.nn.Module,
                 alpha: float = 0.5, temperature: float = 2.0,
                 callback=None):
        self.cfg = student_cfg
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.callback = callback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _notify(self, event, data):
        if self.callback:
            self.callback(event, data)

    def train_one_seed(self, seed: int) -> dict:
        from hecgnn_trainer.models.compact import DistillationLoss

        cfg = self.cfg
        tc = cfg.train
        device = self.device

        torch.manual_seed(seed)
        np.random.seed(seed)

        dc = cfg.data
        train_loader, val_loader, test_loader = make_dataloaders(
            os.path.join(dc.data_dir, dc.train_file),
            os.path.join(dc.data_dir, dc.val_file),
            os.path.join(dc.data_dir, dc.test_file),
            batch_size=tc.batch_size)

        student = build_model(cfg.model).to(device)
        self.teacher.to(device).eval()
        info = model_info(student)
        teacher_info = model_info(self.teacher)

        self._notify("train_start", {
            "name": f"distill_{cfg.name}", "seed": seed,
            "arch": cfg.model.arch,
            "n_params": info["n_params"],
            "teacher_params": teacher_info["n_params"],
            "train_size": len(train_loader.dataset),
        })

        optimizer = build_optimizer(student, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        distill_loss_fn = DistillationLoss(self.alpha, self.temperature)

        best_val_mae = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state = None
        t0 = time.time()

        for epoch in range(tc.epochs):
            student.train()
            epoch_losses = []

            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Student forward
                s_energy, s_cbr, s_rms = student(batch)

                # Teacher forward (no grad)
                with torch.no_grad():
                    t_energy, _, _ = self.teacher(batch)

                # Distillation loss on energy curve
                loss_distill, loss_parts = distill_loss_fn(s_energy, t_energy, batch['energy_curve'])

                # Auxiliary losses (student only, ground truth)
                L_cbr = torch.tensor(0.0, device=device)
                if s_cbr.numel() > 0 and batch['cbr_targets'].numel() > 0:
                    L_cbr = F.binary_cross_entropy_with_logits(
                        s_cbr, batch['cbr_targets'].clamp(0, 1))
                L_rms = F.l1_loss(s_rms, batch['rms_targets'])

                loss = loss_distill + tc.beta_cbr * L_cbr + tc.beta_rms * L_rms

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), tc.grad_clip)
                optimizer.step()
                epoch_losses.append({
                    'total': loss.item(), 'energy': loss_parts['task'],
                    'distill': loss_parts['distill'],
                })

            scheduler.step()
            val_metrics = evaluate_model(student, val_loader, device)
            train_loss = float(np.mean([d['energy'] for d in epoch_losses]))

            self._notify("epoch_end", {
                "name": f"distill_{cfg.name}", "seed": seed,
                "epoch": epoch, "epochs": tc.epochs,
                "train_loss": train_loss,
                "val_mae": val_metrics['mae_r'],
                "best_mae": best_val_mae,
                "elapsed": time.time() - t0,
            })

            if val_metrics['mae_r'] < best_val_mae:
                best_val_mae = val_metrics['mae_r']
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= tc.patience:
                break

        student.load_state_dict(best_state)
        student.to(device)
        test_metrics = evaluate_model(student, test_loader, device)
        elapsed = time.time() - t0

        output_dir = os.path.join(cfg.output_dir, f"distill_{cfg.name}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(best_state, os.path.join(output_dir, f'{cfg.model.arch}_seed{seed}.pt'))

        results = {
            'experiment': f"distill_{cfg.name}",
            'arch': cfg.model.arch, 'seed': seed,
            'n_params_student': info['n_params'],
            'n_params_teacher': teacher_info['n_params'],
            'compression_ratio': teacher_info['n_params'] / max(info['n_params'], 1),
            'best_val_mae_r': best_val_mae,
            'best_epoch': best_epoch + 1,
            'elapsed_sec': elapsed,
            'test_metrics': test_metrics,
            'distill_config': {
                'alpha': self.alpha, 'temperature': self.temperature,
            },
        }

        with open(os.path.join(output_dir, f'results_seed{seed}.json'), 'w') as f:
            json.dump(results, f, indent=2)

        self._notify("train_done", {
            "name": f"distill_{cfg.name}", "seed": seed,
            "test_metrics": test_metrics, "elapsed": elapsed,
        })

        return results

    def run(self) -> list:
        all_results = []
        for seed in self.cfg.train.seeds:
            all_results.append(self.train_one_seed(seed))
        return all_results
