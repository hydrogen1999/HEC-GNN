#!/usr/bin/env python3
"""
train.py -- Training script for HEC-GNN and FlatGNN.

Trains on energy curve prediction with auxiliary CBR + RMS losses.
Matches V3 paper Section 5: AdamW, lr=5e-4, wd=1e-4, batch=32,
cosine annealing 200 epochs, patience 30, gradient clip 1.0.

Usage:
  # Train HEC-GNN on multi-topology SA data
  python train.py --model hec_gnn --data-dir datasets_v3_full

  # Train FlatGNN baseline
  python train.py --model flat_gnn --data-dir datasets_v3_full

  # Train on OOD data
  python train.py --model hec_gnn --data-dir datasets_v3_full --benchmark ood

  # Multiple seeds
  python train.py --model hec_gnn --seeds 42 123 7
"""

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import (
    ChainStrengthDataset, collate_batch, make_dataloaders, GRID, K
)
from src.models.hec_gnn import HECGNN, ModelConfig, parabolic_argmin
from src.models.baselines import build_flat_gnn


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_loss(energy_pred, cbr_pred, rms_pred, batch,
                 beta_cbr=0.1, beta_rms=0.05):
    """Compute total loss = L_energy + beta_cbr * L_cbr + beta_rms * L_rms."""

    # Primary: L1 loss on energy curve
    L_energy = F.l1_loss(energy_pred, batch['energy_curve'])

    # Auxiliary: chain-break BCE
    L_cbr = torch.tensor(0.0, device=energy_pred.device)
    if cbr_pred.numel() > 0 and batch['cbr_targets'].numel() > 0:
        # Clamp targets to [0, 1]
        cbr_targets = batch['cbr_targets'].clamp(0, 1)
        L_cbr = F.binary_cross_entropy_with_logits(cbr_pred, cbr_targets)

    # Auxiliary: RMS L1
    L_rms = F.l1_loss(rms_pred, batch['rms_targets'])

    total = L_energy + beta_cbr * L_cbr + beta_rms * L_rms

    return total, {
        'total': total.item(),
        'energy': L_energy.item(),
        'cbr': L_cbr.item(),
        'rms': L_rms.item(),
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, grid_tensor=None):
    """Evaluate model on a data loader. Returns metrics dict.
    Delegates to evaluate.evaluate_model_on_loader().
    """
    from evaluate import evaluate_model_on_loader
    return evaluate_model_on_loader(model, loader, device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_seed(model_type, data_dir, benchmark, seed, output_dir,
                   epochs=200, batch_size=32, lr=5e-4, weight_decay=1e-4,
                   beta_cbr=0.1, beta_rms=0.05, patience=30, grad_clip=1.0,
                   warmup_epochs=5):
    """Train a single model with a single seed."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    if benchmark == 'multi_topo':
        train_path = os.path.join(data_dir, 'multi_topo_train.pkl')
        val_path = os.path.join(data_dir, 'multi_topo_val.pkl')
        test_path = os.path.join(data_dir, 'multi_topo_test.pkl')
    elif benchmark == 'ood':
        train_path = os.path.join(data_dir, 'ood_train.pkl')
        val_path = os.path.join(data_dir, 'ood_val.pkl')
        test_path = os.path.join(data_dir, 'ood_test.pkl')
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    train_loader, val_loader, test_loader = make_dataloaders(
        train_path, val_path, test_path, batch_size=batch_size)

    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Build model
    if model_type == 'hec_gnn':
        model = HECGNN(ModelConfig()).to(device)
    elif model_type == 'flat_gnn':
        model = build_flat_gnn().to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}, params: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler: linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    grid_tensor = torch.tensor(GRID, dtype=torch.float32, device=device)

    # Training loop
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None
    history = []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            energy_pred, cbr_pred, rms_pred = model(batch)
            loss, loss_dict = compute_loss(
                energy_pred, cbr_pred, rms_pred, batch,
                beta_cbr=beta_cbr, beta_rms=beta_rms)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_losses.append(loss_dict)

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device, grid_tensor)
        train_loss = float(np.mean([d['energy'] for d in epoch_losses]))

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mae_r': val_metrics['mae_r'],
            'val_delta_e_5pct': val_metrics['delta_e_5pct'],
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Early stopping on val MAE(r*)
        if val_metrics['mae_r'] < best_val_mae:
            best_val_mae = val_metrics['mae_r']
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_MAE={val_metrics['mae_r']:.4f}  "
                  f"val_δE≤5%={val_metrics['delta_e_5pct']:.1%}  "
                  f"best={best_val_mae:.4f}@{best_epoch+1}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                  f"({elapsed:.0f}s)")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device, grid_tensor)
    elapsed = time.time() - t0

    print(f"\n=== Test Results (seed={seed}) ===")
    print(f"  MAE(r*)     = {test_metrics['mae_r']:.4f}")
    print(f"  δE ≤ 5%     = {test_metrics['delta_e_5pct']:.1%}")
    print(f"  δE ≤ 2%     = {test_metrics['delta_e_2pct']:.1%}")
    print(f"  Curve MAE   = {test_metrics['curve_mae']:.4f}")
    print(f"  Best epoch  = {best_epoch + 1}")
    print(f"  Elapsed     = {elapsed:.0f}s")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_type}_seed{seed}.pt')
    torch.save(best_state, model_path)

    results = {
        'model_type': model_type,
        'benchmark': benchmark,
        'seed': seed,
        'n_params': n_params,
        'best_val_mae_r': best_val_mae,
        'best_epoch': best_epoch + 1,
        'elapsed_sec': elapsed,
        'test_metrics': test_metrics,
        'config': {
            'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
            'weight_decay': weight_decay, 'beta_cbr': beta_cbr,
            'beta_rms': beta_rms, 'patience': patience,
            'grad_clip': grad_clip, 'warmup_epochs': warmup_epochs,
        },
    }

    results_path = os.path.join(output_dir, f'{model_type}_seed{seed}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train HEC-GNN / FlatGNN')
    parser.add_argument('--model', choices=['hec_gnn', 'flat_gnn'], default='hec_gnn')
    parser.add_argument('--data-dir', default='datasets_v3_full')
    parser.add_argument('--benchmark', choices=['multi_topo', 'ood'], default='multi_topo')
    parser.add_argument('--output-dir', default='results_v3')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Training {args.model} | {args.benchmark} | seed={seed}")
        print(f"{'='*60}")
        results = train_one_seed(
            model_type=args.model,
            data_dir=args.data_dir,
            benchmark=args.benchmark,
            seed=seed,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        all_results.append(results)

    # Aggregate across seeds
    if len(all_results) > 1:
        mae_rs = [r['test_metrics']['mae_r'] for r in all_results]
        de5s = [r['test_metrics']['delta_e_5pct'] for r in all_results]
        print(f"\n{'='*60}")
        print(f"Aggregated ({len(all_results)} seeds):")
        print(f"  MAE(r*) = {np.mean(mae_rs):.4f} ± {np.std(mae_rs):.4f}")
        print(f"  δE≤5%   = {np.mean(de5s):.1%} ± {np.std(de5s):.1%}")

    # Save aggregated
    agg_path = os.path.join(args.output_dir, f'{args.model}_{args.benchmark}_all.json')
    with open(agg_path, 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
