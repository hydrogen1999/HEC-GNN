#!/usr/bin/env python3
"""
ablation.py -- HEC-GNN component ablation study.

Ablation variants:
  1. Full HEC-GNN (all components, beta_cbr=0.1, beta_rms=0.05)
  2. No Auxiliary Losses (same HECGNN architecture, beta_cbr=0, beta_rms=0)
  3. FlatGNN (no hierarchical pooling, 6-layer GIN-E, ~261K params)

For each variant, train with 3 seeds on multi_topo SA data, evaluate on test set.
Reports: MAE(r*), delta_E<=5%, Spearman rho, Top-3 accuracy, param count.

Usage:
  python ablation.py --data-dir datasets_v3_full
  python ablation.py --data-dir datasets_v3_full --seeds 42 123 7 --epochs 200
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import (
    ChainStrengthDataset, collate_batch, make_dataloaders, GRID, K,
)
from src.models.hec_gnn import HECGNN, ModelConfig, parabolic_argmin
from src.models.baselines import build_flat_gnn
from train import compute_loss


# ---------------------------------------------------------------------------
# Extended evaluation: MAE, delta_E<=5%, Spearman, Top-3 accuracy
# ---------------------------------------------------------------------------

def evaluate_ablation(model, loader, device):
    """Evaluate model with all ablation metrics.

    Returns dict with:
      mae_r, delta_e_5pct, spearman_rho, top3_accuracy, n_instances
    """
    grid_tensor = torch.tensor(GRID, dtype=torch.float32, device=device)
    model.eval()

    all_pred_r = []
    all_true_r = []
    all_energy_gap = []
    top3_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            energy_pred, _, _ = model(batch)
            B = energy_pred.size(0)

            # Predicted r* via parabolic interpolation
            r_pred = parabolic_argmin(energy_pred, grid_tensor)
            r_true = batch['r_star']

            all_pred_r.extend(r_pred.cpu().tolist())
            all_true_r.extend(r_true.cpu().tolist())

            for i in range(B):
                ec_true = batch['energy_curve'][i].cpu().numpy()
                ec_pred = energy_pred[i].cpu().numpy()

                # delta_E <= 5%
                true_min_idx = int(np.argmin(ec_true))
                pred_min_idx = int(np.argmin(ec_pred))
                e_true = ec_true[true_min_idx]
                e_at_pred = ec_true[pred_min_idx]
                gap = abs(e_at_pred - e_true) / max(abs(e_true), 1e-8)
                all_energy_gap.append(gap)

                # Top-3 accuracy: is the true optimal grid index among
                # the 3 indices with lowest predicted energy?
                top3_pred_indices = np.argsort(ec_pred)[:3]
                if true_min_idx in top3_pred_indices:
                    top3_correct += 1
                n_total += 1

    pred_r = np.array(all_pred_r)
    true_r = np.array(all_true_r)

    mae_r = float(np.mean(np.abs(pred_r - true_r)))
    delta_e_5pct = float(np.mean([1.0 if g <= 0.05 else 0.0 for g in all_energy_gap]))
    top3_acc = top3_correct / max(n_total, 1)

    # Spearman rho
    try:
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(pred_r, true_r)
        spearman_rho = float(rho) if not np.isnan(rho) else 0.0
    except ImportError:
        # Fallback: manual Spearman with average rank for ties
        def _rankdata(x):
            arr = np.asarray(x)
            sorter = np.argsort(arr)
            ranks = np.empty(len(arr), dtype=float)
            ranks[sorter] = np.arange(1, len(arr) + 1, dtype=float)
            # Average rank for ties
            for val in np.unique(arr):
                mask = arr == val
                ranks[mask] = ranks[mask].mean()
            return ranks
        rp = _rankdata(pred_r)
        rt = _rankdata(true_r)
        n = len(rp)
        d2 = np.sum((rp - rt) ** 2)
        spearman_rho = 1.0 - 6.0 * d2 / (n * (n**2 - 1)) if n > 1 else 0.0

    return {
        'mae_r': mae_r,
        'delta_e_5pct': delta_e_5pct,
        'spearman_rho': spearman_rho,
        'top3_accuracy': top3_acc,
        'n_instances': n_total,
    }


# ---------------------------------------------------------------------------
# Training loop (single seed, single variant)
# ---------------------------------------------------------------------------

def train_variant(variant_name, model, data_dir, seed, device,
                  beta_cbr=0.1, beta_rms=0.05,
                  epochs=200, batch_size=32, lr=5e-4, weight_decay=1e-4,
                  patience=30, grad_clip=1.0, warmup_epochs=5):
    """Train one variant for one seed. Returns test metrics dict."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_path = os.path.join(data_dir, 'multi_topo_train.pkl')
    val_path = os.path.join(data_dir, 'multi_topo_val.pkl')
    test_path = os.path.join(data_dir, 'multi_topo_test.pkl')

    train_loader, val_loader, test_loader = make_dataloaders(
        train_path, val_path, test_path, batch_size=batch_size)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

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
        val_metrics = evaluate_ablation(model, val_loader, device)
        train_loss = float(np.mean([d['energy'] for d in epoch_losses]))

        if val_metrics['mae_r'] < best_val_mae:
            best_val_mae = val_metrics['mae_r']
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"    [{variant_name} seed={seed}] Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_MAE={val_metrics['mae_r']:.4f}  "
                  f"best={best_val_mae:.4f}@{best_epoch+1}  "
                  f"({elapsed:.0f}s)")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate_ablation(model, test_loader, device)
    elapsed = time.time() - t0

    test_metrics['n_params'] = n_params
    test_metrics['best_epoch'] = best_epoch + 1
    test_metrics['elapsed_sec'] = elapsed

    return test_metrics, best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='HEC-GNN component ablation study')
    parser.add_argument('--data-dir', default='datasets_v3_full',
                        help='Directory with multi_topo_{train,val,test}.pkl')
    parser.add_argument('--output-dir', default='results/ablation',
                        help='Output directory for results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7],
                        help='Random seeds')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', default=None,
                        help='Device (default: auto-detect)')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Define ablation variants
    variants = {
        'Full_HEC-GNN': {
            'build_fn': lambda: HECGNN(ModelConfig()),
            'beta_cbr': 0.1,
            'beta_rms': 0.05,
            'description': 'Full HEC-GNN with all components and auxiliary losses',
        },
        'No_Aux_Losses': {
            'build_fn': lambda: HECGNN(ModelConfig()),
            'beta_cbr': 0.0,
            'beta_rms': 0.0,
            'description': 'HEC-GNN architecture, auxiliary loss weights set to 0',
        },
        'FlatGNN': {
            'build_fn': lambda: build_flat_gnn(),
            'beta_cbr': 0.0,
            'beta_rms': 0.0,
            'description': 'Single-level 6-layer GIN-E, no hierarchical pooling',
        },
    }

    all_results = {}

    for vname, vconfig in variants.items():
        print(f"\n{'='*60}")
        print(f"Ablation variant: {vname}")
        print(f"  {vconfig['description']}")
        print(f"{'='*60}")

        seed_results = []
        for seed in args.seeds:
            print(f"\n  --- Seed {seed} ---")
            model = vconfig['build_fn']()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")

            metrics, best_state = train_variant(
                variant_name=vname,
                model=model,
                data_dir=args.data_dir,
                seed=seed,
                device=device,
                beta_cbr=vconfig['beta_cbr'],
                beta_rms=vconfig['beta_rms'],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
            )

            print(f"\n  Test results (seed={seed}):")
            print(f"    MAE(r*)      = {metrics['mae_r']:.4f}")
            print(f"    delta_E<=5%  = {metrics['delta_e_5pct']:.1%}")
            print(f"    Spearman rho = {metrics['spearman_rho']:.4f}")
            print(f"    Top-3 acc    = {metrics['top3_accuracy']:.1%}")
            print(f"    Params       = {metrics['n_params']:,}")

            metrics['seed'] = seed
            seed_results.append(metrics)

            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir,
                                      f'{vname}_seed{seed}.pt')
            torch.save(best_state, ckpt_path)

        # Aggregate across seeds
        agg = {
            'variant': vname,
            'description': vconfig['description'],
            'n_params': seed_results[0]['n_params'],
            'n_seeds': len(seed_results),
            'per_seed': seed_results,
            'mae_r_mean': float(np.mean([r['mae_r'] for r in seed_results])),
            'mae_r_std': float(np.std([r['mae_r'] for r in seed_results])),
            'delta_e_5pct_mean': float(np.mean([r['delta_e_5pct'] for r in seed_results])),
            'delta_e_5pct_std': float(np.std([r['delta_e_5pct'] for r in seed_results])),
            'spearman_rho_mean': float(np.mean([r['spearman_rho'] for r in seed_results])),
            'spearman_rho_std': float(np.std([r['spearman_rho'] for r in seed_results])),
            'top3_accuracy_mean': float(np.mean([r['top3_accuracy'] for r in seed_results])),
            'top3_accuracy_std': float(np.std([r['top3_accuracy'] for r in seed_results])),
        }
        all_results[vname] = agg

        print(f"\n  Aggregated ({len(seed_results)} seeds):")
        print(f"    MAE(r*)      = {agg['mae_r_mean']:.4f} +/- {agg['mae_r_std']:.4f}")
        print(f"    delta_E<=5%  = {agg['delta_e_5pct_mean']:.1%} +/- {agg['delta_e_5pct_std']:.1%}")
        print(f"    Spearman rho = {agg['spearman_rho_mean']:.4f} +/- {agg['spearman_rho_std']:.4f}")
        print(f"    Top-3 acc    = {agg['top3_accuracy_mean']:.1%} +/- {agg['top3_accuracy_std']:.1%}")
        print(f"    Params       = {agg['n_params']:,}")

    # Save all results
    results_path = os.path.join(args.output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("Table 4: Ablation Study Summary")
    print(f"{'='*80}")
    print(f"{'Variant':<20s} {'Params':>8s} {'MAE(r*)':>12s} "
          f"{'dE<=5%':>12s} {'Spearman':>12s} {'Top-3':>12s}")
    print(f"{'-'*80}")
    for vname, agg in all_results.items():
        print(f"{vname:<20s} {agg['n_params']:>8,d} "
              f"{agg['mae_r_mean']:>5.4f}+/-{agg['mae_r_std']:.4f} "
              f"{agg['delta_e_5pct_mean']:>5.1%}+/-{agg['delta_e_5pct_std']:.1%} "
              f"{agg['spearman_rho_mean']:>5.4f}+/-{agg['spearman_rho_std']:.4f} "
              f"{agg['top3_accuracy_mean']:>5.1%}+/-{agg['top3_accuracy_std']:.1%}")


if __name__ == '__main__':
    main()
