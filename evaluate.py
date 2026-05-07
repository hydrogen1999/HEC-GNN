#!/usr/bin/env python3
"""
evaluate.py -- Run all baselines + trained models, produce paper results.

Usage:
  python evaluate.py --benchmark multi_topo --data-dir data/sa_multi_topo --model-dir results/hec_gnn_multi_topo
  python evaluate.py --benchmark ood --data-dir data/sa_ood --model-dir results/hec_gnn_ood
  python evaluate.py --benchmark cross_topo --data-dir data/sa_multi_topo --model-dir results/hec_gnn_multi_topo
  python evaluate.py --benchmark surrogate --data-dir data/sa_multi_topo --boltzmann-dir data/boltzmann_multi_topo
  python evaluate.py --benchmark all --data-dir data/sa_multi_topo --model-dir results/hec_gnn_multi_topo
"""

import argparse
import glob
import json
import math
import os
import pickle

import numpy as np

from src.models.baselines import (
    UTC, ScaledHeuristic, MeanBaseline, LinearRegBaseline,
    FewShotBaseline, BOBaseline, OracleSA,
    evaluate_baseline, run_all_baselines, GRID, K,
)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Model evaluation (shared by train.py and evaluate.py)
# ---------------------------------------------------------------------------

def evaluate_model_on_loader(model, loader, device):
    """Evaluate a trained torch model on a DataLoader. Returns metrics dict."""
    import torch
    from src.models.hec_gnn import parabolic_argmin

    grid_tensor = torch.tensor(GRID, dtype=torch.float32, device=device)
    model.eval()
    all_pred_r = []
    all_true_r = []
    all_energy_gap = []
    all_curve_mae = []
    top3_correct = 0
    n_total = 0
    # Coverage: fraction where pred r* is within 2x of true r*
    coverage_2x = 0

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

            import torch.nn.functional as F
            curve_mae = F.l1_loss(energy_pred, batch['energy_curve'], reduction='none')
            all_curve_mae.extend(curve_mae.mean(dim=1).cpu().tolist())

            for i in range(B):
                ec_true = batch['energy_curve'][i].cpu().numpy()
                ec_pred = energy_pred[i].cpu().numpy()
                true_min_idx = int(np.argmin(ec_true))
                pred_min_idx = int(np.argmin(ec_pred))
                e_true = ec_true[true_min_idx]
                e_at_pred = ec_true[pred_min_idx]
                gap = abs(e_at_pred - e_true) / max(abs(e_true), 1e-8)
                all_energy_gap.append(gap)

                # Top-3: is true argmin among 3 lowest predicted?
                top3_pred = np.argsort(ec_pred)[:3]
                if true_min_idx in top3_pred:
                    top3_correct += 1

                # Coverage ≤2×: is pred r* within [0.5×true, 2×true]?
                rp = r_pred[i].item()
                rt = r_true[i].item()
                if rt > 0 and 0.5 * rt <= rp <= 2.0 * rt:
                    coverage_2x += 1

                n_total += 1

    pred_r = np.array(all_pred_r)
    true_r = np.array(all_true_r)

    # Spearman rho
    try:
        from scipy.stats import spearmanr
        rho, rho_p = spearmanr(pred_r, true_r)
        spearman_rho = float(rho) if not np.isnan(rho) else 0.0
    except ImportError:
        def _rankdata(x):
            arr = np.asarray(x)
            sorter = np.argsort(arr)
            ranks = np.empty(len(arr), dtype=float)
            ranks[sorter] = np.arange(1, len(arr) + 1, dtype=float)
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
        'mae_r': float(np.mean(np.abs(pred_r - true_r))),
        'curve_mae': float(np.mean(all_curve_mae)),
        'delta_e_mean': float(np.mean(all_energy_gap)),
        'delta_e_5pct': float(np.mean([1 if g <= 0.05 else 0 for g in all_energy_gap])),
        'delta_e_2pct': float(np.mean([1 if g <= 0.02 else 0 for g in all_energy_gap])),
        'spearman_rho': spearman_rho,
        'top3_accuracy': top3_correct / max(n_total, 1),
        'coverage_2x': coverage_2x / max(n_total, 1),
        'n_instances': n_total,
    }


def load_and_eval_models(model_dir, test_data, benchmark, batch_size=32):
    """Load all trained model checkpoints and evaluate them."""
    import torch
    from src.data.dataset import ChainStrengthDataset, collate_batch
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_ds = ChainStrengthDataset(test_data)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_batch)

    results = {}

    # Find HEC-GNN checkpoints
    hec_dir = model_dir
    hec_paths = sorted(glob.glob(os.path.join(hec_dir, 'hec_gnn_seed*.pt')))
    if hec_paths:
        from src.models.hec_gnn import HECGNN, ModelConfig
        hec_metrics = []
        for path in hec_paths:
            seed = os.path.basename(path).replace('hec_gnn_seed', '').replace('.pt', '')
            print(f"  Evaluating HEC-GNN seed={seed}...")
            model = HECGNN(ModelConfig())
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.to(device)
            m = evaluate_model_on_loader(model, test_loader, device)
            m['seed'] = seed
            hec_metrics.append(m)
            print(f"    MAE(r*)={m['mae_r']:.4f}, δE≤5%={m['delta_e_5pct']:.1%}")

        results['HEC-GNN'] = {
            'per_seed': hec_metrics,
            'mae_r_mean': float(np.mean([m['mae_r'] for m in hec_metrics])),
            'mae_r_std': float(np.std([m['mae_r'] for m in hec_metrics])),
            'delta_e_5pct_mean': float(np.mean([m['delta_e_5pct'] for m in hec_metrics])),
        }

    # Find FlatGNN checkpoints
    flat_dir = model_dir.replace('hec_gnn', 'flat_gnn')
    flat_paths = sorted(glob.glob(os.path.join(flat_dir, 'flat_gnn_seed*.pt')))
    if flat_paths:
        from src.models.baselines import build_flat_gnn
        flat_metrics = []
        for path in flat_paths:
            seed = os.path.basename(path).replace('flat_gnn_seed', '').replace('.pt', '')
            print(f"  Evaluating FlatGNN seed={seed}...")
            model = build_flat_gnn()
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.to(device)
            m = evaluate_model_on_loader(model, test_loader, device)
            m['seed'] = seed
            flat_metrics.append(m)
            print(f"    MAE(r*)={m['mae_r']:.4f}, δE≤5%={m['delta_e_5pct']:.1%}")

        results['FlatGNN'] = {
            'per_seed': flat_metrics,
            'mae_r_mean': float(np.mean([m['mae_r'] for m in flat_metrics])),
            'mae_r_std': float(np.std([m['mae_r'] for m in flat_metrics])),
            'delta_e_5pct_mean': float(np.mean([m['delta_e_5pct'] for m in flat_metrics])),
        }

    return results


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_multi_topo(data_dir, model_dir, results_dir):
    """Table 1: Multi-topology benchmark."""
    print("=" * 60)
    print("Multi-topology benchmark (Pegasus P4/P8/P16)")
    print("=" * 60)

    train = load_pkl(os.path.join(data_dir, 'multi_topo_train.pkl'))
    test = load_pkl(os.path.join(data_dir, 'multi_topo_test.pkl'))
    print(f"Train: {len(train)}, Test: {len(test)}")

    results = {'baselines': run_all_baselines(train, test)}

    if model_dir and os.path.isdir(model_dir):
        print("\n  --- Trained models ---")
        results['models'] = load_and_eval_models(model_dir, test, 'multi_topo')

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'multi_topo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_cross_topo(data_dir, model_dir, results_dir):
    """Table 2: Cross-topology transfer (Pegasus -> Zephyr Z4)."""
    print("=" * 60)
    print("Cross-topology: Pegasus (train) -> Zephyr Z4 (zero-shot)")
    print("=" * 60)

    train = load_pkl(os.path.join(data_dir, 'multi_topo_train.pkl'))
    z4 = load_pkl(os.path.join(data_dir, 'multi_topo_z4.pkl'))
    print(f"Train: {len(train)}, Z4: {len(z4)}")

    results = {'baselines': run_all_baselines(train, z4)}

    if model_dir and os.path.isdir(model_dir):
        print("\n  --- Trained models on Z4 ---")
        results['models'] = load_and_eval_models(model_dir, z4, 'cross_topo')

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'cross_topo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_ood(data_dir, model_dir, results_dir):
    """Table 3: OOD scaling."""
    print("=" * 60)
    print("OOD scaling benchmark")
    print("=" * 60)

    train = load_pkl(os.path.join(data_dir, 'ood_train.pkl'))
    test = load_pkl(os.path.join(data_dir, 'ood_test.pkl'))
    print(f"Train: {len(train)}, Test: {len(test)}")

    test_by_size = {}
    for inst in test:
        n = inst['n_logical']
        test_by_size.setdefault(n, []).append(inst)

    results = {'overall_baselines': run_all_baselines(train, test)}
    for size in sorted(test_by_size.keys()):
        print(f"\n  --- n={size} ({len(test_by_size[size])} instances) ---")
        results[f'n={size}_baselines'] = run_all_baselines(train, test_by_size[size])

    if model_dir and os.path.isdir(model_dir):
        print("\n  --- Trained models (overall) ---")
        results['overall_models'] = load_and_eval_models(model_dir, test, 'ood')
        for size in sorted(test_by_size.keys()):
            print(f"\n  --- Trained models n={size} ---")
            results[f'n={size}_models'] = load_and_eval_models(
                model_dir, test_by_size[size], 'ood')

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'ood_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_surrogate_disagreement(sa_dir, boltzmann_dir, results_dir):
    """Surrogate disagreement: SA vs Boltzmann optimal r*."""
    print("=" * 60)
    print("Surrogate disagreement analysis")
    print("=" * 60)

    sa_data = load_pkl(os.path.join(sa_dir, 'multi_topo_train.pkl'))
    boltz_data = load_pkl(os.path.join(boltzmann_dir, 'multi_topo_train.pkl'))

    sa_rs = [GRID[np.argmin(inst['energy_curve'])] for inst in sa_data]
    boltz_rs = [GRID[np.argmin(inst['energy_curve'])] for inst in boltz_data]

    from scipy.stats import spearmanr
    rho, p = spearmanr(sa_rs, boltz_rs)

    results = {
        'n_instances': len(sa_data),
        'sa_r_star_mean': float(np.mean(sa_rs)),
        'boltzmann_r_star_mean': float(np.mean(boltz_rs)),
        'spearman_rho': float(rho),
        'spearman_p': float(p),
    }
    print(f"  Spearman rho: {rho:.4f} (p={p:.4e})")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'surrogate_disagreement.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate all methods')
    parser.add_argument('--data-dir', default='data/sa_multi_topo')
    parser.add_argument('--model-dir', default='results/hec_gnn_multi_topo',
                        help='Directory with trained model checkpoints')
    parser.add_argument('--boltzmann-dir', default='data/boltzmann_multi_topo')
    parser.add_argument('--results-dir', default='results/eval')
    parser.add_argument('--benchmark',
                        choices=['multi_topo', 'cross_topo', 'ood', 'surrogate', 'all'],
                        default='all')
    args = parser.parse_args()

    if args.benchmark in ('multi_topo', 'all'):
        run_multi_topo(args.data_dir, args.model_dir, args.results_dir)

    if args.benchmark in ('cross_topo', 'all'):
        run_cross_topo(args.data_dir, args.model_dir, args.results_dir)

    if args.benchmark in ('ood', 'all'):
        ood_data = args.data_dir.replace('multi_topo', 'ood').replace('sa_multi_topo', 'sa_ood')
        ood_model = args.model_dir.replace('multi_topo', 'ood')
        run_ood(ood_data, ood_model, args.results_dir)

    if args.benchmark in ('surrogate', 'all'):
        if os.path.exists(os.path.join(args.boltzmann_dir, 'multi_topo_train.pkl')):
            run_surrogate_disagreement(args.data_dir, args.boltzmann_dir, args.results_dir)
        else:
            print("Skipping surrogate disagreement (Boltzmann data not found)")


if __name__ == '__main__':
    main()
