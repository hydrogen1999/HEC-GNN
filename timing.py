#!/usr/bin/env python3
"""
timing.py -- Benchmark inference timing for all methods.

Neural models (HEC-GNN, FlatGNN):
  - Load trained model, run forward pass on 100 test instances
  - 10 warmup + 100 timed runs, report mean and std

Baselines (UTC, Scaled, Mean, LinearReg, Few-Shot-K, BO-10, Oracle):
  - Run predict_r() on 100 test instances, measure time

Reports per-method timing in milliseconds with speedup vs Oracle.
Saves to results/timing/timing_results.json.

Usage:
  python timing.py --data-dir datasets_v3_full
  python timing.py --data-dir datasets_v3_full --model-dir results_v3 --n-instances 100
"""

import argparse
import json
import math
import os
import pickle
import time

import numpy as np

from src.data.dataset import GRID, K
from src.models.baselines import (
    UTC, ScaledHeuristic, MeanBaseline, LinearRegBaseline,
    FewShotBaseline, BOBaseline, OracleSA,
)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Baseline timing
# ---------------------------------------------------------------------------

def time_baseline(baseline, instances, n_warmup=5, n_timed=100,
                  train_instances=None):
    """Time a baseline's predict_r() on instances.

    Returns dict with mean_ms, std_ms, total_ms.
    """
    # Fit if needed
    if hasattr(baseline, 'fit') and train_instances is not None:
        baseline.fit(train_instances)

    n = min(len(instances), n_timed + n_warmup)
    test_subset = instances[:n]

    # Warmup
    for inst in test_subset[:n_warmup]:
        _ = baseline.predict_r(inst)

    # Timed runs
    times = []
    for inst in test_subset[n_warmup:n_warmup + n_timed]:
        t0 = time.perf_counter()
        _ = baseline.predict_r(inst)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    if not times:
        return {'mean_ms': 0.0, 'std_ms': 0.0, 'total_ms': 0.0,
                'n_runs': 0, 'n_actual_timed': 0}

    return {
        'mean_ms': float(np.mean(times)),
        'n_actual_timed': len(times),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'total_ms': float(np.sum(times)),
        'n_runs': len(times),
    }


# ---------------------------------------------------------------------------
# Neural model timing
# ---------------------------------------------------------------------------

def time_neural_model(model, test_data, device, n_warmup=10, n_timed=100,
                      batch_size=1):
    """Time a neural model's forward pass.

    Uses batch_size=1 for per-instance timing.
    Returns dict with mean_ms, std_ms, etc.
    """
    import torch
    from src.data.dataset import ChainStrengthDataset, collate_batch

    model.eval()
    model.to(device)

    n = min(len(test_data), n_warmup + n_timed)
    subset = test_data[:n]

    # Warmup
    with torch.no_grad():
        for i in range(min(n_warmup, len(subset))):
            batch = collate_batch([subset[i]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            _ = model(batch)

    # Synchronize GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for i in range(n_warmup, min(n_warmup + n_timed, len(subset))):
            batch = collate_batch([subset[i]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(batch)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    if not times:
        return {'mean_ms': 0.0, 'std_ms': 0.0, 'total_ms': 0.0, 'n_runs': 0}

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'total_ms': float(np.sum(times)),
        'n_runs': len(times),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark inference timing for all methods')
    parser.add_argument('--data-dir', default='datasets_v3_full',
                        help='Directory with multi_topo_{train,test}.pkl')
    parser.add_argument('--model-dir', default=None,
                        help='Directory with trained model checkpoints '
                             '(default: auto-detect from results_v3, '
                             'results/ablation, results/hec_gnn_multi_topo)')
    parser.add_argument('--output-dir', default='results/timing',
                        help='Output directory')
    parser.add_argument('--n-instances', type=int, default=100,
                        help='Number of test instances for timing')
    parser.add_argument('--n-warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--device', default=None,
                        help='Device (default: auto-detect)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_path = os.path.join(args.data_dir, 'multi_topo_train.pkl')
    test_path = os.path.join(args.data_dir, 'multi_topo_test.pkl')

    print("Loading data...")
    train_data = load_pkl(train_path)
    test_data = load_pkl(test_path)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    n_inst = min(args.n_instances, len(test_data))
    test_subset = test_data[:n_inst + args.n_warmup]
    print(f"  Using {n_inst} instances for timing "
          f"({args.n_warmup} warmup + {n_inst} timed)\n")

    results = {'methods': {}, 'config': {
        'n_instances': n_inst,
        'n_warmup': args.n_warmup,
    }}

    # -----------------------------------------------------------------------
    # Baseline timing
    # -----------------------------------------------------------------------
    baselines = {
        'UTC': UTC(),
        'Scaled(2.0)': ScaledHeuristic(2.0),
        'Mean': MeanBaseline(),
        'LinearReg': LinearRegBaseline(),
        'Few-Shot-3': FewShotBaseline(n_shots=3),
        'Few-Shot-5': FewShotBaseline(n_shots=5),
        'Few-Shot-10': FewShotBaseline(n_shots=10),
        'BO-10': BOBaseline(n_evals=10),
        'Oracle': OracleSA(),
    }

    print("Timing baselines:")
    for name, bl in baselines.items():
        print(f"  {name}...", end='', flush=True)
        timing = time_baseline(
            bl, test_subset,
            n_warmup=args.n_warmup, n_timed=n_inst,
            train_instances=train_data)
        results['methods'][name] = timing
        print(f"  {timing['mean_ms']:.4f} +/- {timing['std_ms']:.4f} ms")

    # -----------------------------------------------------------------------
    # Neural model timing
    # -----------------------------------------------------------------------
    try:
        import torch
        from src.models.hec_gnn import HECGNN, ModelConfig
        from src.models.baselines import build_flat_gnn

        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")

        results['config']['device'] = str(device)

        # Auto-detect model directory
        model_dir = args.model_dir
        if model_dir is None:
            for candidate in ['results_v3', 'results/hec_gnn_multi_topo',
                              'results/ablation']:
                if os.path.isdir(candidate):
                    model_dir = candidate
                    break

        # HEC-GNN
        print("\nTiming neural models:")
        hec_ckpt = None
        if model_dir:
            for fname in sorted(os.listdir(model_dir)):
                if fname.startswith('hec_gnn_seed') and fname.endswith('.pt'):
                    hec_ckpt = os.path.join(model_dir, fname)
                    break
                if fname.startswith('Full_HEC-GNN_seed') and fname.endswith('.pt'):
                    hec_ckpt = os.path.join(model_dir, fname)
                    break

        if hec_ckpt:
            print(f"  HEC-GNN ({hec_ckpt})...", end='', flush=True)
            model = HECGNN(ModelConfig())
            state = torch.load(hec_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state)
            timing = time_neural_model(
                model, test_subset, device,
                n_warmup=args.n_warmup, n_timed=n_inst)
            n_params = sum(p.numel() for p in model.parameters())
            timing['n_params'] = n_params
            results['methods']['HEC-GNN'] = timing
            print(f"  {timing['mean_ms']:.4f} +/- {timing['std_ms']:.4f} ms "
                  f"({n_params:,} params)")
        else:
            # Time with random weights
            print("  HEC-GNN (random init, no checkpoint)...", end='', flush=True)
            model = HECGNN(ModelConfig())
            timing = time_neural_model(
                model, test_subset, device,
                n_warmup=args.n_warmup, n_timed=n_inst)
            n_params = sum(p.numel() for p in model.parameters())
            timing['n_params'] = n_params
            timing['note'] = 'random_init'
            results['methods']['HEC-GNN'] = timing
            print(f"  {timing['mean_ms']:.4f} +/- {timing['std_ms']:.4f} ms "
                  f"({n_params:,} params) [random init]")

        # FlatGNN
        flat_ckpt = None
        if model_dir:
            for fname in sorted(os.listdir(model_dir)):
                if fname.startswith('flat_gnn_seed') and fname.endswith('.pt'):
                    flat_ckpt = os.path.join(model_dir, fname)
                    break
                if fname.startswith('FlatGNN_seed') and fname.endswith('.pt'):
                    flat_ckpt = os.path.join(model_dir, fname)
                    break

        if flat_ckpt:
            print(f"  FlatGNN ({flat_ckpt})...", end='', flush=True)
            model = build_flat_gnn()
            state = torch.load(flat_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state)
            timing = time_neural_model(
                model, test_subset, device,
                n_warmup=args.n_warmup, n_timed=n_inst)
            n_params = sum(p.numel() for p in model.parameters())
            timing['n_params'] = n_params
            results['methods']['FlatGNN'] = timing
            print(f"  {timing['mean_ms']:.4f} +/- {timing['std_ms']:.4f} ms "
                  f"({n_params:,} params)")
        else:
            print("  FlatGNN (random init, no checkpoint)...", end='', flush=True)
            model = build_flat_gnn()
            timing = time_neural_model(
                model, test_subset, device,
                n_warmup=args.n_warmup, n_timed=n_inst)
            n_params = sum(p.numel() for p in model.parameters())
            timing['n_params'] = n_params
            timing['note'] = 'random_init'
            results['methods']['FlatGNN'] = timing
            print(f"  {timing['mean_ms']:.4f} +/- {timing['std_ms']:.4f} ms "
                  f"({n_params:,} params) [random init]")

    except ImportError as e:
        print(f"\nSkipping neural model timing (torch not available): {e}")

    # -----------------------------------------------------------------------
    # Compute speedup ratios vs Oracle
    # -----------------------------------------------------------------------
    # Oracle cached lookup is trivial; real Oracle cost = K * SA evaluation time
    # Estimate from Few-Shot timing: each shot = 1 SA eval
    fs3_time = results['methods'].get('Few-Shot-3', {}).get('mean_ms', None)
    if fs3_time:
        sa_per_eval = fs3_time / 3.0  # time per single SA evaluation
        real_oracle_ms = sa_per_eval * K  # K=20 grid evaluations
        results['methods']['Oracle']['note'] = (
            f'Cached argmin; real grid search ~{real_oracle_ms:.1f}ms '
            f'({K} SA evals x {sa_per_eval:.2f}ms each)')
        results['methods']['Oracle']['real_grid_search_ms'] = real_oracle_ms
        oracle_time = real_oracle_ms
    else:
        oracle_time = results['methods'].get('Oracle', {}).get('mean_ms', None)

    if oracle_time and oracle_time > 0:
        print(f"\nSpeedup ratios vs Oracle grid search ({oracle_time:.2f} ms):")
        for name, mdata in results['methods'].items():
            if name == 'Oracle':
                mdata['speedup_vs_oracle'] = 1.0
                continue
            mt = mdata.get('mean_ms', 0)
            if mt > 0:
                speedup = oracle_time / mt
                mdata['speedup_vs_oracle'] = float(speedup)
                direction = 'faster' if speedup > 1 else 'slower'
                print(f"  {name:>15s}: {speedup:>8.2f}x {direction}")
            else:
                mdata['speedup_vs_oracle'] = float('inf')

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = os.path.join(args.output_dir, 'timing_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("Timing Summary")
    print(f"{'='*70}")
    print(f"{'Method':<18s} {'Mean (ms)':>10s} {'Std (ms)':>10s} "
          f"{'Median (ms)':>12s} {'Speedup':>10s}")
    print(f"{'-'*70}")
    for name, mdata in sorted(results['methods'].items(),
                                key=lambda x: x[1].get('mean_ms', 0)):
        speedup = mdata.get('speedup_vs_oracle', '')
        if isinstance(speedup, float):
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = ''
        print(f"{name:<18s} {mdata.get('mean_ms', 0):>10.4f} "
              f"{mdata.get('std_ms', 0):>10.4f} "
              f"{mdata.get('median_ms', 0):>12.4f} "
              f"{speedup_str:>10s}")


if __name__ == '__main__':
    main()
