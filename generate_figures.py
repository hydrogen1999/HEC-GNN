#!/usr/bin/env python3
"""
generate_figures.py -- Publication-quality figures for the paper.

Generates four figures:
  1. fig_energy_curves.pdf    -- True vs predicted energy curves (2x2 grid)
  2. fig_ood_scaling.pdf      -- MAE(r*) vs problem size n for OOD eval
  3. fig_pred_scatter.pdf     -- Predicted r* vs true r* scatter, colored by topology
  4. fig_cost_quality.pdf     -- Cost vs quality Pareto plot

Usage:
  python generate_figures.py --results-dir results --data-dir datasets_v3_full
  python generate_figures.py --results-dir results --fig-dir figures --format pdf
"""

import argparse
import json
import math
import os
import pickle

import numpy as np

# ---------------------------------------------------------------------------
# Grid (matches the chain-strength ratio grid used by the trainer)
# ---------------------------------------------------------------------------
K = 20
R_MIN = 0.02
R_MAX = 5.0
GRID = np.logspace(math.log10(R_MIN), math.log10(R_MAX), K).astype(np.float32)


def setup_matplotlib():
    """Configure matplotlib for NeurIPS publication quality."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'standard',
        'savefig.pad_inches': 0.05,
        'text.usetex': False,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
    })
    return plt


def load_json(path):
    """Load a JSON file, return None if not found."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def load_pkl(path):
    """Load a pickle file, return None if not found."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Energy curves (true vs predicted)
# ---------------------------------------------------------------------------

def fig_energy_curves(data_dir, model_dir, fig_dir, fmt, device_str):
    """fig_energy_curves.{fmt} -- 2x2 subplot of example energy curves."""
    plt = setup_matplotlib()

    # Try to load model and data for real predictions
    test_data = load_pkl(os.path.join(data_dir, 'multi_topo_test.pkl'))
    if test_data is None:
        print("  Skipping fig_energy_curves: no test data found")
        return

    # Try loading trained model for predictions
    pred_curves = None
    if model_dir:
        try:
            import torch
            from src.models.hec_gnn import HECGNN, ModelConfig
            from src.data.dataset import ChainStrengthDataset, collate_batch

            device = torch.device(device_str)
            # Find first available checkpoint
            ckpt_path = None
            for fname in os.listdir(model_dir):
                if fname.startswith('hec_gnn_seed') and fname.endswith('.pt'):
                    ckpt_path = os.path.join(model_dir, fname)
                    break
            # Also check ablation dir
            if ckpt_path is None:
                abl_dir = os.path.join(os.path.dirname(model_dir), 'ablation')
                if os.path.isdir(abl_dir):
                    for fname in os.listdir(abl_dir):
                        if fname.startswith('Full_HEC-GNN_seed') and fname.endswith('.pt'):
                            ckpt_path = os.path.join(abl_dir, fname)
                            break

            if ckpt_path:
                print(f"  Loading model from {ckpt_path}")
                model = HECGNN(ModelConfig())
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.to(device)
                model.eval()

                # Get predictions for selected instances
                pred_curves = {}
                selected_indices = _select_diverse_instances(test_data, n=4)
                for idx in selected_indices:
                    batch = collate_batch([test_data[idx]])
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    with torch.no_grad():
                        ep, _, _ = model(batch)
                    pred_curves[idx] = ep[0].cpu().numpy()
        except Exception as e:
            print(f"  Could not load model: {e}")
            pred_curves = None

    # Select 4 diverse instances
    selected = _select_diverse_instances(test_data, n=4)

    fig, axes = plt.subplots(2, 2, figsize=(3.25, 3.0))
    axes = axes.flatten()

    for ax_idx, inst_idx in enumerate(selected):
        ax = axes[ax_idx]
        inst = test_data[inst_idx]

        ec_true = np.array(inst['energy_curve'])
        true_min_idx = np.argmin(ec_true)
        r_star_true = GRID[true_min_idx]

        # Topology label
        topo = inst.get('topology', inst.get('topo', f'inst_{inst_idx}'))
        n_log = inst.get('n_logical', inst.get('n_chains', '?'))

        ax.plot(GRID, ec_true, 'b-', label='True', linewidth=1.0)

        if pred_curves and inst_idx in pred_curves:
            ec_pred = pred_curves[inst_idx]
            ax.plot(GRID, ec_pred, 'r--', label='Predicted', linewidth=1.0)
            pred_min_idx = np.argmin(ec_pred)
            r_star_pred = GRID[pred_min_idx]
            ax.axvline(r_star_pred, color='red', linestyle=':', alpha=0.5,
                       linewidth=0.5)

        ax.axvline(r_star_true, color='blue', linestyle=':', alpha=0.5,
                   linewidth=0.5, label='$r^*$')

        ax.set_xscale('log')
        ax.set_title(f'{topo}, n={n_log}', fontsize=7)
        ax.set_xlabel('$r$')
        ax.set_ylabel('Energy')

        if ax_idx == 0:
            ax.legend(loc='upper right', framealpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(fig_dir, f'fig_energy_curves.{fmt}')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


def _select_diverse_instances(data, n=4):
    """Select n diverse instances spanning different sizes/topologies."""
    if len(data) <= n:
        return list(range(len(data)))

    # Group by topology if available
    by_topo = {}
    for i, inst in enumerate(data):
        topo = inst.get('topology', inst.get('topo', 'unknown'))
        by_topo.setdefault(topo, []).append(i)

    selected = []
    topos = sorted(by_topo.keys())

    if len(topos) >= n:
        # Pick one from each of n different topologies
        for t in topos[:n]:
            # Pick instance with median size
            indices = by_topo[t]
            sizes = [data[i].get('n_logical', data[i].get('n_chains', 0))
                     for i in indices]
            median_idx = indices[np.argsort(sizes)[len(sizes) // 2]]
            selected.append(median_idx)
    else:
        # Pick evenly spaced indices
        step = max(1, len(data) // n)
        selected = [i * step for i in range(n)]
        selected = [min(s, len(data) - 1) for s in selected]

    return selected[:n]


# ---------------------------------------------------------------------------
# Figure 2: OOD scaling
# ---------------------------------------------------------------------------

def fig_ood_scaling(results_dir, fig_dir, fmt):
    """fig_ood_scaling.{fmt} -- MAE(r*) vs logical problem size n."""
    plt = setup_matplotlib()

    # Try loading OOD results
    ood_path = os.path.join(results_dir, 'eval', 'ood_results.json')
    if not os.path.exists(ood_path):
        ood_path = os.path.join(results_dir, 'ood_results.json')
    ood_results = load_json(ood_path)

    if ood_results is None:
        print("  Skipping fig_ood_scaling: no OOD results found.")
        print("  Expected at: results/eval/ood_results.json")
        return

    # Extract MAE by size for each method
    sizes = []
    for key in sorted(ood_results.keys()):
        if key.startswith('n='):
            n = int(key.split('=')[1].split('_')[0])
            sizes.append(n)
    sizes = sorted(set(sizes))

    if not sizes:
        print("  No per-size OOD results found.")
        return

    methods = {
        'HEC-GNN': {'marker': 'o', 'color': '#2171b5', 'ls': '-'},
        'FlatGNN': {'marker': 's', 'color': '#cb181d', 'ls': '--'},
        'LinearReg': {'marker': '^', 'color': '#238b45', 'ls': '-.'},
        'Mean': {'marker': 'D', 'color': '#6a51a3', 'ls': ':'},
    }

    fig, ax = plt.subplots(figsize=(3.25, 2.2))

    for method, style in methods.items():
        mae_vals = []
        valid_sizes = []
        for n in sizes:
            key_bl = f'n={n}_baselines'
            key_mod = f'n={n}_models'

            mae = None
            if method in ('HEC-GNN', 'FlatGNN'):
                data = ood_results.get(key_mod, {})
                if method in data:
                    mae = data[method].get('mae_r_mean', data[method].get('mae_r'))
            else:
                data = ood_results.get(key_bl, {})
                if method in data:
                    mae = data[method].get('mae_r')

            if mae is not None:
                mae_vals.append(mae)
                valid_sizes.append(n)

        if mae_vals:
            ax.plot(valid_sizes, mae_vals, marker=style['marker'],
                    color=style['color'], linestyle=style['ls'],
                    label=method, markersize=4)

    ax.set_xlabel('Logical problem size $n$')
    ax.set_ylabel('MAE($r^*$)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.8)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])

    plt.tight_layout()
    out_path = os.path.join(fig_dir, f'fig_ood_scaling.{fmt}')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Prediction scatter
# ---------------------------------------------------------------------------

def fig_pred_scatter(data_dir, model_dir, fig_dir, fmt, device_str):
    """fig_pred_scatter.{fmt} -- Predicted r* vs true r*, colored by topology."""
    plt = setup_matplotlib()

    test_data = load_pkl(os.path.join(data_dir, 'multi_topo_test.pkl'))
    if test_data is None:
        print("  Skipping fig_pred_scatter: no test data found")
        return

    # Get predictions from trained model
    pred_rs = None
    if model_dir:
        try:
            import torch
            from src.models.hec_gnn import HECGNN, ModelConfig, parabolic_argmin
            from src.data.dataset import ChainStrengthDataset, collate_batch
            from torch.utils.data import DataLoader

            device = torch.device(device_str)

            ckpt_path = None
            for fname in sorted(os.listdir(model_dir)):
                if fname.startswith('hec_gnn_seed') and fname.endswith('.pt'):
                    ckpt_path = os.path.join(model_dir, fname)
                    break
            if ckpt_path is None:
                abl_dir = os.path.join(os.path.dirname(model_dir), 'ablation')
                if os.path.isdir(abl_dir):
                    for fname in sorted(os.listdir(abl_dir)):
                        if fname.startswith('Full_HEC-GNN_seed') and fname.endswith('.pt'):
                            ckpt_path = os.path.join(abl_dir, fname)
                            break

            if ckpt_path:
                print(f"  Loading model from {ckpt_path}")
                model = HECGNN(ModelConfig())
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.to(device)
                model.eval()

                grid_t = torch.tensor(GRID, dtype=torch.float32, device=device)
                ds = ChainStrengthDataset(test_data)
                loader = DataLoader(ds, batch_size=64, shuffle=False,
                                    collate_fn=collate_batch)

                pred_rs = []
                with torch.no_grad():
                    for batch in loader:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                        ep, _, _ = model(batch)
                        rp = parabolic_argmin(ep, grid_t)
                        pred_rs.extend(rp.cpu().tolist())
        except Exception as e:
            print(f"  Could not load model: {e}")
            pred_rs = None

    # Collect true r*
    true_rs = []
    topos = []
    for inst in test_data:
        ec = np.array(inst['energy_curve'])
        true_rs.append(GRID[np.argmin(ec)])
        topos.append(inst.get('topology', inst.get('topo', 'unknown')))

    true_rs = np.array(true_rs)

    if pred_rs is None:
        print("  WARNING: No model predictions available. Skipping scatter plot.")
        return
    pred_rs = np.array(pred_rs)

    # Topology coloring
    unique_topos = sorted(set(topos))
    cmap = plt.cm.get_cmap('tab10')
    topo_colors = {t: cmap(i / max(len(unique_topos) - 1, 1))
                   for i, t in enumerate(unique_topos)}

    # Spearman
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(pred_rs, true_rs)
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
        rp = _rankdata(pred_rs)
        rt = _rankdata(true_rs)
        n = len(rp)
        d2 = np.sum((rp - rt) ** 2)
        rho = 1.0 - 6.0 * d2 / (n * (n**2 - 1)) if n > 1 else 0.0

    fig, ax = plt.subplots(figsize=(3.25, 3.0))

    for topo in unique_topos:
        mask = np.array([t == topo for t in topos])
        ax.scatter(true_rs[mask], pred_rs[mask],
                   c=[topo_colors[topo]], s=8, alpha=0.6,
                   label=topo, edgecolors='none')

    # Diagonal
    lo = min(true_rs.min(), pred_rs.min()) * 0.8
    hi = max(true_rs.max(), pred_rs.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('True $r^*$')
    ax.set_ylabel('Predicted $r^*$')
    ax.legend(title=f'$\\rho$={rho:.3f}', loc='upper left', framealpha=0.8,
              markerscale=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')

    plt.tight_layout()
    out_path = os.path.join(fig_dir, f'fig_pred_scatter.{fmt}')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")

    # Save scatter data as JSON
    scatter_data = {
        'spearman_rho': float(rho),
        'n_instances': len(true_rs),
        'topologies': unique_topos,
    }
    json_path = os.path.join(fig_dir, 'fig_pred_scatter_data.json')
    with open(json_path, 'w') as f:
        json.dump(scatter_data, f, indent=2)


# ---------------------------------------------------------------------------
# Figure 4: Cost vs quality Pareto
# ---------------------------------------------------------------------------

def fig_cost_quality(results_dir, fig_dir, fmt):
    """fig_cost_quality.{fmt} -- Inference time vs delta_E<=5% accuracy."""
    plt = setup_matplotlib()

    # Load timing results
    timing_path = os.path.join(results_dir, 'timing', 'timing_results.json')
    timing = load_json(timing_path)

    # Load accuracy results
    mt_path = os.path.join(results_dir, 'eval', 'multi_topo_results.json')
    if not os.path.exists(mt_path):
        mt_path = os.path.join(results_dir, 'multi_topo_results.json')
    mt_results = load_json(mt_path)

    # Also check ablation results for accuracy
    abl_path = os.path.join(results_dir, 'ablation', 'ablation_results.json')
    abl_results = load_json(abl_path)

    if timing is None and mt_results is None and abl_results is None:
        print("  Skipping fig_cost_quality: no results found.")
        print("  Run timing.py and evaluate.py first.")
        return

    # Collect (time_ms, accuracy) pairs
    points = {}

    # From timing results
    if timing:
        methods_timing = timing.get('methods', timing)
        for method_name, tdata in methods_timing.items():
            if isinstance(tdata, dict):
                ms = tdata.get('mean_ms', tdata.get('time_ms'))
                if ms is not None:
                    points.setdefault(method_name, {})['time_ms'] = ms

    # From multi_topo results
    if mt_results:
        baselines = mt_results.get('baselines', {})
        for name, bdata in baselines.items():
            points.setdefault(name, {})['delta_e_5pct'] = bdata.get('delta_e_5pct', 0)
        models = mt_results.get('models', {})
        for name, mdata in models.items():
            points.setdefault(name, {})['delta_e_5pct'] = mdata.get('delta_e_5pct_mean',
                                                                      mdata.get('delta_e_5pct', 0))

    # From ablation results
    if abl_results:
        for vname, vdata in abl_results.items():
            mapped_name = vname.replace('Full_', '').replace('No_Aux_Losses', 'No-Aux')
            if 'delta_e_5pct_mean' in vdata:
                points.setdefault(mapped_name, {})['delta_e_5pct'] = vdata['delta_e_5pct_mean']

    # Filter to points that have both time and accuracy
    complete = {k: v for k, v in points.items()
                if 'time_ms' in v and 'delta_e_5pct' in v}

    if not complete:
        print("  Still no plottable points. Skipping.")
        return

    # Style map
    style_map = {
        'HEC-GNN': ('o', '#2171b5', 9),
        'FlatGNN': ('s', '#cb181d', 8),
        'UTC': ('^', '#f16913', 7),
        'Scaled(2.0)': ('v', '#d94801', 7),
        'Mean': ('D', '#6a51a3', 7),
        'LinearReg': ('P', '#238b45', 7),
        'Few-Shot-3': ('X', '#e6550d', 6),
        'Few-Shot-5': ('X', '#fd8d3c', 6),
        'Few-Shot-10': ('X', '#fdae6b', 6),
        'BO-10': ('*', '#756bb1', 8),
        'Oracle': ('h', '#31a354', 9),
    }
    default_style = ('o', '#636363', 6)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    for name, vals in complete.items():
        marker, color, size = style_map.get(name, default_style)
        ax.scatter(vals['time_ms'], vals['delta_e_5pct'] * 100,
                   marker=marker, c=color, s=size**2, zorder=3,
                   label=name, edgecolors='white', linewidths=0.3)

    ax.set_xscale('log')
    ax.set_xlabel('Inference time (ms)')
    ax.set_ylabel('$\\delta_E \\leq 5\\%$ accuracy (%)')
    ax.legend(loc='lower right', fontsize=5.5, ncol=2, framealpha=0.8)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out_path = os.path.join(fig_dir, f'fig_cost_quality.{fmt}')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")

    # Save data
    pareto_data = {name: {'time_ms': v['time_ms'],
                          'delta_e_5pct': v['delta_e_5pct']}
                   for name, v in complete.items()}
    json_path = os.path.join(fig_dir, 'fig_cost_quality_data.json')
    with open(json_path, 'w') as f:
        json.dump(pareto_data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality figures for the paper')
    parser.add_argument('--results-dir', default='results',
                        help='Root results directory')
    parser.add_argument('--data-dir', default='datasets_v3_full',
                        help='Directory with dataset pickle files')
    parser.add_argument('--model-dir', default=None,
                        help='Directory with trained model checkpoints '
                             '(default: results_v3 or results/hec_gnn_multi_topo)')
    parser.add_argument('--fig-dir', default='figures',
                        help='Output directory for figures')
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png', 'svg'],
                        help='Figure output format')
    parser.add_argument('--device', default=None,
                        help='Device for model inference (default: auto)')
    parser.add_argument('--figures', nargs='*', default=None,
                        help='Specific figures to generate (e.g., energy_curves ood_scaling)')
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    # Auto-detect model directory
    model_dir = args.model_dir
    if model_dir is None:
        for candidate in ['results_v3', 'results/hec_gnn_multi_topo',
                          'results/ablation']:
            if os.path.isdir(candidate):
                model_dir = candidate
                break

    device_str = args.device or ('cuda' if _has_cuda() else 'cpu')

    # Determine which figures to generate
    all_figs = ['energy_curves', 'ood_scaling', 'pred_scatter', 'cost_quality']
    figs_to_gen = args.figures if args.figures else all_figs

    print("Generating publication figures")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Model dir:   {model_dir}")
    print(f"  Fig dir:     {args.fig_dir}")
    print(f"  Format:      {args.format}")
    print(f"  Device:      {device_str}")
    print()

    if 'energy_curves' in figs_to_gen:
        print("[1/4] Energy curves...")
        fig_energy_curves(args.data_dir, model_dir, args.fig_dir, args.format,
                          device_str)

    if 'ood_scaling' in figs_to_gen:
        print("[2/4] OOD scaling...")
        fig_ood_scaling(args.results_dir, args.fig_dir, args.format)

    if 'pred_scatter' in figs_to_gen:
        print("[3/4] Prediction scatter...")
        fig_pred_scatter(args.data_dir, model_dir, args.fig_dir, args.format,
                         device_str)

    if 'cost_quality' in figs_to_gen:
        print("[4/4] Cost vs quality Pareto...")
        fig_cost_quality(args.results_dir, args.fig_dir, args.format)

    print("\nDone.")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == '__main__':
    main()
