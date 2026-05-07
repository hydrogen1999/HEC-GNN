#!/usr/bin/env python3
"""
qpu_adapt_600.py -- Evaluate & fine-tune all models on 600 QPU instances.

Steps:
  1. Load 600 QPU instances (K=20 RMS-normalized grid, 500 reads)
  2. Build graph representations from stored seeds + embeddings
  3. Handle early-stopped instances (pad to 20 points)
  4. Evaluate pre-trained models ZERO-SHOT on QPU curves:
     - HEC-GNN (3 seeds), FlatGNN (3 seeds)
     - LinearReg, UTC, Scaled(2.0), Mean
  5. Fine-tune HEC-GNN on QPU data (freeze backbone, train energy_head)
  6. Report: δE, Gap%, curve MAE, r* MAE, p_solve comparison

Usage:
  python qpu_adapt_600.py \
    --qpu-data qpu_600/qpu_labeling_large_qpu.json \
    --embeddings qpu_600/embeddings_large.json \
    --hec-dir results/diverse_run/hec_gnn_mt \
    --flat-dir results/diverse_run/flat_gnn_mt \
    --output qpu_600/adaptation_results.json
"""

import argparse
import copy
import json
import math
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW

# Add code paths
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
SRC_DIR = CODE_DIR / 'src'
# Alternate path: script at project root, code at src/
ALT_SRC = SCRIPT_DIR / "src"
for p in [str(SRC_DIR), str(CODE_DIR), str(SCRIPT_DIR), str(ALT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.hec_gnn import HECGNN
from models.baselines import UTC, build_flat_gnn
from data.dataset import collate_batch
from data.generate import (
    build_physical_hamiltonian, compute_qubit_features,
    compute_chain_edge_features, compute_inter_chain_edges,
    compute_logical_edge_features, compute_rms_J, make_grid
)

import networkx as nx
import dimod
import dwave_networkx as dnx
import minorminer
from dwave.embedding.chain_strength import uniform_torque_compensation

GRID_20 = np.array(make_grid(20, 0.02, 5.0))
SEED = 42


# ---------------------------------------------------------------------------
# 1. Load & parse QPU data
# ---------------------------------------------------------------------------
def load_qpu_600(qpu_path, embeddings_path):
    """Load 600-instance QPU data. Returns parsed instances."""
    with open(qpu_path) as f:
        qpu_data = json.load(f)
    with open(embeddings_path) as f:
        emb_data = json.load(f)

    print(f"Loaded {len(qpu_data['instances'])} QPU instances")
    print(f"Loaded {len(emb_data)} embeddings")
    return qpu_data, emb_data


def extract_qpu_curves(qpu_data):
    """Extract 20-point energy/cbr/p_solve curves, handling early stops."""
    instances = qpu_data['instances']
    curves = []
    n_full = 0
    n_padded = 0

    for inst in instances:
        sweep = inst['sweep']
        n_points = len(sweep)

        # Extract measured points
        rs = np.array([s['rel'] for s in sweep])
        energies = np.array([s['mean_energy'] for s in sweep])
        cbrs = np.array([s['cbr'] for s in sweep])
        p_solves = np.array([s['p_solve'] for s in sweep])

        if n_points == 20:
            n_full += 1
            energy_20 = energies
            cbr_20 = cbrs
            psolve_20 = p_solves
        else:
            n_padded += 1
            # Pad: for missing high-r points (early stop at low r due to CBR),
            # use the last measured energy (flat extrapolation)
            energy_20 = np.full(20, energies[-1])
            cbr_20 = np.full(20, cbrs[-1])
            psolve_20 = np.full(20, p_solves[-1])
            # Fill in measured points by matching grid positions
            for j, r in enumerate(rs):
                idx = np.argmin(np.abs(GRID_20 - r))
                energy_20[idx] = energies[j]
                cbr_20[idx] = cbrs[j]
                psolve_20[idx] = p_solves[j]

        r_star_idx = int(np.argmin(energy_20))
        curves.append({
            'energy_curve': energy_20.tolist(),
            'break_curve': cbr_20.tolist(),
            'psolve_curve': psolve_20.tolist(),
            'r_star': float(GRID_20[r_star_idx]),
            'r_star_idx': r_star_idx,
            'n_measured': n_points,
            'opt_energy': float(np.min(energy_20)),
            'opt_p_solve': float(p_solves[np.argmin(energies)]),
        })

    print(f"Curves: {n_full} full (20pt), {n_padded} padded (<20pt)")
    return curves


# ---------------------------------------------------------------------------
# 2. Build graph representations
# ---------------------------------------------------------------------------
def build_graphs(qpu_data, emb_data, curves):
    """Regenerate h/J from seeds, build GNN-compatible graph dicts."""
    # QPU used Advantage_system4.1 which is Pegasus
    # Use P16 for local embedding (instances were generated on actual QPU graph,
    # but embeddings are stored, so we just need the topology for features)
    topology = dnx.pegasus_graph(16)
    print(f"Using Pegasus P16 ({topology.number_of_nodes()} qubits)")

    instances = qpu_data['instances']
    graphs = []
    failed = 0

    for i, (inst, emb_info, curve) in enumerate(zip(instances, emb_data, curves)):
        try:
            seed = inst['seed']
            n = inst['n']
            p = inst['p']

            # Regenerate problem from seed
            rng = np.random.RandomState(seed)
            # Match generate_instances() logic
            _ = int(rng.choice([8, 10, 12, 15, 20, 25, 30, 40]))  # consume size choice
            _ = float(rng.choice([0.3, 0.4, 0.5, 0.6, 0.7]))      # consume density choice
            G = nx.erdos_renyi_graph(n, p, seed=seed)
            h = {node: float(rng.uniform(-1, 1)) for node in G.nodes()}
            J = {(u, v): float(rng.uniform(-2, 2)) for u, v in G.edges()}

            # Load stored embedding
            emb_raw = emb_info['embedding']
            emb = {int(k): v for k, v in emb_raw.items()}

            # Build physical Hamiltonian
            h_phys, J_phys, chain_edge_map, chain_edges_flat = \
                build_physical_hamiltonian(h, J, emb, topology)
            rms = compute_rms_J(J)

            # Compute features
            feats, qubit_list, chain_assign = compute_qubit_features(
                emb, h_phys, J_phys, chain_edge_map, rms)
            qubit_to_local = {q: idx for idx, q in enumerate(qubit_list)}
            chain_ei, chain_ef = compute_chain_edge_features(chain_edge_map, qubit_to_local)
            inter_ei, inter_ef = compute_inter_chain_edges(emb, J_phys, qubit_to_local, rms)
            logical_ei, logical_ef = compute_logical_edge_features(J, emb, J_phys, topology, rms)

            sorted_vars = sorted(emb.keys())
            graph = {
                'qubit_features': feats,
                'chain_assignment': chain_assign,
                'chain_edge_index': chain_ei,
                'chain_edge_features': chain_ef,
                'inter_edge_index': inter_ei,
                'inter_edge_features': inter_ef,
                'logical_edge_index': logical_ei,
                'logical_edge_features': logical_ef,
                'n_chains': len(emb),
                'energy_curve': curve['energy_curve'],
                'break_curve': curve['break_curve'],
                'r_star': curve['r_star'],
                'r_star_idx': curve['r_star_idx'],
                'jc_star_raw': float(curve['r_star'] * rms),
                'rms_J': rms,
                'chain_break_targets': [0.0] * len(emb),
                'family': 'random_ising',
                'instance_id': i,
                'topology': 'QPU',
                'n_logical': n,
                'h_logical': h,
                'J_logical': J,
                'embedding': emb,
                'h_phys': h_phys,
                'J_phys': J_phys,
                'chain_edge_map': chain_edge_map,
            }
            graphs.append(graph)

            if (i + 1) % 100 == 0:
                print(f"  Built {i+1}/{len(instances)} graphs")

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Instance {i}: FAILED ({e})")

    print(f"Built {len(graphs)} graphs ({failed} failed)")
    return graphs


# ---------------------------------------------------------------------------
# 3. Evaluate GNN models (HEC-GNN, FlatGNN)
# ---------------------------------------------------------------------------
def eval_gnn(model, graphs, device='cpu'):
    """Evaluate a GNN model on QPU graphs. Returns per-instance metrics."""
    model.eval()
    model.to(device)

    results = []
    with torch.no_grad():
        for graph in graphs:
            batch = collate_batch([graph])
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            output = model(batch)
            energy_pred = output[0] if isinstance(output, tuple) else output
            pred_curve = energy_pred.squeeze(0).cpu().numpy()

            qpu_curve = np.array(graph['energy_curve'])
            qpu_opt = np.min(qpu_curve)
            pred_r_idx = int(np.argmin(pred_curve))

            # Energy at predicted r* on QPU curve
            e_at_pred = qpu_curve[pred_r_idx]
            delta_e = (e_at_pred - qpu_opt) / max(abs(qpu_opt), 1e-8)

            results.append({
                'delta_e': float(delta_e),
                'curve_mae': float(np.mean(np.abs(pred_curve - qpu_curve))),
                'r_star_mae': float(abs(GRID_20[pred_r_idx] - graph['r_star'])),
                'pred_r_star': float(GRID_20[pred_r_idx]),
                'qpu_r_star': graph['r_star'],
                'n_logical': graph['n_logical'],
            })

    return results


def load_and_eval_gnn(model_dir, model_class, graphs, device='cpu', label=''):
    """Load all seed checkpoints and evaluate."""
    model_dir = Path(model_dir)
    model_files = sorted(model_dir.glob('*.pt'))
    if not model_files:
        print(f"  No models found in {model_dir}")
        return None

    all_results = []
    for mf in model_files:
        checkpoint = torch.load(str(mf), map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model = model_class()
        model.load_state_dict(state_dict)
        results = eval_gnn(model, graphs, device)
        all_results.append(results)
        delta_es = [r['delta_e'] for r in results]
        pct5 = sum(1 for d in delta_es if d <= 0.05) / len(delta_es) * 100
        print(f"  {mf.name}: δE≤5%={pct5:.1f}%, CurveMAE={np.mean([r['curve_mae'] for r in results]):.2f}")

    return all_results


# ---------------------------------------------------------------------------
# 4. Evaluate non-GNN baselines
# ---------------------------------------------------------------------------
def eval_baselines(graphs):
    """Evaluate UTC, Scaled(2.0), Mean, LinearReg on QPU data."""
    utc = UTC()
    results = {}

    # UTC
    utc_deltas = []
    for g in graphs:
        r_pred = utc.predict(g) / max(g['rms_J'], 1e-8)
        idx = int(np.argmin(np.abs(GRID_20 - r_pred)))
        qpu_curve = np.array(g['energy_curve'])
        qpu_opt = np.min(qpu_curve)
        delta = (qpu_curve[idx] - qpu_opt) / max(abs(qpu_opt), 1e-8)
        utc_deltas.append(float(delta))
    results['UTC'] = utc_deltas

    # Scaled(2.0): r = 2.0 always
    scaled_deltas = []
    idx_2 = int(np.argmin(np.abs(GRID_20 - 2.0)))
    for g in graphs:
        qpu_curve = np.array(g['energy_curve'])
        qpu_opt = np.min(qpu_curve)
        delta = (qpu_curve[idx_2] - qpu_opt) / max(abs(qpu_opt), 1e-8)
        scaled_deltas.append(float(delta))
    results['Scaled(2.0)'] = scaled_deltas

    # Mean: predict mean r* from training data (approximate as median of grid)
    idx_mid = 10  # middle of 20-point grid
    mean_deltas = []
    for g in graphs:
        qpu_curve = np.array(g['energy_curve'])
        qpu_opt = np.min(qpu_curve)
        delta = (qpu_curve[idx_mid] - qpu_opt) / max(abs(qpu_opt), 1e-8)
        mean_deltas.append(float(delta))
    results['Mean'] = mean_deltas

    # Oracle
    oracle_deltas = [0.0] * len(graphs)
    results['Oracle'] = oracle_deltas

    return results


# ---------------------------------------------------------------------------
# 5. Fine-tune HEC-GNN
# ---------------------------------------------------------------------------
def finetune_hecgnn(model_path, train_graphs, val_graphs, lr=1e-4,
                    epochs=200, patience=30, device='cpu'):
    """Fine-tune: freeze backbone, train energy_head on QPU curves."""
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model = HECGNN()
    model.load_state_dict(state_dict)

    # Freeze backbone
    for name, param in model.named_parameters():
        if 'energy_head' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} (energy_head only)")

    model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for graph in train_graphs:
            batch = collate_batch([graph])
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            energy_pred, _, _ = model(batch)
            loss = nn.functional.mse_loss(energy_pred, batch['energy_curve'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for graph in val_graphs:
                batch = collate_batch([graph])
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                energy_pred, _, _ = model(batch)
                val_loss += nn.functional.mse_loss(energy_pred, batch['energy_curve']).item()

        train_loss = total_loss / max(len(train_graphs), 1)
        val_loss = val_loss / max(len(val_graphs), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}  train={train_loss:.4f}  val={val_loss:.4f}  best={best_val_loss:.4f}")

        if wait >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# 6. Summary statistics
# ---------------------------------------------------------------------------
def summarize(delta_es, label=''):
    """Compute summary stats from delta_e list."""
    d = np.array(delta_es)
    return {
        'method': label,
        'delta_e_5pct': float(np.mean(d <= 0.05) * 100),
        'delta_e_2pct': float(np.mean(d <= 0.02) * 100),
        'delta_e_1pct': float(np.mean(d <= 0.01) * 100),
        'gap_pct': float(np.mean(d) * 100),
        'n': len(d),
    }


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='QPU adaptation for 600 instances')
    parser.add_argument('--qpu-data', required=True, help='QPU labeling JSON')
    parser.add_argument('--embeddings', required=True, help='Embeddings JSON')
    parser.add_argument('--hec-dir', default=None, help='HEC-GNN model directory (3 seeds)')
    parser.add_argument('--flat-dir', default=None, help='FlatGNN model directory (3 seeds)')
    parser.add_argument('--output', default='qpu_adaptation_results.json')
    parser.add_argument('--finetune', action='store_true', help='Run fine-tuning')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print("=" * 70)
    print("QPU Adaptation: 600 Instances")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading QPU data...")
    qpu_data, emb_data = load_qpu_600(args.qpu_data, args.embeddings)

    # 2. Extract curves
    print("\n[2] Extracting curves...")
    curves = extract_qpu_curves(qpu_data)

    # 3. Build graphs
    print("\n[3] Building graph representations...")
    graphs = build_graphs(qpu_data, emb_data, curves)

    all_results = {}

    # 4. Evaluate non-GNN baselines
    print("\n[4] Evaluating baselines...")
    baseline_results = eval_baselines(graphs)
    for name, deltas in baseline_results.items():
        s = summarize(deltas, name)
        all_results[name] = s
        print(f"  {name}: δE≤5%={s['delta_e_5pct']:.1f}%, Gap%={s['gap_pct']:.2f}%")

    # 5. Evaluate HEC-GNN (zero-shot)
    if args.hec_dir:
        print("\n[5] Evaluating HEC-GNN (zero-shot on QPU)...")
        hec_results = load_and_eval_gnn(args.hec_dir, HECGNN, graphs, args.device, 'HEC-GNN')
        if hec_results:
            # Average over seeds
            per_seed = []
            for seed_results in hec_results:
                delta_es = [r['delta_e'] for r in seed_results]
                per_seed.append(summarize(delta_es, 'HEC-GNN'))
            avg = {k: np.mean([s[k] for s in per_seed]) for k in per_seed[0] if k != 'method'}
            std = {k: np.std([s[k] for s in per_seed]) for k in per_seed[0] if k != 'method'}
            avg['method'] = 'HEC-GNN (SA-trained)'
            all_results['HEC-GNN'] = avg
            print(f"  Average: δE≤5%={avg['delta_e_5pct']:.1f}±{std['delta_e_5pct']:.1f}%, "
                  f"Gap%={avg['gap_pct']:.2f}±{std['gap_pct']:.2f}%")

    # 6. Evaluate FlatGNN (zero-shot)
    if args.flat_dir:
        print("\n[6] Evaluating FlatGNN (zero-shot on QPU)...")
        flat_results = load_and_eval_gnn(args.flat_dir, build_flat_gnn, graphs, args.device, 'FlatGNN')
        if flat_results:
            per_seed = []
            for seed_results in flat_results:
                delta_es = [r['delta_e'] for r in seed_results]
                per_seed.append(summarize(delta_es, 'FlatGNN'))
            avg = {k: np.mean([s[k] for s in per_seed]) for k in per_seed[0] if k != 'method'}
            std = {k: np.std([s[k] for s in per_seed]) for k in per_seed[0] if k != 'method'}
            avg['method'] = 'FlatGNN (SA-trained)'
            all_results['FlatGNN'] = avg
            print(f"  Average: δE≤5%={avg['delta_e_5pct']:.1f}±{std['delta_e_5pct']:.1f}%, "
                  f"Gap%={avg['gap_pct']:.2f}±{std['gap_pct']:.2f}%")

    # 7. Fine-tune HEC-GNN on QPU data
    if args.finetune and args.hec_dir:
        print(f"\n[7] Fine-tuning HEC-GNN ({args.n_folds}-fold CV)...")
        model_files = sorted(Path(args.hec_dir).glob('*.pt'))
        if not model_files:
            print("  No model files found, skipping fine-tuning")
        else:
            rng = np.random.RandomState(42)
            n = len(graphs)
            perm = rng.permutation(n)
            fold_size = n // args.n_folds

            all_fold_results = []
            for fold_i in range(args.n_folds):
                test_idx = perm[fold_i * fold_size: (fold_i + 1) * fold_size].tolist()
                train_idx = [j for j in perm if j not in test_idx]
                train_graphs = [graphs[j] for j in train_idx]
                test_graphs = [graphs[j] for j in test_idx]

                # Split train into train/val
                n_val = max(1, len(train_graphs) // 5)
                val_set = train_graphs[:n_val]
                train_set = train_graphs[n_val:]

                print(f"\n  Fold {fold_i+1}/{args.n_folds} "
                      f"(train={len(train_set)}, val={len(val_set)}, test={len(test_graphs)})")

                # Fine-tune first seed
                model = finetune_hecgnn(
                    model_files[0], train_set, val_set,
                    lr=args.lr, epochs=args.epochs,
                    patience=args.patience, device=args.device)

                results = eval_gnn(model, test_graphs, args.device)
                delta_es = [r['delta_e'] for r in results]
                fold_s = summarize(delta_es, f'fold_{fold_i+1}')
                all_fold_results.append(fold_s)
                print(f"    δE≤5%={fold_s['delta_e_5pct']:.1f}%, Gap%={fold_s['gap_pct']:.2f}%")

            # Average over folds
            ft_avg = {k: np.mean([f[k] for f in all_fold_results])
                      for k in all_fold_results[0] if k != 'method'}
            ft_std = {k: np.std([f[k] for f in all_fold_results])
                      for k in all_fold_results[0] if k != 'method'}
            ft_avg['method'] = 'HEC-GNN (QPU fine-tuned)'
            all_results['HEC-GNN-FT'] = ft_avg
            all_results['HEC-GNN-FT-std'] = ft_std

            print(f"\n  Fine-tuned avg: δE≤5%={ft_avg['delta_e_5pct']:.1f}±{ft_std['delta_e_5pct']:.1f}%, "
                  f"Gap%={ft_avg['gap_pct']:.2f}±{ft_std['gap_pct']:.2f}%")

    # 8. Summary table
    print("\n" + "=" * 70)
    print("QPU ADAPTATION RESULTS (600 instances)")
    print("=" * 70)
    print(f"\n{'Method':<30} | {'δE≤5%':>8} | {'δE≤2%':>8} | {'δE≤1%':>8} | {'Gap%':>8}")
    print("-" * 75)
    for name in ['Oracle', 'HEC-GNN-FT', 'HEC-GNN', 'FlatGNN', 'UTC', 'Scaled(2.0)', 'Mean']:
        if name not in all_results:
            continue
        r = all_results[name]
        std = all_results.get(f'{name}-std', {})
        if std:
            print(f"{r.get('method', name):<30} | "
                  f"{r['delta_e_5pct']:>6.1f}±{std.get('delta_e_5pct',0):.1f} | "
                  f"{r['delta_e_2pct']:>6.1f}±{std.get('delta_e_2pct',0):.1f} | "
                  f"{r['delta_e_1pct']:>6.1f}±{std.get('delta_e_1pct',0):.1f} | "
                  f"{r['gap_pct']:>6.2f}±{std.get('gap_pct',0):.2f}")
        else:
            print(f"{r.get('method', name):<30} | "
                  f"{r['delta_e_5pct']:>8.1f} | {r['delta_e_2pct']:>8.1f} | "
                  f"{r['delta_e_1pct']:>8.1f} | {r['gap_pct']:>8.2f}")

    # 9. Build plot data
    print("\n[9] Building plot data...")
    plot_data = {
        'grid': GRID_20.tolist(),
        'n_instances': len(graphs),
        'sizes': [g['n_logical'] for g in graphs],
    }

    # Per-instance QPU curves
    plot_data['qpu_curves'] = [g['energy_curve'] for g in graphs]
    plot_data['qpu_r_stars'] = [g['r_star'] for g in graphs]
    plot_data['qpu_cbr_curves'] = [g['break_curve'] for g in graphs]

    # Per-instance delta_e for each method (for histogram / CDF)
    plot_data['per_instance'] = {}

    # Baselines per-instance
    for name, deltas in baseline_results.items():
        plot_data['per_instance'][name] = deltas

    # GNN per-instance (first seed)
    if args.hec_dir and hec_results:
        plot_data['per_instance']['HEC-GNN'] = [r['delta_e'] for r in hec_results[0]]
        plot_data['hec_pred_curves'] = []
        # Re-run to get predicted curves
        mf = sorted(Path(args.hec_dir).glob('*.pt'))[0]
        ckpt = torch.load(str(mf), map_location=args.device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model = HECGNN(); model.load_state_dict(sd); model.eval()
        with torch.no_grad():
            for g in graphs:
                batch = collate_batch([g])
                batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                out = model(batch)
                pred = (out[0] if isinstance(out, tuple) else out).squeeze(0).cpu().numpy()
                plot_data['hec_pred_curves'].append(pred.tolist())

    if args.flat_dir and flat_results:
        plot_data['per_instance']['FlatGNN'] = [r['delta_e'] for r in flat_results[0]]

    # Per-size breakdown
    plot_data['per_size'] = {}
    sizes_arr = np.array([g['n_logical'] for g in graphs])
    unique_sizes = sorted(set(sizes_arr))
    for method_name, deltas in plot_data['per_instance'].items():
        deltas_arr = np.array(deltas)
        size_breakdown = {}
        for sz in unique_sizes:
            mask = sizes_arr == sz
            if mask.sum() == 0:
                continue
            d = deltas_arr[mask]
            size_breakdown[int(sz)] = {
                'delta_e_5pct': float(np.mean(d <= 0.05) * 100),
                'delta_e_2pct': float(np.mean(d <= 0.02) * 100),
                'gap_pct': float(np.mean(d) * 100),
                'n': int(mask.sum()),
            }
        plot_data['per_size'][method_name] = size_breakdown

    # 10. Save everything
    output = {
        'summary': all_results,
        'plot_data': plot_data,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
