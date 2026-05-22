#!/usr/bin/env python3
"""
generate_large_scale.py -- Multi-process large-scale OOD dataset generation.

Scalability dataset for n in {100, 200, 300, 500, 1000}.

P16 Pegasus (5640 qubits) embedding limits require adaptive density:
  n=100:  density 0.10-0.30  (~1500 edges, embeds in ~13s)
  n=200:  density 0.05-0.15  (~3000 edges, embeds in ~40s)
  n=300:  density 0.02-0.03  (~1300 edges, embeds in ~13s)
  n=500:  density 0.005-0.01 (~1200 edges, embeds in ~12s)
  n=1000: density 0.002-0.003(~1500 edges, embeds in ~28s)

Only random_ising family (SK is fully-connected, won't embed for n>30).
SA labeling cost reduced for larger n to keep wall time manageable.

Usage:
  # Full generation (32 workers on Apollo)
  python generate_large_scale.py --workers 28

  # Quick test
  python generate_large_scale.py --workers 4 --quick

  # Merge chunks after generation
  python generate_large_scale.py --merge
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np

# ---------------------------------------------------------------------------
# Size configurations: (n, density_range, n_instances, sa_reads, sa_sweeps, embed_timeout)
# ---------------------------------------------------------------------------
SIZE_CONFIGS = [
    # n,   density_lo, density_hi, n_instances, sa_reads, sa_sweeps, embed_timeout
    (100,  0.10, 0.30, 1000, 200, 500, 120),
    (200,  0.05, 0.15, 800,  200, 500, 180),
    (300,  0.02, 0.04, 500,  100, 300, 180),
    (500,  0.005, 0.015, 300, 100, 300, 180),
    (1000, 0.002, 0.004, 100,  50, 200, 300),
]

SIZE_CONFIGS_QUICK = [
    (100,  0.10, 0.30, 10, 50, 100, 60),
    (200,  0.05, 0.15, 8,  50, 100, 120),
    (300,  0.02, 0.04, 5,  50, 100, 120),
    (500,  0.005, 0.015, 5, 50, 100, 120),
    (1000, 0.002, 0.004, 3, 30, 100, 180),
]

SA_GRID_SIZE = 20
SA_R_MIN = 0.02
SA_R_MAX = 5.0
TOPOLOGY = 'P16'


def make_grid(K=SA_GRID_SIZE, r_min=SA_R_MIN, r_max=SA_R_MAX):
    return np.geomspace(r_min, r_max, K).tolist()


def generate_worker(args_tuple):
    """Worker: generate instances for one (size, chunk) and write to disk."""
    (chunk_id, n_logical, density_lo, density_hi, n_target,
     sa_reads, sa_sweeps, embed_timeout, grid_ratios,
     seed_base, output_dir) = args_tuple

    # Lazy imports inside worker (fork-safe)
    import dimod
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.data.generate import (
        find_embedding, build_physical_hamiltonian,
        compute_rms_J, sa_energy_curve, get_hardware_graph, build_instance,
    )

    hw_graph = get_hardware_graph(TOPOLOGY)
    instances = []
    failures = 0
    inst_id = 0
    max_failures = n_target * 10  # give up after too many failures
    t0 = time.time()

    while len(instances) < n_target and failures < max_failures:
        rng = np.random.RandomState(seed_base + inst_id)

        # Adaptive density: sample uniformly in [density_lo, density_hi]
        density = float(rng.uniform(density_lo, density_hi))

        # Generate random Ising problem
        h = {i: float(rng.uniform(-0.5, 0.5)) for i in range(n_logical)}
        J = {}
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                if rng.random() < density:
                    J[(i, j)] = float(rng.uniform(-1.0, 1.0))
        if len(J) == 0:
            inst_id += 1
            failures += 1
            continue

        # Embed
        emb = find_embedding(h, J, hw_graph,
                             seed=seed_base + inst_id,
                             timeout=embed_timeout)
        if emb is None:
            failures += 1
            inst_id += 1
            continue

        # Build physical Hamiltonian
        bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
        h_phys, J_phys, chain_edge_map, chain_edges_flat = \
            build_physical_hamiltonian(h, J, emb, hw_graph)
        rms = compute_rms_J(J)

        # SA labeling
        try:
            sa_result = sa_energy_curve(
                bqm, emb, hw_graph, grid_ratios,
                n_reads=sa_reads, sweeps=sa_sweeps,
                seed=seed_base + inst_id * 100,
            )
        except Exception as e:
            print(f"  [chunk {chunk_id}] SA failed for n={n_logical} inst {inst_id}: {e}")
            failures += 1
            inst_id += 1
            continue

        instance = build_instance(
            h, J, emb, hw_graph,
            chain_edge_map, chain_edges_flat, h_phys, J_phys,
            sa_result, 'random_ising', seed_base + inst_id,
            TOPOLOGY, n_logical,
        )
        instance['labeling'] = 'sa'
        instance['density'] = density
        instances.append(instance)

        if len(instances) % 5 == 0:
            elapsed = time.time() - t0
            rate = len(instances) / elapsed if elapsed > 0 else 0
            print(f"  [chunk {chunk_id} n={n_logical}] {len(instances)}/{n_target} "
                  f"({elapsed:.0f}s, {rate:.2f} inst/s, fails={failures})",
                  flush=True)

        inst_id += 1

    # Write chunk
    os.makedirs(output_dir, exist_ok=True)
    chunk_path = os.path.join(output_dir, f'chunk_n{n_logical}_{chunk_id:04d}.pkl')
    with open(chunk_path, 'wb') as f:
        pickle.dump(instances, f)

    elapsed = time.time() - t0
    print(f"  [chunk {chunk_id} n={n_logical}] DONE: {len(instances)} inst in {elapsed:.0f}s "
          f"(fails={failures}) -> {os.path.basename(chunk_path)}", flush=True)

    return {
        'chunk_id': chunk_id,
        'n_logical': n_logical,
        'n_instances': len(instances),
        'failures': failures,
        'elapsed': elapsed,
        'path': chunk_path,
    }


def run_generation(output_dir, n_workers, quick=False):
    configs = SIZE_CONFIGS_QUICK if quick else SIZE_CONFIGS
    grid = make_grid()

    # Build work chunks: split each size config across workers
    chunks = []
    chunk_id = 0
    for (n, d_lo, d_hi, n_inst, reads, sweeps, timeout) in configs:
        # Split instances across workers, but cap workers per size
        workers_for_size = min(n_workers, n_inst)
        per_worker = max(1, n_inst // workers_for_size)
        remainder = n_inst - per_worker * workers_for_size

        for w in range(workers_for_size):
            n_this = per_worker + (1 if w < remainder else 0)
            if n_this <= 0:
                continue
            seed_base = 42 + chunk_id * 500000
            chunks.append((
                chunk_id, n, d_lo, d_hi, n_this,
                reads, sweeps, timeout, grid,
                seed_base, output_dir,
            ))
            chunk_id += 1

    total_instances = sum(c[4] for c in chunks)
    print(f"=" * 60)
    print(f"Large-Scale OOD Dataset Generation")
    print(f"=" * 60)
    print(f"Sizes: {[c[0] for c in configs]}")
    print(f"Total target: {total_instances} instances")
    print(f"Chunks: {len(chunks)}, Workers: {n_workers}")
    print(f"Output: {output_dir}")
    print(f"{'QUICK MODE' if quick else 'FULL MODE'}")
    print(f"=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    results = []
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(generate_worker, chunks):
            results.append(result)
            done = len(results)
            total_so_far = sum(r['n_instances'] for r in results)
            elapsed = time.time() - t0
            print(f"[{done}/{len(chunks)} chunks] total={total_so_far} inst, "
                  f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total_generated = sum(r['n_instances'] for r in results)
    total_failures = sum(r['failures'] for r in results)

    # Per-size summary
    by_size = defaultdict(lambda: {'instances': 0, 'failures': 0, 'elapsed': 0})
    for r in results:
        by_size[r['n_logical']]['instances'] += r['n_instances']
        by_size[r['n_logical']]['failures'] += r['failures']
        by_size[r['n_logical']]['elapsed'] = max(by_size[r['n_logical']]['elapsed'], r['elapsed'])

    print(f"\n{'=' * 60}")
    print(f"Generation Summary")
    print(f"{'=' * 60}")
    for n in sorted(by_size.keys()):
        s = by_size[n]
        print(f"  n={n:>5d}: {s['instances']:>4d} instances, "
              f"{s['failures']:>4d} failures, {s['elapsed']:.0f}s")
    print(f"  TOTAL:  {total_generated} instances, {total_failures} failures, {elapsed:.0f}s")

    # Save stats
    stats = {
        'total_instances': total_generated,
        'total_failures': total_failures,
        'elapsed_sec': elapsed,
        'n_workers': n_workers,
        'quick': quick,
        'per_size': {str(n): dict(s) for n, s in by_size.items()},
        'configs': [
            {'n': c[0], 'density_lo': c[1], 'density_hi': c[2],
             'n_target': c[3], 'sa_reads': c[4], 'sa_sweeps': c[5]}
            for c in configs
        ],
    }
    with open(os.path.join(output_dir, 'generation_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)


def merge_chunks(output_dir):
    """Merge all chunk files into a single dataset file, split by size."""
    import glob
    chunks = sorted(glob.glob(os.path.join(output_dir, 'chunk_*.pkl')))
    print(f"Found {len(chunks)} chunk files in {output_dir}")

    all_instances = []
    for p in chunks:
        with open(p, 'rb') as f:
            data = pickle.load(f)
            all_instances.extend(data)
    print(f"Total instances: {len(all_instances)}")

    # Group by size
    by_size = defaultdict(list)
    for inst in all_instances:
        by_size[inst['n_logical']].append(inst)

    print(f"\nPer-size counts:")
    for n in sorted(by_size.keys()):
        print(f"  n={n}: {len(by_size[n])} instances")

    # Save combined test file (this is all OOD test data)
    combined_path = os.path.join(output_dir, 'large_scale_ood_test.pkl')
    with open(combined_path, 'wb') as f:
        pickle.dump(all_instances, f)
    print(f"\nCombined: {len(all_instances)} -> {combined_path}")

    # Save per-size files for granular analysis
    for n in sorted(by_size.keys()):
        path = os.path.join(output_dir, f'ood_test_n{n}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(by_size[n], f)
        print(f"  n={n}: {len(by_size[n])} -> {path}")

    # Save merge stats
    stats = {
        'total': len(all_instances),
        'per_size': {str(n): len(v) for n, v in sorted(by_size.items())},
        'sizes': sorted(by_size.keys()),
    }
    with open(os.path.join(output_dir, 'merge_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDone! Use large_scale_ood_test.pkl or per-size ood_test_n*.pkl files.")


def main():
    parser = argparse.ArgumentParser(description='Large-scale OOD dataset generation')
    parser.add_argument('--workers', type=int, default=28,
                        help='Number of parallel workers (default: 28)')
    parser.add_argument('--output-dir', default='data/large_scale_ood',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (fewer instances)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge chunk files into final dataset')
    args = parser.parse_args()

    if args.merge:
        merge_chunks(args.output_dir)
    else:
        run_generation(args.output_dir, args.workers, args.quick)


if __name__ == '__main__':
    main()
