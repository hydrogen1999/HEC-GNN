#!/usr/bin/env python3
"""
parallel_generate.py -- Multi-process dataset generation with INCREMENTAL writes.

Each worker writes its chunk immediately to disk as a separate .pkl file.
Main process just dispatches chunks and monitors — no collection bottleneck.
Partial results survive crashes.

Usage:
  python -m src.data.parallel_generate \
      --benchmark multi_topo --n-instances 100000 --labeling sa \
      --workers 20 --output-dir data/sa_mt_100k

  # Merge chunk files into standard train/val/test/z4 split
  python -m src.data.parallel_generate --merge --output-dir data/sa_mt_100k
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np


def generate_chunk(args_tuple):
    """Worker: generate a chunk of instances and WRITE to disk immediately."""
    (chunk_id, n_instances, families, sizes, topo_key, labeling,
     grid_ratios, seed_offset, sa_reads, sa_sweeps,
     boltzmann_beta, boltzmann_n_chains, boltzmann_n_burn,
     boltzmann_n_sample, boltzmann_thin, embed_timeout,
     output_dir) = args_tuple

    import dimod
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data.generate import (
        generate_problem, find_embedding, build_physical_hamiltonian,
        compute_rms_J, sa_energy_curve, boltzmann_energy_curve,
        get_hardware_graph, build_instance,
    )

    hw_graph = get_hardware_graph(topo_key)
    instances_per_cell = max(1, n_instances // (len(families) * len(sizes)))
    chunk_data = []
    failures = 0
    inst_id = 0
    seed_base = seed_offset + chunk_id * 100000

    for family in families:
        for size in sizes:
            cell_count = 0
            while cell_count < instances_per_cell:
                rng = np.random.RandomState(seed_base + inst_id)
                h, J = generate_problem(family, size, rng)
                if len(J) == 0:
                    inst_id += 1
                    failures += 1
                    if failures > n_instances * 3:
                        break
                    continue

                emb = find_embedding(h, J, hw_graph,
                                     seed=seed_base + inst_id,
                                     timeout=embed_timeout)
                if emb is None:
                    failures += 1
                    inst_id += 1
                    if failures > n_instances * 3:
                        break
                    continue

                bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
                h_phys, J_phys, chain_edge_map, chain_edges_flat = \
                    build_physical_hamiltonian(h, J, emb, hw_graph)
                rms = compute_rms_J(J)

                if labeling == 'sa':
                    result = sa_energy_curve(
                        bqm, emb, hw_graph, grid_ratios,
                        n_reads=sa_reads, sweeps=sa_sweeps,
                        seed=seed_base + inst_id * 100)
                elif labeling == 'boltzmann':
                    result = boltzmann_energy_curve(
                        h_phys, J_phys, chain_edges_flat, emb,
                        h, J, rms, grid_ratios,
                        beta=boltzmann_beta,
                        n_chains=boltzmann_n_chains,
                        n_burn=boltzmann_n_burn,
                        n_sample=boltzmann_n_sample,
                        thin=boltzmann_thin,
                        seed=seed_base + inst_id * 100 + 50000)
                else:
                    raise ValueError(f"Unknown labeling: {labeling}")

                instance = build_instance(
                    h, J, emb, hw_graph,
                    chain_edge_map, chain_edges_flat, h_phys, J_phys,
                    result, family, seed_base + inst_id, topo_key, size)
                instance['labeling'] = labeling

                chunk_data.append(instance)
                cell_count += 1
                inst_id += 1

    # Write chunk immediately
    os.makedirs(output_dir, exist_ok=True)
    chunk_path = os.path.join(output_dir, f'chunk_{chunk_id:06d}.pkl')
    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk_data, f)

    return chunk_id, len(chunk_data), failures, chunk_path


def run_parallel(benchmark, n_instances, labeling, output_dir,
                 n_workers, n_shards=1, shard_id=0, seed=42):
    """Generate dataset using multiprocessing with incremental writes."""
    from src.data.generate import (make_grid, FAMILIES, MULTI_TOPO_SIZES,
                                    OOD_TRAIN_SIZES, OOD_TEST_SIZES,
                                    SA_N_READS, SA_SWEEPS)

    grid = make_grid()

    if benchmark == 'multi_topo':
        topologies = ['P4', 'P8', 'P16']
        sizes = MULTI_TOPO_SIZES
        z4_count = n_instances // 6
        per_topo = (n_instances - z4_count) // len(topologies)
        topo_work = [(t, per_topo, sizes) for t in topologies]
        topo_work.append(('Z4', z4_count, sizes))
    elif benchmark == 'ood_train':
        sizes = OOD_TRAIN_SIZES
        topo_work = [('P16', n_instances, sizes)]
    elif benchmark == 'ood_test':
        sizes = OOD_TEST_SIZES
        topo_work = [('P16', n_instances, sizes)]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    total_work = sum(c for _, c, _ in topo_work)
    shard_size = math.ceil(total_work / n_shards)
    my_count = total_work // n_shards
    if shard_id == n_shards - 1:
        my_count = total_work - my_count * (n_shards - 1)

    print(f"Benchmark: {benchmark}, labeling: {labeling}")
    print(f"Shard {shard_id}/{n_shards}: generating ~{my_count} instances")
    print(f"Workers: {n_workers}")
    print(f"Output dir: {output_dir}")

    # Build chunks (one per worker per topology)
    chunks = []
    chunk_id = shard_id * 10000  # avoid chunk_id collisions across shards
    for topo, count, szs in topo_work:
        my_topo_count = math.ceil(count / n_shards)
        if shard_id == n_shards - 1:
            my_topo_count = count - math.ceil(count / n_shards) * (n_shards - 1)
            my_topo_count = max(0, my_topo_count)

        per_worker = max(1, my_topo_count // n_workers)
        for w in range(n_workers):
            n = per_worker
            if w == n_workers - 1:
                n = my_topo_count - per_worker * (n_workers - 1)
            if n <= 0:
                continue
            chunks.append((
                chunk_id, n, FAMILIES, szs, topo, labeling,
                grid, seed + shard_id * 1000000 + chunk_id * 10000,
                SA_N_READS, SA_SWEEPS,
                2.0, 50, 500, 500, 5,
                120,
                output_dir,
            ))
            chunk_id += 1

    print(f"Dispatching {len(chunks)} chunks to {n_workers} workers...")
    print(f"Each chunk writes to: {output_dir}/chunk_*.pkl (incremental)")
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    completed = 0
    total_failures = 0
    total_instances = 0

    with Pool(processes=n_workers) as pool:
        # imap_unordered gives us results as they complete
        for result in pool.imap_unordered(generate_chunk, chunks):
            chunk_id, n_inst, failures, path = result
            completed += 1
            total_failures += failures
            total_instances += n_inst
            elapsed = time.time() - t0
            rate = total_instances / elapsed if elapsed > 0 else 0
            remaining_chunks = len(chunks) - completed
            eta = (remaining_chunks * elapsed / completed) if completed > 0 else 0
            print(f"  [{completed}/{len(chunks)}] chunk {chunk_id}: {n_inst} inst -> {os.path.basename(path)} "
                  f"({rate:.1f} inst/s, ETA {eta/60:.0f}min)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nGenerated {total_instances} instances in {elapsed:.0f}s "
          f"({total_failures} failures)")

    # Save summary
    with open(os.path.join(output_dir, 'generation_stats.json'), 'w') as f:
        json.dump({
            'benchmark': benchmark,
            'labeling': labeling,
            'n_instances': total_instances,
            'n_failures': total_failures,
            'elapsed_sec': elapsed,
            'n_workers': n_workers,
            'shard_id': shard_id,
            'n_shards': n_shards,
        }, f, indent=2)


def merge_chunks(output_dir, benchmark='multi_topo', seed=42):
    """Merge all chunk_*.pkl files into train/val/test splits."""
    import glob
    chunks = sorted(glob.glob(os.path.join(output_dir, 'chunk_*.pkl')))
    print(f"Found {len(chunks)} chunks in {output_dir}")

    all_instances = []
    for p in chunks:
        with open(p, 'rb') as f:
            all_instances.extend(pickle.load(f))

    print(f"Total instances: {len(all_instances)}")

    if benchmark == 'multi_topo':
        pegasus = [i for i in all_instances if i['topology'] != 'Z4']
        z4 = [i for i in all_instances if i['topology'] == 'Z4']

        rng = np.random.RandomState(seed)
        rng.shuffle(pegasus)
        n_train = int(len(pegasus) * 0.70)
        n_val = int(len(pegasus) * 0.14)
        train = pegasus[:n_train]
        val = pegasus[n_train:n_train + n_val]
        test = pegasus[n_train + n_val:]

        for name, data in [('multi_topo_train', train), ('multi_topo_val', val),
                           ('multi_topo_test', test), ('multi_topo_z4', z4)]:
            path = os.path.join(output_dir, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  {name}: {len(data)} -> {path}")

    elif benchmark == 'ood_train':
        rng = np.random.RandomState(seed)
        rng.shuffle(all_instances)
        n_train = int(len(all_instances) * 0.85)
        train = all_instances[:n_train]
        val = all_instances[n_train:]
        for name, data in [('ood_train', train), ('ood_val', val)]:
            path = os.path.join(output_dir, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  {name}: {len(data)} -> {path}")

    elif benchmark == 'ood_test':
        path = os.path.join(output_dir, 'ood_test.pkl')
        with open(path, 'wb') as f:
            pickle.dump(all_instances, f)
        print(f"  ood_test: {len(all_instances)} -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['multi_topo', 'ood_train', 'ood_test'])
    parser.add_argument('--n-instances', type=int, default=100000)
    parser.add_argument('--labeling', choices=['sa', 'boltzmann'], default='sa')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--merge', action='store_true',
                        help='Merge chunks into train/val/test splits')
    args = parser.parse_args()

    if args.merge:
        merge_chunks(args.output_dir, args.benchmark or 'multi_topo', args.seed)
    else:
        run_parallel(
            benchmark=args.benchmark,
            n_instances=args.n_instances,
            labeling=args.labeling,
            output_dir=args.output_dir,
            n_workers=args.workers,
            n_shards=args.n_shards,
            shard_id=args.shard_id,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
