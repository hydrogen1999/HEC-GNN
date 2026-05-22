#!/usr/bin/env python3
"""
SA Natural (mode 1) config sweep on ALL diverse test sets.
Solve LOGICAL problem directly — no embedding, no chain strength.
Configs: numsweep={1000, 10000, 100000}, num_samples={100, 1000}
Find best-known energy per instance across all configs.
Save per-instance results for p_solve computation later.
"""
import pickle, json, numpy as np, sys, glob, time, os
sys.path.insert(0, "src")
import neal, dimod

sa = neal.SimulatedAnnealingSampler()

SA_CONFIGS = [
    {'num_reads': 100, 'num_sweeps': 1000},
    {'num_reads': 100, 'num_sweeps': 10000},
    {'num_reads': 1000, 'num_sweeps': 1000},
    {'num_reads': 1000, 'num_sweeps': 10000},
]

def load_ds(path):
    chunks = []
    for c in sorted(glob.glob(f"{path}/chunk_*.pkl")):
        with open(c, "rb") as f: chunks.extend(pickle.load(f))
    return chunks

def split(data):
    n = len(data); return data[:int(n*0.70)], data[int(n*0.70):int(n*0.84)], data[int(n*0.84):]

def sa_natural_sweep(h, J, configs, seed=42):
    """Run SA natural across all configs. Return per-config results + best-known."""
    bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)
    best_energy = float('inf')
    config_results = []

    for cfg in configs:
        t0 = time.time()
        result = sa.sample(bqm, num_reads=cfg['num_reads'],
                          num_sweeps=cfg['num_sweeps'], seed=seed)
        elapsed = time.time() - t0

        energies = []
        for rec in result.record:
            for _ in range(rec.num_occurrences):
                energies.append(float(rec.energy))

        best_e = min(energies)
        mean_e = np.mean(energies)
        if best_e < best_energy:
            best_energy = best_e

        config_results.append({
            'reads': cfg['num_reads'],
            'sweeps': cfg['num_sweeps'],
            'best_energy': float(best_e),
            'mean_energy': float(mean_e),
            'time_s': round(elapsed, 3),
            'p_solve_self': sum(1 for e in energies if abs(e - best_e) < 1e-6) / len(energies),
        })

    return best_energy, config_results

def process_dataset(name, path, out_path):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    data = load_ds(path)
    _, _, test = split(data)
    print(f"  Total={len(data)}, Test={len(test)}")

    results = []
    t0 = time.time()

    for idx, inst in enumerate(test):
        h = inst['h_logical']; J = inst['J_logical']
        n = inst.get('n_logical', len(h))

        best_known, cfg_results = sa_natural_sweep(h, J, SA_CONFIGS, seed=42+idx)

        results.append({
            'idx': idx,
            'n_logical': n,
            'family': inst.get('family', ''),
            'topology': inst.get('topology', ''),
            'best_known_energy': best_known,
            'sa_oracle_energy': float(min(inst['energy_curve'])),
            'configs': cfg_results,
        })

        if (idx + 1) % 200 == 0:
            el = time.time() - t0
            eta = el / (idx+1) * (len(test) - idx - 1)
            # Incremental save
            with open(out_path, 'w') as f:
                json.dump({'dataset': name, 'n_test': len(test),
                           'sa_configs': [{'reads':c['num_reads'],'sweeps':c['num_sweeps']} for c in SA_CONFIGS],
                           'instances': results}, f)
            print(f"  {idx+1}/{len(test)} ({el:.0f}s, ETA {eta:.0f}s) saved")

    # Final save
    with open(out_path, 'w') as f:
        json.dump({'dataset': name, 'n_test': len(test),
                   'sa_configs': [{'reads':c['num_reads'],'sweeps':c['num_sweeps']} for c in SA_CONFIGS],
                   'instances': results}, f)

    el = time.time() - t0
    print(f"  DONE: {len(results)} instances in {el:.0f}s ({el/len(results):.2f}s/inst)")

    # Summary: best-known vs SA oracle
    gaps = []
    for r in results:
        sa_oracle = r['sa_oracle_energy']
        best = r['best_known_energy']
        gap = (sa_oracle - best) / max(abs(best), 1e-8) * 100
        gaps.append(gap)
    print(f"  SA oracle vs best-known: mean gap = {np.mean(gaps):.2f}%, median = {np.median(gaps):.2f}%")
    print(f"  SA oracle == best-known: {sum(1 for g in gaps if abs(g) < 0.01) / len(gaps) * 100:.1f}%")

    # Per-config summary
    print(f"\n  {'Reads':>6} {'Sweeps':>8} {'MeanTime':>9} {'AvgBest':>10} {'AvgMean':>10}")
    for ci, cfg in enumerate(SA_CONFIGS):
        times = [r['configs'][ci]['time_s'] for r in results]
        bests = [r['configs'][ci]['best_energy'] for r in results]
        means = [r['configs'][ci]['mean_energy'] for r in results]
        print(f"  {cfg['num_reads']:>6} {cfg['num_sweeps']:>8} {np.mean(times):>8.3f}s {np.mean(bests):>10.2f} {np.mean(means):>10.2f}")

# ── Main ──
os.makedirs("results/sa_natural", exist_ok=True)

DATASETS = [
    ("diverse_sa_mt", "data/diverse_sa_mt", "results/sa_natural/diverse_sa_mt.json"),
    ("diverse_boltz_mt", "data/diverse_boltz_mt", "results/sa_natural/diverse_boltz_mt.json"),
]

for name, path, out in DATASETS:
    process_dataset(name, path, out)

# OOD
print(f"\n{'='*60}")
print("OOD (n=50,75,100)")
print(f"{'='*60}")

ood_test = load_ds("data/diverse_sa_ood_test")
print(f"OOD test: {len(ood_test)}")

results_ood = []
t0 = time.time()

for idx, inst in enumerate(ood_test):
    h = inst['h_logical']; J = inst['J_logical']
    n = inst.get('n_logical', len(h))

    configs = SA_CONFIGS

    best_known, cfg_results = sa_natural_sweep(h, J, configs, seed=42+idx)

    results_ood.append({
        'idx': idx, 'n_logical': n,
        'family': inst.get('family', ''),
        'topology': inst.get('topology', ''),
        'best_known_energy': best_known,
        'sa_oracle_energy': float(min(inst['energy_curve'])),
        'configs': cfg_results,
    })

    if (idx + 1) % 200 == 0:
        el = time.time() - t0
        eta = el / (idx+1) * (len(ood_test) - idx - 1)
        with open("results/sa_natural/ood_test.json", 'w') as f:
            json.dump({'dataset': 'ood_test', 'n_test': len(ood_test),
                       'instances': results_ood}, f)
        print(f"  {idx+1}/{len(ood_test)} ({el:.0f}s, ETA {eta:.0f}s) saved")

with open("results/sa_natural/ood_test.json", 'w') as f:
    json.dump({'dataset': 'ood_test', 'n_test': len(ood_test),
               'instances': results_ood}, f)

print(f"\nOOD DONE: {len(results_ood)} instances in {time.time()-t0:.0f}s")

print(f"\n{'='*60}")
print("ALL DONE")
print(f"{'='*60}")
