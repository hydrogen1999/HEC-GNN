#!/usr/bin/env python3
"""
Step 1: Compute TRUE optimal energy for ALL test instances using GUROBI/ExactSolver.
Save results so p_solve can be computed later for any baseline.

Output per instance: {instance_id, n, family, topology, true_optimal_energy, solver_used, gap}
"""
import pickle, json, numpy as np, math, sys, glob, time, os
sys.path.insert(0, "src")
import dimod

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

def find_true_optimal(h, J, n, time_limit=60):
    """Find true optimal of LOGICAL Ising instance."""
    bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)

    # ExactSolver for n <= 20
    if n <= 20:
        solver = dimod.ExactSolver()
        result = solver.sample(bqm)
        return float(result.first.energy), 'exact', 0.0

    # GUROBI for n > 20
    if HAS_GUROBI:
        try:
            model = gp.Model("Ising")
            model.setParam("TimeLimit", time_limit)
            model.setParam("Threads", 8)
            model.setParam("OutputFlag", 0)

            x = {var: model.addVar(vtype=GRB.BINARY, name=f"x_{var}") for var in bqm.variables}
            obj = gp.QuadExpr()
            offset = 0.0
            for var, hi in bqm.linear.items():
                obj += 2 * hi * x[var]; offset -= hi
            for (i, j), Jij in bqm.quadratic.items():
                obj += 4 * Jij * x[i] * x[j]
                obj -= 2 * Jij * x[i]; obj -= 2 * Jij * x[j]; offset += Jij
            model.setObjective(obj + offset, GRB.MINIMIZE)
            model.optimize()

            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                opt_e = model.objVal
                gap = 0.0
                if model.status == GRB.TIME_LIMIT and model.objBound is not None:
                    gap = abs(opt_e - model.objBound) / max(abs(opt_e), 1e-8)
                return float(opt_e), 'gurobi', float(gap)
        except Exception as e:
            print(f"    GUROBI fail n={n}: {str(e)[:80]}")

    # Fallback: SA with many sweeps
    import neal
    sa = neal.SimulatedAnnealingSampler()
    result = sa.sample(bqm, num_reads=1000, num_sweeps=10000, seed=42)
    return float(result.first.energy), 'sa_fallback', -1.0

def load_ds(path):
    chunks = []
    for c in sorted(glob.glob(f"{path}/chunk_*.pkl")):
        with open(c, "rb") as f: chunks.extend(pickle.load(f))
    return chunks

def split(data):
    n = len(data); return data[:int(n*0.70)], data[int(n*0.70):int(n*0.84)], data[int(n*0.84):]

def process_dataset(name, path, out_path, time_limit=60):
    print(f"\n{'='*60}")
    print(f"Computing TRUE optimal: {name}")
    print(f"{'='*60}")

    data = load_ds(path)
    _, _, test = split(data)
    print(f"  Total={len(data)}, Test={len(test)}")

    results = []
    t0 = time.time()
    n_exact = 0; n_gurobi = 0; n_fallback = 0

    for idx, inst in enumerate(test):
        h = inst['h_logical']; J = inst['J_logical']
        n = inst.get('n_logical', len(h))

        opt_e, method, gap = find_true_optimal(h, J, n, time_limit)

        if method == 'exact': n_exact += 1
        elif method == 'gurobi': n_gurobi += 1
        else: n_fallback += 1

        results.append({
            'idx': idx,
            'n_logical': n,
            'family': inst.get('family', ''),
            'topology': inst.get('topology', ''),
            'true_optimal': opt_e,
            'solver': method,
            'gap': gap,
            'sa_oracle_energy': float(min(inst['energy_curve'])),
        })

        if (idx + 1) % 500 == 0:
            el = time.time() - t0
            eta = el / (idx+1) * (len(test) - idx - 1)
            print(f"  {idx+1}/{len(test)} ({el:.0f}s, ETA {eta:.0f}s) exact={n_exact} gurobi={n_gurobi} fb={n_fallback}")

        # Incremental save every 2000
        if (idx + 1) % 2000 == 0:
            with open(out_path, 'w') as f:
                json.dump({'dataset': name, 'n_test': len(test),
                           'n_exact': n_exact, 'n_gurobi': n_gurobi, 'n_fallback': n_fallback,
                           'instances': results}, f)

    # Final save
    with open(out_path, 'w') as f:
        json.dump({'dataset': name, 'n_test': len(test),
                   'n_exact': n_exact, 'n_gurobi': n_gurobi, 'n_fallback': n_fallback,
                   'instances': results}, f)

    el = time.time() - t0
    print(f"  DONE: {len(results)} instances, exact={n_exact} gurobi={n_gurobi} fb={n_fallback}")
    print(f"  Time: {el:.0f}s ({el/len(results):.2f}s/inst)")
    print(f"  Saved: {out_path}")

    # Summary: SA oracle vs true optimal
    gaps = [r['sa_oracle_energy'] - r['true_optimal'] for r in results]
    gaps_pct = [abs(g) / max(abs(r['true_optimal']), 1e-8) * 100 for g, r in zip(gaps, results)]
    print(f"  SA oracle vs true optimal: mean gap = {np.mean(gaps_pct):.2f}%")
    print(f"  SA oracle == true optimal: {sum(1 for g in gaps_pct if g < 0.01) / len(gaps_pct) * 100:.1f}%")

# ── Main ──
os.makedirs("results/true_optimal", exist_ok=True)
print(f"GUROBI available: {HAS_GUROBI}")

DATASETS = [
    ("diverse_sa_mt", "data/diverse_sa_mt", "results/true_optimal/diverse_sa_mt.json", 30),
    ("diverse_boltz_mt", "data/diverse_boltz_mt", "results/true_optimal/diverse_boltz_mt.json", 30),
]

for name, path, out, tl in DATASETS:
    process_dataset(name, path, out, tl)

# OOD (larger instances, need more time)
print(f"\n{'='*60}")
print("OOD instances (n=50,75,100)")
print(f"{'='*60}")

ood_test = load_ds("data/diverse_sa_ood_test")
print(f"OOD test: {len(ood_test)} instances")

results_ood = []
t0 = time.time()
n_exact = 0; n_gurobi = 0; n_fallback = 0

for idx, inst in enumerate(ood_test):
    h = inst['h_logical']; J = inst['J_logical']
    n = inst.get('n_logical', len(h))

    # OOD instances larger → more GUROBI time
    tl = 60 if n <= 50 else 120 if n <= 75 else 300
    opt_e, method, gap = find_true_optimal(h, J, n, tl)

    if method == 'exact': n_exact += 1
    elif method == 'gurobi': n_gurobi += 1
    else: n_fallback += 1

    results_ood.append({
        'idx': idx, 'n_logical': n,
        'family': inst.get('family', ''), 'topology': inst.get('topology', ''),
        'true_optimal': opt_e, 'solver': method, 'gap': gap,
        'sa_oracle_energy': float(min(inst['energy_curve'])),
    })

    if (idx + 1) % 200 == 0:
        el = time.time() - t0
        eta = el / (idx+1) * (len(ood_test) - idx - 1)
        print(f"  {idx+1}/{len(ood_test)} ({el:.0f}s, ETA {eta:.0f}s) exact={n_exact} gurobi={n_gurobi} fb={n_fallback}")

    # Incremental save
    if (idx + 1) % 1000 == 0:
        with open("results/true_optimal/ood_test.json", 'w') as f:
            json.dump({'dataset': 'ood_test', 'n_test': len(ood_test),
                       'n_exact': n_exact, 'n_gurobi': n_gurobi, 'n_fallback': n_fallback,
                       'instances': results_ood}, f)

with open("results/true_optimal/ood_test.json", 'w') as f:
    json.dump({'dataset': 'ood_test', 'n_test': len(ood_test),
               'n_exact': n_exact, 'n_gurobi': n_gurobi, 'n_fallback': n_fallback,
               'instances': results_ood}, f)

el = time.time() - t0
print(f"  OOD DONE: {len(results_ood)} inst, exact={n_exact} gurobi={n_gurobi} fb={n_fallback}")
print(f"  Time: {el:.0f}s")

print(f"\n{'='*60}")
print("ALL DONE")
print(f"{'='*60}")
