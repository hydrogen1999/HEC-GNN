#!/usr/bin/env python3
"""
p_solve evaluation against the true logical optimum.

1. Find TRUE optimal energy for each logical instance:
   - n ≤ 20: dimod.ExactSolver (exact, feasible)
   - n > 20: GUROBI with time limit (or SA natural as fallback)
2. SA natural (mode 1): solve logical problem directly (no embedding)
3. SA embedded (mode 2): at each method's predicted r*
4. p_solve = P(SA reads finding energy == true_optimal)
5. Optimality gap = |E_found - E_optimal| / |E_optimal|
"""
import pickle, json, numpy as np, math, sys, glob, time, os, torch
sys.path.insert(0, "src")

import neal, dimod
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_strength import uniform_torque_compensation
import dwave_networkx as dnx
from data.dataset import collate_batch
from models.hec_gnn import HECGNN
from models.baselines import LinearRegBaseline, build_flat_gnn

grid = np.logspace(np.log10(0.02), np.log10(5.0), 20)
sa_sampler = neal.SimulatedAnnealingSampler()
tc = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_topo(n):
    if n not in tc:
        m = {'P4':('pegasus',4),'P8':('pegasus',8),'P16':('pegasus',16),'Z4':('zephyr',4)}
        t,s = m[n]; tc[n] = dnx.pegasus_graph(s) if t=='pegasus' else dnx.zephyr_graph(s)
    return tc[n]

def compute_tts(p_solve, total_time, confidence=0.99):
    """TTS in seconds. total_time = time for one SA run."""
    if p_solve <= 1e-10: return float('inf')
    if p_solve >= 1 - 1e-10: return total_time
    return math.ceil(math.log(1 - confidence) / math.log(1 - p_solve)) * total_time

# ── Step 1: Find TRUE optimal ──

def find_true_optimal(h, J, n, time_limit=60):
    """Find true optimal energy for logical Ising instance.
    n <= 20: ExactSolver (exact)
    n > 20: try GUROBI, fallback to SA natural with many sweeps
    Returns: (optimal_energy, method_used, optimality_gap)
    """
    bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)

    # ExactSolver for small instances
    if n <= 20:
        solver = dimod.ExactSolver()
        result = solver.sample(bqm)
        opt_e = result.first.energy
        return float(opt_e), 'exact', 0.0

    # Try GUROBI
    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("Ising")
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", 8)
        model.setParam("OutputFlag", 0)

        # Convert Ising to QUBO: s_i = 2*x_i - 1
        # H = sum h_i s_i + sum J_ij s_i s_j
        # = sum h_i (2x_i-1) + sum J_ij (2x_i-1)(2x_j-1)
        x = {}
        for var in bqm.variables:
            x[var] = model.addVar(vtype=GRB.BINARY, name=f"x_{var}")

        obj = gp.QuadExpr()
        offset = 0.0
        # Linear: h_i * (2x_i - 1) = 2*h_i*x_i - h_i
        for var, hi in bqm.linear.items():
            obj += 2 * hi * x[var]
            offset -= hi
        # Quadratic: J_ij * (2x_i-1)(2x_j-1) = 4*J_ij*x_i*x_j - 2*J_ij*x_i - 2*J_ij*x_j + J_ij
        for (i, j), Jij in bqm.quadratic.items():
            obj += 4 * Jij * x[i] * x[j]
            obj -= 2 * Jij * x[i]
            obj -= 2 * Jij * x[j]
            offset += Jij

        model.setObjective(obj + offset, GRB.MINIMIZE)
        model.optimize()

        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            opt_e = model.objVal
            gap = 0.0
            if model.status == GRB.TIME_LIMIT and model.objBound is not None:
                gap = abs(opt_e - model.objBound) / max(abs(opt_e), 1e-8)
            return float(opt_e), 'gurobi', float(gap)
        else:
            raise RuntimeError(f"Gurobi status {model.status}")

    except (ImportError, RuntimeError) as e:
        # Fallback: SA natural with lots of sweeps
        result = sa_sampler.sample(bqm, num_reads=1000, num_sweeps=10000, seed=42)
        opt_e = result.first.energy
        return float(opt_e), 'sa_natural_fallback', -1.0  # gap unknown


# ── Step 2: SA natural (mode 1) ──

def sa_natural(h, J, num_reads=200, num_sweeps=1000):
    """SA solves logical problem directly. No embedding, no chain strength."""
    bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)
    result = sa_sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=42)
    energies = []
    for rec in result.record:
        for _ in range(rec.num_occurrences):
            energies.append(float(rec.energy))
    return energies, float(result.first.energy)


# ── Step 3: SA embedded (mode 2) ──

def sa_embedded(inst, r_val, num_reads=200, num_sweeps=500):
    """SA solves embedded problem at given chain strength."""
    h = inst.get("h_logical", {}); J = inst.get("J_logical", {})
    emb = inst.get("embedding", {}); rms = inst.get("rms_J", 1.0)
    topo = inst.get("topology", "P16"); hw = get_topo(topo)
    if not emb: return None, None

    bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)
    Jc = r_val * rms
    try: ebqm = embed_bqm(bqm, emb, hw, chain_strength=Jc)
    except: return None, None

    mh = max((abs(v) for v in ebqm.linear.values()), default=0.)
    mj = max((abs(v) for v in ebqm.quadratic.values()), default=0.)
    a = max(mh/4, mj/1, 1)
    if a > 1:
        ebqm = dimod.BinaryQuadraticModel(
            {v:b/a for v,b in ebqm.linear.items()},
            {e:b/a for e,b in ebqm.quadratic.items()},
            ebqm.offset/a, ebqm.vartype)

    raw = sa_sampler.sample(ebqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=42)
    unemb = unembed_sampleset(raw, emb, bqm)

    energies = []
    for rec in unemb.record:
        for _ in range(rec.num_occurrences):
            energies.append(float(rec.energy))
    return energies, float(unemb.first.energy)


# ── Step 4: Compute metrics ──

def compute_metrics(energies, true_optimal, total_time_s):
    """Compute p_solve, TTS, optimality gap against TRUE optimal."""
    if not energies:
        return {'p_solve': 0, 'tts_s': float('inf'), 'gap_pct': 100, 'best_found': float('inf')}

    best_found = min(energies)
    mean_e = np.mean(energies)
    n_reads = len(energies)

    # p_solve: fraction finding TRUE optimal
    p_solve = sum(1 for e in energies if abs(e - true_optimal) < 1e-6) / n_reads

    # TTS
    tts = compute_tts(p_solve, total_time_s)

    # Optimality gap (best found vs true optimal)
    gap_pct = abs(best_found - true_optimal) / max(abs(true_optimal), 1e-8) * 100

    # Mean gap
    mean_gap_pct = abs(mean_e - true_optimal) / max(abs(true_optimal), 1e-8) * 100

    return {
        'p_solve': float(p_solve),
        'tts_s': float(tts),
        'gap_pct': float(gap_pct),
        'mean_gap_pct': float(mean_gap_pct),
        'best_found': float(best_found),
    }


def utc_r(inst):
    J = inst.get("J_logical", {}); h = inst.get("h_logical", {}); rms = inst.get("rms_J", 1.0)
    if not J: return 2.0
    return uniform_torque_compensation(dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)) / max(rms, 1e-8)

def fs(ec, k):
    idx = np.linspace(0, 19, k, dtype=int)
    return idx[np.argmin(ec[idx])]

def load_model(cls_fn, path):
    try:
        m = cls_fn().to(device)
        ck = torch.load(path, map_location=device, weights_only=False)
        sd = ck.get("model_state_dict", ck)
        m.load_state_dict(sd); m.eval(); return m
    except:
        try:
            m = cls_fn().to(device)
            ck = torch.load(path, map_location=device, weights_only=False)
            m.load_state_dict(ck); m.eval(); return m
        except: return None


# ── Main ──

print(f"Device: {device}")
os.makedirs("results/true_optimal", exist_ok=True)

# Load data
print("Loading diverse_sa_mt...")
chunks = []
for c in sorted(glob.glob("data/diverse_sa_mt/chunk_*.pkl")):
    with open(c, "rb") as f: chunks.extend(pickle.load(f))
n_total = len(chunks)
train = chunks[:int(n_total * 0.70)]
test = chunks[int(n_total * 0.84):]
print(f"Train={len(train)} Test={len(test)}")

# Load models
hec = load_model(HECGNN, "results/diverse_run/hec_gnn_mt/hec_gnn_seed42.pt")
flat = load_model(build_flat_gnn, "results/diverse_run/flat_gnn_mt/flat_gnn_seed42.pt")
lr = LinearRegBaseline(); lr.fit(train[:10000])
mean_r = np.mean([i['r_star'] for i in train[:5000]])
if hec: print("  HEC-GNN OK")
if flat: print("  FlatGNN OK")

# Evaluate on 500 instances
rng = np.random.RandomState(42)
sub_idx = rng.choice(len(test), min(500, len(test)), replace=False)
subset = [test[i] for i in sub_idx]

SA_EMBEDDED_TIME = 0.033  # seconds per SA embedded run (200 reads × 500 sweeps)
SA_NATURAL_TIME = 0.033   # seconds per SA natural run

methods = ['SA_natural', 'Oracle', 'HEC-GNN', 'FlatGNN', 'LinearReg', 'UTC',
           'Scaled(2.0)', 'r=min', 'r=max', 'Mean', 'Few-Shot-10']
results = {m: [] for m in methods}

t0 = time.time()
n_exact = 0; n_gurobi = 0; n_fallback = 0

for idx, inst in enumerate(subset):
    h = inst['h_logical']; J = inst['J_logical']
    n = inst.get('n_logical', len(h))
    ec = np.array(inst['energy_curve'])
    oi = int(np.argmin(ec)); rms = inst.get('rms_J', 1.0)

    # Step 1: Find TRUE optimal
    true_opt, opt_method, opt_gap = find_true_optimal(h, J, n, time_limit=30)
    if opt_method == 'exact': n_exact += 1
    elif opt_method == 'gurobi': n_gurobi += 1
    else: n_fallback += 1

    # Step 2: SA natural (solve logical directly)
    nat_energies, nat_best = sa_natural(h, J, num_reads=200, num_sweeps=1000)
    results['SA_natural'].append(compute_metrics(nat_energies, true_opt, SA_NATURAL_TIME))

    # Step 3: Predict r* for each method
    preds = {
        'Oracle': oi,
        'UTC': int(np.argmin(np.abs(grid - utc_r(inst)))),
        'Scaled(2.0)': int(np.argmin(np.abs(grid - 2.0))),
        'r=min': 0,       # grid[0] = 0.02, weakest chain strength
        'r=max': 19,      # grid[19] = 5.0, strongest chain strength
        'Mean': int(np.argmin(np.abs(grid - mean_r))),
        'LinearReg': int(np.argmin(np.abs(grid - lr.predict_r(inst)))),
        'Few-Shot-10': fs(ec, 10),
    }
    if hec:
        b = collate_batch([inst])
        b = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b.items()}
        with torch.no_grad(): p, _, _ = hec(b)
        preds['HEC-GNN'] = int(np.argmin(p.squeeze(0).cpu().numpy()))
    if flat:
        b = collate_batch([inst])
        b = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b.items()}
        with torch.no_grad(): p, _, _ = flat(b)
        preds['FlatGNN'] = int(np.argmin(p.squeeze(0).cpu().numpy()))

    # Step 4: SA embedded at each predicted r*
    for name, pidx in preds.items():
        emb_energies, emb_best = sa_embedded(inst, grid[pidx])
        if emb_energies is None: continue
        results[name].append(compute_metrics(emb_energies, true_opt, SA_EMBEDDED_TIME))

    if (idx + 1) % 50 == 0:
        el = time.time() - t0
        print(f"  {idx+1}/{len(subset)} ({el:.0f}s) exact={n_exact} gurobi={n_gurobi} fallback={n_fallback}")

# ── Print results ──
print(f"\n{'='*70}")
print(f"RESULTS: p_solve vs TRUE OPTIMAL ({len(subset)} instances)")
print(f"Optimal found by: exact={n_exact}, gurobi={n_gurobi}, fallback={n_fallback}")
print(f"{'='*70}")
print(f"\n{'Method':<15} {'p_solve':>8} {'TTS(s)':>8} {'Gap%':>7} {'MeanGap%':>9} {'BestGap%':>9}")
print("-" * 58)

summary = {}
for m in methods:
    r = results[m]
    if not r: continue
    ps = np.mean([x['p_solve'] for x in r])
    tts_v = [x['tts_s'] for x in r if x['tts_s'] < 1e8]
    tts = np.median(tts_v) if tts_v else float('inf')
    gap = np.mean([x['gap_pct'] for x in r])
    mgap = np.mean([x['mean_gap_pct'] for x in r])
    print(f"{m:<15} {ps:>8.4f} {tts:>8.3f} {gap:>6.2f}% {mgap:>8.2f}%")
    summary[m] = {
        'n': len(r), 'p_solve': round(ps, 4),
        'tts_median_s': round(tts, 3),
        'best_gap_pct': round(gap, 2),
        'mean_gap_pct': round(mgap, 2),
    }

with open("results/true_optimal/results.json", "w") as f:
    json.dump({
        'n_instances': len(subset),
        'n_exact': n_exact, 'n_gurobi': n_gurobi, 'n_fallback': n_fallback,
        'methods': summary,
    }, f, indent=2)
print(f"\nSaved results/true_optimal/results.json")
print(f"Total time: {time.time()-t0:.0f}s")
