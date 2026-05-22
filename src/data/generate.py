#!/usr/bin/env python3
"""
generate.py -- Synthetic dataset generation for the chain-strength benchmark.

Generates two benchmark suites matching Section 5 (Experimental setup):

  1. Multi-topology benchmark:
     - Sizes: n in {10, 12, 15, 20, 25, 30}
     - Topologies: P4, P8, P16 (train 70%, val 14%, test 16%), Z4 (zero-shot)
     - ~3000 instances across 3 Pegasus sizes (~1000 per topology)
     - Z4: ~500 instances held out entirely
     - Families: random_ising, sk_model

  2. Large-scale OOD benchmark:
     - Train: 2000 instances on P16, n <= 30
     - Test:  498 instances on P16, n in {50, 75, 100} (166 per size)
     - Families: random_ising, sk_model

  SA labeling: K=20 log grid, r in [0.02, 3.0], 200 reads, 500 sweeps.
  Energy curve stored per instance for energy-curve prediction training.

Usage:
  # Generate multi-topology benchmark
  python generate_v3_datasets.py --benchmark multi_topo

  # Generate OOD benchmark
  python generate_v3_datasets.py --benchmark ood

  # Generate both
  python generate_v3_datasets.py --benchmark all

  # Quick test (10x fewer instances)
  python generate_v3_datasets.py --benchmark all --quick
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import dimod
import dwave_networkx as dnx
import minorminer
from dwave.embedding import embed_bqm, unembed_sampleset

# ---------------------------------------------------------------------------
# Generation constants
# ---------------------------------------------------------------------------
SEED = 42

# SA labeling config
SA_GRID_SIZE = 20          # K = 20 grid points
SA_R_MIN = 0.02            # r_min
SA_R_MAX = 5.0             # r_max (extended from 3.0 to capture all optima)
SA_N_READS = 200           # N_SA = 200 reads
SA_SWEEPS = 500            # S_SA = 500 sweeps

# Problem families
FAMILIES = ['random_ising', 'sk_model']

# Multi-topology benchmark
MULTI_TOPO_SIZES = [10, 12, 15, 20, 25, 30]
MULTI_TOPO_PEGASUS = ['P4', 'P8', 'P16']  # train/val/test
MULTI_TOPO_ZEPHYR = ['Z4']                 # zero-shot held out
MULTI_TOPO_INSTANCES_PER_PEGASUS = 1000    # ~1000 per Pegasus topology => ~3000 total
MULTI_TOPO_INSTANCES_Z4 = 500              # Z4 held out for zero-shot eval
MULTI_TOPO_TRAIN_FRAC = 0.70
MULTI_TOPO_VAL_FRAC = 0.14
MULTI_TOPO_TEST_FRAC = 0.16

# OOD benchmark
OOD_TRAIN_SIZES = [10, 12, 15, 20, 25, 30]
OOD_TEST_SIZES = [50, 75, 100]
OOD_TRAIN_INSTANCES = 2000
OOD_TEST_PER_SIZE = 166   # 166 * 3 = 498
OOD_TOPOLOGY = 'P16'

# Topology definitions
TOPOLOGIES = {
    'P4':  {'type': 'pegasus', 'm': 4},
    'P8':  {'type': 'pegasus', 'm': 8},
    'P16': {'type': 'pegasus', 'm': 16},
    'Z4':  {'type': 'zephyr',  'm': 4},
}


# ---------------------------------------------------------------------------
# Hardware graph cache
# ---------------------------------------------------------------------------
_hw_cache = {}

def get_hardware_graph(topo_key: str) -> nx.Graph:
    if topo_key not in _hw_cache:
        spec = TOPOLOGIES[topo_key]
        if spec['type'] == 'pegasus':
            _hw_cache[topo_key] = dnx.pegasus_graph(spec['m'])
        elif spec['type'] == 'zephyr':
            _hw_cache[topo_key] = dnx.zephyr_graph(spec['m'])
        else:
            raise ValueError(f"Unknown topology type: {spec['type']}")
        print(f"  [hw] Loaded {topo_key}: {_hw_cache[topo_key].number_of_nodes()} qubits, "
              f"{_hw_cache[topo_key].number_of_edges()} edges")
    return _hw_cache[topo_key]


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------
def generate_random_ising(n, rng):
    density = float(rng.choice([0.3, 0.4, 0.5, 0.6, 0.7]))
    h = {i: float(rng.uniform(-0.5, 0.5)) for i in range(n)}
    J = {}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                J[(i, j)] = float(rng.uniform(-1.0, 1.0))
    return h, J


def generate_sk_model(n, rng):
    h = {i: 0.0 for i in range(n)}
    J = {}
    std = 1.0 / math.sqrt(n)
    for i in range(n):
        for j in range(i + 1, n):
            J[(i, j)] = float(rng.normal(0, std))
    return h, J


def generate_problem(family, n, rng):
    if family == 'random_ising':
        return generate_random_ising(n, rng)
    elif family == 'sk_model':
        return generate_sk_model(n, rng)
    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def find_embedding(h, J, hw_graph, seed, timeout=120):
    bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
    src = nx.Graph()
    src.add_nodes_from(bqm.variables)
    src.add_edges_from(bqm.quadratic.keys())
    try:
        emb = minorminer.find_embedding(src, hw_graph, random_seed=seed, timeout=timeout)
    except Exception:
        return None
    if not emb or any(len(c) == 0 for c in emb.values()):
        return None
    return emb


# ---------------------------------------------------------------------------
# Physical Hamiltonian
# ---------------------------------------------------------------------------
def build_physical_hamiltonian(h_logical, J_logical, embedding, hw_graph):
    h_phys = {}
    J_phys = {}
    chain_edge_map = {}
    chain_edges_flat = []

    for var, qubits in embedding.items():
        hi = h_logical.get(var, 0.0)
        share = hi / max(len(qubits), 1)
        for q in qubits:
            h_phys[q] = h_phys.get(q, 0.0) + share

    for var, qubits in embedding.items():
        edges = []
        for i, q1 in enumerate(qubits):
            for q2 in qubits[i + 1:]:
                if hw_graph.has_edge(q1, q2):
                    a, b = min(q1, q2), max(q1, q2)
                    edges.append((a, b))
        chain_edge_map[var] = edges
        chain_edges_flat.extend(edges)

    var_to_qubits = {var: set(qs) for var, qs in embedding.items()}
    for (i, j), Jij in J_logical.items():
        qi_set = var_to_qubits.get(i, set())
        qj_set = var_to_qubits.get(j, set())
        inter_edges = []
        for q1 in qi_set:
            for q2 in qj_set:
                if hw_graph.has_edge(q1, q2):
                    inter_edges.append((min(q1, q2), max(q1, q2)))
        if not inter_edges:
            # minorminer guarantees connectivity; skip if violated
            continue
        share = Jij / len(inter_edges)
        for (a, b) in inter_edges:
            J_phys[(a, b)] = J_phys.get((a, b), 0.0) + share

    return h_phys, J_phys, chain_edge_map, chain_edges_flat


# ---------------------------------------------------------------------------
# RMS(J)
# ---------------------------------------------------------------------------
def compute_rms_J(J_logical):
    if not J_logical:
        return 1e-8
    vals = np.array(list(J_logical.values()))
    return max(float(np.sqrt(np.mean(vals ** 2))), 1e-8)


# ---------------------------------------------------------------------------
# SA grid search — energy curve labeling
# ---------------------------------------------------------------------------
def _rescale_bqm(ebqm, alpha):
    """Rescale an embedded BQM by 1/alpha to get H_eff = H_emb / alpha.

    Returns a new BQM with all coefficients divided by alpha.
    """
    if alpha <= 1.0:
        return ebqm
    new_h = {v: bias / alpha for v, bias in ebqm.linear.items()}
    new_J = {edge: bias / alpha for edge, bias in ebqm.quadratic.items()}
    new_offset = ebqm.offset / alpha
    return dimod.BinaryQuadraticModel(new_h, new_J, new_offset, ebqm.vartype)


def sa_energy_curve(bqm, embedding, hw_graph, grid_ratios, n_reads, sweeps, seed,
                    h_max=4.0, J_max=1.0):
    """Run SA at each grid point on H_eff (rescaled), return energy curve.

    Per the paper (Section 5): SA is executed on the effective Hamiltonian
    H_eff = H_emb / alpha, where alpha is the hardware rescaling factor.
    This produces the characteristic U-shaped energy curve with both
    chain-break (low r) and signal-compression (high r) regimes.

    Returns dict with keys:
      energy_curve: [E(r_1), ..., E(r_K)]  — mean logical energy at each r
      break_curve:  [cbr(r_1), ..., cbr(r_K)]
      r_star: optimal normalized ratio
      r_star_idx: index of optimal grid point
      jc_star_raw: optimal raw chain strength (before normalization)
      per_chain_breaks_at_star: per-chain break rates at optimal r
      sweep: full sweep data
    """
    import neal
    sa = neal.SimulatedAnnealingSampler()

    rms = compute_rms_J(dict(bqm.quadratic))

    sweep = []
    for gi, r in enumerate(grid_ratios):
        cs = r * rms  # J_c = r * RMS(J)

        # Step 1: Build embedded BQM (H_emb) with chain strength cs
        ebqm = embed_bqm(bqm, embedding, hw_graph, chain_strength=cs)

        # Step 2: Compute hardware rescaling alpha(J_c)
        # alpha = max(max|h|/h_max, max(max|J_prob|, J_c)/J_max, 1)
        max_h = max((abs(v) for v in ebqm.linear.values()), default=0.0)
        max_J = max((abs(v) for v in ebqm.quadratic.values()), default=0.0)
        alpha = max(max_h / h_max, max_J / J_max, 1.0)

        # Step 3: Rescale to get H_eff = H_emb / alpha
        ebqm_eff = _rescale_bqm(ebqm, alpha)

        # Step 4: Run SA on H_eff
        raw = sa.sample(ebqm_eff, num_reads=n_reads, num_sweeps=sweeps,
                        seed=seed + gi * 1000)

        # Step 5: Unembed — logical energies come from the original BQM
        unemb = unembed_sampleset(raw, embedding, bqm)

        # Mean logical energy
        energies = []
        for rec in unemb.record:
            for _ in range(rec.num_occurrences):
                energies.append(float(rec.energy))
        mean_e = float(np.mean(energies))

        # Chain break rate + per-chain (occurrence-weighted)
        per_chain_breaks = {var: 0 for var in embedding}
        n_samples = 0
        n_multi_chains = sum(1 for c in embedding.values() if len(c) > 1)
        for s, occ in zip(raw.samples(), raw.record.num_occurrences):
            n_samples += int(occ)
            for var, chain in embedding.items():
                cl = list(chain)
                if len(cl) > 1:
                    if len(set(s.get(q, 0) for q in cl)) > 1:
                        per_chain_breaks[var] += int(occ)
        cbr = float(sum(per_chain_breaks.values()) /
                     max(n_samples * n_multi_chains, 1))

        sweep.append({
            'r': float(r),
            'cs': float(cs),
            'alpha': float(alpha),
            'mean_energy': mean_e,
            'cbr': cbr,
            'per_chain_breaks': {
                var: cnt / max(n_samples, 1)
                for var, cnt in per_chain_breaks.items()
            },
        })

    # Find optimal
    energy_curve = [s['mean_energy'] for s in sweep]
    break_curve = [s['cbr'] for s in sweep]
    best_idx = int(np.argmin(energy_curve))
    best = sweep[best_idx]

    return {
        'energy_curve': energy_curve,
        'break_curve': break_curve,
        'r_star': best['r'],
        'r_star_idx': best_idx,
        'jc_star_raw': best['cs'],
        'per_chain_breaks_at_star': best['per_chain_breaks'],
        'sweep': sweep,
    }


# ---------------------------------------------------------------------------
# Boltzmann (MCMC) grid search — energy curve labeling
# ---------------------------------------------------------------------------
def _compute_rescaling_alpha(h_phys, J_phys, J_c, h_max=4.0, J_max=1.0):
    """Hardware rescaling factor: alpha = max(max|h|/h_max, max(max|J|,Jc)/J_max, 1)."""
    max_h = max((abs(v) for v in h_phys.values()), default=0.0)
    max_J = max((abs(v) for v in J_phys.values()), default=0.0)
    return max(max_h / h_max, max(max_J, J_c) / J_max, 1.0)


def _majority_vote_unembed(sigma, embedding):
    """Majority-vote decoding: physical spins -> logical assignment."""
    logical = {}
    for var, chain in embedding.items():
        vals = [sigma.get(q, 1) for q in chain]
        logical[var] = 1 if sum(vals) >= 0 else -1
    return logical


def _logical_energy(assignment, h_logical, J_logical):
    """Compute H_L(s) = sum h_i s_i + sum J_ij s_i s_j."""
    e = 0.0
    for i, hi in h_logical.items():
        e += hi * assignment.get(i, 1)
    for (i, j), Jij in J_logical.items():
        e += Jij * assignment.get(i, 1) * assignment.get(j, 1)
    return e


def _find_cpp_boltzmann():
    """Find the C++ Boltzmann sampler binary."""
    import shutil
    candidates = [
        os.environ.get('BOLTZMANN_CPP_BIN', ''),
        os.path.join(os.path.dirname(__file__), 'boltzmann_sampler'),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def boltzmann_energy_curve_cpp(h_phys, J_phys, chain_edges_flat, embedding,
                               chain_edge_map, h_logical, J_logical, rms,
                               grid_ratios, beta=2.0, n_chains=50,
                               n_burn=500, n_sample=500, thin=5, seed=42):
    """Boltzmann MCMC via C++ binary (~50-100x faster than Python).

    The C++ binary expects:
      - h_phys, J_phys: raw (unrescaled) physical coefficients
      - utc: uniform torque compensation value
      - grid_points: UTC-relative multipliers (cs = utc * rel)
      - C++ does its own hardware rescaling internally

    We convert our RMS-normalized grid to UTC-relative:
      rel = r * RMS(J) / UTC

    One call per instance (C++ sweeps all grid points internally).
    """
    import json as _json
    import subprocess
    import tempfile

    cpp_bin = _find_cpp_boltzmann()
    if cpp_bin is None:
        raise RuntimeError("C++ Boltzmann binary not found.")

    # Compute UTC for this instance
    import dimod
    from dwave.embedding.chain_strength import uniform_torque_compensation
    bqm = dimod.BinaryQuadraticModel(h_logical, J_logical, 0.0, dimod.SPIN)
    utc = uniform_torque_compensation(bqm, embedding)
    if utc < 1e-10:
        utc = 1.0  # fallback

    # Convert RMS-normalized grid to UTC-relative: J_c = r * rms = utc * rel
    # So rel = r * rms / utc
    utc_grid = [float(r * rms / utc) for r in grid_ratios]

    # Serialize instance
    instance_data = {
        "n_qubits": len(set(h_phys.keys())),
        "h_phys": {str(k): v for k, v in h_phys.items()},
        "J_phys": {f"{a},{b}": v for (a, b), v in J_phys.items()},
        "chain_edges": [[a, b] for a, b in chain_edges_flat],
        "embedding": {str(k): list(v) for k, v in embedding.items()},
        "chain_edge_map": {
            str(k): [[a, b] for a, b in edges]
            for k, edges in (chain_edge_map or {}).items()
        },
        "h_logical": {str(k): v for k, v in h_logical.items()},
        "J_logical": {f"{a},{b}": v for (a, b), v in J_logical.items()},
        "utc": utc,
        "rms_j": rms,
        "grid_points": utc_grid,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
        _json.dump(instance_data, f_in)
        in_path = f_in.name
    out_path = in_path.replace('.json', '_out.json')

    try:
        result = subprocess.run(
            [cpp_bin, in_path, out_path,
             "--beta", str(beta),
             "--n-chains", str(n_chains),
             "--n-burn", str(n_burn),
             "--n-sample", str(n_sample),
             "--thin", str(thin),
             "--seed", str(seed)],
            capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"C++ exit {result.returncode}: {result.stderr[:200]}")

        with open(out_path) as f_out:
            cpp_out = _json.load(f_out)

        # Parse sweep: C++ returns sweep with 'rel', 'cs', 'mean_energy', 'cbr', 'per_chain_breaks'
        cpp_sweep = cpp_out.get('sweep', [])
        if len(cpp_sweep) != len(grid_ratios):
            raise RuntimeError(f"C++ returned {len(cpp_sweep)} sweep points, expected {len(grid_ratios)}")

        sweep = []
        for gi, (r, s) in enumerate(zip(grid_ratios, cpp_sweep)):
            pcb = s.get('per_chain_breaks', {})
            pcb = {int(k): v for k, v in pcb.items()}
            sweep.append({
                'r': float(r),
                'cs': float(r * rms),
                'mean_energy': float(s['mean_energy']),
                'cbr': float(s.get('cbr', 0.0)),
                'per_chain_breaks': {var: pcb.get(var, 0.0) for var in embedding},
            })

    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)

    energy_curve = [s['mean_energy'] for s in sweep]
    break_curve = [s['cbr'] for s in sweep]
    best_idx = int(np.argmin(energy_curve))
    best = sweep[best_idx]

    return {
        'energy_curve': energy_curve,
        'break_curve': break_curve,
        'r_star': best['r'],
        'r_star_idx': best_idx,
        'jc_star_raw': best['cs'],
        'per_chain_breaks_at_star': best['per_chain_breaks'],
        'sweep': sweep,
    }


def boltzmann_energy_curve(h_phys, J_phys, chain_edges_flat, embedding,
                           h_logical, J_logical, rms, grid_ratios,
                           beta=2.0, n_chains=50, n_burn=500, n_sample=500,
                           thin=5, seed=42):
    """Boltzmann MCMC labeling at fixed inverse temperature beta.

    Tries C++ binary first (50-100x faster), falls back to Python MCMC.
    """
    # Try C++ first
    cpp_bin = _find_cpp_boltzmann()
    if cpp_bin:
        try:
            result = boltzmann_energy_curve_cpp(
                h_phys, J_phys, chain_edges_flat, embedding, None,
                h_logical, J_logical, rms, grid_ratios,
                beta=beta, n_chains=n_chains, n_burn=n_burn,
                n_sample=n_sample, thin=thin, seed=seed)
            # Verify result is not all zeros (C++ interface mismatch)
            if any(abs(e) > 1e-10 for e in result['energy_curve']):
                return result
            print("  [boltzmann] C++ returned all-zero curves, falling back to Python")
        except Exception as e:
            print(f"  [boltzmann] C++ failed ({e}), falling back to Python")

    # Python fallback
    rng_master = np.random.RandomState(seed)

    # Build qubit list and adjacency once
    all_qubits = sorted(set(h_phys.keys()))
    qubit_idx = {q: i for i, q in enumerate(all_qubits)}
    n_qubits = len(all_qubits)

    if n_qubits == 0:
        return {
            'energy_curve': [0.0] * len(grid_ratios),
            'break_curve': [0.0] * len(grid_ratios),
            'r_star': grid_ratios[0], 'r_star_idx': 0,
            'jc_star_raw': grid_ratios[0] * rms,
            'per_chain_breaks_at_star': {},
            'sweep': [],
        }

    # Pre-build inter-chain adjacency (coupling values change with rescaling)
    prob_edges = []
    for (a, b), val in J_phys.items():
        ia, ib = qubit_idx.get(a), qubit_idx.get(b)
        if ia is not None and ib is not None:
            prob_edges.append((ia, ib, val))

    chain_edge_idx = []
    for (a, b) in chain_edges_flat:
        ia, ib = qubit_idx.get(a), qubit_idx.get(b)
        if ia is not None and ib is not None:
            chain_edge_idx.append((ia, ib))

    h_arr = np.array([h_phys.get(q, 0.0) for q in all_qubits])

    sweep = []
    for gi, r in enumerate(grid_ratios):
        cs = r * rms
        alpha = _compute_rescaling_alpha(h_phys, J_phys, cs)

        # Effective fields and couplings after rescaling
        eff_h = h_arr / alpha
        eff_prob = [(ia, ib, val / alpha) for ia, ib, val in prob_edges]
        eff_chain_J = cs / alpha  # ferromagnetic: energy = -J_c * s_a * s_b

        # Build neighbor coupling array for fast MCMC
        # neighbor_coupling[i] = list of (j, coupling_value)
        neighbor_coupling = [[] for _ in range(n_qubits)]
        for ia, ib, val in eff_prob:
            neighbor_coupling[ia].append((ib, val))
            neighbor_coupling[ib].append((ia, val))
        for ia, ib in chain_edge_idx:
            neighbor_coupling[ia].append((ib, -eff_chain_J))
            neighbor_coupling[ib].append((ia, -eff_chain_J))

        # Run MCMC chains
        energies = []
        per_chain_breaks = {var: 0 for var in embedding}
        total_samples = 0

        for c in range(n_chains):
            rng = np.random.RandomState(rng_master.randint(0, 2**31))
            spins = rng.choice([-1, 1], size=n_qubits).astype(np.int8)

            # Burn-in
            for _ in range(n_burn):
                for qi in range(n_qubits):
                    # ΔE = -2 s_i (h_i + Σ_j J_ij s_j) for H = Σ h s + Σ J s s
                    dE = -2.0 * spins[qi] * eff_h[qi]
                    for qj, Jval in neighbor_coupling[qi]:
                        dE += -2.0 * Jval * spins[qi] * spins[qj]
                    if dE <= 0 or rng.random() < np.exp(-beta * dE):
                        spins[qi] *= -1

            # Sampling
            for sweep_idx in range(n_sample):
                for qi in range(n_qubits):
                    dE = -2.0 * spins[qi] * eff_h[qi]
                    for qj, Jval in neighbor_coupling[qi]:
                        dE += -2.0 * Jval * spins[qi] * spins[qj]
                    if dE <= 0 or rng.random() < np.exp(-beta * dE):
                        spins[qi] *= -1

                if sweep_idx % thin == 0:
                    # Decode and compute logical energy
                    sigma = {all_qubits[i]: int(spins[i]) for i in range(n_qubits)}
                    assignment = _majority_vote_unembed(sigma, embedding)
                    e = _logical_energy(assignment, h_logical, J_logical)
                    energies.append(e)
                    total_samples += 1

                    # Check chain breaks
                    for var, chain in embedding.items():
                        if len(chain) > 1:
                            vals = set(sigma.get(q, 1) for q in chain)
                            if len(vals) > 1:
                                per_chain_breaks[var] += 1

        mean_e = float(np.mean(energies)) if energies else 0.0
        cbr_total = sum(per_chain_breaks.values())
        cbr_possible = total_samples * sum(1 for c in embedding.values() if len(c) > 1)
        cbr = float(cbr_total / max(cbr_possible, 1))

        sweep.append({
            'r': float(r),
            'cs': float(cs),
            'mean_energy': mean_e,
            'cbr': cbr,
            'per_chain_breaks': {
                var: cnt / max(total_samples, 1)
                for var, cnt in per_chain_breaks.items()
            },
        })

    energy_curve = [s['mean_energy'] for s in sweep]
    break_curve = [s['cbr'] for s in sweep]
    best_idx = int(np.argmin(energy_curve))
    best = sweep[best_idx]

    return {
        'energy_curve': energy_curve,
        'break_curve': break_curve,
        'r_star': best['r'],
        'r_star_idx': best_idx,
        'jc_star_raw': best['cs'],
        'per_chain_breaks_at_star': best['per_chain_breaks'],
        'sweep': sweep,
    }


# ---------------------------------------------------------------------------
# Feature computation (7-dim node features, scale-invariant)
# ---------------------------------------------------------------------------
def compute_qubit_features(embedding, h_phys, J_phys, chain_edge_map, rms_J):
    norm = max(rms_J, 1e-8)
    qubit_to_var = {}
    for var, qubits in embedding.items():
        for q in qubits:
            qubit_to_var[q] = var

    chain_neighbors = {}
    for var, edges in chain_edge_map.items():
        for (a, b) in edges:
            chain_neighbors[a] = chain_neighbors.get(a, 0) + 1
            chain_neighbors[b] = chain_neighbors.get(b, 0) + 1

    inter_degree = {}
    delta_prob = {}
    for (a, b), Jpq in J_phys.items():
        va, vb = qubit_to_var.get(a), qubit_to_var.get(b)
        if va is not None and vb is not None and va != vb:
            inter_degree[a] = inter_degree.get(a, 0) + 1
            inter_degree[b] = inter_degree.get(b, 0) + 1
            delta_prob[a] = delta_prob.get(a, 0.0) + abs(Jpq)
            delta_prob[b] = delta_prob.get(b, 0.0) + abs(Jpq)

    qubit_list = []
    chain_assignment = []
    features = []
    chain_idx = 0
    for var in sorted(embedding.keys()):
        qubits = embedding[var]
        n_i = len(qubits)
        for q in qubits:
            qubit_list.append(q)
            chain_assignment.append(chain_idx)
            h_q = h_phys.get(q, 0.0)
            deg_c = chain_neighbors.get(q, 0)
            deg_x = inter_degree.get(q, 0)
            dp = delta_prob.get(q, 0.0) + abs(h_q)
            features.append([
                h_q / norm,
                abs(h_q) / norm,
                float(deg_c),
                float(deg_x),
                float(n_i),
                dp / norm,
                1.0 if n_i == 1 else 0.0,  # singleton indicator
            ])
        chain_idx += 1
    return features, qubit_list, chain_assignment


def compute_logical_edge_features(J_logical, embedding, J_phys, hw_graph, rms_J):
    norm = max(rms_J, 1e-8)
    sorted_vars = sorted(embedding.keys())
    var_to_idx = {v: i for i, v in enumerate(sorted_vars)}
    edge_list = []
    edge_features = []

    for (i, j), Jij in J_logical.items():
        qi_set = set(embedding.get(i, []))
        qj_set = set(embedding.get(j, []))
        inter_J_vals = []
        for (a, b), Jpq in J_phys.items():
            if (a in qi_set and b in qj_set) or (a in qj_set and b in qi_set):
                inter_J_vals.append(abs(Jpq))
        n_inter = max(len(inter_J_vals), 1)
        if not inter_J_vals:
            inter_J_vals = [abs(Jij)]
        mu_load = float(np.mean(inter_J_vals))
        sigma_load = float(np.std(inter_J_vals)) if len(inter_J_vals) > 1 else 0.0

        idx_i, idx_j = var_to_idx[i], var_to_idx[j]
        feat = [Jij / norm, abs(Jij) / norm, float(n_inter),
                mu_load / norm, sigma_load / norm]
        edge_list.append((idx_i, idx_j))
        edge_list.append((idx_j, idx_i))
        edge_features.append(feat)
        edge_features.append(list(feat))

    return edge_list, edge_features


def compute_chain_edge_features(chain_edge_map, qubit_to_local_idx):
    """Intra-chain edges with 3-dim structural indicator [1, 0, 0]."""
    pairs = []
    feats = []
    for var, edges in chain_edge_map.items():
        for (a, b) in edges:
            ia = qubit_to_local_idx.get(a)
            ib = qubit_to_local_idx.get(b)
            if ia is not None and ib is not None:
                pairs.append((ia, ib))
                pairs.append((ib, ia))
                feats.append([1.0, 0.0, 0.0])
                feats.append([1.0, 0.0, 0.0])
    return pairs, feats


def compute_inter_chain_edges(embedding, J_phys, qubit_to_local_idx, rms_J):
    """Compute qubit-level inter-chain edges for Stage 2.

    For each physical inter-chain edge (p,q) where p in C_i, q in C_j, i != j,
    produces edge features [0, J_pq/RMS(J), |J_pq|/RMS(J)] (3-dim).

    Returns:
        edge_list: list of (local_idx_p, local_idx_q) pairs (bidirectional)
        edge_features: list of 3-dim feature vectors
    """
    norm = max(rms_J, 1e-8)
    qubit_to_var = {}
    for var, qubits in embedding.items():
        for q in qubits:
            qubit_to_var[q] = var

    edge_list = []
    edge_features = []
    for (a, b), Jpq in J_phys.items():
        va, vb = qubit_to_var.get(a), qubit_to_var.get(b)
        if va is not None and vb is not None and va != vb:
            ia = qubit_to_local_idx.get(a)
            ib = qubit_to_local_idx.get(b)
            if ia is not None and ib is not None:
                feat = [0.0, Jpq / norm, abs(Jpq) / norm]
                edge_list.append((ia, ib))
                edge_list.append((ib, ia))
                edge_features.append(feat)
                edge_features.append(feat[:])  # copy
    return edge_list, edge_features


# ---------------------------------------------------------------------------
# Build a complete instance
# ---------------------------------------------------------------------------
def build_instance(h_logical, J_logical, embedding, hw_graph,
                   chain_edge_map, chain_edges_flat, h_phys, J_phys,
                   sa_result, family, instance_id, topology, n_logical):
    rms = compute_rms_J(J_logical)
    feats, qubit_list, chain_assign = compute_qubit_features(
        embedding, h_phys, J_phys, chain_edge_map, rms)
    qubit_to_local = {q: i for i, q in enumerate(qubit_list)}
    chain_ei, chain_ef = compute_chain_edge_features(chain_edge_map, qubit_to_local)
    inter_ei, inter_ef = compute_inter_chain_edges(embedding, J_phys, qubit_to_local, rms)
    logical_ei, logical_ef = compute_logical_edge_features(
        J_logical, embedding, J_phys, hw_graph, rms)

    sorted_vars = sorted(embedding.keys())
    cbr_at_star = sa_result['per_chain_breaks_at_star']
    chain_break_targets = [cbr_at_star.get(v, 0.0) for v in sorted_vars]

    return {
        # GNN inputs
        'qubit_features': feats,
        'chain_assignment': chain_assign,
        'chain_edge_index': chain_ei,
        'chain_edge_features': chain_ef,
        'inter_edge_index': inter_ei,        # qubit-level inter-chain (Stage 2)
        'inter_edge_features': inter_ef,     # [0, J_pq/RMS, |J_pq|/RMS]
        'logical_edge_index': logical_ei,
        'logical_edge_features': logical_ef,
        'n_chains': len(sorted_vars),
        # Energy curve target (K=20 values)
        'energy_curve': sa_result['energy_curve'],
        'break_curve': sa_result['break_curve'],
        # Scalar targets
        'r_star': sa_result['r_star'],
        'r_star_idx': sa_result['r_star_idx'],
        'jc_star_raw': sa_result['jc_star_raw'],
        'rms_J': rms,
        'chain_break_targets': chain_break_targets,
        # Metadata
        'family': family,
        'instance_id': instance_id,
        'topology': topology,
        'n_logical': n_logical,
        # Raw data for evaluation and baselines (UTC, LinearReg)
        'h_logical': h_logical,
        'J_logical': J_logical,
        'embedding': embedding,
        'h_phys': h_phys,
        'J_phys': J_phys,
        'chain_edge_map': chain_edge_map,
    }


# ---------------------------------------------------------------------------
# Grid points: K=20 log grid in [0.02, 3.0]
# ---------------------------------------------------------------------------
def make_grid(K=SA_GRID_SIZE, r_min=SA_R_MIN, r_max=SA_R_MAX):
    return np.geomspace(r_min, r_max, K).tolist()


# ---------------------------------------------------------------------------
# Generate instances for a given topology
# ---------------------------------------------------------------------------
def generate_for_topology(topo_key, n_instances, sizes, families, seed,
                          grid_ratios, labeling='sa',
                          n_reads=SA_N_READS, sweeps=SA_SWEEPS,
                          boltzmann_beta=2.0, boltzmann_n_chains=50,
                          boltzmann_n_burn=500, boltzmann_n_sample=500,
                          boltzmann_thin=5,
                          embed_timeout=120):
    """Generate labeled instances for a single topology.

    Args:
        labeling: 'sa' for SA labels, 'boltzmann' for Boltzmann MCMC labels,
                  'both' for dual-labeled (same instances, two energy curves).
    """
    hw_graph = get_hardware_graph(topo_key)
    n_cells = len(families) * len(sizes)
    # Use ceiling division so we never undershoot the target
    instances_per_cell = max(1, -(-n_instances // n_cells))  # ceil division
    actual_target = n_instances  # we track against the original request

    dataset = []
    embed_failures = 0
    inst_id = 0
    max_retries = 5  # retries per failed instance
    t0 = time.time()

    for family in families:
        for size in sizes:
            cell_count = 0
            cell_target = instances_per_cell
            while cell_count < cell_target:
                rng = np.random.RandomState(seed + inst_id)

                h, J = generate_problem(family, size, rng)
                if len(J) == 0:
                    inst_id += 1
                    embed_failures += 1
                    if embed_failures > actual_target * 2:
                        break  # safety valve
                    continue

                emb = find_embedding(h, J, hw_graph,
                                     seed=seed + inst_id,
                                     timeout=embed_timeout)
                if emb is None:
                    embed_failures += 1
                    inst_id += 1
                    if embed_failures > actual_target * 2:
                        break
                    continue

                bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
                h_phys, J_phys, chain_edge_map, chain_edges_flat = \
                    build_physical_hamiltonian(h, J, emb, hw_graph)
                rms = compute_rms_J(J)

                # --- SA labeling ---
                if labeling in ('sa', 'both'):
                    sa_result = sa_energy_curve(
                        bqm, emb, hw_graph, grid_ratios,
                        n_reads=n_reads, sweeps=sweeps,
                        seed=seed + inst_id * 100,
                    )
                else:
                    sa_result = None

                # --- Boltzmann labeling ---
                if labeling in ('boltzmann', 'both'):
                    boltz_result = boltzmann_energy_curve(
                        h_phys, J_phys, chain_edges_flat, emb,
                        h, J, rms, grid_ratios,
                        beta=boltzmann_beta,
                        n_chains=boltzmann_n_chains,
                        n_burn=boltzmann_n_burn,
                        n_sample=boltzmann_n_sample,
                        thin=boltzmann_thin,
                        seed=seed + inst_id * 100 + 50000,
                    )
                else:
                    boltz_result = None

                # Use SA as primary if available, else Boltzmann
                primary_result = sa_result if sa_result else boltz_result

                instance = build_instance(
                    h, J, emb, hw_graph,
                    chain_edge_map, chain_edges_flat, h_phys, J_phys,
                    primary_result, family, inst_id, topo_key, size,
                )
                instance['labeling'] = labeling

                # Store both curves when dual-labeled
                if labeling == 'both':
                    instance['sa_energy_curve'] = sa_result['energy_curve']
                    instance['sa_break_curve'] = sa_result['break_curve']
                    instance['sa_r_star'] = sa_result['r_star']
                    instance['boltzmann_energy_curve'] = boltz_result['energy_curve']
                    instance['boltzmann_break_curve'] = boltz_result['break_curve']
                    instance['boltzmann_r_star'] = boltz_result['r_star']
                elif labeling == 'boltzmann':
                    instance['boltzmann_beta'] = boltzmann_beta

                dataset.append(instance)
                cell_count += 1

                if len(dataset) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = len(dataset) / elapsed if elapsed > 0 else 0
                    print(f"  [{topo_key}] {len(dataset)}/{actual_target} "
                          f"({elapsed:.0f}s, {rate:.1f} inst/s, "
                          f"failures={embed_failures})")

                inst_id += 1

    # Trim to exact target (ceiling division may overshoot slightly)
    if len(dataset) > n_instances:
        dataset = dataset[:n_instances]

    elapsed = time.time() - t0
    if len(dataset) < n_instances:
        print(f"  [{topo_key}] WARNING: only generated {len(dataset)}/{n_instances} "
              f"instances ({embed_failures} failures). Consider increasing retry budget.")
    print(f"  [{topo_key}] Done: {len(dataset)} instances in {elapsed:.0f}s "
          f"(embed failures: {embed_failures})")
    return dataset


# ---------------------------------------------------------------------------
# Split with stratification by family
# ---------------------------------------------------------------------------
def stratified_split(dataset, train_frac, val_frac, seed=SEED):
    rng = np.random.RandomState(seed)
    by_family = defaultdict(list)
    for inst in dataset:
        by_family[inst['family']].append(inst)

    train, val, test = [], [], []
    for fam, instances in by_family.items():
        rng.shuffle(instances)
        n = len(instances)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(instances[:n_train])
        val.extend(instances[n_train:n_train + n_val])
        test.extend(instances[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Benchmark 1: Multi-topology
# ---------------------------------------------------------------------------
def generate_multi_topology(output_dir, quick=False, labeling='sa'):
    print("=" * 60)
    print(f"Benchmark 1: Multi-topology (labeling={labeling})")
    print("=" * 60)

    grid = make_grid()
    scale = 10 if quick else 1
    n_pegasus = MULTI_TOPO_INSTANCES_PER_PEGASUS // scale
    n_z4 = MULTI_TOPO_INSTANCES_Z4 // scale

    # Generate per topology
    pegasus_data = []
    for topo in MULTI_TOPO_PEGASUS:
        data = generate_for_topology(
            topo, n_pegasus, MULTI_TOPO_SIZES, FAMILIES,
            seed=SEED, grid_ratios=grid, labeling=labeling,
        )
        pegasus_data.extend(data)

    z4_data = generate_for_topology(
        'Z4', n_z4, MULTI_TOPO_SIZES, FAMILIES,
        seed=SEED + 100000, grid_ratios=grid, labeling=labeling,
    )

    # Split Pegasus data into train/val/test
    train, val, test = stratified_split(
        pegasus_data, MULTI_TOPO_TRAIN_FRAC, MULTI_TOPO_VAL_FRAC)

    # Stats
    stats = {
        'benchmark': 'multi_topology',
        'pegasus_total': len(pegasus_data),
        'z4_total': len(z4_data),
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'per_topology': {},
        'grid': {'K': SA_GRID_SIZE, 'r_min': SA_R_MIN, 'r_max': SA_R_MAX},
        'sa': {'n_reads': SA_N_READS, 'sweeps': SA_SWEEPS},
    }
    for topo in MULTI_TOPO_PEGASUS + MULTI_TOPO_ZEPHYR:
        if topo in MULTI_TOPO_PEGASUS:
            subset = [d for d in pegasus_data if d['topology'] == topo]
        else:
            subset = z4_data
        stats['per_topology'][topo] = {
            'count': len(subset),
            'per_family': {f: sum(1 for d in subset if d['family'] == f) for f in FAMILIES},
            'per_size': {str(s): sum(1 for d in subset if d['n_logical'] == s) for s in MULTI_TOPO_SIZES},
        }

    print(f"\nMulti-topology stats:")
    print(f"  Pegasus (P4+P8+P16): {len(pegasus_data)} instances")
    print(f"  Zephyr Z4 (held out): {len(z4_data)} instances")
    print(f"  Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'multi_topo_train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(output_dir, 'multi_topo_val.pkl'), 'wb') as f:
        pickle.dump(val, f)
    with open(os.path.join(output_dir, 'multi_topo_test.pkl'), 'wb') as f:
        pickle.dump(test, f)
    with open(os.path.join(output_dir, 'multi_topo_z4.pkl'), 'wb') as f:
        pickle.dump(z4_data, f)
    with open(os.path.join(output_dir, 'multi_topo_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# Benchmark 2: Large-scale OOD
# ---------------------------------------------------------------------------
def generate_ood(output_dir, quick=False, labeling='sa'):
    print("=" * 60)
    print(f"Benchmark 2: Large-scale OOD (labeling={labeling})")
    print("=" * 60)

    grid = make_grid()
    scale = 10 if quick else 1

    # Generate enough so that after 85/15 split, train has ~2000 instances
    # 2000 / 0.85 ≈ 2353
    n_generate = int(math.ceil(OOD_TRAIN_INSTANCES / 0.85)) // scale
    print(f"\nGenerating OOD pool: {n_generate} instances on {OOD_TOPOLOGY}, "
          f"sizes={OOD_TRAIN_SIZES} (will split to ~{OOD_TRAIN_INSTANCES // scale} train)")
    train_data = generate_for_topology(
        OOD_TOPOLOGY, n_generate, OOD_TRAIN_SIZES, FAMILIES,
        seed=SEED + 200000, grid_ratios=grid, labeling=labeling,
    )

    # Test: 498 instances on P16, n in {50, 75, 100}
    n_test_per = OOD_TEST_PER_SIZE // scale
    test_data = []
    for size in OOD_TEST_SIZES:
        print(f"\nGenerating OOD test: {n_test_per} instances on {OOD_TOPOLOGY}, n={size}")
        data = generate_for_topology(
            OOD_TOPOLOGY, n_test_per, [size], FAMILIES,
            seed=SEED + 300000 + size * 1000, grid_ratios=grid,
            labeling=labeling,
            embed_timeout=180,  # larger instances need more time
        )
        test_data.extend(data)

    # Split train into train/val (85/15); merge remainder into train
    train_split, val_split, remainder = stratified_split(train_data, 0.85, 0.15)
    train_split.extend(remainder)  # don't lose truncation leftovers

    stats = {
        'benchmark': 'ood',
        'topology': OOD_TOPOLOGY,
        'train_total': len(train_data),
        'train_split': len(train_split),
        'val_split': len(val_split),
        'test_total': len(test_data),
        'train_sizes': OOD_TRAIN_SIZES,
        'test_sizes': OOD_TEST_SIZES,
        'test_per_size': {str(s): sum(1 for d in test_data if d['n_logical'] == s)
                          for s in OOD_TEST_SIZES},
        'grid': {'K': SA_GRID_SIZE, 'r_min': SA_R_MIN, 'r_max': SA_R_MAX},
        'sa': {'n_reads': SA_N_READS, 'sweeps': SA_SWEEPS},
    }

    print(f"\nOOD stats:")
    print(f"  Train: {len(train_split)} (+ {len(val_split)} val)")
    size_counts = ', '.join(
        f'n={s}: {stats["test_per_size"][str(s)]}' for s in OOD_TEST_SIZES)
    print(f"  Test: {len(test_data)} ({size_counts})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ood_train.pkl'), 'wb') as f:
        pickle.dump(train_split, f)
    with open(os.path.join(output_dir, 'ood_val.pkl'), 'wb') as f:
        pickle.dump(val_split, f)
    with open(os.path.join(output_dir, 'ood_test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join(output_dir, 'ood_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='V3 dataset generation')
    parser.add_argument('--benchmark', choices=['multi_topo', 'ood', 'all'],
                        default='all')
    parser.add_argument('--output-dir', default='datasets_v3')
    parser.add_argument('--quick', action='store_true',
                        help='10x fewer instances for testing')
    parser.add_argument('--labeling', choices=['sa', 'boltzmann', 'both'],
                        default='sa',
                        help='Labeling mode: sa (default), boltzmann, or both')
    args = parser.parse_args()

    output_dir = args.output_dir
    print(f"Output directory: {output_dir}")
    print(f"Grid: K={SA_GRID_SIZE}, r=[{SA_R_MIN}, {SA_R_MAX}]")
    print(f"Labeling: {args.labeling}")
    if args.labeling in ('sa', 'both'):
        print(f"  SA: {SA_N_READS} reads, {SA_SWEEPS} sweeps")
    if args.labeling in ('boltzmann', 'both'):
        print(f"  Boltzmann: beta=2.0, 50 chains, 500 burn, 500 sample")
    if args.quick:
        print("*** QUICK MODE: 10x fewer instances ***")

    all_stats = {}
    t0 = time.time()

    if args.benchmark in ('multi_topo', 'all'):
        stats = generate_multi_topology(output_dir, quick=args.quick,
                                        labeling=args.labeling)
        all_stats['multi_topology'] = stats

    if args.benchmark in ('ood', 'all'):
        stats = generate_ood(output_dir, quick=args.quick,
                             labeling=args.labeling)
        all_stats['ood'] = stats

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All done in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    with open(os.path.join(output_dir, 'generation_summary.json'), 'w') as f:
        json.dump({
            'elapsed_sec': elapsed,
            'benchmarks': all_stats,
            'quick_mode': args.quick,
        }, f, indent=2)


if __name__ == '__main__':
    main()
