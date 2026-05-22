#!/usr/bin/env python3
"""
gpu_sampler.py -- GPU-accelerated SA and Boltzmann samplers via PyTorch.

Replaces CPU-bound neal SA with batched GPU computation:
- Vectorize across N_reads (200) × K grid points (20) = 4000 parallel chains
- Each sweep updates all spins sequentially but across all chains in parallel
- ~50-200x faster than CPU neal on a single GPU

Usage:
    from src.data.gpu_sampler import gpu_sa_energy_curve, gpu_boltzmann_energy_curve
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def _build_graph_tensors(h_phys, J_phys, chain_edges_flat, embedding,
                         rms, grid_ratios, h_max=4.0, J_max=1.0, device='cuda'):
    """Pre-compute graph structure tensors for GPU sampling.

    Returns tensors that represent the embedded Hamiltonian for each grid point,
    with hardware rescaling applied.
    """
    all_qubits = sorted(set(h_phys.keys()))
    qubit_idx = {q: i for i, q in enumerate(all_qubits)}
    n_q = len(all_qubits)
    K = len(grid_ratios)

    # Build adjacency: for each qubit, list of (neighbor_idx, coupling_value)
    # We need separate prob and chain edges for rescaling
    h_arr = np.array([h_phys.get(q, 0.0) for q in all_qubits], dtype=np.float32)

    # Problem edges
    prob_src, prob_dst, prob_val = [], [], []
    for (a, b), val in J_phys.items():
        ia, ib = qubit_idx.get(a), qubit_idx.get(b)
        if ia is not None and ib is not None:
            prob_src.extend([ia, ib])
            prob_dst.extend([ib, ia])
            prob_val.extend([val, val])

    # Chain edges
    chain_src, chain_dst = [], []
    for (a, b) in chain_edges_flat:
        ia, ib = qubit_idx.get(a), qubit_idx.get(b)
        if ia is not None and ib is not None:
            chain_src.extend([ia, ib])
            chain_dst.extend([ib, ia])

    # For each grid point, compute rescaled H_eff
    # H_eff = (1/alpha) * [h_prob + J_prob - J_c * chain]
    # Store: eff_h[k, q], eff_J_prob[k, e], eff_chain_J[k]
    eff_h = np.zeros((K, n_q), dtype=np.float32)
    eff_prob_val = np.zeros((K, len(prob_val)), dtype=np.float32)
    eff_chain_J = np.zeros(K, dtype=np.float32)

    for ki, r in enumerate(grid_ratios):
        cs = r * rms
        max_h = max((abs(v) for v in h_phys.values()), default=0.0)
        max_J_prob = max((abs(v) for v in J_phys.values()), default=0.0)
        alpha = max(max_h / h_max, max(max_J_prob, cs) / J_max, 1.0)

        eff_h[ki] = h_arr / alpha
        eff_prob_val[ki] = np.array(prob_val, dtype=np.float32) / alpha
        eff_chain_J[ki] = cs / alpha

    return {
        'n_q': n_q,
        'K': K,
        'eff_h': torch.tensor(eff_h, device=device),                    # [K, n_q]
        'prob_src': torch.tensor(prob_src, dtype=torch.long, device=device),
        'prob_dst': torch.tensor(prob_dst, dtype=torch.long, device=device),
        'eff_prob_val': torch.tensor(eff_prob_val, device=device),       # [K, n_prob_edges]
        'chain_src': torch.tensor(chain_src, dtype=torch.long, device=device),
        'chain_dst': torch.tensor(chain_dst, dtype=torch.long, device=device),
        'eff_chain_J': torch.tensor(eff_chain_J, device=device),         # [K]
        'embedding': embedding,
        'all_qubits': all_qubits,
        'qubit_idx': qubit_idx,
    }


def _compute_delta_E_batched(spins, qi, graph, beta_schedule_k=None):
    """Compute ΔE for flipping qubit qi across all [K, N_reads] chains.

    spins: [K, N_reads, n_q] int8
    Returns: dE [K, N_reads]
    """
    K, N, n_q = spins.shape
    s_i = spins[:, :, qi].float()  # [K, N]

    # Field contribution: -2 * s_i * h_i
    dE = -2.0 * s_i * graph['eff_h'][:, qi].unsqueeze(1)  # [K, N]

    # Problem coupling contribution
    prob_src = graph['prob_src']
    prob_dst = graph['prob_dst']
    # Find edges where qi is the source
    mask = prob_src == qi
    if mask.any():
        neighbors = prob_dst[mask]  # neighbor indices
        J_vals = graph['eff_prob_val'][:, mask]  # [K, n_neighbors]
        s_neighbors = spins[:, :, neighbors].float()  # [K, N, n_neighbors]
        # dE += -2 * J * s_i * s_j for each neighbor
        contrib = -2.0 * J_vals.unsqueeze(1) * s_i.unsqueeze(2) * s_neighbors  # [K, N, n_neighbors]
        dE += contrib.sum(dim=2)

    # Chain coupling contribution: -(-J_c) * s_i * s_j = J_c * s_i * s_j
    # Wait: chain coupling in H is -J_c * s_a * s_b, so J_ij = -J_c
    # dE = -2 * (-J_c) * s_i * s_j = 2 * J_c * s_i * s_j
    chain_src = graph['chain_src']
    chain_dst = graph['chain_dst']
    mask_c = chain_src == qi
    if mask_c.any():
        c_neighbors = chain_dst[mask_c]
        s_c_neighbors = spins[:, :, c_neighbors].float()  # [K, N, n_chain_neighbors]
        eff_cJ = graph['eff_chain_J'].unsqueeze(1).unsqueeze(2)  # [K, 1, 1]
        # dE += -2 * (-eff_chain_J) * s_i * s_j = 2 * eff_chain_J * s_i * s_j
        contrib_c = 2.0 * eff_cJ * s_i.unsqueeze(2) * s_c_neighbors
        dE += contrib_c.sum(dim=2)

    return dE


@torch.no_grad()
def gpu_sa_energy_curve(h_phys, J_phys, chain_edges_flat, embedding,
                        h_logical, J_logical, rms, grid_ratios,
                        n_reads=200, n_sweeps=500, seed=42,
                        h_max=4.0, J_max=1.0, device=None):
    """GPU-accelerated Simulated Annealing for energy curve labeling.

    Runs K × N_reads SA chains in parallel on GPU.
    Each chain anneals from high T to low T over n_sweeps.

    Returns same structure as sa_energy_curve().
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    graph = _build_graph_tensors(
        h_phys, J_phys, chain_edges_flat, embedding,
        rms, grid_ratios, h_max, J_max, device)

    n_q = graph['n_q']
    K = graph['K']
    N = n_reads

    # Initialize random spins: [K, N, n_q]
    torch.manual_seed(seed)
    spins = (torch.randint(0, 2, (K, N, n_q), device=device) * 2 - 1).to(torch.int8)

    # SA beta schedule: geometric from beta_min to beta_max
    # Neal default: beta_range based on energy scale
    h_scale = graph['eff_h'].abs().max().item()
    J_scale = max(graph['eff_prob_val'].abs().max().item(),
                  graph['eff_chain_J'].abs().max().item())
    max_coupling = max(h_scale, J_scale, 0.01)
    beta_min = 0.1 / max_coupling
    beta_max = 3.0 / max_coupling
    betas = torch.logspace(
        math.log10(beta_min), math.log10(beta_max), n_sweeps, device=device)

    # SA sweeps
    for sweep_idx in range(n_sweeps):
        beta = betas[sweep_idx]
        for qi in range(n_q):
            dE = _compute_delta_E_batched(spins, qi, graph)  # [K, N]
            # Metropolis acceptance
            accept_prob = torch.exp(-beta * dE).clamp(max=1.0)
            accept = torch.rand(K, N, device=device) < accept_prob
            # Also accept if dE <= 0
            accept = accept | (dE <= 0)
            # Flip accepted spins
            flip_mask = accept.unsqueeze(2)  # [K, N, 1]
            qi_slice = spins[:, :, qi:qi+1]
            spins[:, :, qi:qi+1] = torch.where(
                flip_mask, -qi_slice, qi_slice)

    # Decode: majority vote per chain -> logical assignment -> logical energy
    all_qubits = graph['all_qubits']
    qubit_idx = graph['qubit_idx']
    spins_np = spins.cpu().numpy()  # [K, N, n_q]

    energy_curve = []
    break_curve = []
    all_per_chain_breaks = []

    for ki in range(K):
        energies = []
        per_chain_breaks = {var: 0 for var in embedding}
        n_broken_total = 0
        n_chain_checks = 0

        for ri in range(N):
            sigma = {all_qubits[i]: int(spins_np[ki, ri, i])
                     for i in range(n_q)}

            # Majority vote decode
            logical = {}
            for var, chain in embedding.items():
                vals = [sigma.get(q, 1) for q in chain]
                logical[var] = 1 if sum(vals) >= 0 else -1
                # Check chain break
                if len(chain) > 1:
                    if len(set(sigma.get(q, 1) for q in chain)) > 1:
                        per_chain_breaks[var] += 1
                        n_broken_total += 1
                    n_chain_checks += 1

            # Logical energy
            e = 0.0
            for i, hi in h_logical.items():
                e += hi * logical.get(i, 1)
            for (i, j), Jij in J_logical.items():
                e += Jij * logical.get(i, 1) * logical.get(j, 1)
            energies.append(e)

        energy_curve.append(float(np.mean(energies)))
        cbr = n_broken_total / max(n_chain_checks, 1)
        break_curve.append(cbr)
        all_per_chain_breaks.append(
            {var: cnt / max(N, 1) for var, cnt in per_chain_breaks.items()})

    best_idx = int(np.argmin(energy_curve))
    return {
        'energy_curve': energy_curve,
        'break_curve': break_curve,
        'r_star': float(grid_ratios[best_idx]),
        'r_star_idx': best_idx,
        'jc_star_raw': float(grid_ratios[best_idx] * rms),
        'per_chain_breaks_at_star': all_per_chain_breaks[best_idx],
        'sweep': [{'r': float(grid_ratios[ki]),
                   'cs': float(grid_ratios[ki] * rms),
                   'mean_energy': energy_curve[ki],
                   'cbr': break_curve[ki],
                   'per_chain_breaks': all_per_chain_breaks[ki]}
                  for ki in range(K)],
    }


@torch.no_grad()
def gpu_boltzmann_energy_curve(h_phys, J_phys, chain_edges_flat, embedding,
                               h_logical, J_logical, rms, grid_ratios,
                               beta=2.0, n_chains=50, n_burn=500, n_sample=500,
                               thin=5, seed=42, device=None):
    """GPU-accelerated Boltzmann MCMC at fixed temperature.

    Runs K × n_chains MCMC chains in parallel on GPU.
    Returns same structure as boltzmann_energy_curve().
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    graph = _build_graph_tensors(
        h_phys, J_phys, chain_edges_flat, embedding,
        rms, grid_ratios, device=device)

    n_q = graph['n_q']
    K = graph['K']
    N = n_chains

    torch.manual_seed(seed)
    spins = (torch.randint(0, 2, (K, N, n_q), device=device) * 2 - 1).to(torch.int8)

    # Burn-in
    for _ in range(n_burn):
        for qi in range(n_q):
            dE = _compute_delta_E_batched(spins, qi, graph)
            accept_prob = torch.exp(-beta * dE).clamp(max=1.0)
            accept = (torch.rand(K, N, device=device) < accept_prob) | (dE <= 0)
            qi_slice = spins[:, :, qi:qi+1]
            spins[:, :, qi:qi+1] = torch.where(
                accept.unsqueeze(2), -qi_slice, qi_slice)

    # Sampling
    all_qubits = graph['all_qubits']
    qubit_idx = graph['qubit_idx']
    energy_accum = np.zeros(K)
    break_accum = [{var: 0 for var in embedding} for _ in range(K)]
    n_samples = 0

    for sweep_idx in range(n_sample):
        for qi in range(n_q):
            dE = _compute_delta_E_batched(spins, qi, graph)
            accept_prob = torch.exp(-beta * dE).clamp(max=1.0)
            accept = (torch.rand(K, N, device=device) < accept_prob) | (dE <= 0)
            qi_slice = spins[:, :, qi:qi+1]
            spins[:, :, qi:qi+1] = torch.where(
                accept.unsqueeze(2), -qi_slice, qi_slice)

        if sweep_idx % thin == 0:
            spins_np = spins.cpu().numpy()
            n_samples += N

            for ki in range(K):
                for ri in range(N):
                    sigma = {all_qubits[i]: int(spins_np[ki, ri, i])
                             for i in range(n_q)}
                    # Decode
                    logical = {}
                    for var, chain in embedding.items():
                        vals = [sigma.get(q, 1) for q in chain]
                        logical[var] = 1 if sum(vals) >= 0 else -1
                        if len(chain) > 1 and len(set(sigma.get(q, 1) for q in chain)) > 1:
                            break_accum[ki][var] += 1
                    e = sum(h_logical.get(i, 0) * logical.get(i, 1) for i in h_logical)
                    e += sum(J * logical.get(i, 1) * logical.get(j, 1)
                             for (i, j), J in J_logical.items())
                    energy_accum[ki] += e

    energy_curve = (energy_accum / max(n_samples, 1)).tolist()
    break_curve = [sum(b.values()) / max(n_samples * sum(1 for c in embedding.values() if len(c) > 1), 1)
                   for b in break_accum]

    best_idx = int(np.argmin(energy_curve))
    return {
        'energy_curve': energy_curve,
        'break_curve': break_curve,
        'r_star': float(grid_ratios[best_idx]),
        'r_star_idx': best_idx,
        'jc_star_raw': float(grid_ratios[best_idx] * rms),
        'per_chain_breaks_at_star': {var: cnt / max(n_samples, 1)
                                     for var, cnt in break_accum[best_idx].items()},
        'sweep': [],
    }


def benchmark_gpu_vs_cpu(n_instances=5):
    """Benchmark GPU sampler vs CPU neal."""
    import time
    from src.data.generate import (
        generate_random_ising, find_embedding, build_physical_hamiltonian,
        compute_rms_J, sa_energy_curve, get_hardware_graph, make_grid
    )

    hw = get_hardware_graph("P4")
    grid = make_grid()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Benchmarking GPU ({device}) vs CPU SA on {n_instances} instances...")
    cpu_times = []
    gpu_times = []

    for i in range(n_instances):
        rng = np.random.RandomState(42 + i)
        h, J = generate_random_ising(15, rng)
        emb = find_embedding(h, J, hw, seed=42 + i)
        if emb is None:
            continue

        import dimod
        bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
        h_p, J_p, cem, cef = build_physical_hamiltonian(h, J, emb, hw)
        rms = compute_rms_J(J)

        # CPU
        t0 = time.perf_counter()
        r_cpu = sa_energy_curve(bqm, emb, hw, grid, n_reads=200, sweeps=500, seed=42)
        t_cpu = time.perf_counter() - t0
        cpu_times.append(t_cpu)

        # GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r_gpu = gpu_sa_energy_curve(
            h_p, J_p, cef, emb, h, J, rms, grid,
            n_reads=200, n_sweeps=500, seed=42, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        gpu_times.append(t_gpu)

        # Compare
        ec_cpu = np.array(r_cpu['energy_curve'])
        ec_gpu = np.array(r_gpu['energy_curve'])
        corr = np.corrcoef(ec_cpu, ec_gpu)[0, 1] if len(ec_cpu) > 1 else 0

        print(f"  Instance {i}: CPU={t_cpu:.2f}s, GPU={t_gpu:.2f}s, "
              f"speedup={t_cpu/max(t_gpu,0.001):.1f}x, "
              f"curve_corr={corr:.4f}")

    if cpu_times and gpu_times:
        print(f"\nMean: CPU={np.mean(cpu_times):.2f}s, GPU={np.mean(gpu_times):.2f}s, "
              f"speedup={np.mean(cpu_times)/max(np.mean(gpu_times),0.001):.1f}x")


if __name__ == '__main__':
    benchmark_gpu_vs_cpu(5)
