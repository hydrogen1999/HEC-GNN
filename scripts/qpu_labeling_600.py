#!/usr/bin/env python3
"""
qpu_labeling_600.py -- QPU coin-flip labeling for 600 instances.

FOUR-PHASE DESIGN (test small, then scale):
  1. calibrate: 1 instance  -> measure real QPU timing, validate connectivity
  2. pilot:     10 instances -> check SA-QPU correlation, verify cost model
  3. full:      30 instances -> collect QPU labels (backward compat)
  4. large:     600 instances -> full dataset with K=20 grid, 500 reads

SAFETY:
  - Incremental save after EVERY instance (crash loses at most 1 instance)
  - Rotating backups every 5 instances
  - Embeddings pre-saved before any QPU call
  - Per-submission cost tracking with hard cap
  - Resume from checkpoint after interruption

EMBEDDING:
  - minorminer.find_embedding() with fixed random_seed per instance
  - Same embedding used for both SA and QPU (FixedEmbeddingComposite equivalent)

USAGE (progressive):
  # Step 0: Estimate costs
  python qpu_labeling.py --phase estimate

  # Step 1: SA dry run (FREE) — validate code & embeddings
  python qpu_labeling.py --phase calibrate --mode sa

  # Step 2: Single QPU instance — measure real timing & cost
  DWAVE_API_TOKEN=xxx python qpu_labeling.py --phase calibrate --mode qpu

  # Step 3: Review calibration, then pilot
  DWAVE_API_TOKEN=xxx python qpu_labeling.py --phase pilot --mode qpu

  # Step 4: Full collection
  DWAVE_API_TOKEN=xxx python qpu_labeling.py --phase full --mode qpu

  # Resume interrupted run
  DWAVE_API_TOKEN=xxx python qpu_labeling.py --phase full --mode qpu --resume

  # Analyze SA vs QPU gap
  python qpu_labeling.py --phase analyze

  # Compare with parent run's SA results
  python qpu_labeling.py --phase compare --sa-file qpu_labeling_full_sa.json --qpu-file qpu_labeling_full_qpu.json
"""

import argparse
import json
import os
import signal
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

import dimod
import dwave_networkx as dnx
import minorminer
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_strength import uniform_torque_compensation


# ---------------------------------------------------------------------------
# Graceful shutdown: SIGINT (Ctrl+C), SIGTERM, or STOP file
# ---------------------------------------------------------------------------
_STOP_REQUESTED = False
_STOP_REASON = ""


def _signal_handler(signum, frame):
    global _STOP_REQUESTED, _STOP_REASON
    name = signal.Signals(signum).name
    if _STOP_REQUESTED:
        # Second signal = force kill
        print(f"\n  *** FORCE QUIT ({name}) -- exiting immediately ***")
        sys.exit(1)
    _STOP_REQUESTED = True
    _STOP_REASON = f"signal {name}"
    print(f"\n  *** {name} received -- will stop after current instance ***")
    print(f"  *** Press Ctrl+C again to force quit (may lose current instance) ***")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def check_stop(output_dir: str) -> bool:
    """Check if graceful stop was requested (signal or STOP file)."""
    global _STOP_REQUESTED, _STOP_REASON
    if _STOP_REQUESTED:
        return True
    stop_file = Path(output_dir) / 'STOP'
    if stop_file.exists():
        _STOP_REQUESTED = True
        _STOP_REASON = f"STOP file found: {stop_file}"
        stop_file.unlink()  # consume the file
        print(f"\n  *** STOP file detected -- will stop after current instance ***")
        return True
    return False

# ---------------------------------------------------------------------------
# Freeze protocol
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Cost model (D-Wave Advantage, 2026)
# ---------------------------------------------------------------------------
COST_PER_QPU_SECOND = 2000.0 / 3600.0  # $0.556/s
DEFAULT_US_PER_READ = 150              # ~20us anneal + ~123us readout + overhead
DEFAULT_OVERHEAD_PER_SUBMISSION_US = 15000  # 15ms programming overhead

# ---------------------------------------------------------------------------
# Phase configs — conservative, optimized for minimal QPU usage
# ---------------------------------------------------------------------------

# K=20 log-spaced grid matching SA labeling (generate.py)
import math
GRID_20 = np.geomspace(0.02, 5.0, 20).tolist()

PHASE_CONFIG = {
    'calibrate': {
        'n_instances': 1,
        'sizes': [10],
        'densities': [0.5],
        'n_reads_qpu': 500,
        'n_reads_sa': 500,
        'sa_sweeps': 500,
        'sa_n_runs': 200,
        'grid_points': GRID_20,
        'cost_cap_usd': 5.0,
        'qpu_seconds_cap': 10.0,
        'backup_every': 1,
        'description': 'Calibrate: 1 instance, K=20 grid, measure real QPU timing',
    },
    'pilot': {
        'n_instances': 10,
        'sizes': [10, 12, 15, 20],
        'densities': [0.3, 0.4, 0.5, 0.6, 0.7],
        'n_reads_qpu': 500,
        'n_reads_sa': 500,
        'sa_sweeps': 500,
        'sa_n_runs': 200,
        'grid_points': GRID_20,
        'cost_cap_usd': 15.0,
        'qpu_seconds_cap': 30.0,
        'backup_every': 5,
        'description': 'Pilot: 10 instances, K=20, verify cost model & SA-QPU correlation',
    },
    'full': {
        'n_instances': 30,
        'sizes': [10, 12, 15, 20, 25],
        'densities': [0.3, 0.4, 0.5, 0.6, 0.7],
        'n_reads_qpu': 500,
        'n_reads_sa': 500,
        'sa_sweeps': 500,
        'sa_n_runs': 200,
        'grid_points': GRID_20,
        'cost_cap_usd': 35.0,
        'qpu_seconds_cap': 65.0,
        'backup_every': 5,
        'description': 'Full: 30 instances (backward compat with previous run)',
    },
    'large': {
        'n_instances': 600,
        'sizes': [8, 10, 12, 15, 20, 25, 30, 40],
        'densities': [0.3, 0.4, 0.5, 0.6, 0.7],
        'n_reads_qpu': 500,
        'n_reads_sa': 500,
        'sa_sweeps': 500,
        'sa_n_runs': 200,
        'grid_points': GRID_20,
        'cost_cap_usd': 550.0,          # hard $ cap (pilot avg $0.83/inst × 600 = ~$498)
        'qpu_seconds_cap': 990.0,       # hard QPU-second cap (550/0.556)
        'backup_every': 10,
        'description': 'Large: 600 instances, K=20 grid, 500 reads — coin-flip labeling',
        # Budget guardrails: print warning at these thresholds
        'warn_at_usd': [100, 200, 300, 400],
    },
}


# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Cost tracker (from qpu_finetuning.py, unchanged)
# ---------------------------------------------------------------------------
class CostCapExceeded(Exception):
    pass


class CostTracker:
    """Track QPU usage and enforce cost cap with real-time monitoring."""

    def __init__(self, cap_usd: float, qpu_seconds_cap: float = 60.0,
                 initial_qpu_us: int = 0, initial_submissions: int = 0):
        self.cap_usd = cap_usd
        self.qpu_seconds_cap = qpu_seconds_cap
        self.total_qpu_us = initial_qpu_us
        self.total_submissions = initial_submissions
        self.instance_costs: List[dict] = []
        self._inst_start_us = 0

    @property
    def total_qpu_seconds(self) -> float:
        return self.total_qpu_us / 1e6

    @property
    def total_cost_usd(self) -> float:
        return self.total_qpu_seconds * COST_PER_QPU_SECOND

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.total_cost_usd

    def add_submission(self, qpu_access_us: int):
        self.total_qpu_us += qpu_access_us
        self.total_submissions += 1

    def start_instance(self):
        self._inst_start_us = self.total_qpu_us

    def end_instance(self, instance_id: int):
        inst_us = self.total_qpu_us - self._inst_start_us
        self.instance_costs.append({
            'instance': instance_id,
            'qpu_us': inst_us,
            'qpu_seconds': inst_us / 1e6,
            'cost_usd': (inst_us / 1e6) * COST_PER_QPU_SECOND,
        })

    def should_abort(self) -> bool:
        return (self.total_cost_usd >= self.cap_usd or
                self.total_qpu_seconds >= self.qpu_seconds_cap)

    def check_or_abort(self, context: str = ""):
        if self.total_qpu_seconds >= self.qpu_seconds_cap:
            raise CostCapExceeded(
                f"QPU TIME CAP: {self.total_qpu_seconds:.3f}s >= {self.qpu_seconds_cap:.1f}s "
                f"(${self.total_cost_usd:.4f}, {self.total_submissions} subs) "
                f"[{context}]"
            )
        if self.total_cost_usd >= self.cap_usd:
            raise CostCapExceeded(
                f"COST CAP: ${self.total_cost_usd:.4f} >= ${self.cap_usd:.2f} "
                f"(QPU: {self.total_qpu_seconds:.3f}s, {self.total_submissions} subs) "
                f"[{context}]"
            )

    def estimate_remaining(self, n_remaining: int) -> float:
        """Project cost for remaining instances based on average so far."""
        if not self.instance_costs:
            return 0.0
        avg = np.mean([c['cost_usd'] for c in self.instance_costs])
        return avg * n_remaining

    def summary(self) -> dict:
        return {
            'total_qpu_us': self.total_qpu_us,
            'total_qpu_seconds': self.total_qpu_seconds,
            'total_cost_usd': self.total_cost_usd,
            'cost_cap_usd': self.cap_usd,
            'total_submissions': self.total_submissions,
            'per_instance': self.instance_costs,
            'avg_cost_per_instance': (
                float(np.mean([c['cost_usd'] for c in self.instance_costs]))
                if self.instance_costs else 0
            ),
            'avg_qpu_us_per_submission': (
                self.total_qpu_us / max(self.total_submissions, 1)
            ),
        }


# ---------------------------------------------------------------------------
# Checkpoint manager — incremental saves & backup
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Atomic save after each instance + rotating backups + resume."""

    def __init__(self, phase: str, mode: str, output_dir: str):
        self.phase = phase
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.backup_dir = self.output_dir / 'backups'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / f'qpu_labeling_{phase}_{mode}.json'
        self.checkpoint_file = self.output_dir / f'checkpoint_{phase}_{mode}.json'
        self.embeddings_file = self.output_dir / f'embeddings_{phase}.json'

    def save_embeddings(self, instances: list):
        """Pre-save all embeddings BEFORE any QPU calls."""
        data = []
        for inst in instances:
            data.append({
                'id': inst['id'],
                'seed': inst['seed'],
                'n': inst['n'],
                'p': inst['p'],
                'n_edges': inst['n_edges'],
                'embedding': {str(k): list(v) for k, v in inst['embedding'].items()},
                'utc': float(inst['utc']),
                'rms_j': float(inst['rms_j']),
                'mean_chain': float(inst['mean_chain']),
                'max_chain': int(inst['max_chain']),
            })
        self._atomic_write(self.embeddings_file, data)
        print(f"  Embeddings pre-saved to {self.embeddings_file}")

    def save_after_instance(self, all_results: list, cost_summary: dict,
                            grid_points: list, completed_ids: list,
                            total_instances: int):
        """Atomic save of results + checkpoint update + optional backup."""
        now = datetime.now(timezone.utc).isoformat()

        # Full results file
        output = {
            'phase': self.phase,
            'mode': self.mode,
            'n_completed': len(all_results),
            'n_total': total_instances,
            'grid_points': grid_points,
            'cost': cost_summary,
            'saved_at': now,
            'instances': all_results,
        }
        self._atomic_write(self.results_file, output)

        # Checkpoint
        ckpt = {
            'phase': self.phase,
            'mode': self.mode,
            'completed_ids': completed_ids,
            'n_completed': len(completed_ids),
            'total_instances': total_instances,
            'cost_qpu_us': cost_summary.get('total_qpu_us', 0),
            'cost_submissions': cost_summary.get('total_submissions', 0),
            'cost_usd': cost_summary.get('total_cost_usd', 0),
            'grid_points': grid_points,
            'last_save': now,
        }
        self._atomic_write(self.checkpoint_file, ckpt)

        # Rotating backup
        backup_every = PHASE_CONFIG[self.phase].get('backup_every', 5)
        n = len(completed_ids)
        if n % backup_every == 0 or n == total_instances:
            bk = self.backup_dir / f'qpu_labeling_{self.phase}_{self.mode}_at_{n}.json'
            shutil.copy2(self.results_file, bk)
            print(f"    Backup saved: {bk.name}")

    def load_checkpoint(self) -> Tuple[list, list, dict]:
        """Load checkpoint for resume. Returns (results, completed_ids, ckpt)."""
        if not self.checkpoint_file.exists():
            return [], [], {}
        ckpt = json.loads(self.checkpoint_file.read_text())
        results = []
        if self.results_file.exists():
            data = json.loads(self.results_file.read_text())
            results = data.get('instances', [])
        return results, ckpt.get('completed_ids', []), ckpt

    def _atomic_write(self, path: Path, data):
        """Write to temp file then rename (atomic on POSIX)."""
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, cls=NpEncoder)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Adaptive grid selection
# ---------------------------------------------------------------------------
def build_adaptive_grid(calibrate_file: str) -> Optional[list]:
    """From calibration sweep, select focused 6-point grid."""
    try:
        data = json.loads(Path(calibrate_file).read_text())
        instances = data.get('instances', [])
        if not instances:
            return None
        sweep = instances[0]['sweep']
        energies = [s['mean_energy'] for s in sweep]
        rels = [s['rel'] for s in sweep]

        best_idx = int(np.argmin(energies))
        best_rel = rels[best_idx]

        # Bracket: 2 points below, 1 at optimum, 3 above
        lo = max(best_rel * 0.2, 0.02)
        hi = min(best_rel * 5.0, 3.0)
        grid = sorted(set(round(g, 4) for g in np.geomspace(lo, hi, 6)))
        print(f"  Adaptive grid from calibration (best_rel={best_rel:.3f}): {grid}")
        return grid
    except Exception as e:
        print(f"  Could not build adaptive grid: {e}")
        return None


def resolve_grid(phase: str, calibrate_file: Optional[str], cfg: dict) -> list:
    """Resolve grid points: adaptive > configured > fallback."""
    if cfg['grid_points'] is not None:
        return cfg['grid_points']

    if calibrate_file:
        adaptive = build_adaptive_grid(calibrate_file)
        if adaptive:
            return adaptive

    # Try auto-detect calibrate results in same output dir
    # (useful when running phases sequentially in same dir)
    for mode in ['qpu', 'sa']:
        auto_path = Path(calibrate_file or '.') / f'qpu_labeling_calibrate_{mode}.json'
        if auto_path.exists():
            adaptive = build_adaptive_grid(str(auto_path))
            if adaptive:
                return adaptive

    return cfg.get('grid_points_fallback', [0.05, 0.1, 0.2, 0.5, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Instance generation with FIXED embeddings
# ---------------------------------------------------------------------------
def generate_instances(
    hardware_graph: nx.Graph,
    n_instances: int,
    sizes: list,
    densities: list,
    seed: int = SEED,
) -> list:
    """Generate instances with deterministic fixed embeddings."""
    instances = []

    for idx in range(n_instances * 5):  # oversample for embedding failures
        if len(instances) >= n_instances:
            break

        rng = np.random.RandomState(seed + idx)
        n = int(rng.choice(sizes))
        p = float(rng.choice(densities))

        G = nx.erdos_renyi_graph(n, p, seed=seed + idx)
        if G.number_of_edges() < 3:
            continue

        h = {i: float(rng.uniform(-1, 1)) for i in G.nodes()}
        J = {(i, j): float(rng.uniform(-2, 2)) for i, j in G.edges()}
        bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)

        src = nx.Graph()
        src.add_nodes_from(bqm.variables)
        src.add_edges_from(bqm.quadratic.keys())

        try:
            emb = minorminer.find_embedding(
                src, hardware_graph, random_seed=seed + idx, timeout=30)
        except Exception:
            continue
        if not emb or max(len(c) for c in emb.values()) < 2:
            continue

        utc = uniform_torque_compensation(bqm, emb)
        J_vals = list(bqm.quadratic.values())
        rms_j = float(np.sqrt(np.mean(np.array(J_vals) ** 2))) if J_vals else 1.0
        chains = [len(c) for c in emb.values()]

        instances.append({
            'id': len(instances),
            'seed': seed + idx,
            'n': n, 'p': p,
            'n_edges': G.number_of_edges(),
            'bqm': bqm,
            'embedding': emb,
            'utc': float(utc),
            'rms_j': rms_j,
            'mean_chain': float(np.mean(chains)),
            'max_chain': int(max(chains)),
            'n_chains': len(chains),
        })

    return instances


# ---------------------------------------------------------------------------
# SA grid search (FREE)
# ---------------------------------------------------------------------------
def compute_rich_metrics(raw_sampleset, unemb_sampleset, emb: dict,
                         bqm) -> dict:
    """Compute full measurement vector from a QPU/SA sampleset at one J_c.

    Returns dict with all metrics needed for fine-tuning dataset.
    """
    # --- Logical energies ---
    energies = []
    for r in unemb_sampleset.record:
        for _ in range(r.num_occurrences):
            energies.append(float(r.energy))
    energies = np.array(energies) if energies else np.array([0.0])

    mean_e = float(np.mean(energies))
    best_e = float(np.min(energies))
    std_e = float(np.std(energies))
    q25, q50, q75 = np.percentile(energies, [25, 50, 75]).tolist()
    n_unique = len(set(round(e, 8) for e in energies))

    # --- Residual energy (gap from best) ---
    residual = float(mean_e - best_e) if abs(best_e) > 1e-12 else 0.0
    residual_pct = float(abs(residual) / abs(best_e) * 100) if abs(best_e) > 1e-12 else 0.0

    # --- p_solve (fraction of reads finding best or near-best) ---
    # Threshold = best_e + 1% of |best_e| (always ABOVE best_e, regardless of sign)
    threshold_e = best_e + 0.01 * abs(best_e) if abs(best_e) > 1e-12 else 1e-8
    n_good = sum(1 for e in energies if e <= threshold_e + 1e-8)
    p_solve = n_good / max(len(energies), 1)

    # --- Time to solution (TTS) at 99% confidence ---
    if p_solve > 0 and p_solve < 1:
        import math
        tts_99 = float(20e-6 * math.log(1 - 0.99) / math.log(1 - p_solve))
    elif p_solve >= 1:
        tts_99 = 20e-6  # one read suffices
    else:
        tts_99 = float('inf')

    # --- Chain break analysis ---
    per_chain_breaks = {var: 0 for var in emb}
    n_samples = 0
    total_breaks = 0
    total_chains_checked = 0

    for record in raw_sampleset.record:
        sample = dict(zip(raw_sampleset.variables, record.sample))
        num_occ = record.num_occurrences
        n_samples += num_occ
        for var, chain in emb.items():
            cl = list(chain)
            if len(cl) > 1:
                if len(set(sample.get(q, 0) for q in cl)) > 1:
                    per_chain_breaks[var] += num_occ
                    total_breaks += num_occ
            total_chains_checked += num_occ

    cbr = float(total_breaks / max(total_chains_checked, 1))
    per_chain_cbr = {
        str(var): float(cnt / max(n_samples, 1))
        for var, cnt in per_chain_breaks.items()
    }
    n_chains_ever_broken = sum(1 for v in per_chain_breaks.values() if v > 0)
    max_chain_cbr = float(max(per_chain_breaks.values()) / max(n_samples, 1)) if per_chain_breaks else 0.0
    weakest_chain = max(per_chain_breaks, key=per_chain_breaks.get) if per_chain_breaks else None

    # --- Energy recovery (vs best energy across entire sweep, filled later) ---
    # This is filled in by the caller after the full sweep

    return {
        'mean_energy': mean_e,
        'best_energy': best_e,
        'std_energy': std_e,
        'q25_energy': float(q25),
        'median_energy': float(q50),
        'q75_energy': float(q75),
        'residual_energy': residual,
        'residual_energy_pct': residual_pct,
        'p_solve': float(p_solve),
        'tts_99': tts_99,
        'cbr': cbr,
        'per_chain_cbr': per_chain_cbr,
        'n_chains_ever_broken': n_chains_ever_broken,
        'max_chain_cbr': max_chain_cbr,
        'weakest_chain': int(weakest_chain) if weakest_chain is not None else None,
        'n_unique_solutions': n_unique,
        'n_samples': len(energies),
    }


def sa_grid_search(inst: dict, grid_points: list, n_runs: int,
                   sweeps: int, topology: nx.Graph) -> Tuple[list, dict]:
    """SA grid search using dwave embed_bqm + neal. Returns (sweep, best).

    grid_points are RMS-normalized ratios r: cs = r * RMS(J).
    This matches generate.py SA labeling for direct curve comparison.
    """
    import neal
    sa = neal.SimulatedAnnealingSampler()
    bqm, emb = inst['bqm'], inst['embedding']
    rms_j = inst['rms_j']
    sweep = []

    for r in grid_points:
        cs = r * rms_j  # RMS-normalized: cs = r * RMS(J)
        ebqm = embed_bqm(bqm, emb, topology, chain_strength=cs)
        raw = sa.sample(ebqm, num_reads=n_runs, num_sweeps=sweeps, seed=SEED)
        unemb = unembed_sampleset(raw, emb, bqm)

        metrics = compute_rich_metrics(raw, unemb, emb, bqm)
        metrics['rel'] = float(r)
        metrics['cs'] = float(cs)
        sweep.append(metrics)

    # Post-process: compute energy_recovery relative to global best
    global_best = min(s['best_energy'] for s in sweep)
    for s in sweep:
        if abs(global_best) > 1e-12:
            s['energy_recovery'] = float(s['mean_energy'] / global_best)
        else:
            s['energy_recovery'] = 1.0

    best_idx = int(np.argmin([s['mean_energy'] for s in sweep]))
    return sweep, sweep[best_idx]


# ---------------------------------------------------------------------------
# QPU grid search (COSTS $$$) — with early stopping
# ---------------------------------------------------------------------------
def qpu_grid_search(
    inst: dict,
    sampler,
    hardware_graph: nx.Graph,
    grid_points: list,
    n_reads: int,
    cost_tracker: CostTracker,
) -> Tuple[list, dict]:
    """QPU grid search with rich metrics, cost tracking, and early stopping.

    grid_points are RMS-normalized ratios r: cs = r * RMS(J).
    This matches generate.py SA labeling for direct curve comparison.
    """
    bqm, emb = inst['bqm'], inst['embedding']
    rms_j = inst['rms_j']
    sweep = []
    consecutive_high_cbr = 0

    for r in grid_points:
        cost_tracker.check_or_abort(f"inst {inst['id']}, r={r}")

        cs = r * rms_j  # RMS-normalized: cs = r * RMS(J)
        ebqm = embed_bqm(bqm, emb, hardware_graph, chain_strength=cs)
        raw = sampler.sample(
            ebqm,
            num_reads=n_reads,
            auto_scale=True,
            reduce_intersample_correlation=False,  # D-Wave: "drastically increases run times" if True
            readout_thermalization=0,               # default, no added delay between reads
            programming_thermalization=1000,         # default 1ms, safe
            annealing_time=20,                       # default 20us
        )

        # Track QPU time
        timing = raw.info.get('timing', {})
        qpu_us = timing.get('qpu_access_time', 0)
        cost_tracker.add_submission(qpu_us)

        # Unembed and compute full metrics
        unemb = unembed_sampleset(raw, emb, bqm)
        metrics = compute_rich_metrics(raw, unemb, emb, bqm)
        metrics['rel'] = float(r)
        metrics['cs'] = float(cs)
        metrics['qpu_us'] = int(qpu_us)

        # QPU-specific timing details
        metrics['qpu_timing'] = {
            'qpu_access_time_us': timing.get('qpu_access_time', 0),
            'qpu_programming_time_us': timing.get('qpu_programming_time', 0),
            'qpu_sampling_time_us': timing.get('qpu_sampling_time', 0),
            'qpu_anneal_time_per_run_us': timing.get('qpu_anneal_time_per_run', 0),
            'qpu_readout_time_per_run_us': timing.get('qpu_readout_time_per_run', 0),
            'total_post_processing_time_us': timing.get('total_post_processing_time', 0),
        }

        sweep.append(metrics)

        # Early stopping: only on DESCENDING cbr trend (high-r side).
        # On ascending grid (low→high r), low-r always has high CBR (weak chains),
        # so we must NOT stop early before reaching the good region.
        # Early-stop only when: (a) we've seen the CBR basin (low CBR region),
        # AND (b) CBR rises again on the high-r side (over-strong chains).
        cbr = metrics['cbr']
        saw_low_cbr = any(s['cbr'] < 0.3 for s in sweep)
        if saw_low_cbr and cbr > 0.5:
            consecutive_high_cbr += 1
        else:
            consecutive_high_cbr = 0
        if consecutive_high_cbr >= 2 and len(sweep) >= 5:
            print(f"      Early stop at r={r:.3f} (past basin, cbr>{0.5} x2)")
            break

    # Post-process: energy_recovery relative to global best
    global_best = min(s['best_energy'] for s in sweep)
    for s in sweep:
        if abs(global_best) > 1e-12:
            s['energy_recovery'] = float(s['mean_energy'] / global_best)
        else:
            s['energy_recovery'] = 1.0

    best_idx = int(np.argmin([s['mean_energy'] for s in sweep]))
    return sweep, sweep[best_idx]


# ---------------------------------------------------------------------------
# Format a single instance result
# ---------------------------------------------------------------------------
def format_result(inst: dict, sweep: list, best: dict) -> dict:
    """Format a single instance result with full sweep data for fine-tuning.

    Structure per instance:
      - Instance metadata (n, p, embedding stats)
      - Optimal point summary (jc_star, metrics at optimum)
      - Full sweep: for each J_c grid point, the complete measurement vector
        (energy stats, p_solve, cbr, per-chain breaks, TTS, etc.)
    """
    return {
        # --- Instance identity ---
        'id': inst['id'],
        'seed': inst['seed'],

        # --- Problem structure ---
        'n': inst['n'],
        'p': inst['p'],
        'n_edges': inst['n_edges'],

        # --- Embedding structure ---
        'n_chains': inst['n_chains'],
        'mean_chain': inst['mean_chain'],
        'max_chain': inst['max_chain'],
        'utc': inst['utc'],
        'rms_j': inst['rms_j'],

        # --- Optimal point (label for fine-tuning) ---
        'jc_star': best['cs'],
        'r_star': best['rel'],
        'target': best['cs'] / inst['rms_j'],

        # --- Metrics at optimal J_c* ---
        'opt_mean_energy': best['mean_energy'],
        'opt_best_energy': best['best_energy'],
        'opt_std_energy': best.get('std_energy', 0),
        'opt_p_solve': best.get('p_solve', 0),
        'opt_tts_99': best.get('tts_99', float('inf')),
        'opt_cbr': best.get('cbr', 0),
        'opt_n_unique': best.get('n_unique_solutions', 0),
        'opt_energy_recovery': best.get('energy_recovery', 1.0),
        'opt_per_chain_cbr': best.get('per_chain_cbr', {}),
        'opt_weakest_chain': best.get('weakest_chain'),

        # --- Full sweep: rich measurement at every grid point ---
        # Each entry has: rel, cs, mean/best/std/Q25/Q50/Q75 energy,
        # residual, p_solve, tts_99, cbr, per_chain_cbr, n_unique, etc.
        'sweep': sweep,

        # --- Sweep summary statistics ---
        'n_grid_points': len(sweep),
        'energy_range': [
            min(s['mean_energy'] for s in sweep),
            max(s['mean_energy'] for s in sweep),
        ],
        'cbr_range': [
            min(s['cbr'] for s in sweep),
            max(s['cbr'] for s in sweep),
        ],
        'p_solve_range': [
            min(s['p_solve'] for s in sweep),
            max(s['p_solve'] for s in sweep),
        ],
    }


# ---------------------------------------------------------------------------
# Calibration timing output
# ---------------------------------------------------------------------------
def save_calibration_timing(cost_tracker: CostTracker, output_dir: str,
                            n_reads: int, n_grid: int):
    """Save empirical timing for accurate cost projection."""
    s = cost_tracker.summary()
    if s['total_submissions'] == 0:
        return

    avg_us = s['avg_qpu_us_per_submission']
    avg_us_per_read = avg_us / max(n_reads, 1)

    # Project costs for pilot and full
    for phase in ['pilot', 'full']:
        cfg = PHASE_CONFIG[phase]
        n = cfg['n_instances']
        g = len(cfg.get('grid_points_fallback', [0]*6))
        r = cfg['n_reads_qpu']
        total_us = n * g * avg_us
        projected = (total_us / 1e6) * COST_PER_QPU_SECOND
        PHASE_CONFIG[phase]['_projected_cost'] = projected
        PHASE_CONFIG[phase]['_projected_qpu_s'] = total_us / 1e6

    timing = {
        'measured_avg_qpu_us_per_submission': float(avg_us),
        'measured_avg_qpu_us_per_read': float(avg_us_per_read),
        'n_reads_used': n_reads,
        'n_submissions': s['total_submissions'],
        'total_qpu_seconds': s['total_qpu_seconds'],
        'total_cost_usd': s['total_cost_usd'],
        'projected_pilot_cost_usd': PHASE_CONFIG['pilot'].get('_projected_cost', 0),
        'projected_pilot_qpu_seconds': PHASE_CONFIG['pilot'].get('_projected_qpu_s', 0),
        'projected_full_cost_usd': PHASE_CONFIG['full'].get('_projected_cost', 0),
        'projected_full_qpu_seconds': PHASE_CONFIG['full'].get('_projected_qpu_s', 0),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    path = Path(output_dir) / 'calibration_timing.json'
    with open(path, 'w') as f:
        json.dump(timing, f, indent=2)
    print(f"\n  Calibration timing saved to {path}")
    print(f"    Measured: {avg_us:.0f} us/submission, {avg_us_per_read:.1f} us/read")
    print(f"    Projected pilot: ${timing['projected_pilot_cost_usd']:.4f} "
          f"({timing['projected_pilot_qpu_seconds']:.3f}s QPU)")
    print(f"    Projected full:  ${timing['projected_full_cost_usd']:.4f} "
          f"({timing['projected_full_qpu_seconds']:.3f}s QPU)")


# ---------------------------------------------------------------------------
# Cost estimator (pre-execution)
# ---------------------------------------------------------------------------
def estimate_cost(phase: str, calibration_file: Optional[str] = None) -> dict:
    cfg = PHASE_CONFIG[phase]
    n = cfg['n_instances']
    grid = cfg.get('grid_points') or cfg.get('grid_points_fallback', [0]*6)
    g = len(grid)
    r = cfg['n_reads_qpu']

    # Use calibration data if available
    us_per_read = DEFAULT_US_PER_READ
    overhead_us = DEFAULT_OVERHEAD_PER_SUBMISSION_US
    source = 'theoretical'

    if calibration_file and Path(calibration_file).exists():
        cal = json.loads(Path(calibration_file).read_text())
        us_per_read = cal.get('measured_avg_qpu_us_per_read', us_per_read)
        overhead_us = 0  # already included in measured avg
        source = 'empirical (from calibration)'

    total_reads = n * g * r
    total_submissions = n * g
    qpu_us = total_reads * us_per_read + total_submissions * overhead_us
    qpu_s = qpu_us / 1e6
    cost = qpu_s * COST_PER_QPU_SECOND

    return {
        'phase': phase,
        'description': cfg['description'],
        'n_instances': n,
        'n_grid_points': g,
        'grid_points': grid,
        'n_reads': r,
        'total_reads': total_reads,
        'total_submissions': total_submissions,
        'estimated_qpu_seconds': round(qpu_s, 3),
        'estimated_cost_usd': round(cost, 4),
        'cost_cap_usd': cfg['cost_cap_usd'],
        'estimate_source': source,
        'within_free_tier': qpu_s < 60,
        'free_tier_usage': f'{qpu_s:.1f}s of 60s/month',
    }


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------
def run_collection(phase: str, mode: str, resume: bool = False,
                   output_dir: str = '.', calibrate_file: Optional[str] = None):
    cfg = PHASE_CONFIG[phase]

    print("=" * 70)
    print(f"QPU Labeling -- Phase: {phase.upper()}, Mode: {mode.upper()}")
    print(f"  {cfg['description']}")
    print("=" * 70)

    # --- Resolve grid ---
    grid_points = resolve_grid(phase, calibrate_file, cfg)
    print(f"\n  Grid points ({len(grid_points)}): {grid_points}")

    # --- Cost estimate ---
    cal_timing = Path(output_dir) / 'calibration_timing.json'
    est = estimate_cost(phase, str(cal_timing) if cal_timing.exists() else None)
    print(f"  Cost estimate ({est['estimate_source']}):")
    print(f"    {est['n_instances']} instances x {est['n_grid_points']} grid x {est['n_reads']} reads")
    print(f"    = {est['total_reads']:,} total reads")
    print(f"    Est. QPU time: {est['estimated_qpu_seconds']:.3f}s")
    print(f"    Est. cost: ${est['estimated_cost_usd']:.4f}")
    print(f"    Cost cap: ${est['cost_cap_usd']:.2f}")
    print(f"    Free tier: {est['free_tier_usage']}")

    # --- Checkpoint manager ---
    ckpt_mgr = CheckpointManager(phase, mode, output_dir)

    # --- Resume ---
    completed_results = []
    completed_ids = []
    initial_qpu_us = 0
    initial_subs = 0

    if resume:
        completed_results, completed_ids, ckpt_data = ckpt_mgr.load_checkpoint()
        if completed_ids:
            initial_qpu_us = ckpt_data.get('cost_qpu_us', 0)
            initial_subs = ckpt_data.get('cost_submissions', 0)
            print(f"\n  RESUMING: {len(completed_ids)}/{cfg['n_instances']} done, "
                  f"${ckpt_data.get('cost_usd', 0):.4f} spent")
        else:
            print(f"\n  No checkpoint found, starting fresh")

    # --- Connect to QPU or build topology ---
    sampler = None
    if mode == 'qpu':
        token = os.environ.get("DWAVE_API_TOKEN")
        if not token:
            print("\nERROR: Set DWAVE_API_TOKEN environment variable")
            sys.exit(1)

        # Safety confirmation (skip if --yes flag or non-interactive)
        if sys.stdin.isatty() and not os.environ.get('QPU_CONFIRM_YES'):
            print(f"\n  *** QPU MODE — REAL MONEY/FREE-TIER TIME WILL BE USED ***")
            print(f"  *** Cost cap: ${cfg['cost_cap_usd']:.2f}, "
                  f"QPU-second cap: {cfg.get('qpu_seconds_cap', 60):.0f}s ***")
            answer = input("  Type 'yes' to proceed: ").strip().lower()
            if answer != 'yes':
                print("  Aborted by user.")
                sys.exit(0)

        from dwave.system import DWaveSampler
        sampler = DWaveSampler(token=token)
        hardware_graph = sampler.to_networkx_graph()
        chip = sampler.properties.get('chip_id', 'unknown')
        topo = sampler.properties.get('topology', {}).get('type', 'unknown')
        n_qubits = len(sampler.nodelist)
        print(f"\n  QPU: {chip} ({topo}, {n_qubits} qubits)")
    else:
        # Real Pegasus P16 for SA (matches D-Wave Advantage)
        hardware_graph = dnx.pegasus_graph(16)
        print(f"\n  SA mode -- Pegasus P16 ({hardware_graph.number_of_nodes()} qubits)")

    # --- Cost tracker (with resume state) ---
    qpu_sec_cap = cfg.get('qpu_seconds_cap', 60.0)
    cost_tracker = CostTracker(
        cfg['cost_cap_usd'], qpu_sec_cap, initial_qpu_us, initial_subs)

    try:
        # --- Generate instances (deterministic) ---
        print(f"\n  Generating {cfg['n_instances']} instances...")
        instances = generate_instances(
            hardware_graph, cfg['n_instances'],
            cfg['sizes'], cfg['densities'], seed=SEED)
        print(f"  Generated {len(instances)} instances")
        for inst in instances:
            print(f"    #{inst['id']}: n={inst['n']}, p={inst['p']}, "
                  f"chains={inst['n_chains']}(max={inst['max_chain']}), "
                  f"UTC={inst['utc']:.3f}")

        # --- Pre-save embeddings ---
        ckpt_mgr.save_embeddings(instances)

        # --- Run sweeps ---
        print(f"\n  Running {mode.upper()} grid search...")
        print(f"  Stop methods: Ctrl+C (graceful), or `touch {output_dir}/STOP`")
        t0 = time.time()
        stop_reason = None

        for inst in instances:
            iid = inst['id']
            if iid in completed_ids:
                print(f"    #{iid}: skipped (already completed)")
                continue  # skip completed (resume)

            # --- Check for graceful stop BEFORE starting next instance ---
            if check_stop(output_dir):
                stop_reason = _STOP_REASON
                print(f"\n  *** GRACEFUL STOP: {stop_reason} ***")
                print(f"  *** Saved {len(completed_results)}/{len(instances)} instances ***")
                print(f"  *** Resume with: --resume ***")
                break

            cost_tracker.start_instance()
            inst_t0 = time.time()

            try:
                if mode == 'qpu':
                    sweep, best = qpu_grid_search(
                        inst, sampler, hardware_graph, grid_points,
                        cfg['n_reads_qpu'], cost_tracker)
                else:
                    sweep, best = sa_grid_search(
                        inst, grid_points, cfg['sa_n_runs'],
                        cfg['sa_sweeps'], hardware_graph)
            except CostCapExceeded as e:
                stop_reason = f"cost cap: {e}"
                print(f"\n  *** {e} ***")
                print(f"  *** ABORTING -- saved {len(completed_results)}/{len(instances)} ***")
                print(f"  *** Resume with: --resume (after raising cost cap) ***")
                break
            except Exception as e:
                print(f"  Instance {iid}: FAILED ({e}) -- skipping")
                import traceback
                traceback.print_exc()
                continue

            cost_tracker.end_instance(iid)
            inst_elapsed = time.time() - inst_t0

            result = format_result(inst, sweep, best)
            completed_results.append(result)
            completed_ids.append(iid)

            # --- SAVE AFTER EVERY INSTANCE ---
            ckpt_mgr.save_after_instance(
                completed_results, cost_tracker.summary(),
                grid_points, completed_ids, len(instances))

            # --- Budget warning checkpoints ---
            warn_thresholds = cfg.get('warn_at_usd', [])
            current_cost = cost_tracker.total_cost_usd
            for threshold in warn_thresholds:
                # Warn once when crossing each threshold
                prev_cost = current_cost - (cost_tracker.instance_costs[-1]['cost_usd']
                                            if cost_tracker.instance_costs else 0)
                if prev_cost < threshold <= current_cost:
                    pct = len(completed_results) / len(instances) * 100
                    proj_total = cost_tracker.estimate_remaining(
                        len(instances) - len(completed_results)) + current_cost
                    print(f"\n  {'!'*60}")
                    print(f"  !!! BUDGET WARNING: ${current_cost:.2f} spent "
                          f"({pct:.0f}% done, {len(completed_results)}/{len(instances)})")
                    print(f"  !!! Projected total: ${proj_total:.2f} "
                          f"(cap: ${cfg['cost_cap_usd']:.2f})")
                    print(f"  !!! To stop gracefully: touch {ckpt_mgr.output_dir}/STOP")
                    print(f"  {'!'*60}\n")

            # Progress with detailed stats
            elapsed = time.time() - t0
            remaining = len(instances) - len(completed_results)
            cost_str = ""
            if mode == 'qpu':
                proj = cost_tracker.estimate_remaining(remaining)
                cost_str = (f", cost: ${cost_tracker.total_cost_usd:.4f}"
                            f", proj remaining: ${proj:.4f}")
            eta_str = ""
            if len(completed_results) > 0 and remaining > 0:
                avg_time = elapsed / len(completed_results)
                eta = avg_time * remaining
                eta_str = f", ETA: {eta:.0f}s"
            print(f"  [{len(completed_results)}/{len(instances)}] "
                  f"r*={best['rel']:.3f}, E={best['mean_energy']:.3f}, "
                  f"cbr={best['cbr']:.3f}, p_solve={best['p_solve']:.3f}"
                  f"{cost_str} ({inst_elapsed:.1f}s this, {elapsed:.0f}s total{eta_str})")

    finally:
        # --- ALWAYS disconnect QPU ---
        if sampler is not None:
            try:
                sampler.client.close()
                print(f"\n  *** QPU DISCONNECTED ***")
            except Exception:
                pass

        # --- Final emergency save (in case loop exited unexpectedly) ---
        if completed_results:
            try:
                ckpt_mgr.save_after_instance(
                    completed_results, cost_tracker.summary(),
                    grid_points, completed_ids, len(instances))
                print(f"  Final checkpoint saved: {len(completed_results)} instances")
            except Exception as e:
                print(f"  WARNING: Final save failed: {e}")

    elapsed = time.time() - t0

    # --- Save calibration timing ---
    if phase == 'calibrate' and mode == 'qpu' and cost_tracker.total_submissions > 0:
        save_calibration_timing(
            cost_tracker, output_dir, cfg['n_reads_qpu'], len(grid_points))

    # --- Final summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {phase.upper()} {mode.upper()}")
    print(f"{'='*70}")
    print(f"  Completed: {len(completed_results)}/{len(instances)}")
    print(f"  Wall time: {elapsed:.0f}s")

    if completed_results:
        r_stars = [r['r_star'] for r in completed_results]
        targets = [r['target'] for r in completed_results]
        print(f"  r* range: [{min(r_stars):.3f}, {max(r_stars):.3f}]")
        print(f"  target range: [{min(targets):.3f}, {max(targets):.3f}]")

    if mode == 'qpu':
        cs = cost_tracker.summary()
        print(f"  QPU time: {cs['total_qpu_seconds']:.3f}s")
        print(f"  Total cost: ${cs['total_cost_usd']:.4f}")
        print(f"  Avg cost/instance: ${cs['avg_cost_per_instance']:.4f}")
        print(f"  Avg us/submission: {cs['avg_qpu_us_per_submission']:.0f}")

    if stop_reason:
        print(f"\n  STOPPED: {stop_reason}")
        print(f"  Resume: python qpu_labeling.py --phase {phase} --mode {mode} "
              f"--output-dir {output_dir} --resume")

    print(f"\n  Results: {ckpt_mgr.results_file}")
    print(f"  Checkpoint: {ckpt_mgr.checkpoint_file}")

    # --- Verify saved data ---
    verify_checkpoint(output_dir, phase, mode)


# ---------------------------------------------------------------------------
# Checkpoint verification
# ---------------------------------------------------------------------------
def verify_checkpoint(output_dir: str, phase: str, mode: str):
    """Verify that saved checkpoint and results are consistent and readable."""
    od = Path(output_dir)
    results_file = od / f'qpu_labeling_{phase}_{mode}.json'
    ckpt_file = od / f'checkpoint_{phase}_{mode}.json'
    emb_file = od / f'embeddings_{phase}.json'

    print(f"\n  Verifying saved data...")
    ok = True

    # Check results file
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text())
            n_inst = len(data.get('instances', []))
            n_total = data.get('n_total', '?')
            print(f"    Results:     {results_file.name} -- {n_inst}/{n_total} instances, "
                  f"{results_file.stat().st_size / 1024:.1f} KB")
            # Verify each instance has sweep data
            for inst in data.get('instances', []):
                if 'sweep' not in inst or len(inst['sweep']) == 0:
                    print(f"    WARNING: instance {inst.get('id', '?')} has no sweep data!")
                    ok = False
                if 'jc_star' not in inst:
                    print(f"    WARNING: instance {inst.get('id', '?')} missing jc_star!")
                    ok = False
        except Exception as e:
            print(f"    ERROR: Cannot read {results_file.name}: {e}")
            ok = False
    else:
        print(f"    Results:     MISSING")
        ok = False

    # Check checkpoint file
    if ckpt_file.exists():
        try:
            ckpt = json.loads(ckpt_file.read_text())
            n_done = ckpt.get('n_completed', 0)
            ids = ckpt.get('completed_ids', [])
            print(f"    Checkpoint:  {ckpt_file.name} -- {n_done} completed, IDs: {ids}")
        except Exception as e:
            print(f"    ERROR: Cannot read {ckpt_file.name}: {e}")
            ok = False
    else:
        print(f"    Checkpoint:  MISSING")

    # Check embeddings
    if emb_file.exists():
        try:
            embs = json.loads(emb_file.read_text())
            print(f"    Embeddings:  {emb_file.name} -- {len(embs)} instances, "
                  f"{emb_file.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"    ERROR: Cannot read {emb_file.name}: {e}")
            ok = False
    else:
        print(f"    Embeddings:  MISSING")

    # Check backups
    backup_dir = od / 'backups'
    if backup_dir.exists():
        backups = sorted(backup_dir.glob(f'qpu_labeling_{phase}_{mode}_at_*.json'))
        print(f"    Backups:     {len(backups)} snapshots in {backup_dir.name}/")
        for b in backups:
            print(f"      {b.name} ({b.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"    Backups:     none")

    if ok:
        print(f"    STATUS: ALL OK")
    else:
        print(f"    STATUS: ISSUES FOUND (see warnings above)")


# ---------------------------------------------------------------------------
# Analyze SA vs QPU gap
# ---------------------------------------------------------------------------
def analyze_gap(sa_file: str, qpu_file: str, output_dir: str = '.'):
    from scipy.stats import spearmanr, pearsonr

    sa = json.loads(Path(sa_file).read_text())
    qpu = json.loads(Path(qpu_file).read_text())

    print("=" * 70)
    print("SA <-> QPU Gap Analysis")
    print("=" * 70)

    sa_map = {r['id']: r for r in sa['instances']}
    qpu_map = {r['id']: r for r in qpu['instances']}
    common = sorted(set(sa_map.keys()) & set(qpu_map.keys()))

    if len(common) < 3:
        print(f"  Only {len(common)} common instances -- not enough")
        return

    print(f"  Common instances: {len(common)}")

    sa_jc = np.array([sa_map[i]['jc_star'] for i in common])
    qpu_jc = np.array([qpu_map[i]['jc_star'] for i in common])
    sa_r = np.array([sa_map[i]['r_star'] for i in common])
    qpu_r = np.array([qpu_map[i]['r_star'] for i in common])

    sp_jc, sp_jc_p = spearmanr(sa_jc, qpu_jc)
    sp_r, sp_r_p = spearmanr(sa_r, qpu_r)
    pr_jc, pr_jc_p = pearsonr(sa_jc, qpu_jc)

    gap = qpu_jc - sa_jc
    ratio = qpu_jc / np.clip(sa_jc, 1e-6, None)
    mae_gap = np.mean(np.abs(gap))

    print(f"\n  Correlation:")
    print(f"    Spearman(J_c*): rho = {sp_jc:.3f} (p = {sp_jc_p:.1e})")
    print(f"    Spearman(r*):   rho = {sp_r:.3f} (p = {sp_r_p:.1e})")
    print(f"    Pearson(J_c*):  r   = {pr_jc:.3f} (p = {pr_jc_p:.1e})")

    print(f"\n  Gap Statistics:")
    print(f"    Mean gap (QPU-SA):  {np.mean(gap):.4f}")
    print(f"    Std gap:            {np.std(gap):.4f}")
    print(f"    MAE gap:            {mae_gap:.4f}")
    print(f"    Mean ratio QPU/SA:  {np.mean(ratio):.3f} +/- {np.std(ratio):.3f}")
    within = np.mean((ratio >= 1/1.5) & (ratio <= 1.5)) * 100
    print(f"    Within 1.5x:        {within:.0f}%")

    # Deployment analysis: SA-predicted J_c* on QPU energy landscape
    deploy = []
    for i in common:
        sa_jc_i = sa_map[i]['jc_star']
        qpu_sweep = qpu_map[i]['sweep']
        jcs = np.array([s['cs'] for s in qpu_sweep])
        Es = np.array([s['mean_energy'] for s in qpu_sweep])
        PSs = np.array([s.get('p_solve', 0) for s in qpu_sweep])

        E_oracle = Es.min()
        E_at_sa = float(np.interp(np.clip(sa_jc_i, jcs.min(), jcs.max()), jcs, Es))
        PS_oracle = float(PSs[np.argmin(Es)])
        PS_at_sa = float(np.interp(np.clip(sa_jc_i, jcs.min(), jcs.max()), jcs, PSs))

        if abs(E_oracle) > 1e-6:
            deploy.append({
                'id': i,
                'energy_gap_pct': abs(E_at_sa - E_oracle) / abs(E_oracle) * 100,
                'ps_recovery': PS_at_sa / max(PS_oracle, 1e-6),
            })

    if deploy:
        egaps = np.array([d['energy_gap_pct'] for d in deploy])
        ps_rec = np.array([d['ps_recovery'] for d in deploy])
        print(f"\n  SA-trained model -> QPU deployment:")
        print(f"    Energy <5%:  {np.mean(egaps < 5)*100:.0f}%")
        print(f"    Energy <10%: {np.mean(egaps < 10)*100:.0f}%")
        print(f"    p_solve >=95%: {np.mean(ps_rec >= 0.95)*100:.0f}%")
        print(f"    Mean energy gap: {np.mean(egaps):.1f}%")

    # Per-instance details
    print(f"\n  Per-instance:")
    for i in common:
        s, q = sa_map[i], qpu_map[i]
        r = q['jc_star'] / max(s['jc_star'], 1e-6)
        print(f"    #{i}: SA={s['jc_star']:.3f}, QPU={q['jc_star']:.3f}, "
              f"ratio={r:.2f}, n={s['n']}, chains={s['max_chain']}")

    # Decision
    print(f"\n  Fine-tuning Decision:")
    if sp_jc >= 0.5:
        print(f"    GOOD: Spearman >= 0.5 -- SA ordering preserved, fine-tuning viable")
        print(f"    Recommended: Option A (freeze backbone, replace head)")
    elif sp_jc >= 0.3:
        print(f"    MARGINAL: Spearman 0.3-0.5 -- try Option C (dual-head)")
    else:
        print(f"    POOR: Spearman < 0.3 -- SA labels unreliable for QPU")

    # Save
    analysis = {
        'n_common': len(common),
        'spearman_jc': float(sp_jc), 'spearman_jc_p': float(sp_jc_p),
        'spearman_r': float(sp_r), 'spearman_r_p': float(sp_r_p),
        'pearson_jc': float(pr_jc), 'pearson_jc_p': float(pr_jc_p),
        'mae_gap': float(mae_gap),
        'mean_ratio': float(np.mean(ratio)),
        'within_1_5x_pct': float(within),
        'deploy_stats': deploy,
        'recommendation': 'option_a' if sp_jc >= 0.5 else (
            'option_c' if sp_jc >= 0.3 else 'not_viable'),
        'qpu_cost': qpu.get('cost', {}),
    }
    out = Path(output_dir) / 'qpu_labeling_gap_analysis.json'
    with open(out, 'w') as f:
        json.dump(analysis, f, indent=2, cls=NpEncoder)
    print(f"\n  Analysis saved to {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='QPU Label Collection — Optimized with Checkpointing')
    parser.add_argument('--phase',
        choices=['calibrate', 'pilot', 'full', 'large', 'estimate', 'analyze', 'compare', 'verify'],
        default='estimate')
    parser.add_argument('--mode', choices=['sa', 'qpu'], default='sa')
    parser.add_argument('--resume', action='store_true',
        help='Resume from checkpoint')
    parser.add_argument('--output-dir', default='.',
        help='Output directory for results and backups')
    parser.add_argument('--calibrate-file', default=None,
        help='Path to calibrate results for adaptive grid')
    parser.add_argument('--sa-file', default=None,
        help='SA results file (for analyze/compare)')
    parser.add_argument('--qpu-file', default=None,
        help='QPU results file (for analyze/compare)')
    parser.add_argument('--dry-run', action='store_true',
        help='Print plan without connecting to QPU')
    args = parser.parse_args()

    if args.phase == 'estimate':
        cal = Path(args.output_dir) / 'calibration_timing.json'
        cal_str = str(cal) if cal.exists() else None
        for p in ['calibrate', 'pilot', 'full', 'large']:
            est = estimate_cost(p, cal_str)
            print(f"\n{'='*50}")
            print(f"  Phase: {p.upper()} -- {est['description']}")
            print(f"  {est['n_instances']} inst x {est['n_grid_points']} grid x {est['n_reads']} reads")
            print(f"  = {est['total_reads']:,} reads, {est['total_submissions']} submissions")
            print(f"  Est. QPU: {est['estimated_qpu_seconds']:.3f}s")
            print(f"  Est. cost: ${est['estimated_cost_usd']:.4f}")
            print(f"  Cap: ${est['cost_cap_usd']:.2f}")
            print(f"  Free tier: {est['free_tier_usage']}")
            print(f"  Source: {est['estimate_source']}")
        return

    if args.phase == 'verify':
        for p in ['calibrate', 'pilot', 'full', 'large']:
            for m in ['sa', 'qpu']:
                rf = Path(args.output_dir) / f'qpu_labeling_{p}_{m}.json'
                if rf.exists():
                    print(f"\n--- {p.upper()} {m.upper()} ---")
                    verify_checkpoint(args.output_dir, p, m)
        return

    if args.phase in ('analyze', 'compare'):
        sa_file = args.sa_file or str(Path(args.output_dir) / 'qpu_labeling_full_sa.json')
        qpu_file = args.qpu_file or str(Path(args.output_dir) / 'qpu_labeling_full_qpu.json')
        # Try pilot files if full not found
        if not Path(sa_file).exists():
            sa_file = str(Path(args.output_dir) / 'qpu_labeling_pilot_sa.json')
        if not Path(qpu_file).exists():
            qpu_file = str(Path(args.output_dir) / 'qpu_labeling_pilot_qpu.json')
        analyze_gap(sa_file, qpu_file, args.output_dir)
        return

    if args.dry_run:
        cal = Path(args.output_dir) / 'calibration_timing.json'
        est = estimate_cost(args.phase, str(cal) if cal.exists() else None)
        print(f"\n  DRY RUN -- would run {args.phase} in {args.mode} mode")
        print(f"  {est['n_instances']} instances, {est['n_grid_points']} grid points")
        print(f"  Est. cost: ${est['estimated_cost_usd']:.4f}")
        print(f"  Cap: ${est['cost_cap_usd']:.2f}")
        return

    run_collection(
        phase=args.phase,
        mode=args.mode,
        resume=args.resume,
        output_dir=args.output_dir,
        calibrate_file=args.calibrate_file,
    )


if __name__ == '__main__':
    main()
