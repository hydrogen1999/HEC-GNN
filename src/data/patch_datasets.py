#!/usr/bin/env python3
"""
patch_datasets.py -- Add h_phys/J_phys/chain_edge_map to existing datasets.

Reconstructs physical Hamiltonian from stored h_logical/J_logical/embedding.
Safe to run on datasets that already have these fields (skips them).

Usage:
  python patch_datasets.py datasets_v3_full/multi_topo_train.pkl
  python patch_datasets.py datasets_v3_full/*.pkl
"""

import pickle
import sys
from pathlib import Path

import dwave_networkx as dnx

from src.data.generate import (
    build_physical_hamiltonian, get_hardware_graph, TOPOLOGIES
)


def patch_instance(inst):
    """Add h_phys, J_phys, chain_edge_map if missing."""
    if 'h_phys' in inst and inst['h_phys'] is not None:
        return False  # already patched

    topo = inst.get('topology', 'P16')
    hw_graph = get_hardware_graph(topo)

    h_phys, J_phys, chain_edge_map, chain_edges_flat = build_physical_hamiltonian(
        inst['h_logical'], inst['J_logical'], inst['embedding'], hw_graph)

    inst['h_phys'] = h_phys
    inst['J_phys'] = J_phys
    inst['chain_edge_map'] = chain_edge_map
    return True


def patch_file(path):
    """Patch all instances in a pickle file."""
    print(f"Patching {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    n_patched = 0
    for inst in data:
        if patch_instance(inst):
            n_patched += 1

    if n_patched > 0:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Patched {n_patched}/{len(data)} instances")
    else:
        print(f"  Already up to date ({len(data)} instances)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_datasets.py <file.pkl> [file2.pkl ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        patch_file(path)
