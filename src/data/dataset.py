"""
dataset.py -- DataLoader and collation for HEC-GNN.

Converts instance dicts (from generate_v3_datasets.py) into batched tensors
that the model expects. Handles variable-size graphs via offset-based batching.
"""

import math
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Grid matching V3 paper
K = 20
R_MIN = 0.02
R_MAX = 5.0
GRID = np.logspace(math.log10(R_MIN), math.log10(R_MAX), K).astype(np.float32)


class ChainStrengthDataset(Dataset):
    """Dataset wrapping a list of instance dicts or a pickle file."""

    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'rb') as f:
                self.instances = pickle.load(f)
        else:
            self.instances = data

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


def collate_batch(instances: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-size graph instances into a single batched dict.

    Handles offset-based batching: edge indices are shifted by cumulative
    node/chain counts so all graphs can be processed as one big disconnected graph.

    Returns batch dict with keys matching HECGNN.forward() expectations.
    """
    all_x = []
    all_chain_edge_src = []
    all_chain_edge_dst = []
    all_inter_edge_src = []
    all_inter_edge_dst = []
    all_inter_edge_attr = []
    all_logical_edge_src = []
    all_logical_edge_dst = []
    all_logical_edge_attr = []
    all_chain_batch = []
    all_graph_batch = []
    all_chain_lengths = []
    all_energy_curves = []
    all_cbr_targets = []
    all_rms_targets = []
    all_r_star = []

    qubit_offset = 0
    chain_offset = 0

    for gi, inst in enumerate(instances):
        n_qubits = len(inst['qubit_features'])
        n_chains = inst['n_chains']

        # Node features
        all_x.append(torch.tensor(inst['qubit_features'], dtype=torch.float32))

        # Chain assignment (qubit -> chain index, offset by chain_offset)
        ca = torch.tensor(inst['chain_assignment'], dtype=torch.long) + chain_offset
        all_chain_batch.append(ca)

        # Graph batch (chain -> graph index)
        all_graph_batch.append(torch.full((n_chains,), gi, dtype=torch.long))

        # Chain lengths
        chain_assign = inst['chain_assignment']
        lengths = [0] * n_chains
        for c in chain_assign:
            lengths[c] += 1
        all_chain_lengths.append(torch.tensor(lengths, dtype=torch.long))

        # Intra-chain edges (offset by qubit_offset)
        if inst['chain_edge_index']:
            cei = torch.tensor(inst['chain_edge_index'], dtype=torch.long).T  # [2, E]
            all_chain_edge_src.append(cei[0] + qubit_offset)
            all_chain_edge_dst.append(cei[1] + qubit_offset)

        # Inter-chain edges (offset by qubit_offset)
        if inst['inter_edge_index']:
            iei = torch.tensor(inst['inter_edge_index'], dtype=torch.long).T  # [2, E]
            all_inter_edge_src.append(iei[0] + qubit_offset)
            all_inter_edge_dst.append(iei[1] + qubit_offset)
            all_inter_edge_attr.append(
                torch.tensor(inst['inter_edge_features'], dtype=torch.float32))

        # Logical edges (offset by chain_offset)
        if inst['logical_edge_index']:
            lei = torch.tensor(inst['logical_edge_index'], dtype=torch.long).T  # [2, E]
            all_logical_edge_src.append(lei[0] + chain_offset)
            all_logical_edge_dst.append(lei[1] + chain_offset)
            all_logical_edge_attr.append(
                torch.tensor(inst['logical_edge_features'], dtype=torch.float32))

        # Targets
        all_energy_curves.append(torch.tensor(inst['energy_curve'], dtype=torch.float32))
        all_cbr_targets.append(torch.tensor(inst['chain_break_targets'], dtype=torch.float32))
        all_rms_targets.append(inst['rms_J'])
        all_r_star.append(inst['r_star'])

        qubit_offset += n_qubits
        chain_offset += n_chains

    # Stack/concatenate everything
    batch = {
        'x': torch.cat(all_x, dim=0),
        'chain_edge_index': _cat_edge_index(all_chain_edge_src, all_chain_edge_dst),
        'inter_edge_index': _cat_edge_index(all_inter_edge_src, all_inter_edge_dst),
        'inter_edge_attr': torch.cat(all_inter_edge_attr, dim=0) if all_inter_edge_attr
                           else torch.zeros(0, 3),
        'chain_batch': torch.cat(all_chain_batch, dim=0),
        'logical_edge_index': _cat_edge_index(all_logical_edge_src, all_logical_edge_dst),
        'logical_edge_attr': torch.cat(all_logical_edge_attr, dim=0) if all_logical_edge_attr
                             else torch.zeros(0, 5),
        'graph_batch': torch.cat(all_graph_batch, dim=0),
        'chain_lengths': torch.cat(all_chain_lengths, dim=0),
        'batch_size': len(instances),
        # Targets
        'energy_curve': torch.stack(all_energy_curves, dim=0),  # [B, K]
        'cbr_targets': torch.cat(all_cbr_targets, dim=0),       # [N_chains_total]
        'rms_targets': torch.tensor(all_rms_targets, dtype=torch.float32),  # [B]
        'r_star': torch.tensor(all_r_star, dtype=torch.float32),  # [B]
    }
    return batch


def _cat_edge_index(src_list, dst_list):
    if src_list:
        return torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
    return torch.zeros(2, 0, dtype=torch.long)


def make_dataloaders(train_path, val_path, test_path=None, batch_size=32,
                     num_workers=0):
    """Create DataLoaders from pickle files."""
    train_ds = ChainStrengthDataset(train_path)
    val_ds = ChainStrengthDataset(val_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_batch, num_workers=num_workers)

    test_loader = None
    if test_path:
        test_ds = ChainStrengthDataset(test_path)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_batch, num_workers=num_workers)

    return train_loader, val_loader, test_loader
