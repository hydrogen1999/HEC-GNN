"""
raw_features.py -- Transform RMS-normalized features to raw/hardware scale.

Current features (RMS-normalized):
  [h/RMS, |h|/RMS, deg_C, deg_x, |C|, delta/RMS, 1_single]

Option A: Raw scale (un-normalize)
  [h, |h|, deg_C, deg_x, |C|, delta, 1_single, RMS(J)]
  → model sees actual physical values + RMS as explicit feature

Option B: Hardware scale
  [h/J_ext_min, |h|/h_max, deg_C, deg_x, |C|, delta/J_ext_min, 1_single, J_ext_min]
  → features in hardware coordinate system

Both can be done at training time (no data regeneration) since
rms_J is stored per instance.
"""

import torch


def transform_features_to_raw(batch):
    """Transform batch features from RMS-normalized to raw scale.

    Features 0,1,5 are divided by RMS(J) in the data.
    This function multiplies them back and appends RMS(J) as feature 7.

    Modifies batch in-place, returns new feature dim (8).
    """
    x = batch['x']  # [N_qubits, 7]
    rms = batch['rms_targets']  # [B]
    chain_batch = batch['chain_batch']
    graph_batch = batch['graph_batch']

    # Map each qubit to its graph's RMS
    qubit_graph = graph_batch[chain_batch]  # [N_qubits]
    qubit_rms = rms[qubit_graph]  # [N_qubits]

    # Un-normalize features 0, 1, 5
    x_raw = x.clone()
    x_raw[:, 0] = x[:, 0] * qubit_rms  # h_raw
    x_raw[:, 1] = x[:, 1] * qubit_rms  # |h|_raw
    x_raw[:, 5] = x[:, 5] * qubit_rms  # delta_raw

    # Append RMS(J) as 8th feature
    x_raw = torch.cat([x_raw, qubit_rms.unsqueeze(-1)], dim=-1)  # [N, 8]

    batch['x'] = x_raw
    return 8  # new feature dim


def transform_features_to_hardware(batch):
    """Transform batch features to hardware scale.

    Normalize by D-Wave Advantage limits instead of RMS(J):
      h features → divide by h_max (4.0)
      J features → divide by |J_ext_min| (2.0)

    Modifies batch in-place, returns new feature dim (9).
    """
    H_MAX = 4.0
    J_EXT_MIN = 2.0

    x = batch['x']  # [N_qubits, 7]
    rms = batch['rms_targets']  # [B]
    chain_batch = batch['chain_batch']
    graph_batch = batch['graph_batch']
    qubit_graph = graph_batch[chain_batch]
    qubit_rms = rms[qubit_graph]

    x_hw = x.clone()
    # Un-normalize from RMS, then re-normalize by hardware limits
    x_hw[:, 0] = (x[:, 0] * qubit_rms) / H_MAX    # h / h_max
    x_hw[:, 1] = (x[:, 1] * qubit_rms) / H_MAX    # |h| / h_max
    x_hw[:, 5] = (x[:, 5] * qubit_rms) / J_EXT_MIN  # delta / J_ext_min

    # Append RMS/J_ext_min ratio and RMS as extra features
    rms_ratio = qubit_rms / J_EXT_MIN
    x_hw = torch.cat([x_hw, qubit_rms.unsqueeze(-1), rms_ratio.unsqueeze(-1)], dim=-1)  # [N, 9]

    batch['x'] = x_hw
    return 9  # new feature dim
