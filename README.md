# HEC-GNN: A Hierarchical Hardware-aware GNN for Chain Strength Selection in Quantum Annealing

Official code for the NeurIPS 2026 submission.

Code: [https://anonymous.4open.science/r/HEC-GNN](https://anonymous.4open.science/r/HEC-GNN)

## Setup

```bash
conda create -n hecgnn python=3.10 -y
conda activate hecgnn
pip install -r requirements.txt
```

**Hardware requirements:**
- Training: 1 GPU, ~2-4 hours per seed
- Inference: CPU only, 1.4 ms per instance
- Dataset generation: multi-core CPU, 8-24 hours depending on size
- QPU labeling: D-Wave Leap account (~$335 for 600 instances)

## Project Structure

```
hecgnn/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/default.yaml          # All hyperparameters
├── train.py                      # Training (HEC-GNN, FlatGNN, baselines)
├── evaluate.py                   # Full evaluation pipeline
├── src/
│   ├── models/
│   │   ├── hec_gnn.py            # HEC-GNN (2.75M params, 3-stage hierarchy)
│   │   ├── layers.py             # GIN, GINE, MPNN (no PyG dependency)
│   │   ├── baselines.py          # UTC, Scaled, Mean, LinearReg, FlatGNN, BO, Few-Shot, Oracle
│   │   └── all_baselines.py      # R-GCN, HGT, GPS, Flat-GAT, SAGE-HEC, GAT-HEC, FlatGNN-Large
│   ├── data/
│   │   ├── generate.py           # Dataset generation (SA + Boltzmann MCMC on embedded graph)
│   │   ├── dataset.py            # PyTorch Dataset + collate_batch
│   │   └── parallel_generate.py  # Multi-process generation
│   └── utils/
│       └── ising_util.py         # p_solve, TTS, Gap_best, CBR
├── scripts/
│   ├── generate_large_scale.py   # Large-scale OOD generation (n=100-1000)
│   ├── qpu_labeling_600.py       # QPU label collection (D-Wave Advantage)
│   └── qpu_adapt_600.py          # QPU adaptation (head-only fine-tuning)
└── figures/
    └── fig_qpu_adaptation.py     # Self-contained QPU figure (Figure 3)
```

## Model Architecture (Section 4)

HEC-GNN is a three-stage hierarchical GNN (2.75M parameters) that mirrors the physical hierarchy of minor embedding:

**Input:** The embedded graph G_E with scale-invariant features (Section 4.1, App. B.1):
- 7-dim qubit features: h/RMS, |h|/RMS, intra-chain degree, inter-chain degree, |C_i|, defection incentive, singleton indicator
- All coupling-dependent features normalized by RMS(J) for scale invariance

**Three-stage hierarchy (Section 4.1):**
- **Stage 1: Intra-chain encoding.** Each chain C_i processed independently by L_1=3 GIN layers. Qubit states pooled via sum, mean, max, and min aggregators. Max/min branches expose weakest-link and min-cut vulnerabilities (Prop. 3).
- **Stage 2: Inter-chain pairing.** A single MPNN layer over inter-chain edges E_x reconstructs qubit-level states from [c_i^(1) || x_q], propagates external loads, and re-pools to refined chain representations. Residual connection preserves Stage-1 encoding.
- **Stage 3: Logical-graph integration.** Chain representations augmented with log|C_i| placed on G_L nodes. L_3=3 GINEConv layers with 5-dim logical edge features. Global readout: sum + mean + max pool.

**Hardware-conditioned curve head (Section 4.2):**

The prediction head maps the global readout to a curve:

```
E_hat = MLP_curve(z, alpha)
```

where **alpha = [alpha(r_1 RMS(J)), ..., alpha(r_K RMS(J))]** is the closed-form hardware rescaling vector evaluated on the same grid as the output curve. Injecting alpha makes the compression regime explicit: the model does not need to infer from data where increasing J_c begins to flatten the logical signal.

**Decoding (Section 4.2):** At inference, k* = argmin_k E_hat(r_k), refined via local parabolic interpolation on the log-grid. Deployed chain strength: J_c* = r* RMS(J). Single forward pass, 1.4 ms.

**No PyTorch Geometric dependency.** All message passing via scatter operations in `layers.py`.

## Key Contributions

1. **Scalar reduction under global rescaling (Theorem 1).** Under Boltzmann sampling with global hardware rescaling and chain-consistency-rejection unembedding, per-chain strengths are dominated by their uniform envelope. The deployed control is structurally scalar.

2. **Hardware-aware curve prediction.** HEC-GNN predicts the full energy-response curve E(r) with closed-form alpha(J_c) injection, rather than a point estimate. Curve supervision exposes basin geometry; Lemma 4 gives a regret bound for curve error while Remark 1 shows scalar error alone cannot control regret.

3. **Surrogate-to-QPU adaptation.** Surrogate training on classical curves + head-only fine-tuning on 20 QPU-labeled calibration instances per fold. 2.1% of parameters updated.

## Training Details

- **Loss:** L = L_energy + beta_cbr L_cbr, where L_energy is L1 on normalized excess-energy curves (Eq. 1), L_cbr is per-chain break BCE auxiliary
- **Optimizer:** AdamW (lr=5e-4, wd=1e-4), linear warmup (5 epochs) + cosine decay, 200 epochs
- **Early stopping:** patience 30 on validation MAE(r*)
- **Gradient clipping:** max_norm=1.0
- **Seeds:** 3 independent runs (42, 123, 7)
- **Batch size:** 32

## Datasets

### Surrogate labels (Section 5, App. C.2)
- **Boltzmann** (n <= 40): MCMC on rescaled embedded Hamiltonian H_eff = H_emb/alpha, beta=2.0, 50 chains x 500 burn-in x 500 sampling
- **SA** (OOD, n > 40): neal SimulatedAnnealingSampler, 200 reads x 500 sweeps, on H_eff
- Both operate on the **embedded** (physical) graph, decode via majority vote, report logical energy
- True logical optima from GUROBI 11 (n <= 40) or SA natural mode

### Main benchmark (~96K instances, Table 1)
- 5 families: Random Ising, Sherrington-Kirkpatrick, weighted MaxCut, planted solution, 3-regular MaxCut
- Logical sizes: n in {8, 10, 12, 15, 20, 25, 30, 35, 40}
- Topologies: Pegasus P4/P8/P16 (70/14/16 train/val/test) + Zephyr Z4 (zero-shot)
- Embedding: minorminer, fixed seeds, 120s timeout

### Embedding transfer (~29K instances, Table 2 / App. C.9)
- Tuned minorminer (multi-seed best-of-10), Pegasus/Zephyr clique, clique-init+minorminer hybrid

### OOD size extrapolation (Figure 2, App. C.11)
- Train: ~120K instances on P16, n <= 40
- Test: n in {50, 75, 100, 200, 500, 1000}

### QPU benchmark (600 instances, Table 3, Figure 3)
- D-Wave Advantage (Pegasus P16, 5627 qubits)
- n in {8, 10, 12, 15, 20, 25, 30, 40}, 5 families
- K=20 grid points, 1000 anneals per J_c, majority-vote decoding
- 5-fold head-fine-tuning: 20 QPU-labeled calibration instances per fold

## Baselines (23 methods, App. C.6)

| Category | Methods |
|----------|---------|
| Boundary diagnostics | J_c=0, J_c=J_max |
| Heuristics | UTC, Scaled(2.0), Mean |
| Learned scalar zero-shot | LinearReg, XGBoost, MLP-18 |
| Regret-aligned HEC backbone | SPO+, Plackett-Luce |
| Learned curve: flat | FlatGNN (262K), FlatGNN-Large (2.7M) |
| Learned curve: heterogeneous | R-GCN, HGT, GPS, Flat-GAT |
| Learned curve: hierarchical | SAGE-HEC, GAT-HEC, **HEC-GNN (2.75M)** |
| Search-based | Few-Shot-3/5/10, BO-10, Oracle (grid) |

## Reproducing Paper Results

### Table 1: Multi-topology benchmark

```bash
# Generate dataset (~96K instances, 5 families, 4 topologies)
python -m src.data.parallel_generate \
    --benchmark multi_topo --n-instances 100000 --labeling sa \
    --workers 20 --output-dir data/diverse_sa_mt
python -m src.data.parallel_generate --merge --output-dir data/diverse_sa_mt

# Train HEC-GNN (3 seeds)
python train.py --model hec_gnn --data-dir data/diverse_sa_mt \
    --benchmark multi_topo --seeds 42 123 7 --output-dir results/hec_gnn_mt

# Train FlatGNN baseline (3 seeds)
python train.py --model flat_gnn --data-dir data/diverse_sa_mt \
    --benchmark multi_topo --seeds 42 123 7 --output-dir results/flat_gnn_mt

# Evaluate all 23 methods
python evaluate.py --benchmark all --data-dir data/diverse_sa_mt \
    --model-dir results/hec_gnn_mt
```

**Expected:** HEC-GNN alpha_5% = 97.9 +/- 0.4%, alpha_2% = 88.5 +/- 0.4%, Gap 0.9%, CBR 10.8%, at 1.4 ms/instance.

### Table 2: Instance stratification

```bash
# Uses same multi-topology data, stratified by basin width and family
python evaluate.py --benchmark stratification --data-dir data/diverse_sa_mt \
    --model-dir results/hec_gnn_mt
```

**Expected:** HEC-GNN leads on narrow basins (+8.7 pp over FlatGNN-L, +11.9 pp over XGBoost) and planted-solution family (+19.2/+22.1 pp).

### Figure 2 / OOD size extrapolation

```bash
# Generate OOD data (train n<=40, test n=50..1000)
python -m src.data.parallel_generate \
    --benchmark ood_train --n-instances 120000 --labeling sa \
    --workers 20 --output-dir data/diverse_sa_ood_train
python -m src.data.parallel_generate \
    --benchmark ood_test --n-instances 10000 --labeling sa \
    --workers 20 --output-dir data/diverse_sa_ood_test

# Large-scale OOD (n=100..1000)
python scripts/generate_large_scale.py --workers 28 --output-dir data/large_scale_ood

# Train + evaluate
python train.py --model hec_gnn --data-dir data/diverse_sa_ood_train \
    --benchmark ood --seeds 42 123 7 --output-dir results/hec_gnn_ood
python evaluate.py --benchmark ood --data-dir data/diverse_sa_ood_test \
    --model-dir results/hec_gnn_ood
```

**Expected:** HEC-GNN degrades from 96.3% (n=50) to 38.7% (n=1000), +9.2 pp gap over FlatGNN-Large at n=1000 (25x extrapolation).

### Table 3 / Figure 3: QPU transfer

```bash
# Step 1: Collect QPU labels (requires DWAVE_API_TOKEN, ~$335)
DWAVE_API_TOKEN=xxx python scripts/qpu_labeling_600.py \
    --phase calibrate --mode qpu --output-dir qpu_data    # $0.69, measure timing
DWAVE_API_TOKEN=xxx python scripts/qpu_labeling_600.py \
    --phase pilot --mode qpu --output-dir qpu_data        # $8, verify correlation
DWAVE_API_TOKEN=xxx python scripts/qpu_labeling_600.py \
    --phase large --mode qpu --output-dir qpu_data        # ~$335, full dataset

# Step 2: Evaluate all methods + head-only fine-tuning (5-fold CV)
python scripts/qpu_adapt_600.py \
    --qpu-data qpu_data/qpu_labeling_large_qpu.json \
    --embeddings qpu_data/embeddings_large.json \
    --hec-dir results/hec_gnn_mt \
    --flat-dir results/flat_gnn_mt \
    --finetune --n-folds 5 --output qpu_adaptation_results.json
```

**Expected:** HEC-GNN (FT) alpha_5% = 92.7 +/- 1.6%, alpha_2% = 81.2 +/- 3.4%, Gap 1.42 +/- 0.42%, leading every aggregate metric. Head-only fine-tuning updates 2.1% of parameters (58,836 / 2,753,756).

## Citation

```bibtex
@inproceedings{hecgnn2026,
  title={{HEC-GNN}: A Hierarchical Hardware-aware {GNN} for Chain Strength Selection in Quantum Annealing},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT
