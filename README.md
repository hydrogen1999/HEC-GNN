# HEC-GNN: Hierarchical Hardware-aware GNN for Chain Strength Selection in Quantum Annealing

Anonymous code release accompanying the paper submission.

This repository contains the training, evaluation, and dataset generation code for HEC-GNN, a three-stage hierarchical graph neural network that predicts the optimal chain strength on a per-instance basis for minor-embedded Ising problems on D-Wave-style hardware.

**Start here:** read [`DESIGN.md`](DESIGN.md) for the paper-to-code mapping, design decisions, and full experimental results inventory. This README covers installation and quick-start; `DESIGN.md` covers everything else.

## Repository layout

```
.
├── src/                            # Original paper code
│   ├── models/
│   │   ├── hec_gnn.py              # HEC-GNN architecture (three-stage hierarchy)
│   │   ├── baselines.py            # FlatGNN and FlatGNN-Large baselines
│   │   ├── all_baselines.py        # SAGE-HEC, GAT-HEC, RGCN, GPS, HGT, FlatGAT
│   │   ├── gnn_baselines.py        # Additional GNN variants
│   │   ├── layers.py               # Low-level GNN layers (GINE, scatter ops, etc.)
│   │   └── gurobi_solver.py        # Reference exact solver
│   ├── data/
│   │   ├── generate.py             # Single-process dataset generator
│   │   ├── parallel_generate.py    # Multi-worker generator
│   │   ├── dataset.py              # PyTorch dataset interface
│   │   ├── gpu_sampler.py          # GPU-resident SA sampler
│   │   └── patch_datasets.py       # Dataset post-processing
│   └── utils/
│       └── ising_util.py           # Ising-model utilities
│
├── hecgnn_trainer/                 # Modular training package
│   ├── config.py                   # YAML-driven experiment configs (ModelConfig.alpha_mode default = "hardware")
│   ├── registry.py                 # @register decorator + build_model factory (52 architectures)
│   ├── engine.py                   # TrainingEngine + DistillEngine
│   ├── cli.py                      # python -m hecgnn_trainer.cli {list,train,sweep}
│   └── models/
│       ├── alpha_injection.py      # V2 D-Wave auto_scale α + AlphaInjectionWrapper
│       ├── jc_injection.py         # JCDirectHEC, HardwareNormHEC
│       ├── scalar_ensemble.py      # ScalarEnsembleWrapper, ScalarDropoutWrapper
│       ├── agnn.py                 # Anisotropic GNN (Flat + HEC)
│       ├── extended.py             # GCN/GIN/SAGE/GATv2/PNA/GatedGCN/EdgeConv/APPNP/DeepSets + HECVariant
│       ├── compact.py              # CompactHEC, TinyHEC, SharedHEC, distillation methods, pruning
│       ├── mlp_model.py            # MLP baselines on 18 hand-crafted features
│       └── raw_features.py         # Feature normalization
│
├── experiments/                    # Sweep YAML configs
│   ├── sweep_all_architectures.yaml
│   ├── sweep_27_architectures.yaml
│   ├── ablation_loss_and_norm.yaml
│   ├── flatgnn_alpha_ablation.yaml
│   └── distillation_sweep.yaml
│
├── scripts/                        # Standalone utilities
│   ├── v2_smoke_test.py            # Build every architecture (verify install)
│   ├── qpu_labeling_600.py         # QPU benchmark dataset construction
│   ├── qpu_adapt_600.py            # Head-only QPU fine-tuning protocol
│   ├── generate_large_scale.py     # OOD generator for n ∈ {50, 75, 100, 200, 500, 1000}
│   ├── fast_emb_v2.py              # Embedding utilities
│   ├── sa_natural_sweep.py         # Simulated-annealing sweep
│   ├── compute_true_optimal_all.py # GUROBI ground-truth solver
│   └── eval_true_optimal.py
│
├── train.py                        # Single-config training entry point
├── evaluate.py                     # Evaluation entry point
├── launch_single.py                # Lightweight per-arch launcher (used by sweeps)
├── timing.py                       # Inference-time benchmark
├── ablation.py                     # Aggregated ablation runner
├── generate_figures.py             # Paper figures from result JSONs
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT
├── DESIGN.md                       # Design document (paper-to-code mapping, design decisions, results)
└── README.md                       # this file
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.0+, and an NVIDIA GPU with ≥40 GB memory for the large architectures.

## Quick verification

A smoke test that instantiates every registered architecture under the default V2 hardware α-injection:

```bash
python scripts/v2_smoke_test.py
```

Expected output: 29 architectures built cleanly, all reporting `alpha_mode=hardware`.

List all registered architectures:

```bash
python -m hecgnn_trainer.cli list
```

## Training

Single-architecture training (uses the default `alpha_mode="hardware"`):

```bash
python -m hecgnn_trainer.cli train \
    --arch hec_gnn \
    --data-dir data/diverse_sa_mt \
    --epochs 200 --patience 30 --seeds 42,123,7
```

YAML sweep:

```bash
python -m hecgnn_trainer.cli sweep --config experiments/sweep_all_architectures.yaml
```

To disable V2 α-injection for ablation, set `model.alpha_mode: "none"` in the YAML, or pass `--config` with a YAML that sets it explicitly. The V1 legacy formula is also available as `alpha_mode: "rms"`.

## Dataset

Synthetic datasets are generated by `src/data/generate.py` (single-process) or `src/data/parallel_generate.py` (multi-worker). Five problem families (Random Ising, SK, weighted MaxCut, planted, 3-reg MaxCut) are embedded onto four hardware topologies (Pegasus P4/P8/P16 and Zephyr Z4) via `minorminer`.

The QPU benchmark protocol is in `scripts/qpu_labeling_600.py` (label collection) and `scripts/qpu_adapt_600.py` (head-only fine-tuning).

The C++ Boltzmann sampler used for surrogate labelling is located via the `BOLTZMANN_CPP_BIN` environment variable, or it is expected at `src/data/boltzmann_sampler`.

## Reproducibility notes

- **Default normalization is V2** (D-Wave Advantage `auto_scale`). Every registered architecture is automatically wrapped with the V2 α-injection at the prediction head unless the architecture already manages α internally (the `_SKIP_ALPHA_WRAP` set in `hecgnn_trainer/registry.py`).
- **Scalar ensemble vs curve head.** Two prediction strategies are supported. `scalar_ensemble_hec` predicts r\* directly using K=20 independent scalar heads and reports the median; `hec_gnn` (curve mode) regresses a K=20 normalized excess-energy curve and decodes by argmin with parabolic refinement.
- **Seeds.** All results in the paper use three seeds (42, 123, 7). The training engine writes one checkpoint and one JSON result file per seed.
- **Determinism.** PyTorch deterministic-mode is not enforced by default. For exact reproduction, set `torch.use_deterministic_algorithms(True)` and the appropriate CUDA flags before calling `TrainingEngine.run()`.

## License

MIT — see `LICENSE`.
