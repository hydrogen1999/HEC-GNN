# HEC-GNN — Design Document

> Companion to the paper "HEC-GNN: A Hierarchical Hardware-aware GNN for Chain Strength Selection in Quantum Annealing." This document is the contract between the paper's claims and the code in this release.

---

## 0. How to read this document

| You want to ... | Section |
|---|---|
| Understand the problem we solve | §1 |
| Find the file that implements paper §X | §10 *Code-to-paper mapping* |
| Reproduce a specific table or figure | §9 *Results* + §11 *Reproducibility* |
| Add a new architecture | §12 *Extension points* |
| Understand why we chose option A over B | §6 *Design decisions* |
| See the V2 hardware-rescaling formula | §2.3 + §4.2 |
| See the scalar-ensemble head | §4.3 |

The document is self-contained: a reviewer with the paper and this code release should not need any external context.

---

## 1. Problem statement

A D-Wave Advantage quantum annealer with Pegasus topology has 5 627 working qubits laid out in a sparse coupling graph. A logical Ising problem $H_L = \langle h, \sigma\rangle + \sigma^\top J \sigma$ with $n$ logical spins is compiled onto the device by *minor embedding*: each logical variable $i$ becomes a *chain* $C_i$ of $|C_i|$ physical qubits coupled through ferromagnetic chain couplers of strength $-J_c$. The chain strength $J_c$ is a free control:

- Too small: chains fracture under thermal noise (high chain-break rate, decoded solutions become noise).
- Too large: D-Wave's `auto_scale` rescales every Hamiltonian coefficient by $\alpha(J_c) = 1/S(J_c)$, where $S$ grows linearly with $J_c$ once chain couplers dominate. The logical signal is compressed below the annealer's noise floor.

The optimal $J_c^\star$ depends jointly on $(h, J)$, the embedding geometry (chain sizes, defection structure), and the hardware rescaling response. Production approaches either ignore most of these factors (UTC, scaled-RMS, the D-Wave default $J_c \propto \max |J_{ij}|$) or require many simulator / QPU evaluations (PIMC, Bayesian optimization, oracle grid search).

This work delivers a single-instance predictor that takes the embedded Ising problem and emits $\hat J_c^\star$ in milliseconds. Section §3 of the paper formalises the deployed quantity as a scalar (Theorem 1). Section §4 develops HEC-GNN. Section §5 evaluates ~96K simulated instances across five problem families and four hardware topologies, plus 600 D-Wave Advantage instances. This document maps every piece of that into running code.

---

## 2. Theoretical foundation

### 2.1 Notation (paper §3.1)

| Symbol | Meaning | Code |
|---|---|---|
| $h \in \mathbb R^n$ | Logical biases | `batch['x'][:, 0]` (after RMS normalization) |
| $J \in \mathbb R^{n \times n}$ | Logical couplings | `batch['logical_edge_attr']` |
| $C_i$ | Chain of logical variable $i$ | `batch['chain_batch']` (qubit → chain index) |
| $J_c$ | Chain strength | scalar; the quantity we predict |
| $r = J_c / \mathrm{RMS}(J)$ | Dimensionless chain-strength ratio | `batch['r_star']` (target) |
| $r_k$ | $k$-th grid point | `GRID = np.logspace(log10(0.02), log10(5.0), K)` |
| $\alpha(J_c)$ | Hardware rescaling factor | `compute_alpha_vector(mode="hardware")` |
| $\bar E(r_k)$ | Normalized excess-energy curve | training target |
| $\delta_E(\tau)$ | Compliance at tolerance $\tau$ | `engine.evaluate_model` |

### 2.2 The scalar reduction (paper Theorem 1)

> Under Boltzmann sampling on the sparse Ising machine model with global hardware rescaling and chain-consistency-rejection unembedding, no per-chain assignment $\boldsymbol\lambda = (\lambda_1, \ldots, \lambda_n)$ achieves strictly higher solution probability than its uniform envelope $\max_i \lambda_i$.

Consequence: the deployed control is one-dimensional. The whole code base is built around this scalar interface. The architecture is allowed to use per-chain heterogeneity *as input information* (via Stage 1's qubit-level features), but the prediction it emits is a single $\hat r^\star \in \mathbb R_{>0}$.

This is the central reason the registry contains both *curve mode* (predict $\bar E(r_k)$ for $K=20$ grid points, decode by argmin) and *scalar ensemble mode* (predict $r^\star$ directly with K independent scalar heads, take the median). Both interfaces emit a single scalar at deployment.

### 2.3 Hardware rescaling (paper §3.3)

D-Wave Advantage applies `auto_scale` at submission time:

$$
S(r_k) = \max\{\frac{\max h}{H_{\max}},\; \frac{-\min h}{H_{\max}},\; \frac{\max J}{J^+_{\max}},\; \frac{-\min J}{J^-_{\max}},\; \frac{r_k \cdot \mathrm{RMS}(J)}{J^-_{\max}},\; 1\}, \qquad \alpha(r_k) = \frac{1}{S(r_k)}
$$

with the Pegasus hardware constants $H_{\max} = 4$, $J^+_{\max} = 1$, $J^-_{\max} = 2$. This formula is the contract that the prediction head must respect: changing $J_c$ changes $\alpha$, which compresses every Hamiltonian coefficient proportionally. The model must reason about which side of the threshold $S = 1$ each grid point lies on.

The code computes this formula at `hecgnn_trainer/models/alpha_injection.py::compute_alpha_vector` with `mode="hardware"`. A simpler alternative formula $\alpha_{\mathrm{rms}}(r) = 1/\max(1, r)$ is also retained at `mode="rms"` for the ablation table.

### 2.4 Cliff-shape regret bound (paper Lemma 1, Remark 1)

> If $\hat{\bar E}$ approximates $\bar E$ within $\epsilon$ in $L^1$ on the grid, the deployed regret is bounded by $C \cdot \epsilon$ for a Lipschitz constant $C$ depending on basin width. The bound is tight on cliff-shaped basins; scalar regression of $r^\star$ alone admits arbitrarily large regret on the same family.

This is what justifies *curve* supervision in the original paper. Empirically, the K=20 scalar-ensemble head bypasses the cliff failure by averaging over K independent regressors (each with its own random init), which is why both heads coexist in the registry.

---

## 3. System architecture

### 3.1 Layered view

```
┌─────────────────────────────────────────────────────────────────┐
│                       Entry points (CLI / scripts)               │
│   cli.py · train.py · evaluate.py · launch_single.py · ablation │
└─────────────────┬───────────────────────────────────────────────┘
                  │ build_model(cfg) + TrainingEngine(cfg).run()
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│             hecgnn_trainer/ — modular training package           │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  Registry            │    │   Engine                     │   │
│  │  registry.py         │    │   engine.py                  │   │
│  │  • @register         │    │   • TrainingEngine.run       │   │
│  │  • build_model       │    │   • DistillEngine            │   │
│  │  • _maybe_wrap_alpha │    │   • compute_loss             │   │
│  └────────┬─────────────┘    └─────────┬────────────────────┘   │
│           │ wraps                       │ orchestrates           │
│           ▼                             ▼                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Models layer                                             │   │
│  │  • alpha_injection.py  — V2 D-Wave alpha + Wrapper        │   │
│  │  • scalar_ensemble.py  — K-head ensemble + MC dropout     │   │
│  │  • jc_injection.py     — JCDirectHEC, HardwareNormHEC     │   │
│  │  • agnn / extended / compact — 30+ backbone variants       │   │
│  │  • _wrapper_utils.py   — shared CaptureZ + head-detection │   │
│  └──────────────────────┬───────────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────────┘
                          │ imports
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           src/ — paper code (canonical implementations)          │
│   models/hec_gnn.py  ←  paper §4 architecture                    │
│   models/layers.py   ←  GINELayer, scatter_{add,mean,max,min}    │
│   models/baselines.py · models/all_baselines.py                  │
│   data/generate.py   ←  paper §5.1 dataset construction          │
│   data/dataset.py    ←  PyTorch Dataset + collate                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module responsibilities

| Module | Lines | Responsibility | Paper section |
|---|---|---|---|
| `src/models/hec_gnn.py` | 365 | Three-stage hierarchical backbone (Stage 1 GIN, Stage 2 inter-chain pairing, Stage 3 logical GINEConv), four-way pooling, energy / CBR / RMS heads. | §4.1, §4.2 |
| `src/models/layers.py` | 360 | Primitive layers: `GINELayer`, `MPLayer`, scatter ops. | §4.1 |
| `src/models/baselines.py` | 350 | `FlatGNN`, `FlatGNNLarge` — parameter-matched flat ablation. Defines the canonical `GRID` constant and head sizes. | §5.2 capacity-matched ablation |
| `src/models/all_baselines.py` | 470 | `SAGE-HEC`, `GAT-HEC`, `R-GCN`, `HGT`, `GPS`, `FlatGAT`, `HeteroSAGE`. | §5.3 hierarchy-vs-aggregator ablation |
| `src/data/generate.py` | 1185 | Multi-topology dataset generation. Embeds Random Ising / SK / weighted MaxCut / planted / 3-reg MaxCut onto Pegasus P4/P8/P16 and Zephyr Z4. Labels with SA or Boltzmann sampling at $K=20$ log-spaced ratios. | §5.1 |
| `src/data/dataset.py` | 280 | `MultiTopologyDataset`, `collate_batches`. Defines the batch dict that flows through every forward pass. | §5.1 |
| `hecgnn_trainer/config.py` | 218 | `ModelConfig` (with the **`alpha_mode = "hardware"` default**), `TrainConfig`, `DataConfig`, `ExperimentConfig`, `SweepConfig`, YAML loaders. | §5.1 hyperparameters |
| `hecgnn_trainer/registry.py` | 644 | `@register` decorator, `build_model`, 52 canonical builders, the alpha-wrapping decision tree. | §5.2-5.5 |
| `hecgnn_trainer/engine.py` | 569 | `TrainingEngine` (AdamW + cosine schedule, four loss modes, patience-based early stop, per-seed checkpointing). `DistillEngine` (knowledge distillation, 11 methods). | §5.1 training recipe |
| `hecgnn_trainer/models/alpha_injection.py` | 318 | `compute_alpha_vector` (D-Wave auto_scale + RMS fallback), `AlphaInjectionWrapper` (generic head replacement), `FlatGNNAlpha`, `FlatGNNLargeAlpha`. | §4.2 hardware conditioning |
| `hecgnn_trainer/models/scalar_ensemble.py` | 165 | `ScalarEnsembleWrapper` (K=20 independent scalar heads, median decoding), `ScalarDropoutWrapper` (1 head + MC dropout). | §4.4 (extended) |
| `hecgnn_trainer/models/jc_injection.py` | 165 | `JCDirectHEC` (raw $J_c$ at head), `HardwareNormHEC` (V2 alpha at head, full formula). | §4.2 ablation |
| `hecgnn_trainer/models/_wrapper_utils.py` | 116 | Shared `CaptureZ` stub, `detect_head_input_dim`, `install_capture_head`. Used by all three wrappers. | (infrastructure) |
| `hecgnn_trainer/models/{agnn,extended,compact,mlp_model,raw_features}.py` | 2 066 | Architecture zoo: AGNN, GCN/GIN/SAGE/GATv2/PNA/GatedGCN/EdgeConv/APPNP, CompactHEC/SharedHEC/TinyHEC, 11 distillation methods, MLP baselines, raw-feature transforms. | §5.3 |

---

## 4. Component design

### 4.1 The three-stage hierarchical backbone (`src/models/hec_gnn.py`)

The backbone mirrors minor embedding's physical hierarchy.

**Stage 1 — intra-chain encoding** (`HECGNN.stage1`). Each chain $C_i$ is processed as an independent subgraph with $L_1$ GIN layers. GIN's multiset-injective sum aggregation preserves the intra-chain coupling distribution that mean / softmax aggregators collapse — empirically a 3.0-3.5 pp gap vs SAGE-HEC and GAT-HEC. Qubit states pool into chain representations via four aggregators (sum, mean, max, min); min/max expose weakest-link and min-cut vulnerabilities.

**Stage 2 — inter-chain pairing** (`HECGNN.stage2` + `stage2_proj`). A single MPNN layer over inter-chain edges $E_\times$. Reconstructs per-qubit states as $[c_i^{(1)} \| x_q]$, propagates external loads between chains, and re-pools. A residual connection preserves the Stage 1 encoding.

**Stage 3 — logical-graph integration** (`HECGNN.stage3`). Refined chain representations become node features on $G_L$, augmented with $\log |C_i|$, and processed by $L_3$ GINEConv layers. The global readout $z = [\text{sum} \| \text{mean} \| \text{max}]$ is the input to the head.

**Three heads** (paper §4.1):
- `energy_head`: $\mathbb R^{3H} \to \mathbb R^K$, predicts the normalized excess-energy curve.
- `cbr_head`: $\mathbb R^H \to \mathbb R^{N_{\text{chains}}}$, predicts per-chain break frequency (auxiliary).
- `rms_head`: $\mathbb R^{3H} \to \mathbb R$, predicts $\mathrm{RMS}(J)$ (auxiliary).

The auxiliary heads are trained with $\beta_{\mathrm{cbr}} = 0.1$ and $\beta_{\mathrm{rms}} = 0.05$ as in paper §5.1.

### 4.2 Hardware-rescaling alpha vector (`hecgnn_trainer/models/alpha_injection.py`)

`compute_alpha_vector(batch, K, mode)` is the single canonical implementation of $\alpha(J_c)$. Two modes are supported:

- `mode="hardware"` (default): the full D-Wave `auto_scale` formula of §2.3. Includes the problem $h$ and $J$ contributions in addition to the chain coupler term — relevant when $|h|$ or $|J|$ is large enough to dominate $S$ at small $J_c$.
- `mode="rms"`: the simpler $\alpha(r) = 1/\max(1, r)$ formula, retained for the ablation table. It ignores the actual hardware limits and is included only to demonstrate that hardware-grounded conditioning matters.

`AlphaInjectionWrapper` is the *generic* wrapper. Given any base model whose energy/curve head consumes a $3H$-dim graph representation $z$, it:

1. detects the head's input dimension via `_wrapper_utils.detect_head_input_dim`;
2. replaces the head with `CaptureZ`, a stub that records $z$ and returns a zero curve;
3. installs a fresh head MLP `Linear(3H+K, H) → Linear(H, H/2) → Linear(H/2, K)` that consumes $[z \| \alpha]$.

At forward time, the base model runs to populate $z$ and the auxiliary outputs; the new head emits the conditioned energy curve. `cbr_pred` and `rms_pred` pass through unchanged.

`wrap_with_alpha(base, K, dropout, alpha_mode)` is the registry's entry point. With `alpha_mode="none"` it returns the base model unmodified. This is the *only* knob a reviewer needs to disable the V2 normalization globally.

### 4.3 Scalar ensemble head (`hecgnn_trainer/models/scalar_ensemble.py`)

Theorem 1 says the deployed quantity is a scalar. Two implementations exploit this:

`ScalarEnsembleWrapper` — K=20 independent scalar heads, each a small MLP $\mathbb R^{3H+K} \to \mathbb R^{H/6} \to \mathbb R$. Training loss is $\mathcal L = \frac{1}{K} \sum_k |r_k^{\text{pred}} - r^\star|$. Inference takes the median of K predictions. Diversity comes from independent random init.

`ScalarDropoutWrapper` — single head, MC dropout at inference (K stochastic forwards, median). Smaller, but tied init means lower diversity.

Both wrappers use the shared head-detection + `CaptureZ` plumbing in `_wrapper_utils.py`, and both accept `alpha_mode` so the V2 alpha vector can be concatenated to $z$.

### 4.4 Registry (`hecgnn_trainer/registry.py`)

The registry is the single source of truth for model construction.

```python
@register("hec_gnn", aliases=["hecgnn", "hec"])
def _build_hec_gnn(cfg: ModelConfig):
    from src.models.hec_gnn import HECGNN, ModelConfig as HECModelConfig
    return HECGNN(HECModelConfig(...))
```

`build_model(cfg)` follows this decision tree:

1. Resolve `cfg.arch` (lowercase, normalize `-` and spaces) against `_REGISTRY`.
2. Call the builder to get a base model.
3. Canonicalise the registry key (handle aliases).
4. Apply `_maybe_wrap_alpha(canonical, model, cfg)`:
   - `cfg.alpha_mode == "none"` → return base unwrapped.
   - canonical name in `_SKIP_ALPHA_WRAP` → return base (model manages alpha itself).
   - base is already a wrapper instance → return base.
   - otherwise → `AlphaInjectionWrapper(base, K, dropout, alpha_mode)`.

`_SKIP_ALPHA_WRAP` lists architectures that consume an `alpha_mode` argument internally (`flat_gnn_alpha*`, `hw_norm_hec`, `jc_direct_hec`) or have no graph head to inject into (`mlp_curve`, `mlp_scalar`).

The registry contains **52 canonical builders** organised into six groups (sum verified by `python -m hecgnn_trainer.cli list`):

| Group | Count | Examples |
|---|---|---|
| HEC hierarchical | 9 | `hec_gnn`, `hec_agnn`, `hec_gatedgcn`, `hec_gatv2`, `hec_gcn`, `hec_pna`, `hec_edgeconv`, `sage_hec`, `gat_hec` |
| Flat GNN | 15 | `flat_gnn`, `flat_gnn_large`, `gcn`, `gin`, `sage`, `gatv2`, `pna`, `gated_gcn`, `edgeconv`, `appnp`, `flat_gat`, `rgcn`, `gps`, `hetero_sage`, `agnn` (flat AGNN) |
| Compact | 3 | `compact_hec` (455K), `tiny_hec` (82K), `shared_hec` |
| Non-GNN | 4 | `deepsets`, `true_deepsets`, `mlp_curve`, `mlp_scalar` |
| Alpha / J_c ablation | 6 | `flat_gnn_alpha`, `flat_gnn_alpha_hw`, `flat_gnn_large_alpha`, `flat_gnn_large_alpha_hw`, `jc_direct_hec`, `hw_norm_hec` |
| Scalar variants | 15 | `scalar_ensemble_hec`, `scalar_dropout_hec`, 8 × `scalar_ens_{backbone}`, 5 × `scalar_ens_{backbone}_hw` |

Smoke test: `scripts/v2_smoke_test.py` instantiates 29 representative architectures under the default V2 alpha-injection and confirms parameter counts.

### 4.5 Training engine (`hecgnn_trainer/engine.py`)

`TrainingEngine.run()` orchestrates one experiment, which is one or more seed runs of a single architecture:

```
for seed in cfg.train.seeds:
    set_seeds(seed)
    train_one_seed(seed)        # AdamW + cosine, patience-based early stop
    evaluate_on_test_set()
    save: best_state_dict.pt, results_seed{N}.json
```

Loss decomposition (paper §4.3):

```
L_total = L_energy + beta_cbr * L_cbr + beta_rms * L_rms
       = L1(curve_pred, curve_true) + 0.1 * BCE(cbr) + 0.05 * MSE(rms)
```

Four loss modes are exposed via `cfg.train.loss_mode`:

| Mode | Energy | CBR | RMS | Use |
|---|---|---|---|---|
| `standard` | ✓ | ✓ | ✓ | Default (paper §4.3) |
| `energy_only` | ✓ | — | — | Ablation: how much do auxiliaries help? |
| `auxiliary_only` | — | ✓ | ✓ | Ablation: can break patterns alone predict $r^\star$? |
| `cbr_only` | — | ✓ | — | Ablation: chain-break alone. |
| `scalar_ensemble` | (synthetic) | ✓ | $\sum_k |r_k - r^\star|$ | Used by `ScalarEnsembleWrapper` / `ScalarDropoutWrapper` |

Evaluation metrics (paper §5.1):

- $\delta_E(\tau) = \frac{1}{N}\sum_i \mathbf 1[|E_i(\hat r_i) - E_i(r^\star_i)| / E_i(r^\star_i) \le \tau]$ at $\tau \in \{1, 2, 5\}\%$ — the headline *compliance* metric.
- $\mathrm{MAE}(r^\star) = \frac{1}{N}\sum_i |\hat r_i - r^\star_i|$ — direct prediction error.
- Inference time per instance — measured in `timing.py`.

---

## 5. Data pipeline

### 5.1 Dataset construction (`src/data/generate.py`)

Per paper §5.1, the multi-topology benchmark consists of ~96K instances drawn from five families across four topologies:

| Family | $n$ range | Generator | Embeds on |
|---|---|---|---|
| Random Ising | 8–40 | `generate_random_ising` (density-stratified) | P4 / P8 / P16 / Z4 |
| Sherrington-Kirkpatrick | 8–40 | `generate_sk_model` | P4 / P8 / P16 / Z4 |
| Weighted MaxCut | 8–40 | `generate_weighted_maxcut` | P4 / P8 / P16 / Z4 |
| 3-reg MaxCut | 8–40 | `generate_3reg_maxcut` | P4 / P8 / P16 / Z4 |
| Planted-solution | 8–40 | `generate_planted` | P4 / P8 / P16 / Z4 |

Embeddings use `minorminer.find_embedding` (fixed seed). Each embedded instance is labelled at $K=20$ log-spaced ratios $r_k \in [0.02, 5.0]$ by simulated annealing (5 000 reads per grid point) or single-spin-flip MCMC at $\beta = 2$ (Boltzmann surrogate).

Five OOD splits ship:

- `diverse_sa_mt` — main SA benchmark (~96K instances).
- `diverse_boltz_mt` — same instances, Boltzmann labels.
- `diverse_sa_ood_{train, test}` — size-OOD ($n \in \{50, 75, 100\}$).
- `large_scale_ood` — extreme OOD ($n \in \{200, 500, 1000\}$).
- `emb_{clique, clique_init, tuned}_sa_mt` — alternative embedding algorithms.

### 5.2 Batch structure

The dataset emits dicts; the collate fn batches them. Every wrapper / model reads from this shape:

```python
batch = {
    'x': Tensor[N_qubits, 7],         # qubit features
    'chain_edge_index': LongTensor[2, E_chain],
    'inter_edge_index': LongTensor[2, E_inter],
    'inter_edge_attr':  Tensor[E_inter, 3],
    'logical_edge_index': LongTensor[2, E_logical],
    'logical_edge_attr':  Tensor[E_logical, 5],
    'chain_batch':  LongTensor[N_qubits],  # qubit -> chain idx
    'graph_batch':  LongTensor[N_chains],  # chain -> graph idx
    'chain_lengths': LongTensor[N_chains],
    'rms_targets':  Tensor[B],
    'r_star':       Tensor[B],         # ground truth r* (training target)
    'energy_curve': Tensor[B, K],      # normalized excess energy
    'break_curve':  Tensor[B, K],      # per-instance chain-break rate
    'batch_size':   int,
}
```

Node features (`x[:, :7]`): `[h/RMS, |h|/RMS, deg_C, deg_x, |C_i|, delta/RMS, 1_{singleton}]`. The first six are scale-invariant by construction; the singleton indicator is binary.

### 5.3 Training recipe (paper §5.1)

| Hyperparameter | Value | Where |
|---|---|---|
| Optimizer | AdamW, weight decay $10^{-4}$ | `engine.py:_build_optimizer` |
| Learning rate | $5 \times 10^{-4}$ | `TrainConfig.lr` |
| Schedule | Linear warmup 5 epochs + cosine to 0 | `engine.py:_build_scheduler` |
| Batch size | 32 | `TrainConfig.batch_size` |
| Epochs | 200 (capped by patience) | `TrainConfig.epochs` |
| Patience | 30 epochs | `TrainConfig.patience` |
| Gradient clip | 1.0 | `engine.py:_train_step` |
| Seeds | 42, 123, 7 | `TrainConfig.seeds` |
| Grid | 20 log-spaced $r_k$ in $[0.02, 5.0]$ | `src.models.baselines.GRID` |

---

## 6. Key design decisions

For each decision, we state **what** we chose, **what we considered**, and **why we picked this**.

### 6.1 Curve target vs scalar regression at the head

**Chose:** dual interface — curve head ($K=20$ dim output) *and* scalar ensemble ($K$ independent scalar heads, median decoding).

**Considered:** (a) curve only; (b) single scalar regression; (c) listwise ranking (Plackett–Luce); (d) regret-aligned scalar (SPO+).

**Why:** the paper's Lemma 1 shows curve supervision admits a tight regret bound on cliff-shaped basins where single-scalar regression fails. The scalar ensemble retains the deployment scalar (Theorem 1) while recovering bound-style robustness via ensemble diversity. Empirically (§10), both heads dominate naive scalar regression by 5+ pp $\delta_E(5\%)$.

### 6.2 V2 D-Wave `auto_scale` as the default alpha formula

**Chose:** `alpha_mode="hardware"` by default in `ModelConfig.alpha_mode`. The full formula of §2.3 is computed by `compute_alpha_vector`.

**Considered:** (a) the simpler $1/\max(1, r)$ formula (`alpha_mode="rms"`); (b) raw $J_c$ injection at the head (`jc_direct_hec`); (c) no alpha conditioning at all (`alpha_mode="none"`).

**Why:** the paper §4.2 specifies the closed-form rescaling factor as the conditioning signal. The hardware formula matches what the QPU physically applies, and including the problem $h$ / $J$ contributions matters when $|h|$ or $|J|$ pushes $S$ above the chain-coupler term. The RMS formula and the no-alpha baselines are retained as ablation arms (`flat_gnn_alpha` for the former, `--alpha-mode none` for the latter).

### 6.3 Wrapper pattern vs subclassing

**Chose:** `AlphaInjectionWrapper` *wraps* any base model in the registry. The base's energy head is intercepted by `CaptureZ`, and a new $[z \| \alpha]$ head is built on top.

**Considered:** (a) inheriting from each base class and overriding `forward`; (b) duplicating the alpha-injection logic into each base class.

**Why:** the registry has 52 architectures. Subclassing or duplicating would force every reviewer to verify 52 implementations. The wrapper is one ~80-line class that handles all of them uniformly, including the small subset (4 architectures) that opt into custom alpha behaviour through the `_SKIP_ALPHA_WRAP` list. This is the *primary* reason the V2 hardware normalization is uniformly applied across the full registry without per-architecture edits.

### 6.4 K=20 ensemble vs K=1 dropout for scalar prediction

**Chose:** ship both. `ScalarEnsembleWrapper` is the higher-capacity, higher-diversity variant; `ScalarDropoutWrapper` is the parameter-frugal alternative.

**Considered:** a single batched head $\mathbb R^{3H+K} \to \mathbb R^K$ — same parameter count, simpler — but it eliminates the independent-init diversity that is the entire reason the ensemble works.

**Why:** the ensemble's diversity comes from K independent random initialisations; collapsing into one MLP destroys that. The MC dropout variant is the natural cost-down option for memory-constrained deployment.

### 6.5 Three shared helpers in `_wrapper_utils.py`

**Chose:** lift `CaptureZ`, `detect_head_input_dim`, and `install_capture_head` out of the three wrappers into a single module.

**Considered:** keeping the per-wrapper copies.

**Why:** before the lift, all three wrappers redefined a near-identical `CaptureZ` class inline, and two of them had a silent bug — they hard-coded the dummy curve length to `20` instead of using `self.K`, which would have produced a shape mismatch if a user ever set `K != 20`. The lift fixes the bug at the source and gives reviewers a single 110-line file to read instead of three copies.

### 6.6 What we did *not* refactor

A code review (parallel reuse / quality / efficiency agents) flagged several other simplifications that we explicitly skipped:

- **Vectorising `compute_alpha_vector`'s per-graph loop.** Would change the reduction order and produce small numerical drift; invalidates the released checkpoints.
- **Collapsing K=20 scalar heads into a single batched MLP.** Destroys ensemble diversity (§6.4).
- **`Literal[...]` types for `alpha_mode` / `loss_mode`.** Runtime semantics unchanged, but mypy/IDE behaviour shifts; deferred to a future cleanup.
- **Data-driven registry table.** ~150-line LOC saving, but the per-builder pattern is the canonical PyG / torchvision style and easier to audit one architecture at a time.

These are documented to make the *omission* legible: a reviewer who notices the duplication should know it was a deliberate choice, not an oversight.

---

## 7. Theoretical guarantees from the paper

| Statement | Paper location | Code dependence |
|---|---|---|
| **Theorem 1** — Scalar reduction under global rescaling. | §3.4, App. A | License for the scalar interface in `ScalarEnsembleWrapper` and the deployment of $\hat r^\star$ as a single value. |
| **Lemma 1** — Curve-error regret bound. | §4.3, App. B | Justifies the curve head and the L1 loss on $\bar E$ in `engine.compute_loss`. |
| **Remark 1** — Cliff failure of scalar regression. | §4.3 | Justifies adding K=20 scalar heads instead of K=1. |
| **Proposition 1** — Min-cut chain vulnerability. | §4.1 | Justifies min/max pooling in `HECGNN.stage1`. |
| **Proposition 2** — Hardware rescaling continuity. | §4.2 | Justifies the closed-form $\alpha(J_c)$ injection in `compute_alpha_vector`. |

---

## 8. Experimental setup

### 8.1 Architecture coverage

29 architectures were trained under the V2 D-Wave alpha-injection on `diverse_boltz_mt` (seed 42). Breakdown:

| Group | Count | Architectures |
|---|---|---|
| HEC hierarchical (curve + V2 wrapper) | 9 | `hec_gnn`, `hec_agnn`, `hec_gatedgcn`, `hec_gatv2`, `hec_gcn`, `hec_pna`, `hec_edgeconv`, `sage_hec`, `gat_hec` |
| Flat GNN (curve + V2 wrapper) | 10 | `flat_gnn`, `flat_gnn_large`, `gcn`, `gin`, `sage`, `gatv2`, `pna`, `gated_gcn`, `edgeconv`, `appnp` |
| Compact (curve + V2 wrapper) | 3 | `compact_hec`, `tiny_hec`, `shared_hec` |
| No-MP (curve + V2 wrapper) | 1 | `true_deepsets` |
| Scalar ensemble + V2 alpha (`scalar_ens_*_hw`) | 5 | `scalar_ens_{hec, gatedgcn, agnn, gatv2, gcn}_hw` |
| Scalar dropout + V2 alpha | 1 | `scalar_dropout_hec` |
| **Total** | **29** | |

Six additional architectures manage alpha internally (without the generic wrapper) and were trained in an earlier sweep — their results live in `results/MASTER_BOLTZ_RESULTS.json` (search for the per-arch keys, e.g. `flat_gnn_alpha_boltz`, `flat_gnn_la_boltz`, `hw_norm_hec_boltz`). These six are *not* part of the §9 V2 sweep tables: `flat_gnn_alpha`, `flat_gnn_alpha_hw`, `flat_gnn_large_alpha`, `flat_gnn_large_alpha_hw`, `jc_direct_hec`, `hw_norm_hec`.

### 8.2 Datasets

| Dataset | Instances | Use |
|---|---|---|
| `diverse_sa_mt` | ~96K | Paper main table (Table 1) |
| `diverse_boltz_mt` | ~96K | All V2 sweep results in §10 |
| `diverse_sa_ood_test` | ~10K | OOD size extrapolation (Figure 3) |
| `large_scale_ood` | ~13K | $n \in \{200, 500, 1000\}$ extreme OOD |
| `emb_{clique, clique_init, tuned}_sa_mt` | ~29K each | Embedding-algorithm transfer |
| `qpu_data_600` | 600 | D-Wave Advantage validation (paper Table 4) |

### 8.3 Metrics

- **$\delta_E(5\%)$** — paper's headline compliance metric, $\tau = 5\%$.
- **$\delta_E(2\%)$**, **$\delta_E(1\%)$** — stricter tolerance regimes.
- **MAE($r^\star$)** — absolute deviation from oracle.
- **Inference time** — `timing.py` (1.4 ms for HEC-GNN at $B=1$ on RTX 6000).

---

## 9. Code-to-paper mapping

| Paper section | What it claims | Code that implements it |
|---|---|---|
| §3 Problem setup | Definitions of $H_L$, $C_i$, $J_c$, $\alpha$. | `src/data/generate.py:build_physical_hamiltonian` |
| §3.4 Theorem 1 | Scalar reduction under global rescaling. | (theory only — licenses the scalar interface in `ScalarEnsembleWrapper`) |
| §4.1 Three-stage hierarchy | Intra-chain → inter-chain → logical. | `src/models/hec_gnn.py:HECGNN.stage1/stage2/stage3` |
| §4.1 Min/max pooling (Prop 1) | Weakest-link / min-cut exposure. | `HECGNN.stage1` 4-way pool |
| §4.2 Closed-form $\alpha(J_c)$ injection | Inject hardware rescaling at the head. | `hecgnn_trainer/models/alpha_injection.py:compute_alpha_vector(mode="hardware")` |
| §4.3 Curve target $\bar E(r_k)$ | Normalized excess-energy curve. | `engine.compute_loss` L1 on `energy_curve` |
| §4.3 Auxiliary losses ($\beta_{\mathrm{cbr}}=0.1$, $\beta_{\mathrm{rms}}=0.05$) | CBR + RMS aux. | `engine.compute_loss` weighted sum |
| §4.4 Curve argmin + parabolic refinement | Sub-grid decoding. | `engine.evaluate_model` |
| §5.1 Multi-topology benchmark | ~96K instances across 5 families × 4 topologies. | `src/data/generate.py:generate_multi_topo_benchmark` |
| §5.2 Capacity-matched FlatGNN-Large | Parameter-matched flat ablation. | `flat_gnn_large` in registry, builds `src.models.baselines.FlatGNN(hidden_dim=192, num_layers=8)` |
| §5.3 Hierarchy-vs-aggregator ablation | SAGE-HEC, GAT-HEC. | `sage_hec`, `gat_hec` in registry |
| §5.4 QPU validation (D-Wave Advantage, Pegasus P16) | 600 instances, head-only fine-tuning. | `scripts/qpu_labeling_600.py` + `scripts/qpu_adapt_600.py` |
| §5.5 OOD evaluation | Size extrapolation up to $n=1000$. | `scripts/generate_large_scale.py` + `diverse_sa_ood_test` |
| §5.6 Embedding-algorithm transfer | Clique embedding, tuned minorminer. | `emb_clique_sa_mt`, `emb_clique_init_sa_mt`, `emb_tuned_sa_mt` datasets |
| Appendix A — Uniform-envelope proof | Proof of Theorem 1. | (theory only) |
| Appendix B — Regret bound | Proof of Lemma 1 / Remark 1. | (theory only) |

---

## 10. Reproducibility

### 10.1 One-line examples

```bash
# Verify install: build 29 architectures with V2 default.
python scripts/v2_smoke_test.py

# Train HEC-GNN with V2 default on Boltzmann labels.
python -m hecgnn_trainer.cli train \
    --arch hec_gnn --data-dir data/diverse_boltz_mt \
    --seeds 42,123,7

# Train the best scalar+V2 configuration.
python -m hecgnn_trainer.cli train \
    --arch scalar_ens_gatedgcn_hw --data-dir data/diverse_boltz_mt \
    --seeds 42,123,7

# Run the alpha-injection ablation table.
python -m hecgnn_trainer.cli sweep --config experiments/flatgnn_alpha_ablation.yaml

# Ablate V2 wrapper off (back to no-alpha baseline).
python -m hecgnn_trainer.cli train --arch hec_gnn \
    --config experiments/sweep_all_architectures.yaml \
    # then set model.alpha_mode: "none" in the YAML
```

### 10.2 Expected runtimes (single RTX 6000, $B=32$, 200 epochs)

| Configuration | Time |
|---|---|
| HEC-GNN (2.75 M params) | ~40 min |
| FlatGNN (263 K params) | ~25 min |
| Compact-HEC (455 K params) | ~30 min |
| Scalar ensemble + V2 (3.2 M params) | ~50 min |

Multi-CPU machines may be 1.5-2× slower due to single-process data loading; see §13 *Extension points* for the planned fix.

### 10.3 QPU protocol

`scripts/qpu_labeling_600.py` constructs the 600-instance benchmark. `scripts/qpu_adapt_600.py` performs the head-only fine-tuning protocol (5-fold CV; 20 QPU labels per fold; ~\$5/fold of D-Wave Advantage time). The QPU number in §9.6's caveat list will be recomputed against the fixed pipeline before camera-ready.

