"""
baselines.py -- All baselines for V3 paper (self-contained, no V2 dependencies).

Baselines from Section 5:
  1. UTC: Uniform Torque Compensation (closed-form heuristic)
  2. Scaled(2.0): J_c = 2.0 * RMS(J)
  3. Mean: predict training set mean r* for all instances
  4. LinearReg: 18-feature OLS regression on r*
  5. FlatGNN: Single-level GIN-E on full embedded graph -> K energies
  6. Few-Shot-K: Randomly sample K grid points, pick best SA energy
  7. BO-10: Bayesian optimization (GP-UCB) on the 1D energy landscape
  8. Oracle SA: Full grid search (argmin of energy curve)

All baselines operate on the same data format as HEC-GNN:
  - Instance dicts with energy_curve, embedding, h_logical, J_logical, rms_J, etc.
  - Grid: K=20 log-spaced r in [0.02, 3.0]
"""

import math
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Grid (matching V3 paper)
# ---------------------------------------------------------------------------
K = 20
R_MIN = 0.02
R_MAX = 5.0
GRID = np.logspace(math.log10(R_MIN), math.log10(R_MAX), K).astype(np.float32)


# ============================================================================
# 1. UTC: Uniform Torque Compensation
# ============================================================================

class UTC:
    """UTC heuristic: J_c = max_q (delta_q^prob / deg_C(q))."""

    def predict(self, instance: Dict) -> float:
        embedding = instance["embedding"]
        h_phys = instance.get("h_phys")
        J_phys = instance.get("J_phys")
        chain_edge_map = instance.get("chain_edge_map")

        # If h_phys/J_phys not stored, recompute from logical
        if h_phys is None or J_phys is None:
            return self._predict_from_features(instance)

        qubit_to_var = {}
        for var, qubits in embedding.items():
            for q in qubits:
                qubit_to_var[q] = var

        chain_deg = {}
        if chain_edge_map:
            for var, edges in chain_edge_map.items():
                for a, b in edges:
                    chain_deg[a] = chain_deg.get(a, 0) + 1
                    chain_deg[b] = chain_deg.get(b, 0) + 1

        max_torque = 0.0
        for q in qubit_to_var:
            var = qubit_to_var[q]
            deg = chain_deg.get(q, 0)
            if deg == 0:
                continue  # singleton chain: no chain couplers to compensate
            delta_q = abs(h_phys.get(q, 0.0))
            for (a, b), Jpq in J_phys.items():
                if a == q or b == q:
                    other = b if a == q else a
                    other_var = qubit_to_var.get(other)
                    if other_var is not None and other_var != var:
                        delta_q += abs(Jpq)
            torque = delta_q / deg
            max_torque = max(max_torque, torque)

        return max_torque

    def _predict_from_features(self, instance: Dict) -> float:
        """Approximate UTC from stored qubit features (delta_prob)."""
        feats = instance["qubit_features"]
        rms = instance["rms_J"]
        # Feature 5 (index 5) is delta_prob/RMS, feature 2 (index 2) is deg_C
        max_torque = 0.0
        for f in feats:
            deg = f[2]
            if deg == 0:
                continue  # singleton chain
            dp = f[5] * rms  # un-normalize
            max_torque = max(max_torque, dp / deg)
        return max_torque

    def predict_r(self, instance: Dict) -> float:
        jc = self.predict(instance)
        rms = instance["rms_J"]
        return jc / max(rms, 1e-8)


# ============================================================================
# 2. Scaled Heuristic
# ============================================================================

class ScaledHeuristic:
    """J_c = scale_factor * RMS(J). Default scale=2.0."""

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def predict_r(self, instance: Dict) -> float:
        return self.scale


# ============================================================================
# 3. Mean Baseline
# ============================================================================

class MeanBaseline:
    """Predict training set mean r* for all test instances."""

    def __init__(self):
        self.mean_r = None

    def fit(self, train_instances: List[Dict]):
        rs = []
        for inst in train_instances:
            ec = np.array(inst["energy_curve"])
            rs.append(GRID[np.argmin(ec)])
        self.mean_r = float(np.mean(rs))

    def predict_r(self, instance: Dict) -> float:
        return self.mean_r


# ============================================================================
# 4. Linear Regression (18 features)
# ============================================================================

class LinearRegBaseline:
    """OLS linear regression on 18 hand-crafted features -> r*."""

    def __init__(self):
        self.weights = None
        self.bias = None

    def _extract_features(self, inst: Dict) -> np.ndarray:
        embedding = inst["embedding"]
        J_logical = inst["J_logical"]
        h_logical = inst["h_logical"]
        rms = inst["rms_J"]

        chain_lens = [len(c) for c in embedding.values()]
        J_vals = [abs(v) for v in J_logical.values()] if J_logical else [0.0]
        h_vals = [abs(v) for v in h_logical.values()] if h_logical else [0.0]

        n = len(embedding)
        n_edges = len(J_logical)
        density = 2 * n_edges / max(n * (n - 1), 1)

        inter_deg = {v: 0 for v in embedding}  # all vars start at 0
        for (i, j) in J_logical:
            inter_deg[i] = inter_deg.get(i, 0) + 1
            inter_deg[j] = inter_deg.get(j, 0) + 1
        deg_vals = list(inter_deg.values()) if inter_deg else [0]

        total_qubits = sum(chain_lens)

        utc = UTC()
        utc_val = utc.predict(inst) / max(rms, 1e-8)

        return np.array([
            np.mean(chain_lens), np.max(chain_lens),
            np.min(chain_lens), np.std(chain_lens),
            np.mean(J_vals), np.max(J_vals),
            np.min(J_vals), np.std(J_vals),
            float(n), float(n_edges), density, rms,
            np.max(h_vals), np.mean(h_vals),
            utc_val,
            np.max(deg_vals), np.mean(deg_vals),
            float(total_qubits),
        ], dtype=np.float64)

    def fit(self, train_instances: List[Dict]):
        X = np.stack([self._extract_features(inst) for inst in train_instances])
        y = np.array([GRID[np.argmin(inst["energy_curve"])] for inst in train_instances])

        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        try:
            w = np.linalg.lstsq(X_bias, y, rcond=None)[0]
            self.weights = w[:-1]
            self.bias = w[-1]
        except np.linalg.LinAlgError:
            self.weights = np.zeros(X.shape[1])
            self.bias = float(np.mean(y))

    def predict_r(self, instance: Dict) -> float:
        feats = self._extract_features(instance)
        return float(feats @ self.weights + self.bias)


# ============================================================================
# 5. Flat GNN (requires torch -- imported lazily)
# ============================================================================

def build_flat_gnn(node_dim=7, edge_dim=3, hidden_dim=128, num_layers=6, K=20):
    """Build FlatGNN model. Requires torch and layers.py."""
    import torch
    import torch.nn as nn
    from src.models.layers import GINELayer, scatter_add, scatter_mean, scatter_max

    class FlatGNN(nn.Module):
        """Single-level GIN-E baseline on full embedded graph -> K energies."""

        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(node_dim, hidden_dim)
            self.layers = nn.ModuleList([
                GINELayer(hidden_dim, edge_dim) for _ in range(num_layers)
            ])
            self.head = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, K),
            )

        def forward(self, batch):
            x = batch["x"]
            # Merge chain + inter edges
            edge_indices = [batch["chain_edge_index"]]
            edge_attrs = []

            n_chain = batch["chain_edge_index"].size(1)
            if n_chain > 0:
                chain_feat = torch.zeros(n_chain, 3, device=x.device)
                chain_feat[:, 0] = 1.0
                edge_attrs.append(chain_feat)

            if batch["inter_edge_index"].numel() > 0:
                edge_indices.append(batch["inter_edge_index"])
                edge_attrs.append(batch["inter_edge_attr"])

            if edge_attrs:
                all_edges = torch.cat(edge_indices, dim=1)
                all_attrs = torch.cat(edge_attrs, dim=0)
            else:
                all_edges = torch.zeros(2, 0, dtype=torch.long, device=x.device)
                all_attrs = torch.zeros(0, 3, device=x.device)

            chain_batch = batch["chain_batch"]
            graph_batch_chains = batch["graph_batch"]
            qubit_graph_batch = graph_batch_chains[chain_batch]

            h = self.encoder(x)
            for layer in self.layers:
                h = layer(h, all_edges, all_attrs)

            n_graphs = batch["batch_size"]
            z_sum = scatter_add(h, qubit_graph_batch, n_graphs)
            z_mean = scatter_mean(h, qubit_graph_batch, n_graphs)
            z_max = scatter_max(h, qubit_graph_batch, n_graphs)
            z = torch.cat([z_sum, z_mean, z_max], dim=-1)

            energy_pred = self.head(z)

            # Dummy auxiliaries
            n_chains = batch["graph_batch"].size(0)
            cbr_pred = torch.zeros(n_chains, device=x.device)
            rms_pred = torch.zeros(n_graphs, device=x.device)

            return energy_pred, cbr_pred, rms_pred

    return FlatGNN()


# ============================================================================
# 6. Few-Shot Chain Strength Heuristic
# ============================================================================

class FewShotBaseline:
    """Randomly sample n_shots grid points from energy curve, pick best."""

    def __init__(self, n_shots: int = 5, seed: int = 42):
        self.n_shots = n_shots
        self.seed = seed

    def predict_r(self, instance: Dict) -> float:
        rng = np.random.RandomState(
            self.seed + instance.get("instance_id", 0))
        ec = np.array(instance["energy_curve"])
        n_shots = min(self.n_shots, len(ec))
        indices = rng.choice(len(ec), size=n_shots, replace=False)
        best_idx = indices[np.argmin(ec[indices])]
        return float(GRID[best_idx])


# ============================================================================
# 7. Bayesian Optimization (1D GP-UCB)
# ============================================================================

class BOBaseline:
    """1D Bayesian optimization on energy landscape via GP-UCB."""

    def __init__(self, n_evals: int = 10, kappa: float = 2.0, seed: int = 42):
        self.n_evals = n_evals
        self.kappa = kappa
        self.seed = seed

    def _gp_predict(self, X_obs, y_obs, X_pred, length_scale=0.5, noise=1e-4):
        def rbf(A, B):
            sq_dist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            return np.exp(-0.5 * sq_dist / (length_scale ** 2))

        K_mat = rbf(X_obs, X_obs) + noise * np.eye(len(X_obs))
        K_s = rbf(X_obs, X_pred)
        K_ss = rbf(X_pred, X_pred)

        try:
            L = np.linalg.cholesky(K_mat)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            mu = K_s.T @ alpha
            v = np.linalg.solve(L, K_s)
            cov = K_ss - v.T @ v
            sigma = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        except np.linalg.LinAlgError:
            mu = np.full(len(X_pred), np.mean(y_obs))
            sigma = np.ones(len(X_pred))

        return mu, sigma

    def predict_r(self, instance: Dict) -> float:
        ec = np.array(instance["energy_curve"])
        n_evals = min(self.n_evals, len(ec))
        rng = np.random.RandomState(
            self.seed + instance.get("instance_id", 0))

        log_grid = np.log(GRID).reshape(-1, 1)

        observed_idx = [0, len(ec) - 1]
        interior = rng.randint(1, len(ec) - 1)
        if interior not in observed_idx:
            observed_idx.append(interior)

        for _ in range(n_evals - len(observed_idx)):
            X_obs = log_grid[observed_idx]
            y_obs = ec[observed_idx]
            mu, sigma = self._gp_predict(X_obs, y_obs, log_grid)
            ucb = mu - self.kappa * sigma
            ucb[observed_idx] = np.inf
            next_idx = int(np.argmin(ucb))
            observed_idx.append(next_idx)

        best_idx = observed_idx[np.argmin(ec[observed_idx])]
        return float(GRID[best_idx])


# ============================================================================
# 8. Oracle SA (Grid Search)
# ============================================================================

class OracleSA:
    """Perfect oracle: returns argmin of the pre-computed energy curve."""

    def predict_r(self, instance: Dict) -> float:
        ec = np.array(instance["energy_curve"])
        return float(GRID[np.argmin(ec)])


# ============================================================================
# Evaluation utilities
# ============================================================================

def evaluate_baseline(baseline, test_instances: List[Dict],
                      train_instances: List[Dict] = None) -> Dict:
    """Evaluate a baseline on test instances.

    Returns dict with MAE(r*), delta_E<=5%, spearman_rho, etc.
    """
    if hasattr(baseline, 'fit') and train_instances is not None:
        baseline.fit(train_instances)

    pred_rs = []
    true_rs = []
    energy_gaps = []
    top3_correct = 0
    coverage_2x = 0

    for inst in test_instances:
        ec = np.array(inst["energy_curve"])
        true_idx = int(np.argmin(ec))
        true_r = float(GRID[true_idx])
        pred_r = baseline.predict_r(inst)

        pred_rs.append(pred_r)
        true_rs.append(true_r)

        # Energy gap: find closest grid point to pred_r
        pred_idx = int(np.argmin(np.abs(GRID - pred_r)))
        e_true = ec[true_idx]
        e_at_pred = ec[pred_idx]
        gap = abs(e_at_pred - e_true) / max(abs(e_true), 1e-8)
        energy_gaps.append(gap)

        # Coverage ≤2×: pred within [0.5×true, 2×true]
        if true_r > 0 and 0.5 * true_r <= pred_r <= 2.0 * true_r:
            coverage_2x += 1

    n = len(test_instances)
    pred_rs = np.array(pred_rs)
    true_rs = np.array(true_rs)

    results = {
        'mae_r': float(np.mean(np.abs(pred_rs - true_rs))),
        'delta_e_5pct': float(np.mean([1 if g <= 0.05 else 0 for g in energy_gaps])),
        'delta_e_2pct': float(np.mean([1 if g <= 0.02 else 0 for g in energy_gaps])),
        'delta_e_mean': float(np.mean(energy_gaps)),
        'coverage_2x': coverage_2x / max(n, 1),
        'n_test': n,
    }

    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(pred_rs, true_rs)
        results['spearman_rho'] = float(rho) if not np.isnan(rho) else 0.0
    except ImportError:
        results['spearman_rho'] = 0.0

    return results


def run_all_baselines(train_instances: List[Dict],
                      test_instances: List[Dict]) -> Dict[str, Dict]:
    """Run all baselines and return results dict."""
    baselines = {
        'UTC': UTC(),
        'Scaled(2.0)': ScaledHeuristic(2.0),
        'r=min': ScaledHeuristic(R_MIN),    # boundary: weakest chain strength
        'r=max': ScaledHeuristic(R_MAX),    # boundary: strongest chain strength
        'Mean': MeanBaseline(),
        'LinearReg': LinearRegBaseline(),
        'Few-Shot-3': FewShotBaseline(n_shots=3),
        'Few-Shot-5': FewShotBaseline(n_shots=5),
        'Few-Shot-10': FewShotBaseline(n_shots=10),
        'BO-10': BOBaseline(n_evals=10),
        'Oracle': OracleSA(),
    }

    results = {}
    for name, bl in baselines.items():
        print(f"  Evaluating {name}...")
        results[name] = evaluate_baseline(bl, test_instances, train_instances)
        r = results[name]
        print(f"    MAE(r*)={r['mae_r']:.4f}, "
              f"δE≤5%={r['delta_e_5pct']:.1%}")

    return results
