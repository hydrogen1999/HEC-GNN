"""
registry.py -- Unified model registry for all architectures.

All models follow the same interface:
  model(batch) -> (energy_pred [B, K], cbr_pred [N_chains], rms_pred [B])

Registry pattern inspired by PyG's GraphGym and torchvision model builders.
"""

import sys
import os
import torch.nn as nn

# Ensure src/ is importable
_CODE_ROOT = os.path.join(os.path.dirname(__file__), '..')
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from hecgnn_trainer.config import ModelConfig

# ---------------------------------------------------------------
# Registry
# ---------------------------------------------------------------
_REGISTRY = {}


def register(name: str, aliases: list = None):
    """Decorator to register a model builder."""
    def decorator(fn):
        _REGISTRY[name] = fn
        for alias in (aliases or []):
            _REGISTRY[alias] = fn
        return fn
    return decorator


def list_models():
    """Return sorted list of unique model names."""
    seen = set()
    result = []
    for name, fn in _REGISTRY.items():
        if fn not in seen:
            result.append(name)
            seen.add(fn)
    return sorted(result)


def _canonical_arch_name(key: str) -> str:
    """Resolve an alias to the canonical name registered against the same builder.

    `_REGISTRY` maps both canonical names and aliases to the same builder
    function. The canonical name is the FIRST key inserted for that function;
    aliases come later. Walk in insertion order and pick the first key that
    points at the same builder.
    """
    fn = _REGISTRY.get(key)
    if fn is None:
        return key
    for name, builder in _REGISTRY.items():
        if builder is fn:
            return name
    return key


def build_model(cfg: ModelConfig) -> nn.Module:
    """Build a model from config.

    Every model is wrapped with the D-Wave hardware alpha-injection by
    default. To bypass for ablation, set `cfg.alpha_mode = "none"`; to use
    the simpler 1/max(1, r) formula, set `cfg.alpha_mode = "rms"`.
    """
    key = cfg.arch.lower().replace("-", "_").replace(" ", "_")
    if key not in _REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown architecture '{cfg.arch}'. Available: {available}")
    model = _REGISTRY[key](cfg)
    # Alias-safe: skip-list checks below use the canonical name, not the alias.
    return _maybe_wrap_alpha(_canonical_arch_name(key), model, cfg)


# ---------------------------------------------------------------
# Alpha-injection helpers (primary normalization for the registry).
# ---------------------------------------------------------------

# Builders that already handle α-injection themselves and should NOT be
# double-wrapped: they either consume an `alpha_mode` argument internally
# (flat_gnn_alpha, flat_gnn_large_alpha, jc_direct_hec, hw_norm_hec) or wrap
# the base model in their own way (scalar_ens_*, scalar_dropout_*, mlp_*).
_SKIP_ALPHA_WRAP = {
    "flat_gnn_alpha", "flat_gnn_large_alpha",
    "flat_gnn_alpha_hw", "flat_gnn_large_alpha_hw",
    "jc_direct_hec", "hw_norm_hec",
    "mlp_curve", "mlp_scalar",
}


def _maybe_wrap_alpha(name: str, model: nn.Module, cfg: ModelConfig) -> nn.Module:
    """Wrap `model` with the V2 α-injection wrapper unless explicitly skipped.

    Triggered when:
      - cfg.alpha_mode != "none"
      - and the architecture is not in _SKIP_ALPHA_WRAP
      - and the model is not already a scalar-ensemble / dropout wrapper.
    """
    if getattr(cfg, "alpha_mode", "hardware") == "none":
        return model
    if name in _SKIP_ALPHA_WRAP:
        return model
    # Avoid double-wrapping models that already manage α at the head.
    from hecgnn_trainer.models.scalar_ensemble import (
        ScalarEnsembleWrapper, ScalarDropoutWrapper,
    )
    from hecgnn_trainer.models.alpha_injection import AlphaInjectionWrapper
    if isinstance(model, (ScalarEnsembleWrapper, ScalarDropoutWrapper,
                          AlphaInjectionWrapper)):
        return model
    from hecgnn_trainer.models.alpha_injection import wrap_with_alpha
    return wrap_with_alpha(
        model, K=cfg.K, dropout=cfg.dropout, alpha_mode=cfg.alpha_mode,
    )


def model_info(model: nn.Module) -> dict:
    """Get model parameter count and layer info."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "n_params": n_params,
        "n_trainable": n_trainable,
        "class": model.__class__.__name__,
    }


# ---------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------

@register("hec_gnn", aliases=["hecgnn", "hec"])
def _build_hec_gnn(cfg: ModelConfig):
    from src.models.hec_gnn import HECGNN, ModelConfig as HECModelConfig
    hcfg = HECModelConfig(
        node_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3,
        K=cfg.K, dropout=cfg.dropout, eps_init=cfg.eps_init,
    )
    return HECGNN(hcfg)


@register("flat_gnn", aliases=["flatgnn", "flat"])
def _build_flat_gnn(cfg: ModelConfig):
    from src.models.baselines import build_flat_gnn
    return build_flat_gnn(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, K=cfg.K,
    )


@register("flat_gnn_large", aliases=["flatgnn_large"])
def _build_flat_gnn_large(cfg: ModelConfig):
    from src.models.baselines import build_flat_gnn
    return build_flat_gnn(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim,
        hidden_dim=max(cfg.hidden_dim, 192), num_layers=max(cfg.num_layers, 8),
        K=cfg.K,
    )


@register("sage_hec", aliases=["sagehec"])
def _build_sage_hec(cfg: ModelConfig):
    from src.models.all_baselines import build_hec_variant
    return build_hec_variant('sage')


@register("gat_hec", aliases=["gathec"])
def _build_gat_hec(cfg: ModelConfig):
    from src.models.all_baselines import build_hec_variant
    return build_hec_variant('gat')


@register("rgcn", aliases=["r_gcn", "r-gcn"])
def _build_rgcn(cfg: ModelConfig):
    from src.models.all_baselines import RGCN
    return RGCN(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K,
    )


@register("flat_gat", aliases=["flatgat"])
def _build_flat_gat(cfg: ModelConfig):
    from src.models.all_baselines import FlatGATModel
    return FlatGATModel(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim, num_layers=cfg.num_layers,
        heads=cfg.heads, K=cfg.K,
    )


@register("gps", aliases=["graph_transformer"])
def _build_gps(cfg: ModelConfig):
    from src.models.all_baselines import GPSModel
    return GPSModel(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, heads=cfg.heads, K=cfg.K,
    )


@register("hetero_sage", aliases=["heterosage"])
def _build_hetero_sage(cfg: ModelConfig):
    from src.models.all_baselines import HeteroSAGEModel
    return HeteroSAGEModel(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K,
    )


@register("agnn", aliases=["flat_agnn", "anisotropic_gnn"])
def _build_flat_agnn(cfg: ModelConfig):
    from hecgnn_trainer.models.agnn import FlatAGNN
    return FlatAGNN(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim, num_layers=cfg.num_layers, K=cfg.K,
        dropout=cfg.dropout,
    )


@register("hec_agnn", aliases=["hecagnn", "anisotropic_hec"])
def _build_hec_agnn(cfg: ModelConfig):
    from hecgnn_trainer.models.agnn import HECAGNN
    return HECAGNN(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K,
        dropout=cfg.dropout, eps_init=cfg.eps_init,
    )


@register("mlp_curve", aliases=["mlp", "mlp18"])
def _build_mlp_curve(cfg: ModelConfig):
    from hecgnn_trainer.models.mlp_model import MLPCurve
    return MLPCurve(
        feature_dim=cfg.mlp_feature_dim,
        hidden_layers=cfg.mlp_hidden_layers, K=cfg.K, dropout=cfg.dropout,
    )


@register("mlp_scalar")
def _build_mlp_scalar(cfg: ModelConfig):
    from hecgnn_trainer.models.mlp_model import MLPScalar
    return MLPScalar(
        feature_dim=cfg.mlp_feature_dim,
        hidden_layers=cfg.mlp_hidden_layers, K=cfg.K, dropout=cfg.dropout,
    )


# ---------------------------------------------------------------
# Extended architectures (from extended.py)
# ---------------------------------------------------------------

@register("gcn", aliases=["flat_gcn"])
def _build_gcn(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import FlatGNNGeneric, GCNLayer
    return FlatGNNGeneric(
        lambda: GCNLayer(cfg.hidden_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K, uses_edge_attr=False,
    )


@register("gin", aliases=["gin_pure", "flat_gin"])
def _build_gin(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import FlatGNNGeneric, GINPureLayer
    return FlatGNNGeneric(
        lambda: GINPureLayer(cfg.hidden_dim, cfg.eps_init),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K, uses_edge_attr=False,
    )


@register("sage", aliases=["graphsage", "flat_sage"])
def _build_sage(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import FlatGNNGeneric, SAGEFlatLayer
    return FlatGNNGeneric(
        lambda: SAGEFlatLayer(cfg.hidden_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K, uses_edge_attr=False,
    )


@register("gatv2", aliases=["gat_v2", "flat_gatv2"])
def _build_gatv2(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import FlatGNNGeneric, GATv2Layer
    return FlatGNNGeneric(
        lambda: GATv2Layer(cfg.hidden_dim, cfg.heads, cfg.edge_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K, uses_edge_attr=True,
    )


@register("pna", aliases=["flat_pna"])
def _build_pna(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import FlatGNNGeneric, PNALayer
    return FlatGNNGeneric(
        lambda: PNALayer(cfg.hidden_dim, cfg.edge_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K, uses_edge_attr=True,
    )


@register("gated_gcn", aliases=["gatedgcn"])
def _build_gated_gcn(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import FlatGNNGeneric, GatedGCNLayer
    return FlatGNNGeneric(
        lambda: GatedGCNLayer(cfg.hidden_dim, cfg.edge_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K,
        uses_edge_attr=True, updates_edges=True,
    )


@register("edgeconv", aliases=["dgcnn"])
def _build_edgeconv(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import FlatGNNGeneric, EdgeConvLayer
    return FlatGNNGeneric(
        lambda: EdgeConvLayer(cfg.hidden_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, K=cfg.K, uses_edge_attr=False,
    )


@register("appnp")
def _build_appnp(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import APPNPModel
    alpha = cfg.extra.get("appnp_alpha", 0.1)
    K_hops = cfg.extra.get("appnp_hops", 10)
    return APPNPModel(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        K_hops=K_hops, alpha=alpha, K=cfg.K, dropout=cfg.dropout,
    )


@register("deepsets", aliases=["no_mp", "deepsets_chain"])
def _build_deepsets(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import DeepSetsModel
    return DeepSetsModel(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        K=cfg.K, dropout=cfg.dropout,
    )


@register("true_deepsets", aliases=["deepsets_flat", "no_structure"])
def _build_true_deepsets(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import TrueDeepSetsModel
    return TrueDeepSetsModel(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        K=cfg.K, dropout=cfg.dropout,
    )


# --- Hierarchical variants with different Stage 1 ---

@register("hec_pna", aliases=["hecpna", "pna_hec"])
def _build_hec_pna(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import HECVariant, PNALayer
    return HECVariant(
        lambda: PNALayer(cfg.hidden_dim, edge_dim=0),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K, dropout=cfg.dropout,
    )


@register("hec_gatedgcn", aliases=["hecgatedgcn"])
def _build_hec_gatedgcn(cfg: ModelConfig):
    """True HEC-GatedGCN: EdgeConv at Stage 1, GatedGCN at Stage 3.
    Stage 3 uses actual GatedGCN layers with edge gating + edge updates."""
    from hecgnn_trainer.models.extended import HECGatedGCNTrue
    return HECGatedGCNTrue(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K, dropout=cfg.dropout,
    )


@register("hec_gatv2", aliases=["hecgatv2"])
def _build_hec_gatv2(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import HECVariant, GATv2Layer
    return HECVariant(
        lambda: GATv2Layer(cfg.hidden_dim, cfg.heads, edge_dim=0),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K, dropout=cfg.dropout,
    )


@register("hec_gcn", aliases=["hecgcn"])
def _build_hec_gcn(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import HECVariant, GCNLayer
    return HECVariant(
        lambda: GCNLayer(cfg.hidden_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K, dropout=cfg.dropout,
    )


@register("hec_edgeconv", aliases=["hec_dgcnn"])
def _build_hec_edgeconv(cfg: ModelConfig):
    from hecgnn_trainer.models.extended import HECVariant, EdgeConvLayer
    return HECVariant(
        lambda: EdgeConvLayer(cfg.hidden_dim),
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K, dropout=cfg.dropout,
    )


# ---------------------------------------------------------------
# J_C / normalization ablation models
# ---------------------------------------------------------------

@register("jc_direct_hec", aliases=["hec_jc", "jc_hec"])
def _build_jc_direct_hec(cfg: ModelConfig):
    from hecgnn_trainer.models.jc_injection import JCDirectHEC
    return JCDirectHEC(
        hidden_dim=cfg.hidden_dim, L1=cfg.num_layers,
        L3=cfg.num_layers_stage3, K=cfg.K,
        dropout=cfg.dropout, eps_init=cfg.eps_init,
    )


@register("hw_norm_hec", aliases=["hardware_norm", "hec_hw"])
def _build_hw_norm_hec(cfg: ModelConfig):
    from hecgnn_trainer.models.jc_injection import HardwareNormHEC
    return HardwareNormHEC(
        hidden_dim=cfg.hidden_dim, L1=cfg.num_layers,
        L3=cfg.num_layers_stage3, K=cfg.K,
        dropout=cfg.dropout, eps_init=cfg.eps_init,
    )


# ---------------------------------------------------------------
# Alpha-injection ablation models (FlatGNN backbone, explicit alpha modes)
# ---------------------------------------------------------------

@register("flat_gnn_alpha", aliases=["flatgnn_alpha"])
def _build_flat_gnn_alpha(cfg: ModelConfig):
    """FlatGNN with alpha-injection using the 1/max(1, r) (rms) formula."""
    from hecgnn_trainer.models.alpha_injection import FlatGNNAlpha
    return FlatGNNAlpha(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, K=cfg.K,
        dropout=cfg.dropout, alpha_mode="rms",
    )


@register("flat_gnn_alpha_hw")
def _build_flat_gnn_alpha_hw(cfg: ModelConfig):
    """FlatGNN with the D-Wave auto_scale alpha formula."""
    from hecgnn_trainer.models.alpha_injection import FlatGNNAlpha
    return FlatGNNAlpha(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, K=cfg.K,
        dropout=cfg.dropout, alpha_mode="hardware",
    )


@register("flat_gnn_large_alpha")
def _build_flat_gnn_large_alpha(cfg: ModelConfig):
    """FlatGNN-Large (parameter-matched to HEC-GNN) with the rms alpha formula."""
    from hecgnn_trainer.models.alpha_injection import FlatGNNLargeAlpha
    return FlatGNNLargeAlpha(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim,
        hidden_dim=max(cfg.hidden_dim, 192),
        num_layers=max(cfg.num_layers, 8), K=cfg.K, dropout=cfg.dropout,
        alpha_mode="rms",
    )


@register("flat_gnn_large_alpha_hw")
def _build_flat_gnn_large_alpha_hw(cfg: ModelConfig):
    """FlatGNN-Large with the D-Wave auto_scale alpha formula."""
    from hecgnn_trainer.models.alpha_injection import FlatGNNLargeAlpha
    return FlatGNNLargeAlpha(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim,
        hidden_dim=max(cfg.hidden_dim, 192),
        num_layers=max(cfg.num_layers, 8), K=cfg.K, dropout=cfg.dropout,
        alpha_mode="hardware",
    )


# ---------------------------------------------------------------
# Scalar ensemble models (predict r* directly, ×K ensemble)
# ---------------------------------------------------------------

@register("scalar_ensemble_hec", aliases=["hec_scalar_ens", "scalar_x20"])
def _build_scalar_ensemble(cfg: ModelConfig):
    """Scalar ×K ensemble with HEC-GNN backbone (V2 hardware α by default)."""
    from hecgnn_trainer.models.scalar_ensemble import ScalarEnsembleWrapper
    base = _build_hec_gnn(cfg)  # HEC-GNN backbone
    return ScalarEnsembleWrapper(
        base, K=cfg.K, dropout=cfg.dropout,
        alpha_mode=getattr(cfg, "alpha_mode", "hardware"),
    )


@register("scalar_dropout_hec", aliases=["hec_scalar_mc", "mc_dropout"])
def _build_scalar_dropout(cfg: ModelConfig):
    """Scalar MC dropout with HEC-GNN backbone (V2 hardware α by default)."""
    from hecgnn_trainer.models.scalar_ensemble import ScalarDropoutWrapper
    base = _build_hec_gnn(cfg)
    return ScalarDropoutWrapper(
        base, K=cfg.K, dropout=max(cfg.dropout, 0.2),
        alpha_mode=getattr(cfg, "alpha_mode", "hardware"),
    )


def _build_scalar_ens_variant(base_builder, cfg, alpha_mode="__config__"):
    """Scalar ensemble on any base model, with alpha-injection.

    By default reads `cfg.alpha_mode` so a single config switch controls the
    normalization across the whole registry. Pass an explicit `alpha_mode`
    to pin a specific formula (used by the `scalar_ens_*_hw` shortcut names).
    """
    from hecgnn_trainer.models.scalar_ensemble import ScalarEnsembleWrapper
    if alpha_mode == "__config__":
        alpha_mode = getattr(cfg, "alpha_mode", "hardware")
    base = base_builder(cfg)
    return ScalarEnsembleWrapper(base, K=cfg.K, dropout=cfg.dropout, alpha_mode=alpha_mode)


# --- Scalar ensemble + V2 hardware α-injection ---

@register("scalar_ens_hec_hw", aliases=["scalar_hw_hec"])
def _build_se_hec_hw(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_gnn, cfg, alpha_mode="hardware")

@register("scalar_ens_gatedgcn_hw", aliases=["scalar_hw_gatedgcn"])
def _build_se_gatedgcn_hw(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_gatedgcn, cfg, alpha_mode="hardware")

@register("scalar_ens_agnn_hw", aliases=["scalar_hw_agnn"])
def _build_se_agnn_hw(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_agnn, cfg, alpha_mode="hardware")

@register("scalar_ens_gatv2_hw", aliases=["scalar_hw_gatv2"])
def _build_se_gatv2_hw(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_gatv2, cfg, alpha_mode="hardware")

@register("scalar_ens_gcn_hw", aliases=["scalar_hw_gcn"])
def _build_se_gcn_hw(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_gcn, cfg, alpha_mode="hardware")


# --- Scalar ensemble without α (original) ---

@register("scalar_ens_gatedgcn", aliases=["scalar_hec_gatedgcn"])
def _build_se_gatedgcn(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_gatedgcn, cfg)

@register("scalar_ens_agnn", aliases=["scalar_hec_agnn"])
def _build_se_agnn(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_agnn, cfg)

@register("scalar_ens_edgeconv", aliases=["scalar_hec_edgeconv"])
def _build_se_edgeconv(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_edgeconv, cfg)

@register("scalar_ens_gcn", aliases=["scalar_hec_gcn"])
def _build_se_gcn(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_gcn, cfg)

@register("scalar_ens_gatv2", aliases=["scalar_hec_gatv2"])
def _build_se_gatv2(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_gatv2, cfg)

@register("scalar_ens_pna", aliases=["scalar_hec_pna"])
def _build_se_pna(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_hec_pna, cfg)

@register("scalar_ens_sage", aliases=["scalar_sage_hec"])
def _build_se_sage(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_sage_hec, cfg)

@register("scalar_ens_gat", aliases=["scalar_gat_hec"])
def _build_se_gat(cfg: ModelConfig):
    return _build_scalar_ens_variant(_build_gat_hec, cfg)


# ---------------------------------------------------------------
# Compact / optimized models
# ---------------------------------------------------------------

@register("compact_hec", aliases=["hec_compact", "hec_small"])
def _build_compact_hec(cfg: ModelConfig):
    from hecgnn_trainer.models.compact import CompactHEC
    return CompactHEC(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K,
        dropout=cfg.dropout, eps_init=cfg.eps_init,
    )


@register("shared_hec", aliases=["hec_shared"])
def _build_shared_hec(cfg: ModelConfig):
    from hecgnn_trainer.models.compact import SharedHEC
    return SharedHEC(
        input_dim=cfg.node_dim, hidden_dim=cfg.hidden_dim,
        L1=cfg.num_layers, L3=cfg.num_layers_stage3, K=cfg.K,
        dropout=cfg.dropout, eps_init=cfg.eps_init,
    )


@register("tiny_hec", aliases=["hec_tiny", "hec_micro"])
def _build_tiny_hec(cfg: ModelConfig):
    from hecgnn_trainer.models.compact import TinyHEC
    h = cfg.extra.get("tiny_hidden", 64)
    l1 = cfg.extra.get("tiny_L1", 2)
    l3 = cfg.extra.get("tiny_L3", 2)
    return TinyHEC(
        input_dim=cfg.node_dim, hidden_dim=h, L1=l1, L3=l3,
        K=cfg.K, dropout=cfg.dropout, eps_init=cfg.eps_init,
    )


# ---------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------

def print_registry():
    """Print all registered models with param counts."""
    print(f"{'Model':<20} {'Aliases':<25} {'Params':>12}")
    print("-" * 60)
    seen = set()
    for name, fn in sorted(_REGISTRY.items()):
        if fn in seen:
            continue
        seen.add(fn)
        aliases = [k for k, v in _REGISTRY.items() if v is fn and k != name]
        try:
            cfg = ModelConfig(arch=name)
            model = fn(cfg)
            info = model_info(model)
            print(f"{name:<20} {', '.join(aliases):<25} {info['n_params']:>12,}")
        except Exception as e:
            print(f"{name:<20} {', '.join(aliases):<25} {'ERROR':>12} ({e})")


if __name__ == "__main__":
    print_registry()
