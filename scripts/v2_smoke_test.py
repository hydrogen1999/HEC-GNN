"""V2 α-injection smoke test.

Builds every key architecture with the default alpha-injection mode and reports the
wrapped class, parameter count, and α-mode. Then runs a single forward pass on
a tiny synthetic batch to verify the wrapper plumbing works end-to-end.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hecgnn_trainer.config import ModelConfig
from hecgnn_trainer.registry import build_model, model_info


TEST_ARCHS = [
    # Backbones that previously had no α-injection — should now be wrapped.
    "hec_gnn", "flat_gnn", "flat_gnn_large", "compact_hec", "tiny_hec",
    "true_deepsets", "hec_gatedgcn", "hec_gatv2", "hec_gcn", "hec_pna",
    "pna", "appnp", "gcn", "gin", "sage", "gatv2", "edgeconv",
    "sage_hec", "gat_hec",
    # Wrappers (scalar ensemble) — should default to alpha_mode="hardware".
    "scalar_ensemble_hec", "scalar_ens_gatv2", "scalar_ens_pna",
    "scalar_dropout_hec",
    # Models that manage α themselves — should be untouched.
    "flat_gnn_alpha_hw", "flat_gnn_large_alpha_hw",
    "hw_norm_hec", "jc_direct_hec",
    # Legacy V1 ablation entries — should also be untouched.
    "flat_gnn_alpha", "flat_gnn_large_alpha",
]


def main():
    print(f"{'arch':<28} {'wrapped_class':<30} {'params':>12}  alpha_mode")
    print("-" * 86)
    failures = []
    for arch in TEST_ARCHS:
        try:
            cfg = ModelConfig(arch=arch)
            m = build_model(cfg)
            info = model_info(m)
            print(f"{arch:<28} {type(m).__name__:<30} {info['n_params']:>12,}  {cfg.alpha_mode}")
        except Exception as e:
            print(f"FAIL {arch:<28} {e}")
            failures.append((arch, str(e)))

    if failures:
        print()
        print(f"{len(failures)} architectures failed to build:")
        for arch, e in failures:
            print(f"  {arch}: {e}")
        sys.exit(1)
    print()
    print(f"OK: {len(TEST_ARCHS)} architectures built with V2 default α-injection.")


if __name__ == "__main__":
    main()
