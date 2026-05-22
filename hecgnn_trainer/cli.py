#!/usr/bin/env python3
"""Local CLI entry point for the HEC-GNN modular training system.

Usage:
  # List all registered architectures
  python -m hecgnn_trainer.cli list

  # Train a single model from CLI flags
  python -m hecgnn_trainer.cli train --arch hec_gnn --data-dir data/diverse_sa_mt

  # Train from a YAML config
  python -m hecgnn_trainer.cli train --config experiments/sweep_all_architectures.yaml

  # Run a sweep defined by a YAML config
  python -m hecgnn_trainer.cli sweep --config experiments/sweep_all_architectures.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = os.path.join(os.path.dirname(__file__), '..')
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from hecgnn_trainer.config import (
    ExperimentConfig, ModelConfig, TrainConfig, DataConfig,
    load_config, load_sweep,
)
from hecgnn_trainer.registry import build_model, model_info, print_registry
from hecgnn_trainer.engine import TrainingEngine


def cmd_list(args):
    print_registry()


def cmd_train(args):
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = ExperimentConfig(
            name=args.name or args.arch,
            model=ModelConfig(
                arch=args.arch,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_layers_stage3=args.num_layers_stage3,
                dropout=args.dropout,
            ),
            train=TrainConfig(
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seeds=[int(s) for s in args.seeds.split(",")],
            ),
            data=DataConfig(
                data_dir=args.data_dir,
                train_file=args.train_file,
                val_file=args.val_file,
                test_file=args.test_file,
            ),
            output_dir=args.output_dir,
        )

    model = build_model(cfg.model)
    info = model_info(model)
    print(f"Model: {cfg.model.arch}  params={info['n_params']:,}  "
          f"hidden={cfg.model.hidden_dim}  L1={cfg.model.num_layers} "
          f"L3={cfg.model.num_layers_stage3}  seeds={cfg.train.seeds}  "
          f"data={cfg.data.data_dir}")

    engine = TrainingEngine(cfg)
    results = engine.run()

    print("\nRESULTS")
    print("-" * 60)
    for r in results:
        tm = r['test_metrics']
        print(f"  seed {r['seed']}: "
              f"MAE(r*)={tm['mae_r']:.4f}  "
              f"dE<=5%={tm['delta_e_5pct']:.1%}  "
              f"dE<=2%={tm['delta_e_2pct']:.1%}  "
              f"dE<=1%={tm.get('delta_e_1pct', 0):.1%}  "
              f"({r['elapsed_sec']:.0f}s)")

    if len(results) > 1:
        import numpy as np
        maes = [r['test_metrics']['mae_r'] for r in results]
        de5s = [r['test_metrics']['delta_e_5pct'] for r in results]
        print(f"  mean: MAE(r*)={np.mean(maes):.4f}+-{np.std(maes):.4f}  "
              f"dE<=5%={np.mean(de5s):.1%}+-{np.std(de5s):.1%}")


def cmd_sweep(args):
    sweep = load_sweep(args.config)
    configs = sweep.expand()
    print(f"Sweep: {sweep.sweep_name} ({len(configs)} experiments)")

    all_results = {}
    for i, cfg in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {cfg.name}")
        engine = TrainingEngine(cfg)
        all_results[cfg.name] = engine.run()

    print("\nSWEEP SUMMARY")
    print("-" * 80)
    print(f"{'Experiment':<25} {'Arch':<15} {'Params':>10} "
          f"{'MAE(r*)':>10} {'dE<=5%':>10} {'dE<=2%':>10}")
    print("-" * 80)
    import numpy as np
    for name, results in all_results.items():
        maes = [r['test_metrics']['mae_r'] for r in results]
        de5s = [r['test_metrics']['delta_e_5pct'] for r in results]
        de2s = [r['test_metrics']['delta_e_2pct'] for r in results]
        arch = results[0]['arch']
        params = results[0]['n_params']
        print(f"{name:<25} {arch:<15} {params:>10,} "
              f"{np.mean(maes):>9.4f} {np.mean(de5s):>9.1%} {np.mean(de2s):>9.1%}")

    summary_path = os.path.join(configs[0].output_dir,
                                f"{sweep.sweep_name}_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HEC-GNN local training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s list                                    # List architectures
  %(prog)s train --arch hec_gnn --data-dir data/   # Train locally
  %(prog)s train --config exp.yaml                 # Train from config
  %(prog)s sweep --config sweep.yaml               # Local sweep
""")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all registered architectures")

    p_train = sub.add_parser("train", help="Train locally")
    p_train.add_argument("--config", help="YAML config file")
    p_train.add_argument("--arch", default="hec_gnn")
    p_train.add_argument("--name")
    p_train.add_argument("--hidden-dim", type=int, default=128)
    p_train.add_argument("--num-layers", type=int, default=3)
    p_train.add_argument("--num-layers-stage3", type=int, default=3)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--patience", type=int, default=30)
    p_train.add_argument("--seeds", default="42,123,7")
    p_train.add_argument("--data-dir", default="data/diverse_sa_mt")
    p_train.add_argument("--train-file", default="multi_topo_train.pkl")
    p_train.add_argument("--val-file", default="multi_topo_val.pkl")
    p_train.add_argument("--test-file", default="multi_topo_test.pkl")
    p_train.add_argument("--output-dir", default="results")

    p_sweep = sub.add_parser("sweep", help="Run a local sweep")
    p_sweep.add_argument("--config", required=True)

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
