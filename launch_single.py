#!/usr/bin/env python3
"""Train a single model. No timeout. Saves result immediately on completion.
Usage: CUDA_VISIBLE_DEVICES=0 python3 launch_single.py --name gcn --arch gcn --hidden 128 --layers 4
       CUDA_VISIBLE_DEVICES=0 python3 launch_single.py --name scalar_ens --arch scalar_ensemble_hec --loss-mode scalar_ensemble
"""
import sys, os, json, argparse, time
sys.path.insert(0, '.')

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--arch', required=True)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--layers3', type=int, default=3)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--data', default='data/diverse_sa_mt')
parser.add_argument('--output', default='results/phase1')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--loss-mode', default='standard',
                    choices=['standard', 'auxiliary_only', 'energy_only', 'cbr_only', 'scalar_ensemble'])
args = parser.parse_args()

from hecgnn_trainer.config import ExperimentConfig, ModelConfig, TrainConfig, DataConfig
from hecgnn_trainer.engine import TrainingEngine

cfg = ExperimentConfig(
    name=args.name,
    model=ModelConfig(arch=args.arch, hidden_dim=args.hidden,
                      num_layers=args.layers, num_layers_stage3=args.layers3,
                      heads=args.heads, dropout=args.dropout),
    train=TrainConfig(seeds=[args.seed], epochs=args.epochs, patience=args.patience,
                      loss_mode=args.loss_mode),
    data=DataConfig(data_dir=args.data, train_file='multi_topo_train.pkl',
                    val_file='multi_topo_val.pkl', test_file='multi_topo_test.pkl'),
    output_dir=args.output,
)

print(f'[{args.name}] Starting on GPU:{os.environ.get("CUDA_VISIBLE_DEVICES","?")} loss_mode={args.loss_mode}', flush=True)
t0 = time.time()
results = TrainingEngine(cfg).run()
tm = results[0]['test_metrics']
elapsed = time.time() - t0
print(f'[{args.name}] DONE dE5={tm["delta_e_5pct"]:.1%} dE2={tm["delta_e_2pct"]:.1%} dE1={tm["delta_e_1pct"]:.1%} MAE={tm["mae_r"]:.4f} {elapsed:.0f}s', flush=True)
