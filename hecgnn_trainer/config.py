"""
config.py -- YAML-driven experiment configuration.

Each experiment config specifies: model architecture, hyperparameters,
dataset, training settings, and target server.
"""

import copy
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Architecture configuration."""
    arch: str = "hec_gnn"  # Registry key
    hidden_dim: int = 128
    num_layers: int = 3       # Layers for main message-passing (L1 for HEC-GNN)
    num_layers_stage3: int = 3  # L3 for HEC-GNN Stage 3
    dropout: float = 0.1
    eps_init: float = 0.0
    heads: int = 4            # For attention-based models (GAT, AGNN)
    edge_dim: int = 3         # Inter-edge feature dim
    logical_edge_dim: int = 5  # Logical edge feature dim
    node_dim: int = 7
    K: int = 20               # Grid points
    # AGNN-specific
    agnn_beta_init: float = 1.0
    agnn_learn_beta: bool = True
    # MLP-specific
    mlp_feature_dim: int = 18  # Number of hand-crafted features
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    # Hardware-rescaling normalization. "hardware" = D-Wave auto_scale
    # (default), "rms" = the simpler 1/max(1, r) formula (for ablation),
    # "none" = no alpha-injection.
    alpha_mode: str = "hardware"
    # Extra kwargs for custom models
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 32
    grad_clip: float = 1.0
    patience: int = 30
    beta_cbr: float = 0.1
    beta_rms: float = 0.05
    warmup_epochs: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 7])
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine_warmup"  # cosine_warmup, step, none
    step_size: int = 50       # For step scheduler
    step_gamma: float = 0.5
    loss_mode: str = "standard"  # standard, auxiliary_only, energy_only, cbr_only


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: str = "data/diverse_boltz_mt"  # Default: Boltzmann dataset
    benchmark: str = "multi_topo"  # multi_topo, ood, boltz
    train_file: str = "multi_topo_train.pkl"
    val_file: str = "multi_topo_val.pkl"
    test_file: str = "multi_topo_test.pkl"
    num_workers: int = 0


@dataclass
class ServerConfig:
    """Target server for distributed training.

    All fields default to localhost placeholders. Set the environment
    variables HECGNN_HOST / HECGNN_USER / HECGNN_WORKDIR or pass an explicit
    ServerConfig to use a remote machine.
    """
    name: str = "local"
    address: str = os.environ.get("HECGNN_HOST", "localhost")
    user: str = os.environ.get("HECGNN_USER", os.environ.get("USER", "user"))
    gpu: str = "unspecified"
    activate: str = "source ~/.venv/bin/activate"
    workdir: str = os.environ.get("HECGNN_WORKDIR", "~/hecgnn")
    gpu_id: int = 0


# Empty by default. Populate at runtime or load from YAML if needed.
SERVERS: Dict[str, "ServerConfig"] = {
    "local": ServerConfig(),
}


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    output_dir: str = "results"
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SweepConfig:
    """Multi-experiment sweep configuration."""
    sweep_name: str = "sweep"
    base: ExperimentConfig = field(default_factory=ExperimentConfig)
    experiments: List[Dict[str, Any]] = field(default_factory=list)

    def expand(self) -> List[ExperimentConfig]:
        """Expand sweep into individual experiment configs."""
        configs = []
        for i, overrides in enumerate(self.experiments):
            cfg = copy.deepcopy(self.base)
            cfg.name = overrides.pop("name", f"{self.sweep_name}_{i}")
            _apply_overrides(cfg, overrides)
            configs.append(cfg)
        return configs


def _apply_overrides(cfg: ExperimentConfig, overrides: Dict[str, Any]):
    """Apply nested dict overrides to a config."""
    for key, value in overrides.items():
        if key == "model" and isinstance(value, dict):
            for k, v in value.items():
                setattr(cfg.model, k, v)
        elif key == "train" and isinstance(value, dict):
            for k, v in value.items():
                setattr(cfg.train, k, v)
        elif key == "data" and isinstance(value, dict):
            for k, v in value.items():
                setattr(cfg.data, k, v)
        elif key == "server" and isinstance(value, dict):
            if "name" in value and value["name"] in SERVERS:
                cfg.server = copy.deepcopy(SERVERS[value["name"]])
            for k, v in value.items():
                if k != "name":
                    setattr(cfg.server, k, v)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)


def load_config(path: str) -> ExperimentConfig:
    """Load a single experiment config from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = ExperimentConfig()
    if "name" in raw:
        cfg.name = raw["name"]
    if "model" in raw:
        for k, v in raw["model"].items():
            setattr(cfg.model, k, v)
    if "train" in raw:
        for k, v in raw["train"].items():
            setattr(cfg.train, k, v)
    if "data" in raw:
        for k, v in raw["data"].items():
            setattr(cfg.data, k, v)
    if "server" in raw:
        sname = raw["server"].get("name", "local")
        if sname in SERVERS:
            cfg.server = copy.deepcopy(SERVERS[sname])
        for k, v in raw["server"].items():
            if k != "name":
                setattr(cfg.server, k, v)
    if "output_dir" in raw:
        cfg.output_dir = raw["output_dir"]
    if "tags" in raw:
        cfg.tags = raw["tags"]
    if "notes" in raw:
        cfg.notes = raw["notes"]

    return cfg


def load_sweep(path: str) -> SweepConfig:
    """Load a sweep config from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    sweep = SweepConfig()
    sweep.sweep_name = raw.get("sweep_name", "sweep")

    base_raw = raw.get("base", {})
    base_cfg = ExperimentConfig()
    if "model" in base_raw:
        for k, v in base_raw["model"].items():
            setattr(base_cfg.model, k, v)
    if "train" in base_raw:
        for k, v in base_raw["train"].items():
            setattr(base_cfg.train, k, v)
    if "data" in base_raw:
        for k, v in base_raw["data"].items():
            setattr(base_cfg.data, k, v)
    if "server" in base_raw:
        sname = base_raw["server"].get("name", "local")
        if sname in SERVERS:
            base_cfg.server = copy.deepcopy(SERVERS[sname])
    if "output_dir" in base_raw:
        base_cfg.output_dir = base_raw["output_dir"]

    sweep.base = base_cfg
    sweep.experiments = raw.get("experiments", [])
    return sweep


def config_to_yaml(cfg: ExperimentConfig) -> str:
    """Serialize config to YAML string."""
    d = asdict(cfg)
    return yaml.dump(d, default_flow_style=False, sort_keys=False)
