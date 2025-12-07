from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TemporalConfig:
    backbone: str = "gru"  # "gru", "lstm", "tcn", "transformer"
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    activation: str = "relu"  # used in non-RNN backbones / projection


@dataclass
class MLPConfig:
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    activation: str = "relu"  # "relu", "gelu", "tanh"


@dataclass
class ModelConfig:
    d_embed: int
    d_blend: int
    d_other: int

    d_window: int = 256

    # Separate temporal output sizes per path
    d_temporal_embed: int = 128  # e.g. for 768-d frame embeddings
    d_temporal_blend: int = 32  # e.g. for 52-d blendshapes
    d_temporal_other: int = 8  # e.g. for 4-d other features

    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    window_mlp: MLPConfig = field(default_factory=MLPConfig)
    reg_head_mlp: MLPConfig = field(default_factory=MLPConfig)

    attn_hidden_size: int = 128
    attn_activation: str = "tanh"  # activation in attention MLP


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    batch_size: int = 4
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)  # for Adam/AdamW
    momentum: float = 0.9  # for SGD
    loss: str = "smooth_l1"  # "mse" or "smooth_l1"
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_scheduler: bool = False
    step_lr_step_size: int = 20
    step_lr_gamma: float = 0.1


@dataclass
class SelectionConfig:
    window_size: int = 64
    stride: int = 32
    n_windows: int = 16

    pick_mean_top_begin: int = 2
    pick_max_top_begin: int = 2
    pick_random_begin: int = 12  # must add up to n_windows

    pick_mean_top_end: int = 12
    pick_max_top_end: int = 2
    pick_random_end: int = 2  # must add up to n_windows
