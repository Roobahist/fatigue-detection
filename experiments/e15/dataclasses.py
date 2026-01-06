from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    seed: int = 42
    num_epochs: int = 50
    batch_size: int = 4
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)  # for Adam/AdamW
    momentum: float = 0.9  # for SGD
    loss: str = (
        "smooth_l1"  # "mse", "smooth_l1", "mae", "huber", "cross_entropy", "emd"
    )
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_scheduler: bool = False
    step_lr_step_size: int = 20
    step_lr_gamma: float = 0.1
    training_time_limit: Optional[float] = 43200  # in seconds, e.g. 12 hours


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

    converge_ratio: float = 0.5


@dataclass
class CrossValidationConfig:
    search_strategy: str = "random"  # "grid", "random", "bayesian"
    n_trials: int = 20  # for random/bayesian search
    outer_folds: int = 5
    inner_folds: int = 4


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str  # e.g. "UTA"
    embeddings_dir: str
    features_dir: str
    eventfulness_scores_dir: str
    normalization_info_path: str
    field_normalization: Optional[Dict[str, Dict]] = field(default_factory=dict)

    # TASK CONFIGURATION
    task_type: str = "regression"  # "regression" or "classification"
    num_classes: Optional[int] = None  # Number of classes for classification
    class_labels: Optional[List[float]] = (
        None  # Explicit class labels (e.g., [0.0, 5.0, 10.0])
    )

    # Label mapping - maps original labels to new labels
    # e.g., {1.0: 1.0, 2.0: 1.0, 3.0: 1.0, 4.0: 2.0, ...} for both tasks
    label_mapping: Optional[Dict[float, float]] = None

    # For regression with binning for evaluation metrics
    thresholds: Optional[List[float]] = None  # e.g., [2.5, 7.5] for UTA


@dataclass
class InnerFoldResult:
    """Result from one inner fold."""

    inner_fold: int
    hparams: Dict[str, Any]
    inner_train_metric: float
    inner_val_metric: float
    outer_test_metric: float
    best_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InnerFoldResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OuterFoldResult:
    """Result from one outer fold."""

    outer_fold: int
    best_hparams: Dict[str, Any]
    chosen_num_epochs: int
    inner_results: List["InnerFoldResult"]
    outer_train_metric: Optional[float] = None
    outer_test_metric: Optional[float] = None
    model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "outer_fold": self.outer_fold,
            "best_hparams": self.best_hparams,
            "chosen_num_epochs": self.chosen_num_epochs,
            "inner_results": [
                ir.to_dict() if hasattr(ir, "to_dict") else asdict(ir)
                for ir in self.inner_results
            ],
            "outer_train_metric": self.outer_train_metric,
            "outer_test_metric": self.outer_test_metric,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OuterFoldResult":
        """Create from dictionary."""
        # Convert inner results back
        inner_results = []
        for ir_data in data.get("inner_results", []):
            if isinstance(ir_data, dict):
                inner_results.append(InnerFoldResult.from_dict(ir_data))
            else:
                inner_results.append(ir_data)

        return cls(
            outer_fold=data["outer_fold"],
            best_hparams=data["best_hparams"],
            chosen_num_epochs=data["chosen_num_epochs"],
            inner_results=inner_results,
            outer_train_metric=data.get("outer_train_metric"),
            outer_test_metric=data.get("outer_test_metric"),
            model_path=data.get("model_path"),
        )
