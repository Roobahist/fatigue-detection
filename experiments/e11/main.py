import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==============================
# Configs and helpers
# ==============================

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
    d_temporal_embed: int = 128   # e.g. for 768-d frame embeddings
    d_temporal_blend: int = 32    # e.g. for 52-d blendshapes
    d_temporal_other: int = 8    # e.g. for 4-d other features

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
    pick_random_begin: int = 12 # must add up to n_windows

    pick_mean_top_end: int = 12
    pick_max_top_end: int = 2
    pick_random_end: int = 2 # must add up to n_windows



def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation: {name}")


# ==============================
# Positional Encoding for Transformer
# ==============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, D]
        """
        T = x.size(0)
        return x + self.pe[:T]


# ==============================
# Core model components
# ==============================

class WindowTemporalEncoder(nn.Module):
    """
    Encodes a single temporal sequence [T, D_in] -> [D_out].
    Uses a selectable backbone: GRU, LSTM, TCN, Transformer.
    Aggregation over time is mean pooling over outputs.
    """

    def __init__(self, d_in: int, d_out: int, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone_type = cfg.backbone.lower()
        self.activation = get_activation(cfg.activation)
        self.dropout = nn.Dropout(cfg.dropout)

        if self.backbone_type in ("gru", "lstm"):
            self.rnn_type = self.backbone_type
            self.hidden_size = cfg.hidden_size
            self.num_layers = cfg.num_layers
            self.bidirectional = True
            rnn_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
            if self.rnn_type == "gru":
                self.rnn = nn.GRU(
                    input_size=d_in,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=rnn_dropout,
                    bidirectional=self.bidirectional,
                )
            else:
                self.rnn = nn.LSTM(
                    input_size=d_in,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=rnn_dropout,
                    bidirectional=self.bidirectional,
                )
            backbone_out_dim = self.hidden_size * 2  # bidirectional

        elif self.backbone_type == "tcn":
            self.hidden_size = cfg.hidden_size
            self.num_layers = max(cfg.num_layers, 1)
            layers = []
            in_channels = d_in
            for _ in range(self.num_layers):
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.hidden_size,
                    kernel_size=3,
                    padding=1,
                )
                layers.append(nn.Sequential(conv, self.activation, nn.Dropout(cfg.dropout)))
                in_channels = self.hidden_size
            self.tcn = nn.ModuleList(layers)
            backbone_out_dim = self.hidden_size

        elif self.backbone_type == "transformer":
            d_model = cfg.hidden_size
            self.input_proj = nn.Linear(d_in, d_model)
            nhead = 4 if d_model % 4 == 0 else 1

            activation_arg = get_activation(cfg.activation)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=cfg.dropout,
                batch_first=False,
                activation=activation_arg,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.num_layers,
            )
            self.pos_encoder = PositionalEncoding(d_model)
            backbone_out_dim = d_model

        else:
            raise ValueError(f"Unsupported temporal backbone: {self.backbone_type}")

        self.out_proj = nn.Linear(backbone_out_dim, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, D_in]
        Returns: [D_out]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected [T, D_in], got shape {x.shape}")
        T, _ = x.shape
        if T == 0:
            raise ValueError("Sequence length T must be > 0")

        if self.backbone_type in ("gru", "lstm"):
            # RNN expects [T, B, D]
            x_in = x.unsqueeze(1)  # [T, 1, D_in]
            outputs, _ = self.rnn(x_in)  # [T, 1, H*2]
            pooled = outputs.mean(dim=0).squeeze(0)  # [H*2]

        elif self.backbone_type == "tcn":
            # Conv1d expects [B, C, T]
            x_in = x.transpose(0, 1).unsqueeze(0)  # [1, D_in, T]
            for layer in self.tcn:
                x_in = layer(x_in)  # [1, hidden_size, T]
            pooled = x_in.mean(dim=2).squeeze(0)  # [hidden_size]

        elif self.backbone_type == "transformer":
            # [T, 1, D_model]
            x_in = self.input_proj(x)  # [T, D_model]
            x_in = x_in.unsqueeze(1)   # [T, 1, D_model]
            x_in = self.pos_encoder(x_in)
            outputs = self.transformer(x_in)  # [T, 1, D_model]
            pooled = outputs.mean(dim=0).squeeze(0)  # [D_model]

        else:
            raise ValueError(f"Unsupported temporal backbone: {self.backbone_type}")

        pooled = self.activation(pooled)
        pooled = self.dropout(pooled)
        out = self.out_proj(pooled)  # [D_out]
        return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cfg: MLPConfig):
        super().__init__()
        layers: List[nn.Module] = []
        activation = get_activation(cfg.activation)
        current_dim = in_dim

        if cfg.num_layers <= 1:
            layers.append(nn.Linear(current_dim, out_dim))
        else:
            for _ in range(cfg.num_layers - 1):
                layers.append(nn.Linear(current_dim, cfg.hidden_size))
                layers.append(activation)
                layers.append(nn.Dropout(cfg.dropout))
                current_dim = cfg.hidden_size
            layers.append(nn.Linear(current_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiPathWindowEncoder(nn.Module):
    """
    Encodes three modalities for a single window:
      - embeddings: [T, d_embed]
      - blendshapes: [T, d_blend]
      - others: [T, d_other]
    into a single window embedding [d_window].
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Use separate temporal output dims for each path
        self.embed_encoder = WindowTemporalEncoder(
            cfg.d_embed, cfg.d_temporal_embed, cfg.temporal
        )
        self.blend_encoder = WindowTemporalEncoder(
            cfg.d_blend, cfg.d_temporal_blend, cfg.temporal
        )
        self.other_encoder = WindowTemporalEncoder(
            cfg.d_other, cfg.d_temporal_other, cfg.temporal
        )

        concat_dim = (
            cfg.d_temporal_embed
            + cfg.d_temporal_blend
            + cfg.d_temporal_other
        )
        self.mlp = MLP(concat_dim, cfg.d_window, cfg.window_mlp)

    def forward(
        self,
        embed_seq: torch.Tensor,
        blend_seq: torch.Tensor,
        other_seq: torch.Tensor,
    ) -> torch.Tensor:
        h_embed = self.embed_encoder(embed_seq)   # [d_temporal_embed]
        h_blend = self.blend_encoder(blend_seq)   # [d_temporal_blend]
        h_other = self.other_encoder(other_seq)   # [d_temporal_other]
        h = torch.cat([h_embed, h_blend, h_other], dim=-1)  # [sum d_temporal_*]
        out = self.mlp(h)  # [d_window]
        return out


class AttentionMILPooling(nn.Module):
    """
    Ilse-style attention pooling over window embeddings.
    Input:  H: [N_windows, D_window]
    Output: z: [D_window], a: [N_windows]
    """

    def __init__(self, d_window: int, attn_hidden_size: int, attn_activation: str):
        super().__init__()
        self.V = nn.Linear(d_window, attn_hidden_size)
        self.w = nn.Linear(attn_hidden_size, 1, bias=False)
        self.activation = get_activation(attn_activation)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: [N_windows, D_window]
        """
        if H.ndim != 2:
            raise ValueError(f"Expected [N_windows, D_window], got {H.shape}")
        if H.size(0) == 0:
            raise ValueError("N_windows must be > 0")

        A_hidden = self.activation(self.V(H))  # [N, H_attn]
        logits = self.w(A_hidden).squeeze(-1)  # [N]
        a = torch.softmax(logits, dim=0)       # [N]
        z = torch.sum(a.unsqueeze(-1) * H, dim=0)  # [D_window]
        return z, a


class MultiPathMILRegressor(nn.Module):
    """
    Main model:
      - For each video:
        - For each window: MultiPathWindowEncoder -> window embedding
        - AttentionMILPooling over windows -> video embedding
        - Regression head -> scalar fatigue score
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.window_encoder = MultiPathWindowEncoder(cfg)
        self.mil_pool = AttentionMILPooling(cfg.d_window, cfg.attn_hidden_size, cfg.attn_activation)
        self.reg_head = MLP(cfg.d_window, 1, cfg.reg_head_mlp)

    def forward(self, batch: List[Dict]) -> torch.Tensor:
        """
        batch: list of video dicts. Each video dict:
          {
            "embeddings": List[Tensor [T_w, d_embed]],
            "blendshapes": List[Tensor [T_w, d_blend]],
            "others": List[Tensor [T_w, d_other]],
            "target": float (ignored here)
          }
        Assumes all tensors in batch are already on the same device as model.
        Returns:
          preds: [B] tensor of scalar predictions.
        """
        video_embeddings = []

        for video in batch:
            emb_list: List[torch.Tensor] = video["embeddings"]
            blend_list: List[torch.Tensor] = video["blendshapes"]
            other_list: List[torch.Tensor] = video["others"]

            assert len(emb_list) == len(blend_list) == len(other_list), "Window lists must have same length"
            n_windows = len(emb_list)
            if n_windows == 0:
                raise ValueError("Each video must have at least one window")

            window_embs = []
            for i in range(n_windows):
                e = emb_list[i]
                b = blend_list[i]
                o = other_list[i]
                window_emb = self.window_encoder(e, b, o)  # [d_window]
                window_embs.append(window_emb)

            H = torch.stack(window_embs, dim=0)  # [N_windows, d_window]
            z, _ = self.mil_pool(H)              # [d_window]
            video_embeddings.append(z)

        video_embeddings = torch.stack(video_embeddings, dim=0)  # [B, d_window]
        preds = self.reg_head(video_embeddings).squeeze(-1)      # [B]
        return preds


# ==============================
# Optimizer, loss, training
# ==============================

def build_optimizer(model: nn.Module, cfg: TrainingConfig):
    params = model.parameters()
    opt_name = cfg.optimizer.lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_lr_step_size,
            gamma=cfg.step_lr_gamma,
        )

    return optimizer, scheduler


def get_loss_fn(name: str):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss: {name}")


def move_batch_to_device(batch: List[Dict], device: torch.device) -> List[Dict]:
    """
    Move all tensors in a batch of videos to the given device.
    Keeps scalar targets as Python floats (they're re-tensorized in train loop).
    """
    new_batch: List[Dict] = []
    for video in batch:
        new_video = {
            "embeddings": [t.to(device, non_blocking=True) for t in video["embeddings"]],
            "blendshapes": [t.to(device, non_blocking=True) for t in video["blendshapes"]],
            "others": [t.to(device, non_blocking=True) for t in video["others"]],
            "target": video["target"],
        }
        new_batch.append(new_video)
    return new_batch


def train_one_split(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainingConfig,
) -> Tuple[nn.Module, float]:
    """
    Train on a single train/val split with early stopping.
    Returns best model (loaded with best weights) and best validation loss.
    """
    device_str = train_cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model.to(device)
    optimizer, scheduler = build_optimizer(model, train_cfg)
    loss_fn = get_loss_fn(train_cfg.loss)

    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(train_cfg.num_epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            # Move batch to device
            batch_on_device = move_batch_to_device(batch, device)
            targets = torch.tensor(
                [video["target"] for video in batch_on_device],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            preds = model(batch_on_device)  # [B]
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_on_device)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / max(train_samples, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = move_batch_to_device(batch, device)
                targets = torch.tensor(
                    [video["target"] for video in batch_on_device],
                    dtype=torch.float32,
                    device=device,
                )
                preds = model(batch_on_device)
                loss = loss_fn(preds, targets)

                batch_size = len(batch_on_device)
                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size

        val_loss = val_loss_sum / max(val_samples, 1)

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{train_cfg.num_epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Best Val: {best_val_loss:.4f}"
        )

        if epochs_without_improvement >= train_cfg.early_stopping_patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    return model, best_val_loss


# ==============================
# Dummy dataset and DataLoader
# ==============================

class DummyVideoDataset(Dataset):
    """
    Generates random videos with variable number of windows and frame lengths.
    Each item is a dict:
      {
        "embeddings": List[Tensor [T_w, d_embed]],
        "blendshapes": List[Tensor [T_w, d_blend]],
        "others": List[Tensor [T_w, d_other]],
        "target": float
      }
    """

    def __init__(
        self,
        num_videos: int,
        d_embed: int,
        d_blend: int,
        d_other: int,
        min_windows: int = 2,
        max_windows: int = 5,
        min_T: int = 5,
        max_T: int = 20,
    ):
        super().__init__()
        self.num_videos = num_videos
        self.d_embed = d_embed
        self.d_blend = d_blend
        self.d_other = d_other
        self.min_windows = min_windows
        self.max_windows = max_windows
        self.min_T = min_T
        self.max_T = max_T

    def __len__(self) -> int:
        return self.num_videos

    def __getitem__(self, idx: int) -> Dict:
        n_windows = random.randint(self.min_windows, self.max_windows)
        embeddings_list: List[torch.Tensor] = []
        blend_list: List[torch.Tensor] = []
        other_list: List[torch.Tensor] = []

        for _ in range(n_windows):
            T_w = random.randint(self.min_T, self.max_T)
            embeddings_list.append(torch.randn(T_w, self.d_embed))
            blend_list.append(torch.randn(T_w, self.d_blend))
            other_list.append(torch.randn(T_w, self.d_other))

        target = random.random()  # scalar in [0, 1)
        return {
            "embeddings": embeddings_list,
            "blendshapes": blend_list,
            "others": other_list,
            "target": float(target),
        }


def collate_videos(batch: List[Dict]) -> List[Dict]:
    # Just return the list of video dicts, no padding.
    return batch


# =========================================================
# Nested Cross-Validation utilities (index-level)
# =========================================================

def kfold_split_indices(indices: List[int], n_splits: int, fold_idx: int) -> Tuple[List[int], List[int]]:
    """
    Simple K-fold split on indices.
    Returns (train_indices, val_indices) for a given fold_idx.
    """
    assert 0 <= fold_idx < n_splits
    n = len(indices)
    fold_size = n // n_splits
    remainder = n % n_splits

    # Create fold boundaries
    fold_sizes = [fold_size + (1 if i < remainder else 0) for i in range(n_splits)]
    starts = [sum(fold_sizes[:i]) for i in range(n_splits)]
    ends = [starts[i] + fold_sizes[i] for i in range(n_splits)]

    val_start, val_end = starts[fold_idx], ends[fold_idx]
    val_indices = indices[val_start:val_end]
    train_indices = indices[:val_start] + indices[val_end:]
    return train_indices, val_indices


def nested_cv(
    video_ids: List[str],
    model_cfg: ModelConfig,
    base_train_cfg: TrainingConfig,
    selection_cfg: SelectionConfig,
    eventfulness_scores_root: str,
    embeddings_root: str,
    features_root: str,
    outer_folds: int,
    inner_folds: int,
) -> None:
    """
    Nested cross-validation scaffold (pseudocode):

    Outer loop:
      - Split video_ids into outer_train (for model selection) and outer_test
    Inner loop:
      - Split outer_train into inner_train and inner_val
      - Train and evaluate to select best hyperparameters (you'll add grid/search)
    Then:
      - Retrain on full outer_train with best hyperparams
      - Evaluate on outer_test

    This function shows the structure; you will:
      - plug RealVideoDataset in,
      - possibly modify TrainingConfig per hyperparam setting,
      - log metrics, etc.
    """
    all_indices = list(range(len(video_ids)))

    for outer_idx in range(outer_folds):
        print(f"\n====== Outer fold {outer_idx + 1}/{outer_folds} ======")
        outer_train_idx, outer_test_idx = kfold_split_indices(all_indices, outer_folds, outer_idx)

        outer_train_ids = [video_ids[i] for i in outer_train_idx]
        outer_test_ids = [video_ids[i] for i in outer_test_idx]

        # INNER LOOP: model selection on outer_train_ids
        # You can define a hyperparam grid here. For now, we just show a single config.
        best_inner_val_loss = float("inf")
        best_inner_state = None
        best_inner_cfg = deepcopy(base_train_cfg)  # you can vary LR, weight decay, etc. inside inner loop

        for inner_idx in range(inner_folds):
            print(f"  -- Inner fold {inner_idx + 1}/{inner_folds}")
            inner_all_idx = list(range(len(outer_train_ids)))
            inner_train_idx, inner_val_idx = kfold_split_indices(inner_all_idx, inner_folds, inner_idx)

            inner_train_ids = [outer_train_ids[i] for i in inner_train_idx]
            inner_val_ids = [outer_train_ids[i] for i in inner_val_idx]

            # Build datasets (RealVideoDataset with your memmap logic)
            # Replace window_size/stride with your actual config:
            window_size = 30
            stride = 10

            train_dataset = RealVideoDataset(
                video_ids=inner_train_ids,
                embeddings_root=embeddings_root,
                features_root=features_root,
                d_embed=model_cfg.d_embed,
                d_blend=model_cfg.d_blend,
                d_other=model_cfg.d_other,
                window_size=selection_cfg.window_size,
                stride=selection_cfg.stride,
            )
            val_dataset = RealVideoDataset(
                video_ids=inner_val_ids,
                embeddings_root=embeddings_root,
                features_root=features_root,
                d_embed=model_cfg.d_embed,
                d_blend=model_cfg.d_blend,
                d_other=model_cfg.d_other,
                window_size=window_size,
                stride=stride,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=base_train_cfg.batch_size,
                shuffle=True,
                collate_fn=collate_videos,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=base_train_cfg.batch_size,
                shuffle=False,
                collate_fn=collate_videos,
            )

            # Fresh model for each inner fold
            model = MultiPathMILRegressor(model_cfg)
            model, val_loss = train_one_split(model, train_loader, val_loader, base_train_cfg)

            if val_loss < best_inner_val_loss:
                best_inner_val_loss = val_loss
                best_inner_state = deepcopy(model.state_dict())
                # If you change hyperparams per inner fold, track which config is best here.
                # best_inner_cfg = <cfg_for_this_inner_fold>

        print(f"  Best inner val loss for outer fold {outer_idx + 1}: {best_inner_val_loss:.4f}")

        # Retrain on full outer_train_ids using best_inner_cfg and best_inner_state if needed
        # (You might want to retrain from scratch here using best hyperparameters.)
        # For simplicity, we just re-initialize and train once on all outer_train_ids:

        print("  Retraining on full outer_train set with selected configuration...")

        window_size = 30
        stride = 10

        outer_train_dataset = RealVideoDataset(
            video_ids=outer_train_ids,
            embeddings_root=embeddings_root,
            features_root=features_root,
            d_embed=model_cfg.d_embed,
            d_blend=model_cfg.d_blend,
            d_other=model_cfg.d_other,
            window_size=window_size,
            stride=stride,
        )
        outer_test_dataset = RealVideoDataset(
            video_ids=outer_test_ids,
            embeddings_root=embeddings_root,
            features_root=features_root,
            d_embed=model_cfg.d_embed,
            d_blend=model_cfg.d_blend,
            d_other=model_cfg.d_other,
            window_size=window_size,
            stride=stride,
        )

        outer_train_loader = DataLoader(
            outer_train_dataset,
            batch_size=base_train_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_videos,
        )
        outer_test_loader = DataLoader(
            outer_test_dataset,
            batch_size=base_train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_videos,
        )

        model = MultiPathMILRegressor(model_cfg)
        # Optionally: start from best_inner_state, or from scratch
        # model.load_state_dict(best_inner_state)
        model, _ = train_one_split(model, outer_train_loader, outer_test_loader, base_train_cfg)

        # Here you would compute and store metrics on outer_test_loader
        # (MAE, RMSE, correlation, etc.) as your unbiased estimate.
        print(f"====== End of outer fold {outer_idx + 1}/{outer_folds} ======")


# =========================================================
# Main example: choose between dummy data and real nested CV
# =========================================================

def main():
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)

    USE_DUMMY_DATA = True  # Set to False to use RealVideoDataset + nested CV

    # Model configuration
    temporal_cfg = TemporalConfig(
        backbone="gru",      # try "lstm", "tcn", "transformer"
        hidden_size=128,
        num_layers=1,
        dropout=0.1,
        activation="relu",
    )

    window_mlp_cfg = MLPConfig(
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        activation="relu",
    )

    reg_head_mlp_cfg = MLPConfig(
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        activation="relu",
    )

    model_cfg = ModelConfig(
        d_embed=768,
        d_blend=52,
        d_other=4,
        d_window=256,

        # Choose per-path temporal sizes
        d_temporal_embed=256,   # more capacity for image embeddings
        d_temporal_blend=64,    # moderate for blendshapes
        d_temporal_other=32,    # smaller for low-dim features

        temporal=temporal_cfg,
        window_mlp=window_mlp_cfg,
        reg_head_mlp=reg_head_mlp_cfg,
        attn_hidden_size=128,
        attn_activation="tanh",
    )

    train_cfg = TrainingConfig(
        num_epochs=5,
        batch_size=4,
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-4,
        loss="smooth_l1",
        early_stopping_patience=10,
        device="cuda",
        use_scheduler=False,
    )

    selection_cfg = SelectionConfig(
        window_size = 64,
        stride = 32,
        n_windows = 16,

        pick_mean_top_begin = 2,
        pick_max_top_begin = 2,
        pick_random_begin = 12, # must add up to n_windows

        pick_mean_top_end = 12,
        pick_max_top_end = 2,
        pick_random_end = 2, # must add up to n_windows
    )

    if USE_DUMMY_DATA:
        # Simple single train/val split on dummy data (no nested CV)
        train_dataset = DummyVideoDataset(
            num_videos=64,
            d_embed=model_cfg.d_embed,
            d_blend=model_cfg.d_blend,
            d_other=model_cfg.d_other,
        )
        val_dataset = DummyVideoDataset(
            num_videos=16,
            d_embed=model_cfg.d_embed,
            d_blend=model_cfg.d_blend,
            d_other=model_cfg.d_other,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_videos,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_videos,
        )

        model = MultiPathMILRegressor(model_cfg)
        model, best_val_loss = train_one_split(model, train_loader, val_loader, train_cfg)
        print(f"Training complete on dummy data. Best validation loss: {best_val_loss:.4f}")

    else:
        nested_cv(
            video_ids=video_ids,
            model_cfg=model_cfg,
            base_train_cfg=train_cfg,
            selection_cfg=selection_cfg,
            eventfulness_scores_root=eventfulness_scores_root,
            embeddings_root=embeddings_root,
            features_root=features_root,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
        )


if __name__ == "__main__":
    main()