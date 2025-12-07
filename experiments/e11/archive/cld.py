# """
# Multi-Path Temporal Architecture with Attention-Based MIL for Fatigue Regression.

# Architecture:
# - WindowTemporalEncoder: Encodes [T, D] -> [D_out] using GRU/LSTM/TCN/Transformer
# - MultiPathWindowEncoder: Three parallel encoders + fusion MLP -> window embedding
# - AttentionMILPooling: Ilse-style attention over windows -> video embedding
# - MultiPathMILRegressor: Full model with regression head
# """

# import math
# import copy
# from dataclasses import dataclass, field
# from typing import List, Dict, Tuple, Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader


# # =============================================================================
# # Configuration Dataclasses #cld
# # =============================================================================

# @dataclass
# class TemporalConfig:
#     """Config for WindowTemporalEncoder."""
#     backbone: str = "gru"  # "gru", "lstm", "tcn", "transformer"
#     hidden_size: int = 128
#     num_layers: int = 1
#     dropout: float = 0.1
#     kernel_size: int = 3 # TCN-specific
#     nhead: int = 4 # Transformer-specific
#     dim_feedforward: int = 256 # Transformer-specific


# @dataclass
# class MLPConfig:
#     """Config for MLP blocks."""
#     hidden_size: int = 128
#     num_layers: int = 1  # hidden layers (total layers = num_layers + 1)
#     dropout: float = 0.1
#     activation: str = "relu"  # "relu", "gelu", "tanh"


# @dataclass
# class ModelConfig:
#     """Full model configuration."""
#     d_embed: int = 512
#     d_blend: int = 52
#     d_other: int = 8
    
#     d_window: int = 256

#     # d_temporal is missing

#     temporal: TemporalConfig = field(default_factory=TemporalConfig)
#     window_mlp: MLPConfig = field(default_factory=MLPConfig)
#     reg_head_mlp: MLPConfig = field(default_factory=MLPConfig)

#     attn_hidden_size: int = 128up
#     attn_activation: str = "tanh"


# @dataclass
# class TrainingConfig:
#     """Training pipeline configuration."""
#     num_epochs: int = 50
#     batch_size: int = 4
#     optimizer: str = "adamw"  # "adam", "adamw", "sgd"
#     lr: float = 1e-3
#     weight_decay: float = 1e-4
#     betas: Tuple[float, float] = (0.9, 0.999)
#     momentum: float = 0.9
#     loss: str = "smooth_l1"  # "mse", "smooth_l1"
#     early_stopping_patience: int = 10
#     scheduler: Optional[str] = None  # None, "step", "plateau"
#     scheduler_step_size: int = 10
#     scheduler_gamma: float = 0.5
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"


# # =============================================================================
# # Helpers
# # =============================================================================

# def get_activation(name: str) -> nn.Module:
#     name = name.lower()
#     if name == "relu":
#         return nn.ReLU()
#     elif name == "gelu":
#         return nn.GELU()
#     elif name == "tanh":
#         return nn.Tanh()
#     elif name == "leaky_relu":
#         return nn.LeakyReLU()
#     elif name == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         raise ValueError(f"Unknown activation: {name}")


# def build_mlp(
#     in_dim: int,
#     out_dim: int,
#     cfg: MLPConfig
# ) -> nn.Sequential:
#     """Build MLP with configurable layers, activation, dropout."""
#     layers = []
#     current_dim = in_dim
#     for _ in range(cfg.num_layers):
#         layers.append(nn.Linear(current_dim, cfg.hidden_size))
#         layers.append(get_activation(cfg.activation))
#         if cfg.dropout > 0:
#             layers.append(nn.Dropout(cfg.dropout))
#         current_dim = cfg.hidden_size
#     layers.append(nn.Linear(current_dim, out_dim))
#     return nn.Sequential(*layers)


# # =============================================================================
# # Temporal Backbones
# # =============================================================================

# class TCNBlock(nn.Module):
#     """Single TCN block with residual connection."""
#     def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
#         super().__init__()
#         padding = (kernel_size - 1) * dilation // 2
#         self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
#         self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, T]
#         out = self.dropout(self.relu(self.conv1(x)))
#         out = self.dropout(self.relu(self.conv2(out)))
#         return self.relu(out + self.downsample(x))


# class TCN(nn.Module):
#     """Temporal Convolutional Network."""
#     def __init__(self, in_dim: int, hidden_size: int, num_layers: int, kernel_size: int, dropout: float):
#         super().__init__()
#         layers = []
#         for i in range(num_layers):
#             d = 2 ** i
#             in_ch = in_dim if i == 0 else hidden_size
#             layers.append(TCNBlock(in_ch, hidden_size, kernel_size, d, dropout))
#         self.net = nn.Sequential(*layers)
#         self.out_dim = hidden_size

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [T, D] -> output: [hidden_size]
#         x = x.transpose(0, 1).unsqueeze(0)  # [1, D, T]
#         x = self.net(x)  # [1, hidden_size, T]
#         return x.mean(dim=2).squeeze(0)  # mean pool over time -> [hidden_size]


# class PositionalEncoding(nn.Module):
#     """Sinusoidal positional encoding for Transformer."""
#     def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [T, D]
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


# # =============================================================================
# # Model Components
# # =============================================================================

# class WindowTemporalEncoder(nn.Module):
#     """
#     Encodes a single temporal sequence [T, D_in] -> [D_out].
    
#     Output strategy: For RNN, uses final hidden state (concatenated for BiRNN).
#     For TCN/Transformer, uses mean pooling over time.
#     """
#     def __init__(self, d_in: int, cfg: TemporalConfig):
#         super().__init__()
#         self.cfg = cfg
#         self.backbone_type = cfg.backbone.lower()
#         h = cfg.hidden_size

#         if self.backbone_type == "gru":
#             self.rnn = nn.GRU(d_in, h, cfg.num_layers, dropout=cfg.dropout if cfg.num_layers > 1 else 0,
#                              bidirectional=True, batch_first=False)
#             self.out_dim = h * 2
#         elif self.backbone_type == "lstm":
#             self.rnn = nn.LSTM(d_in, h, cfg.num_layers, dropout=cfg.dropout if cfg.num_layers > 1 else 0,
#                               bidirectional=True, batch_first=False)
#             self.out_dim = h * 2
#         elif self.backbone_type == "tcn":
#             self.tcn = TCN(d_in, h, cfg.num_layers, cfg.kernel_size, cfg.dropout)
#             self.out_dim = h
#         elif self.backbone_type == "transformer":
#             self.input_proj = nn.Linear(d_in, h)
#             self.pos_enc = PositionalEncoding(h, dropout=cfg.dropout)
#             enc_layer = nn.TransformerEncoderLayer(d_model=h, nhead=cfg.nhead,
#                                                    dim_feedforward=cfg.dim_feedforward,
#                                                    dropout=cfg.dropout, batch_first=False)
#             self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
#             self.out_dim = h
#         else:
#             raise ValueError(f"Unknown backbone: {cfg.backbone}")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: [T, D_in] - single window sequence
#         Returns:
#             [D_out] - fixed-size embedding
#         """
#         if self.backbone_type in ("gru", "lstm"):
#             x = x.unsqueeze(1)  # [T, 1, D]
#             _, h_n = self.rnn(x) if self.backbone_type == "gru" else self.rnn(x)
#             if self.backbone_type == "lstm":
#                 h_n = h_n[0]  # take hidden, not cell
#             # h_n: [num_layers*2, 1, h] for bidirectional
#             h_n = h_n[-2:, 0, :]  # last layer fwd/bwd: [2, h]
#             return h_n.flatten()  # [2*h]
#         elif self.backbone_type == "tcn":
#             return self.tcn(x)
#         else:  # transformer
#             x = self.input_proj(x)  # [T, h]
#             x = self.pos_enc(x)
#             x = x.unsqueeze(1)  # [T, 1, h]
#             x = self.transformer(x)  # [T, 1, h]
#             return x.mean(dim=0).squeeze(0)  # [h]


# class MultiPathWindowEncoder(nn.Module):
#     """
#     Three parallel WindowTemporalEncoders + fusion MLP.
    
#     For each window:
#     - Encode embeddings, blendshapes, other features separately
#     - Concatenate outputs
#     - Pass through fusion MLP -> D_window
#     """
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.enc_embed = WindowTemporalEncoder(cfg.d_embed, cfg.temporal)
#         self.enc_blend = WindowTemporalEncoder(cfg.d_blend, cfg.temporal)
#         self.enc_other = WindowTemporalEncoder(cfg.d_other, cfg.temporal)
        
#         concat_dim = self.enc_embed.out_dim + self.enc_blend.out_dim + self.enc_other.out_dim
#         self.fusion_mlp = build_mlp(concat_dim, cfg.d_window, cfg.window_mlp)

#     def forward(self, embeddings: torch.Tensor, blendshapes: torch.Tensor, 
#                 others: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             embeddings: [T, D_embed]
#             blendshapes: [T, D_blend]
#             others: [T, D_other]
#         Returns:
#             [D_window]
#         """
#         z_e = self.enc_embed(embeddings)
#         z_b = self.enc_blend(blendshapes)
#         z_o = self.enc_other(others)
#         z_cat = torch.cat([z_e, z_b, z_o], dim=-1)
#         return self.fusion_mlp(z_cat)


# class AttentionMILPooling(nn.Module):
#     """
#     Ilse-style attention pooling over instances (windows).
    
#     a_i = softmax(w^T tanh(V h_i))
#     z = sum_i a_i * h_i
#     """
#     def __init__(self, d_window: int, attn_hidden: int, activation: str = "tanh"):
#         super().__init__()
#         self.V = nn.Linear(d_window, attn_hidden)
#         self.w = nn.Linear(attn_hidden, 1)
#         self.act = get_activation(activation)

#     def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             H: [N_windows, D_window]
#         Returns:
#             z: [D_window] - video embedding
#             attn: [N_windows] - attention weights
#         """
#         # a_i = softmax(w^T act(V h_i))
#         scores = self.w(self.act(self.V(H)))  # [N, 1]
#         attn = F.softmax(scores, dim=0).squeeze(-1)  # [N]
#         z = (attn.unsqueeze(-1) * H).sum(dim=0)  # [D_window]
#         return z, attn


# class MultiPathMILRegressor(nn.Module):
#     """
#     Full model: MultiPathWindowEncoder + AttentionMILPooling + regression head.
    
#     Input: batch of videos, each video is a dict with:
#         - "embeddings": List[Tensor] of [T_w, D_embed]
#         - "blendshapes": List[Tensor] of [T_w, D_blend]  
#         - "others": List[Tensor] of [T_w, D_other]
#         - "target": float (only needed for training)
#     """
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.cfg = cfg
#         self.window_encoder = MultiPathWindowEncoder(cfg)
#         self.mil_pooling = AttentionMILPooling(cfg.d_window, cfg.attn_hidden_size, cfg.attn_activation)
#         self.reg_head = build_mlp(cfg.d_window, 1, cfg.reg_head_mlp)

#     def encode_video(self, video: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Encode a single video to embedding and attention weights."""
#         window_embeds = []
#         n_windows = len(video["embeddings"])
#         for i in range(n_windows):
#             emb = video["embeddings"][i]
#             blend = video["blendshapes"][i]
#             other = video["others"][i]
#             w_emb = self.window_encoder(emb, blend, other)
#             window_embeds.append(w_emb)
#         H = torch.stack(window_embeds, dim=0)  # [N_windows, D_window]
#         z, attn = self.mil_pooling(H)
#         return z, attn

#     def forward(self, batch: List[Dict]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         """
#         Args:
#             batch: List of video dicts
#         Returns:
#             preds: [B] tensor of predictions
#             attns: List of [N_windows_i] attention tensors per video
#         """
#         preds = []
#         attns = []
#         for video in batch:
#             z, attn = self.encode_video(video)
#             pred = self.reg_head(z).squeeze(-1)
#             preds.append(pred)
#             attns.append(attn)
#         return torch.stack(preds), attns


# # =============================================================================
# # Dummy Dataset
# # =============================================================================

# class DummyVideoDataset(Dataset):
#     """
#     Generates random videos for testing.
    
#     Each video has N_windows windows, each window has T frames.
#     """
#     def __init__(self, num_videos: int, cfg: ModelConfig, 
#                  n_windows_range: Tuple[int, int] = (3, 8),
#                  t_range: Tuple[int, int] = (10, 30),
#                  device: str = "cpu"):
#         self.num_videos = num_videos
#         self.cfg = cfg
#         self.n_windows_range = n_windows_range
#         self.t_range = t_range
#         self.device = device
#         self.videos = self._generate_videos()

#     def _generate_videos(self) -> List[Dict]:
#         videos = []
#         for _ in range(self.num_videos):
#             n_win = torch.randint(self.n_windows_range[0], self.n_windows_range[1] + 1, (1,)).item()
#             embeddings, blendshapes, others = [], [], []
#             for _ in range(n_win):
#                 t = torch.randint(self.t_range[0], self.t_range[1] + 1, (1,)).item()
#                 embeddings.append(torch.randn(t, self.cfg.d_embed, device=self.device))
#                 blendshapes.append(torch.randn(t, self.cfg.d_blend, device=self.device))
#                 others.append(torch.randn(t, self.cfg.d_other, device=self.device))
#             target = torch.rand(1, device=self.device).item() * 10  # fatigue 0-10
#             videos.append({
#                 "embeddings": embeddings,
#                 "blendshapes": blendshapes,
#                 "others": others,
#                 "target": target
#             })
#         return videos

#     def __len__(self) -> int:
#         return self.num_videos

#     def __getitem__(self, idx: int) -> Dict:
#         return self.videos[idx]


# def collate_videos(batch: List[Dict]) -> List[Dict]:
#     """Simple collate: just return list of video dicts."""
#     return batch


# # =============================================================================
# # Training Pipeline
# # =============================================================================

# def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
#     """Build optimizer from config."""
#     opt_name = cfg.optimizer.lower()
#     if opt_name == "adam":
#         return torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
#     elif opt_name == "adamw":
#         return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
#     elif opt_name == "sgd":
#         return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
#     else:
#         raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


# def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig):
#     """Build optional LR scheduler."""
#     if cfg.scheduler is None:
#         return None
#     elif cfg.scheduler == "step":
#         return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
#     elif cfg.scheduler == "plateau":
#         return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler_gamma, patience=5)
#     else:
#         raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


# def get_loss_fn(cfg: TrainingConfig):
#     """Return loss function based on config."""
#     if cfg.loss == "mse":
#         return nn.MSELoss()
#     elif cfg.loss == "smooth_l1":
#         return nn.SmoothL1Loss()
#     else:
#         raise ValueError(f"Unknown loss: {cfg.loss}")


# def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
#                 loss_fn: nn.Module, device: str) -> float:
#     """Train for one epoch, return average loss."""
#     model.train()
#     total_loss = 0.0
#     n_batches = 0
#     for batch in loader:
#         optimizer.zero_grad()
#         targets = torch.tensor([v["target"] for v in batch], device=device, dtype=torch.float32)
#         preds, _ = model(batch)
#         loss = loss_fn(preds, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         n_batches += 1
#     return total_loss / max(n_batches, 1)


# @torch.no_grad()
# def eval_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str) -> float:
#     """Evaluate model, return average loss."""
#     model.eval()
#     total_loss = 0.0
#     n_batches = 0
#     for batch in loader:
#         targets = torch.tensor([v["target"] for v in batch], device=device, dtype=torch.float32)
#         preds, _ = model(batch)
#         loss = loss_fn(preds, targets)
#         total_loss += loss.item()
#         n_batches += 1
#     return total_loss / max(n_batches, 1)


# def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
#           train_cfg: TrainingConfig) -> Dict:
#     """
#     Full training loop with early stopping.
    
#     Returns:
#         dict with 'best_val_loss', 'best_epoch', 'model_state_dict'
#     """
#     device = train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu"
#     model = model.to(device)
    
#     optimizer = build_optimizer(model, train_cfg)
#     scheduler = build_scheduler(optimizer, train_cfg)
#     loss_fn = get_loss_fn(train_cfg)
    
#     best_val_loss = float('inf')
#     best_epoch = 0
#     best_state = None
#     patience_counter = 0
    
#     print(f"Training on {device} for up to {train_cfg.num_epochs} epochs")
#     print(f"Early stopping patience: {train_cfg.early_stopping_patience}")
#     print("-" * 60)
    
#     for epoch in range(1, train_cfg.num_epochs + 1):
#         train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
#         val_loss = eval_epoch(model, val_loader, loss_fn, device)
        
#         # Scheduler step
#         if scheduler is not None:
#             if train_cfg.scheduler == "plateau":
#                 scheduler.step(val_loss)
#             else:
#                 scheduler.step()
        
#         # Check for improvement
#         improved = val_loss < best_val_loss
#         if improved:
#             best_val_loss = val_loss
#             best_epoch = epoch
#             best_state = copy.deepcopy(model.state_dict())
#             patience_counter = 0
#             marker = " *"
#         else:
#             patience_counter += 1
#             marker = ""
        
#         lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
#               f"Best: {best_val_loss:.4f} (ep {best_epoch}) | LR: {lr:.2e}{marker}")
        
#         # Early stopping
#         if patience_counter >= train_cfg.early_stopping_patience:
#             print(f"\nEarly stopping triggered after {epoch} epochs")
#             break
    
#     # Restore best model
#     if best_state is not None:
#         model.load_state_dict(best_state)
    
#     print("-" * 60)
#     print(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
#     return {
#         "best_val_loss": best_val_loss,
#         "best_epoch": best_epoch,
#         "model_state_dict": best_state
#     }


# # =============================================================================
# # Main
# # =============================================================================

# def main():
#     # Set device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Model config
#     model_cfg = ModelConfig(
#         d_embed=512,
#         d_blend=52,
#         d_other=8,
#         d_window=256,
#         temporal=TemporalConfig(
#             backbone="gru",  # try: "gru", "lstm", "tcn", "transformer"
#             hidden_size=128,
#             num_layers=1,
#             dropout=0.1
#         ),
#         window_mlp=MLPConfig(
#             hidden_size=128,
#             num_layers=1,
#             dropout=0.2,
#             activation="relu"
#         ),
#         reg_head_mlp=MLPConfig(
#             hidden_size=64,
#             num_layers=1,
#             dropout=0.2,
#             activation="relu"
#         ),
#         attn_hidden_size=128,
#         attn_activation="tanh"
#     )
    
#     # Training config
#     train_cfg = TrainingConfig(
#         num_epochs=30,
#         batch_size=4,
#         optimizer="adamw",
#         lr=1e-3,
#         weight_decay=1e-4,
#         loss="smooth_l1",
#         early_stopping_patience=5,
#         scheduler="plateau",
#         device=device
#     )
    
#     # Create dummy datasets
#     print("Creating dummy datasets...")
#     train_dataset = DummyVideoDataset(num_videos=40, cfg=model_cfg, device=device)
#     val_dataset = DummyVideoDataset(num_videos=10, cfg=model_cfg, device=device)
    
#     train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, 
#                               shuffle=True, collate_fn=collate_videos)
#     val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size,
#                             shuffle=False, collate_fn=collate_videos)
    
#     # Build model
#     print(f"Building model with {model_cfg.temporal.backbone.upper()} backbone...")
#     model = MultiPathMILRegressor(model_cfg)
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Model parameters: {n_params:,}")
#     print()
    
#     # Train
#     result = train(model, train_loader, val_loader, train_cfg)
    
#     # Demo inference
#     print("\n" + "=" * 60)
#     print("Demo inference on validation set:")
#     model.eval()
#     with torch.no_grad():
#         sample_batch = [val_dataset[i] for i in range(min(3, len(val_dataset)))]
#         preds, attns = model(sample_batch)
#         for i, (pred, attn, video) in enumerate(zip(preds, attns, sample_batch)):
#             print(f"  Video {i+1}: pred={pred.item():.2f}, target={video['target']:.2f}, "
#                   f"n_windows={len(attn)}, top_attn={attn.max().item():.3f}")


# if __name__ == "__main__":
#     main()