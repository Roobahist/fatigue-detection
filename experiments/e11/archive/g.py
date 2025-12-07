# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from dataclasses import dataclass, field
# from typing import List, Dict, Tuple, Optional, Union
# import math
# import copy
# import random

# # =============================================================================
# # 1. CONFIGURATION #g
# # =============================================================================

# @dataclass
# class TemporalConfig:
#     backbone: str = "gru"           # "gru", "lstm", "tcn", "transformer"
#     hidden_size: int = 64
#     num_layers: int = 1
#     dropout: float = 0.1
#     kernel_size: int = 3            # Specific to TCN
#     nhead: int = 4                  # Specific to Transformer

# @dataclass
# class MLPConfig:
#     hidden_size: int = 128
#     num_layers: int = 1
#     dropout: float = 0.1
#     activation: str = "relu"        # "relu", "gelu", "tanh", "leaky_relu"

# @dataclass
# class ModelConfig:
#     # Input Dimensions
#     d_embed: int
#     d_blend: int
#     d_other: int
    
#     # Embedding Dimensions (Output of temporal encoders)
#     d_temporal_out: int = 64        
    
#     # Window Level
#     d_window: int = 256             # Size of the fused window embedding
    
#     # Sub-configs
#     temporal: TemporalConfig = field(default_factory=TemporalConfig)
#     window_mlp: MLPConfig = field(default_factory=MLPConfig)
#     reg_head_mlp: MLPConfig = field(default_factory=MLPConfig)
    
#     # Attention MIL
#     attn_hidden_size: int = 128
#     attn_activation: str = "tanh"

# @dataclass
# class TrainingConfig:
#     num_epochs: int = 20
#     batch_size: int = 2             # Small batch size since we process lists of videos
#     optimizer: str = "adamw"        # "adam", "adamw", "sgd"
#     lr: float = 1e-3
#     weight_decay: float = 1e-4
#     betas: Tuple[float, float] = (0.9, 0.999)
#     momentum: float = 0.9
#     loss: str = "smooth_l1"         # "mse", "smooth_l1"
#     early_stopping_patience: int = 5
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"

# # =============================================================================
# # 2. HELPERS
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

# class MLP(nn.Module):
#     """Generic configurable MLP."""
#     def __init__(self, input_dim: int, output_dim: int, cfg: MLPConfig):
#         super().__init__()
#         layers = []
#         in_dim = input_dim
        
#         for _ in range(cfg.num_layers):
#             layers.append(nn.Linear(in_dim, cfg.hidden_size))
#             layers.append(get_activation(cfg.activation))
#             if cfg.dropout > 0:
#                 layers.append(nn.Dropout(cfg.dropout))
#             in_dim = cfg.hidden_size
        
#         # Final projection to output_dim
#         layers.append(nn.Linear(in_dim, output_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)

# # =============================================================================
# # 3. MODEL COMPONENTS
# # =============================================================================

# class WindowTemporalEncoder(nn.Module):
#     """
#     Encodes a sequence [T, D_in] -> [D_out].
#     Handles GRU, LSTM, TCN, Transformer.
#     """
#     def __init__(self, input_dim: int, output_dim: int, cfg: TemporalConfig):
#         super().__init__()
#         self.backbone_type = cfg.backbone.lower()
#         self.output_dim = output_dim
        
#         if self.backbone_type == "gru":
#             self.model = nn.GRU(
#                 input_size=input_dim,
#                 hidden_size=output_dim,
#                 num_layers=cfg.num_layers,
#                 batch_first=True,
#                 dropout=cfg.dropout if cfg.num_layers > 1 else 0
#             )
#         elif self.backbone_type == "lstm":
#             self.model = nn.LSTM(
#                 input_size=input_dim,
#                 hidden_size=output_dim,
#                 num_layers=cfg.num_layers,
#                 batch_first=True,
#                 dropout=cfg.dropout if cfg.num_layers > 1 else 0
#             )
#         elif self.backbone_type == "tcn":
#             # Simple stack of Conv1d
#             layers = []
#             in_ch = input_dim
#             for _ in range(cfg.num_layers):
#                 layers.append(nn.Conv1d(in_ch, output_dim, kernel_size=cfg.kernel_size, padding='same'))
#                 layers.append(nn.ReLU())
#                 layers.append(nn.Dropout(cfg.dropout))
#                 in_ch = output_dim
#             self.model = nn.Sequential(*layers)
#         elif self.backbone_type == "transformer":
#             # Simple Transformer Encoder.
#             # Note: For simplicity, using a learnable positional embedding up to max T=500.
#             self.pos_encoder = nn.Parameter(torch.randn(1, 500, input_dim) * 0.02)
#             self.project_in = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
            
#             enc_layer = nn.TransformerEncoderLayer(
#                 d_model=output_dim,
#                 nhead=cfg.nhead,
#                 dim_feedforward=output_dim * 2,
#                 dropout=cfg.dropout,
#                 activation="relu",
#                 batch_first=True
#             )
#             self.model = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
#         else:
#             raise ValueError(f"Unknown backbone: {cfg.backbone}")

#     def forward(self, x):
#         # x shape: [T, D_in] or [B, T, D_in]. We ensure [B, T, D_in] locally.
#         if x.dim() == 2:
#             x = x.unsqueeze(0) # [1, T, D]
        
#         B, T, _ = x.shape

#         if self.backbone_type in ["gru", "lstm"]:
#             output, _ = self.model(x)
#             # Take last time step
#             return output[:, -1, :]
            
#         elif self.backbone_type == "tcn":
#             # Conv1d expects [B, C, T]
#             x_perm = x.permute(0, 2, 1)
#             out = self.model(x_perm) # [B, D_out, T]
#             # Global Max Pooling over time
#             out = F.max_pool1d(out, kernel_size=out.shape[2]).squeeze(2)
#             return out
            
#         elif self.backbone_type == "transformer":
#             # Add PE
#             if self.project_in is not None:
#                 x = self.project_in(x) # Now [B, T, D_out]
            
#             # Crop PE to current length
#             pe = self.pos_encoder[:, :T, :] if T <= 500 else self.pos_encoder[:, :500, :]
#             # If T > 500, we just repeat or zero pad (simplification for this prompt)
#             if T > 500:
#                  # Fallback for very long sequences in dummy code
#                  pe = F.pad(pe, (0,0,0, T-500))
                 
#             x = x + pe
#             out = self.model(x) # [B, T, D_out]
#             # Mean pooling over time
#             return out.mean(dim=1)

# class MultiPathWindowEncoder(nn.Module):
#     """
#     Processes 3 streams (Embed, Blend, Other) and fuses them.
#     """
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
        
#         # Temporal Encoders
#         self.enc_embed = WindowTemporalEncoder(cfg.d_embed, cfg.d_temporal_out, cfg.temporal)
#         self.enc_blend = WindowTemporalEncoder(cfg.d_blend, cfg.d_temporal_out, cfg.temporal)
#         self.enc_other = WindowTemporalEncoder(cfg.d_other, cfg.d_temporal_out, cfg.temporal)
        
#         # Fusion MLP
#         # Input size is 3 * d_temporal_out
#         fusion_input_dim = cfg.d_temporal_out * 3
#         self.fusion_mlp = MLP(fusion_input_dim, cfg.d_window, cfg.window_mlp)

#     def forward(self, embed_seq, blend_seq, other_seq):
#         # Each input is [T, D]. 
#         # Temporal encoders handle adding batch dim internally.
#         h1 = self.enc_embed(embed_seq) # [1, D_temp]
#         h2 = self.enc_blend(blend_seq) # [1, D_temp]
#         h3 = self.enc_other(other_seq) # [1, D_temp]
        
#         concat = torch.cat([h1, h2, h3], dim=1) # [1, 3*D_temp]
#         window_emb = self.fusion_mlp(concat)    # [1, D_window]
#         return window_emb

# class AttentionMILPooling(nn.Module):
#     """
#     Ilse et al. Attention Mechanism.
#     z = sum(a_i * h_i)
#     a_i = softmax(w^T * tanh(V * h_i))
#     """
#     def __init__(self, input_dim: int, cfg: ModelConfig):
#         super().__init__()
#         self.V = nn.Linear(input_dim, cfg.attn_hidden_size)
#         self.w = nn.Linear(cfg.attn_hidden_size, 1)
#         self.act = get_activation(cfg.attn_activation)

#     def forward(self, window_embeddings):
#         # window_embeddings: [N_windows, D_window] (stacked)
        
#         # 1. Projection
#         attn_proj = self.V(window_embeddings) # [N, D_attn]
#         attn_proj = self.act(attn_proj)
        
#         # 2. Attention Scores
#         attn_logits = self.w(attn_proj)       # [N, 1]
        
#         # 3. Weights
#         attn_weights = torch.softmax(attn_logits, dim=0) # [N, 1]
        
#         # 4. Weighted Sum
#         # Broadcast weights: [N, 1] * [N, D_win] -> sum -> [D_win]
#         video_embedding = torch.sum(attn_weights * window_embeddings, dim=0)
        
#         return video_embedding, attn_weights

# class MultiPathMILRegressor(nn.Module):
#     """
#     Main Model.
#     Video -> Windows -> MultiPath Encoder -> MIL Pooling -> Regression Head.
#     """
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.cfg = cfg
#         self.window_encoder = MultiPathWindowEncoder(cfg)
#         self.mil_pool = AttentionMILPooling(cfg.d_window, cfg)
#         self.reg_head = MLP(cfg.d_window, 1, cfg.reg_head_mlp)

#     def forward(self, batch_videos: List[Dict[str, List[torch.Tensor]]]):
#         """
#         Args:
#             batch_videos: A list of video dicts. Each dict contains lists of tensors 
#                           for 'embeddings', 'blendshapes', 'others'.
#                           We loop because T varies per window and N varies per video.
#         """
#         video_scores = []
        
#         for video in batch_videos:
#             embeddings_list = video['embeddings']
#             blendshapes_list = video['blendshapes']
#             others_list = video['others']
            
#             num_windows = len(embeddings_list)
#             w_embs = []

#             # Process each window
#             # Note: In a highly optimized setting, one might group windows by length
#             # or use PackedSequence, but looping is robust for variable T.
#             for i in range(num_windows):
#                 # Each is [T_i, D]
#                 emb = embeddings_list[i]
#                 blend = blendshapes_list[i]
#                 oth = others_list[i]
                
#                 # Encode -> [1, D_window]
#                 w_out = self.window_encoder(emb, blend, oth)
#                 w_embs.append(w_out)
            
#             if len(w_embs) == 0:
#                 # Handle edge case of empty video (should not happen in real data)
#                 video_scores.append(torch.zeros(1, device=self.get_device()))
#                 continue

#             # Stack: [N_windows, D_window]
#             w_embs_stack = torch.cat(w_embs, dim=0)
            
#             # MIL Pooling -> [D_window]
#             vid_emb, _ = self.mil_pool(w_embs_stack)
            
#             # Regression Head -> [1]
#             score = self.reg_head(vid_emb)
#             video_scores.append(score)
            
#         # Stack all video scores into batch tensor: [Batch_Size, 1]
#         return torch.stack(video_scores)

#     def get_device(self):
#         return next(self.parameters()).device

# # =============================================================================
# # 4. TRAINING PIPELINE
# # =============================================================================

# def build_optimizer(model: nn.Module, cfg: TrainingConfig):
#     params = [p for p in model.parameters() if p.requires_grad]
    
#     if cfg.optimizer == "adam":
#         return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
#     elif cfg.optimizer == "adamw":
#         return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
#     elif cfg.optimizer == "sgd":
#         return optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
#     else:
#         raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

# def train_model(model: MultiPathMILRegressor, 
#                 train_loader: DataLoader, 
#                 val_loader: DataLoader, 
#                 train_cfg: TrainingConfig):
    
#     model.to(train_cfg.device)
#     optimizer = build_optimizer(model, train_cfg)
    
#     if train_cfg.loss == "mse":
#         criterion = nn.MSELoss()
#     elif train_cfg.loss == "smooth_l1":
#         criterion = nn.SmoothL1Loss()
#     else:
#         raise ValueError(f"Unknown loss: {train_cfg.loss}")

#     best_val_loss = float('inf')
#     patience_counter = 0
#     best_model_state = None

#     print(f"Starting training on {train_cfg.device}...")
#     print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Status'}")
#     print("-" * 45)

#     for epoch in range(1, train_cfg.num_epochs + 1):
#         # --- Training ---
#         model.train()
#         train_loss_sum = 0.0
        
#         for batch in train_loader:
#             # batch is a list of dicts. 
#             # Targets need to be extracted and stacked.
#             targets = torch.tensor([v['target'] for v in batch], 
#                                    dtype=torch.float32, 
#                                    device=train_cfg.device).unsqueeze(1)
            
#             # Move inputs to device (tensors inside list/dict)
#             # Since input is nested list of tensors, we move them just-in-time or here.
#             # Deep move helper:
#             batch_on_device = []
#             for video in batch:
#                 vid_gpu = {
#                     'embeddings': [t.to(train_cfg.device) for t in video['embeddings']],
#                     'blendshapes': [t.to(train_cfg.device) for t in video['blendshapes']],
#                     'others': [t.to(train_cfg.device) for t in video['others']]
#                 }
#                 batch_on_device.append(vid_gpu)

#             optimizer.zero_grad()
#             preds = model(batch_on_device)
#             loss = criterion(preds, targets)
#             loss.backward()
#             optimizer.step()
            
#             train_loss_sum += loss.item() * len(batch)

#         avg_train_loss = train_loss_sum / len(train_loader.dataset)

#         # --- Validation ---
#         model.eval()
#         val_loss_sum = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 targets = torch.tensor([v['target'] for v in batch], 
#                                        dtype=torch.float32, 
#                                        device=train_cfg.device).unsqueeze(1)
                
#                 batch_on_device = []
#                 for video in batch:
#                     vid_gpu = {
#                         'embeddings': [t.to(train_cfg.device) for t in video['embeddings']],
#                         'blendshapes': [t.to(train_cfg.device) for t in video['blendshapes']],
#                         'others': [t.to(train_cfg.device) for t in video['others']]
#                     }
#                     batch_on_device.append(vid_gpu)
                
#                 preds = model(batch_on_device)
#                 loss = criterion(preds, targets)
#                 val_loss_sum += loss.item() * len(batch)
        
#         avg_val_loss = val_loss_sum / len(val_loader.dataset)

#         # --- Early Stopping Check ---
#         status = ""
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_state = copy.deepcopy(model.state_dict())
#             patience_counter = 0
#             status = "Best *"
#         else:
#             patience_counter += 1
#             status = f"Wait {patience_counter}"

#         print(f"{epoch:<6} | {avg_train_loss:.6f}     | {avg_val_loss:.6f}     | {status}")

#         if patience_counter >= train_cfg.early_stopping_patience:
#             print("Early stopping triggered.")
#             break
    
#     print(f"\nTraining complete. Best Val Loss: {best_val_loss:.6f}")
#     model.load_state_dict(best_model_state)
#     return model

# # =============================================================================
# # 5. DUMMY DATASET & MAIN
# # =============================================================================

# class DummyVideoDataset(Dataset):
#     def __init__(self, num_samples=20, d_embed=32, d_blend=10, d_other=5):
#         self.data = []
#         for _ in range(num_samples):
#             n_windows = random.randint(3, 8)
#             video = {
#                 'embeddings': [],
#                 'blendshapes': [],
#                 'others': [],
#                 'target': random.random() * 10.0 # Score 0-10
#             }
#             for _ in range(n_windows):
#                 t_w = random.randint(10, 30) # Variable time length
#                 video['embeddings'].append(torch.randn(t_w, d_embed))
#                 video['blendshapes'].append(torch.randn(t_w, d_blend))
#                 video['others'].append(torch.randn(t_w, d_other))
#             self.data.append(video)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# def collate_fn_list(batch):
#     # Simply return the list of video dicts; model handles the looping
#     return batch

# def main():
#     # 1. Setup Configuration
#     d_embed = 32
#     d_blend = 10
#     d_other = 5

#     # Tunable knobs
#     t_conf = TemporalConfig(backbone="gru", hidden_size=32, num_layers=1, dropout=0.0)
#     mlp_conf = MLPConfig(hidden_size=64, num_layers=1, activation="gelu", dropout=0.1)
#     reg_conf = MLPConfig(hidden_size=32, num_layers=1, activation="relu", dropout=0.0)
    
#     model_cfg = ModelConfig(
#         d_embed=d_embed, 
#         d_blend=d_blend, 
#         d_other=d_other,
#         d_window=128,
#         temporal=t_conf,
#         window_mlp=mlp_conf,
#         reg_head_mlp=reg_conf,
#         attn_activation="tanh"
#     )
    
#     train_cfg = TrainingConfig(
#         num_epochs=15,
#         batch_size=4,
#         optimizer="adamw",
#         lr=0.001,
#         loss="smooth_l1",
#         early_stopping_patience=4
#     )

#     # 2. Instantiate Model
#     model = MultiPathMILRegressor(model_cfg)
    
#     # 3. Prepare Dummy Data
#     train_ds = DummyVideoDataset(num_samples=50, d_embed=d_embed, d_blend=d_blend, d_other=d_other)
#     val_ds = DummyVideoDataset(num_samples=10, d_embed=d_embed, d_blend=d_blend, d_other=d_other)
    
#     train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_fn_list)
#     val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_fn_list)

#     # 4. Train
#     best_model = train_model(model, train_loader, val_loader, train_cfg)

#     # 5. Inference Check
#     print("\n--- Inference Check on one batch ---")
#     best_model.eval()
#     with torch.no_grad():
#         sample_batch = next(iter(val_loader))
#         # Move to device
#         batch_gpu = []
#         for video in sample_batch:
#             batch_gpu.append({
#                 'embeddings': [t.to(train_cfg.device) for t in video['embeddings']],
#                 'blendshapes': [t.to(train_cfg.device) for t in video['blendshapes']],
#                 'others': [t.to(train_cfg.device) for t in video['others']]
#             })
        
#         preds = best_model(batch_gpu)
#         print(f"Predicted scores: {preds.view(-1).cpu().numpy()}")
#         print(f"Actual targets:   {[v['target'] for v in sample_batch]}")

# if __name__ == "__main__":
#     main()