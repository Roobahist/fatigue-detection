# """
# models/mil_temporal.py

# Multi-path temporal model with attention-based MIL pooling for regression.
# """

# from typing import List, Dict, Optional, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -----------------------------------------------------------
# # 1. WindowTemporalEncoder
# # -----------------------------------------------------------

# class WindowTemporalEncoder(nn.Module):
#     """
#     Temporal encoder for a single modality within a window.

#     Supports BiGRU + attention pooling over time.

#     Expected input:
#         x: Tensor of shape [T, D_in] or [B, T, D_in]
#            - T: number of frames in the window
#            - B: number of windows (if batched over windows)

#     Output:
#         Tensor of shape [D_out]  (if input was [T, D_in])
#         or [B, D_out]            (if input was [B, T, D_in])
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: Optional[int] = None,
#         num_layers: int = 1,
#         bidirectional: bool = True,
#         use_attention_pooling: bool = True,
#         attn_hidden_dim: Optional[int] = None,
#         input_proj_dim: Optional[int] = None,
#         dropout: float = 0.0,
#     ):
#         super().__init__()

#         self.use_attention_pooling = use_attention_pooling
#         self.bidirectional = bidirectional

#         # Optional input projection: e.g., D_embed -> smaller dim
#         if input_proj_dim is not None:
#             self.input_proj = nn.Linear(input_dim, input_proj_dim)
#             rnn_input_dim = input_proj_dim
#         else:
#             self.input_proj = None
#             rnn_input_dim = input_dim

#         self.gru = nn.GRU(
#             input_size=rnn_input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,   # input as [B, T, D]
#             bidirectional=bidirectional,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )

#         num_directions = 2 if bidirectional else 1
#         rnn_output_dim = hidden_dim * num_directions

#         # Temporal attention over GRU outputs
#         if self.use_attention_pooling:
#             if attn_hidden_dim is None:
#                 attn_hidden_dim = rnn_output_dim // 2 or 1
#             self.time_attn = nn.Sequential(
#                 nn.Linear(rnn_output_dim, attn_hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(attn_hidden_dim, 1),
#             )
#         else:
#             self.time_attn = None

#         # Optional output projection (e.g. to D_out)
#         if output_dim is not None and output_dim != rnn_output_dim:
#             self.output_proj = nn.Linear(rnn_output_dim, output_dim)
#             self.output_dim = output_dim
#         else:
#             self.output_proj = None
#             self.output_dim = rnn_output_dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass.

#         Args:
#             x: Tensor
#                Shape [T, D_in]  or  [B, T, D_in]

#         Returns:
#             Tensor
#                Shape [D_out]    or  [B, D_out]
#         """
#         # Normalize to [B, T, D]
#         single_instance = False
#         if x.dim() == 2:
#             # [T, D] -> [1, T, D]
#             x = x.unsqueeze(0)
#             single_instance = True
#         elif x.dim() != 3:
#             raise ValueError(f"Expected input dim 2 or 3, got {x.shape}")

#         if self.input_proj is not None:
#             x = self.input_proj(x)  # [B, T, input_proj_dim]

#         # GRU outputs: output [B, T, H_out], h_n [num_layers*num_directions, B, hidden_dim]
#         output, _ = self.gru(x)

#         if self.use_attention_pooling:
#             # Attention over time dimension T
#             # output: [B, T, H]
#             e = self.time_attn(output)         # [B, T, 1]
#             alpha = torch.softmax(e, dim=1)    # [B, T, 1]
#             rep = (alpha * output).sum(dim=1)  # [B, H]
#         else:
#             # Use final time step (last frame) representation
#             rep = output[:, -1, :]  # [B, H]

#         if self.output_proj is not None:
#             rep = self.output_proj(rep)  # [B, D_out]

#         if single_instance:
#             rep = rep.squeeze(0)  # [D_out]

#         return rep


# # -----------------------------------------------------------
# # 2. MultiPathWindowEncoder
# # -----------------------------------------------------------

# class MultiPathWindowEncoder(nn.Module):
#     """
#     Encodes a window using three modalities (paths):

#         1) Frame embeddings path
#         2) Blendshape / AU / pose path
#         3) Other low-dim time-varying features path

#     For each video, you pass all its windows for each modality:
#         embed_seq: [N_windows, T, D_embed]
#         blend_seq: [N_windows, T, D_blend]
#         other_seq: [N_windows, T, D_other]

#     It runs three WindowTemporalEncoders over the windows (N_windows as batch),
#     concatenates the three embeddings, and passes through a fusion MLP:

#         concat_dim = D_emb_path + D_blend_path + D_other_path
#         window_embedding = fusion_mlp(concat)

#     Output:
#         window_embeddings: [N_windows, D_window]
#     """

#     def __init__(
#         self,
#         embed_input_dim: int,
#         blend_input_dim: int,
#         other_input_dim: int,
#         embed_hidden_dim: int = 128,
#         blend_hidden_dim: int = 64,
#         other_hidden_dim: int = 64,
#         embed_output_dim: int = 128,
#         blend_output_dim: int = 64,
#         other_output_dim: int = 64,
#         fusion_hidden_dim: int = 256,
#         window_output_dim: int = 256,
#         use_attention_pooling: bool = True,
#         fusion_dropout: float = 0.1,
#     ):
#         super().__init__()

#         # Path 1: frame embeddings
#         self.embed_encoder = WindowTemporalEncoder(
#             input_dim=embed_input_dim,
#             hidden_dim=embed_hidden_dim,
#             output_dim=embed_output_dim,
#             bidirectional=True,
#             use_attention_pooling=use_attention_pooling,
#         )

#         # Path 2: blendshapes / facial dynamics
#         self.blend_encoder = WindowTemporalEncoder(
#             input_dim=blend_input_dim,
#             hidden_dim=blend_hidden_dim,
#             output_dim=blend_output_dim,
#             bidirectional=True,
#             use_attention_pooling=use_attention_pooling,
#         )

#         # Path 3: other time-varying features
#         self.other_encoder = WindowTemporalEncoder(
#             input_dim=other_input_dim,
#             hidden_dim=other_hidden_dim,
#             output_dim=other_output_dim,
#             bidirectional=True,
#             use_attention_pooling=use_attention_pooling,
#         )

#         concat_dim = embed_output_dim + blend_output_dim + other_output_dim
#         self.window_output_dim = window_output_dim

#         # Fusion MLP to produce final window-level embedding
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(concat_dim, fusion_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(fusion_dropout),
#             nn.Linear(fusion_hidden_dim, window_output_dim),
#             nn.ReLU(),  # keep some non-linearity at window level
#         )

#     def forward(
#         self,
#         embed_seq: torch.Tensor,
#         blend_seq: torch.Tensor,
#         other_seq: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Args:
#             embed_seq: [N_windows, T, D_embed]
#             blend_seq: [N_windows, T, D_blend]
#             other_seq: [N_windows, T, D_other]

#         Returns:
#             window_embeddings: [N_windows, D_window]
#         """
#         if not (embed_seq.dim() == blend_seq.dim() == other_seq.dim() == 3):
#             raise ValueError(
#                 "All modality inputs must be 3D: [N_windows, T, D_mod]. "
#                 f"Got shapes embed={embed_seq.shape}, blend={blend_seq.shape}, other={other_seq.shape}"
#             )

#         # N_windows is treated as batch size inside each temporal encoder
#         emb_path = self.embed_encoder(embed_seq)   # [N_windows, D_emb_path]
#         blend_path = self.blend_encoder(blend_seq) # [N_windows, D_blend_path]
#         other_path = self.other_encoder(other_seq) # [N_windows, D_other_path]

#         # Concatenate along feature dimension
#         concat = torch.cat([emb_path, blend_path, other_path], dim=-1)  # [N_windows, concat_dim]

#         # Final window embedding
#         window_embeddings = self.fusion_mlp(concat)  # [N_windows, D_window]

#         return window_embeddings


# # -----------------------------------------------------------
# # 3. AttentionMILPooling
# # -----------------------------------------------------------

# class AttentionMILPooling(nn.Module):
#     """
#     Attention-based Multiple Instance Learning pooling (Ilse et al., 2018 style).

#     Given window embeddings for a single video:
#         H = [N_windows, D_window]

#     It computes:
#         s_i = f_attn(h_i)                 (scalar score for each window)
#         a_i = softmax(s_i)                (attention weights)
#         z   = sum_i a_i * h_i             (video-level embedding)

#     Optionally supports batched inputs [B, N_windows, D_window], where attention
#     is computed independently per video.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         attn_hidden_dim: int = 128,
#     ):
#         super().__init__()
#         self.attn_mlp = nn.Sequential(
#             nn.Linear(input_dim, attn_hidden_dim),
#             nn.Tanh(),
#             nn.Linear(attn_hidden_dim, 1),  # scalar per window
#         )

#     def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             H: window embeddings
#                Shape [N_windows, D_window] or [B, N_windows, D_window]

#         Returns:
#             z: video-level embedding
#                Shape [D_window] or [B, D_window]
#             alpha: attention weights over windows
#                Shape [N_windows] or [B, N_windows]
#         """
#         if H.dim() == 2:
#             # [N, D]
#             scores = self.attn_mlp(H)           # [N, 1]
#             alpha = torch.softmax(scores, dim=0)  # [N, 1]
#             z = (alpha * H).sum(dim=0)          # [D]
#             return z, alpha.squeeze(-1)         # [N]

#         elif H.dim() == 3:
#             # [B, N, D]
#             scores = self.attn_mlp(H)           # [B, N, 1]
#             alpha = torch.softmax(scores, dim=1)  # [B, N, 1]
#             z = (alpha * H).sum(dim=1)          # [B, D]
#             return z, alpha.squeeze(-1)         # [B, N]

#         else:
#             raise ValueError(f"Expected H dim 2 or 3, got {H.shape}")


# # -----------------------------------------------------------
# # 4. MultiPathMILRegressor (main model)
# # -----------------------------------------------------------

# class MultiPathMILRegressor(nn.Module):
#     """
#     Full model:

#         - Per-window multi-path temporal encoder over frames
#         - MIL attention pooling over windows within each video
#         - Regression head to output a scalar fatigue score per video

#     Expected forward inputs (simple, loop-based batching):

#         embed_batch: List[Tensor], length B
#             embed_batch[i]: [N_windows_i, T, D_embed]

#         blend_batch: List[Tensor], length B
#             blend_batch[i]: [N_windows_i, T, D_blend]

#         other_batch: List[Tensor], length B
#             other_batch[i]: [N_windows_i, T, D_other]

#     Output:
#         {
#             "scores": Tensor [B],                    # fatigue scores
#             "video_embeddings": Tensor [B, D_window],
#             "attentions": List[Tensor], len B,      # each [N_windows_i]
#         }
#     """

#     def __init__(
#         self,
#         embed_input_dim: int,
#         blend_input_dim: int,
#         other_input_dim: int,
#         # Window encoder config
#         embed_hidden_dim: int = 128,
#         blend_hidden_dim: int = 64,
#         other_hidden_dim: int = 64,
#         embed_output_dim: int = 128,
#         blend_output_dim: int = 64,
#         other_output_dim: int = 64,
#         fusion_hidden_dim: int = 256,
#         window_output_dim: int = 256,
#         use_attention_pooling: bool = True,
#         fusion_dropout: float = 0.1,
#         # MIL config
#         mil_attn_hidden_dim: int = 128,
#         # Regression head config
#         reg_hidden_dim: int = 128,
#         reg_dropout: float = 0.1,
#     ):
#         super().__init__()

#         # Multi-path per-window encoder
#         self.window_encoder = MultiPathWindowEncoder(
#             embed_input_dim=embed_input_dim,
#             blend_input_dim=blend_input_dim,
#             other_input_dim=other_input_dim,
#             embed_hidden_dim=embed_hidden_dim,
#             blend_hidden_dim=blend_hidden_dim,
#             other_hidden_dim=other_hidden_dim,
#             embed_output_dim=embed_output_dim,
#             blend_output_dim=blend_output_dim,
#             other_output_dim=other_output_dim,
#             fusion_hidden_dim=fusion_hidden_dim,
#             window_output_dim=window_output_dim,
#             use_attention_pooling=use_attention_pooling,
#             fusion_dropout=fusion_dropout,
#         )

#         # MIL attention pooling over windows
#         self.mil_pool = AttentionMILPooling(
#             input_dim=window_output_dim,
#             attn_hidden_dim=mil_attn_hidden_dim,
#         )

#         # Regression head: z -> scalar fatigue score
#         self.reg_head = nn.Sequential(
#             nn.Linear(window_output_dim, reg_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(reg_dropout),
#             nn.Linear(reg_hidden_dim, 1),
#         )

#     def forward(
#         self,
#         embed_batch: List[torch.Tensor],
#         blend_batch: List[torch.Tensor],
#         other_batch: List[torch.Tensor],
#     ) -> Dict[str, object]:
#         """
#         Args:
#             embed_batch: list of [N_windows_i, T, D_embed]
#             blend_batch: list of [N_windows_i, T, D_blend]
#             other_batch: list of [N_windows_i, T, D_other]

#         Returns:
#             dict with:
#                 scores: Tensor [B]
#                 video_embeddings: Tensor [B, D_window]
#                 attentions: list of attention vectors per video
#         """
#         if not (len(embed_batch) == len(blend_batch) == len(other_batch)):
#             raise ValueError("All modality lists must have the same batch length.")

#         B = len(embed_batch)
#         video_embeddings = []
#         all_attentions = []

#         for i in range(B):
#             embed_seq = embed_batch[i]  # [N_i, T, D_embed]
#             blend_seq = blend_batch[i]  # [N_i, T, D_blend]
#             other_seq = other_batch[i]  # [N_i, T, D_other]

#             # Encode each window via multi-path temporal encoders
#             window_embs = self.window_encoder(
#                 embed_seq=embed_seq,
#                 blend_seq=blend_seq,
#                 other_seq=other_seq,
#             )  # [N_i, D_window]

#             # MIL attention pooling over windows
#             z_i, attn_i = self.mil_pool(window_embs)  # [D_window], [N_i]

#             video_embeddings.append(z_i)
#             all_attentions.append(attn_i)

#         video_embeddings = torch.stack(video_embeddings, dim=0)  # [B, D_window]
#         scores = self.reg_head(video_embeddings).squeeze(-1)     # [B]

#         return {
#             "scores": scores,
#             "video_embeddings": video_embeddings,
#             "attentions": all_attentions,
#         }


# # -----------------------------------------------------------
# # 5. Example: model construction & dummy training step
# # -----------------------------------------------------------

# if __name__ == "__main__":
#     # Device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Example dimensions
#     D_embed = 512   # frame embedding dim
#     D_blend = 30    # blendshape / AU dim
#     D_other = 10    # other features dim

#     # Construct model with concrete dimensions (example)
#     model = MultiPathMILRegressor(
#         embed_input_dim=D_embed,
#         blend_input_dim=D_blend,
#         other_input_dim=D_other,
#         embed_hidden_dim=128,
#         blend_hidden_dim=64,
#         other_hidden_dim=64,
#         embed_output_dim=128,
#         blend_output_dim=64,
#         other_output_dim=64,
#         fusion_hidden_dim=256,
#         window_output_dim=256,        # D_window
#         use_attention_pooling=True,
#         fusion_dropout=0.1,
#         mil_attn_hidden_dim=128,
#         reg_hidden_dim=128,
#         reg_dropout=0.1,
#     ).to(device)

#     # Dummy optimizer and loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.SmoothL1Loss()  # or nn.MSELoss()

#     # -------------------------------------------------------
#     # Create a dummy batch of videos with random data
#     # -------------------------------------------------------
#     B = 3          # number of videos in batch
#     T = 20         # frames per window (fixed within this example)

#     embed_batch: List[torch.Tensor] = []
#     blend_batch: List[torch.Tensor] = []
#     other_batch: List[torch.Tensor] = []

#     for b in range(B):
#         N_windows_b = torch.randint(low=3, high=8, size=(1,)).item()  # variable # of windows per video

#         embed_batch.append(
#             torch.randn(N_windows_b, T, D_embed, device=device)
#         )  # [N_windows_b, T, D_embed]
#         blend_batch.append(
#             torch.randn(N_windows_b, T, D_blend, device=device)
#         )  # [N_windows_b, T, D_blend]
#         other_batch.append(
#             torch.randn(N_windows_b, T, D_other, device=device)
#         )  # [N_windows_b, T, D_other]

#     # Dummy ground-truth fatigue scores for each video
#     targets = torch.randn(B, device=device)  # continuous labels

#     # -------------------------------------------------------
#     # One training step
#     # -------------------------------------------------------
#     model.train()
#     optimizer.zero_grad()

#     out = model(
#         embed_batch=embed_batch,
#         blend_batch=blend_batch,
#         other_batch=other_batch,
#     )

#     scores = out["scores"]  # [B]
#     attentions = out["attentions"]  # list of length B, each [N_windows_b]

#     loss = criterion(scores, targets)
#     loss.backward()
#     optimizer.step()

#     print("Dummy training step:")
#     print(f"  scores.shape         = {scores.shape}")             # [B]
#     print(f"  video_embeddings.shape = {out['video_embeddings'].shape}")  # [B, D_window]
#     print(f"  loss                 = {loss.item():.4f}")
#     for i, a in enumerate(attentions):
#         print(f"  video {i} attention shape: {a.shape}, sum={a.sum().item():.3f}")