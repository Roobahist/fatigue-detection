import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from Zh.experiments.e11.dataclasses import *
from Zh.experiments.e11.helpers import *

# ==============================
# Positional Encoding for Transformer
# ==============================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
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
                layers.append(
                    nn.Sequential(conv, self.activation, nn.Dropout(cfg.dropout))
                )
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
            x_in = x_in.unsqueeze(1)  # [T, 1, D_model]
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

        concat_dim = cfg.d_temporal_embed + cfg.d_temporal_blend + cfg.d_temporal_other
        self.mlp = MLP(concat_dim, cfg.d_window, cfg.window_mlp)

    def forward(
        self,
        embed_seq: torch.Tensor,
        blend_seq: torch.Tensor,
        other_seq: torch.Tensor,
    ) -> torch.Tensor:
        h_embed = self.embed_encoder(embed_seq)  # [d_temporal_embed]
        h_blend = self.blend_encoder(blend_seq)  # [d_temporal_blend]
        h_other = self.other_encoder(other_seq)  # [d_temporal_other]
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
        a = torch.softmax(logits, dim=0)  # [N]
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
        self.mil_pool = AttentionMILPooling(
            cfg.d_window, cfg.attn_hidden_size, cfg.attn_activation
        )
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

            assert (
                len(emb_list) == len(blend_list) == len(other_list)
            ), "Window lists must have same length"
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
            z, _ = self.mil_pool(H)  # [d_window]
            video_embeddings.append(z)

        video_embeddings = torch.stack(video_embeddings, dim=0)  # [B, d_window]
        preds = self.reg_head(video_embeddings).squeeze(-1)  # [B]
        return preds
