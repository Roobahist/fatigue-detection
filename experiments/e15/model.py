# Zh/experiments/e15/model.py

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from Zh.experiments.e15.dataclasses import *
from Zh.experiments.e15.helpers import *


def _safe_str(x: Any, max_len: int = 220) -> str:
    s = str(x)
    return s if len(s) <= max_len else (s[:max_len] + "...")


def _fmt_kv(meta: Dict[str, Any]) -> str:
    if not meta:
        return "(no meta)"
    parts = []
    for k in sorted(meta.keys()):
        v = meta[k]
        if v is None:
            continue
        parts.append(f"{k}={_safe_str(v)}")
    return " ".join(parts) if parts else "(no meta)"


def _seq_stats(name: str, x: Any) -> str:
    if not torch.is_tensor(x):
        return f"{name}: type={type(x)}"
    shape = tuple(x.shape)
    if x.numel() == 0:
        return f"{name}: shape={shape} numel=0"
    with torch.no_grad():
        finite = bool(torch.isfinite(x).all().item())
        minv = float(x.min().item())
        maxv = float(x.max().item())
    return f"{name}: shape={shape} finite={finite} min={minv:.4g} max={maxv:.4g}"


def _log_block(title: str, lines: List[str]) -> None:
    border = "=" * 88
    print("\n" + border)
    print(title)
    for ln in lines:
        print(ln)
    print(border + "\n")


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
        self.dropout = nn.Dropout(float(cfg.dropout))

        if self.backbone_type in ("gru", "lstm"):
            self.rnn_type = self.backbone_type
            self.hidden_size = int(cfg.hidden_size)
            self.num_layers = int(cfg.num_layers)
            self.bidirectional = True
            rnn_dropout = float(cfg.dropout) if cfg.num_layers > 1 else 0.0
            if self.rnn_type == "gru":
                self.rnn = nn.GRU(
                    input_size=int(d_in),
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=rnn_dropout,
                    bidirectional=self.bidirectional,
                )
            else:
                self.rnn = nn.LSTM(
                    input_size=int(d_in),
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=rnn_dropout,
                    bidirectional=self.bidirectional,
                )
            backbone_out_dim = self.hidden_size * 2

        elif self.backbone_type == "tcn":
            self.hidden_size = int(cfg.hidden_size)
            self.num_layers = max(int(cfg.num_layers), 1)
            layers = []
            in_channels = int(d_in)
            for _ in range(self.num_layers):
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.hidden_size,
                    kernel_size=3,
                    padding=1,
                )
                layers.append(
                    nn.Sequential(conv, self.activation, nn.Dropout(float(cfg.dropout)))
                )
                in_channels = self.hidden_size
            self.tcn = nn.ModuleList(layers)
            backbone_out_dim = self.hidden_size

        elif self.backbone_type == "transformer":
            d_model = int(cfg.hidden_size)
            self.input_proj = nn.Linear(int(d_in), d_model)
            nhead = 4 if d_model % 4 == 0 else 1

            activation_arg = get_activation(cfg.activation)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=float(cfg.dropout),
                batch_first=False,
                activation=activation_arg,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(cfg.num_layers),
            )
            self.pos_encoder = PositionalEncoding(d_model)
            backbone_out_dim = d_model

        else:
            raise ValueError(f"Unsupported temporal backbone: {self.backbone_type}")

        self.out_proj = nn.Linear(int(backbone_out_dim), int(d_out))

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
            x_in = x.unsqueeze(1)  # [T, 1, D_in]
            outputs, _ = self.rnn(x_in)  # [T, 1, H*2]
            pooled = outputs.mean(dim=0).squeeze(0)  # [H*2]

        elif self.backbone_type == "tcn":
            x_in = x.transpose(0, 1).unsqueeze(0)  # [1, D_in, T]
            for layer in self.tcn:
                x_in = layer(x_in)  # [1, hidden_size, T]
            pooled = x_in.mean(dim=2).squeeze(0)  # [hidden_size]

        elif self.backbone_type == "transformer":
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
        current_dim = int(in_dim)

        if cfg.num_layers <= 1:
            layers.append(nn.Linear(current_dim, int(out_dim)))
        else:
            for _ in range(cfg.num_layers - 1):
                layers.append(nn.Linear(current_dim, int(cfg.hidden_size)))
                layers.append(activation)
                layers.append(nn.Dropout(float(cfg.dropout)))
                current_dim = int(cfg.hidden_size)
            layers.append(nn.Linear(current_dim, int(out_dim)))

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

        self.embed_encoder = WindowTemporalEncoder(
            int(cfg.d_embed), int(cfg.d_temporal_embed), cfg.temporal
        )
        self.blend_encoder = WindowTemporalEncoder(
            int(cfg.d_blend), int(cfg.d_temporal_blend), cfg.temporal
        )
        self.other_encoder = WindowTemporalEncoder(
            int(cfg.d_other), int(cfg.d_temporal_other), cfg.temporal
        )

        concat_dim = (
            int(cfg.d_temporal_embed)
            + int(cfg.d_temporal_blend)
            + int(cfg.d_temporal_other)
        )
        self.mlp = MLP(concat_dim, int(cfg.d_window), cfg.window_mlp)

    def forward(
        self,
        embed_seq: torch.Tensor,
        blend_seq: torch.Tensor,
        other_seq: torch.Tensor,
        *,
        meta: Optional[Dict[str, Any]] = None,
        window_index: Optional[int] = None,
    ) -> torch.Tensor:
        # Detect empty sequences early and fail with identifiers
        t_e = int(embed_seq.shape[0]) if torch.is_tensor(embed_seq) and embed_seq.ndim >= 1 else None
        t_b = int(blend_seq.shape[0]) if torch.is_tensor(blend_seq) and blend_seq.ndim >= 1 else None
        t_o = int(other_seq.shape[0]) if torch.is_tensor(other_seq) and other_seq.ndim >= 1 else None

        if (t_e == 0) or (t_b == 0) or (t_o == 0):
            meta_str = _fmt_kv(meta or {})
            wi = window_index if window_index is not None else "?"
            _log_block(
                "EMPTY WINDOW SEQUENCE DETECTED in MultiPathWindowEncoder",
                [
                    f"where: {meta_str} window_index={wi}",
                    _seq_stats("embed_seq", embed_seq),
                    _seq_stats("blend_seq", blend_seq),
                    _seq_stats("other_seq", other_seq),
                    f"note: error will be raised to stop the run and surface identifiers",
                ],
            )
            raise ValueError(
                "Empty temporal sequence (T==0). "
                f"{meta_str} window_index={wi} "
                f"embed_T={t_e} blend_T={t_b} other_T={t_o}"
            )

        h_embed = self.embed_encoder(embed_seq)
        h_blend = self.blend_encoder(blend_seq)
        h_other = self.other_encoder(other_seq)
        h = torch.cat([h_embed, h_blend, h_other], dim=-1)
        out = self.mlp(h)
        return out


class AttentionMILPooling(nn.Module):
    """
    Gated Attention MIL Pooling from Ilse et al. 2018
    "Attention-based Deep Multiple Instance Learning"
    """

    def __init__(self, d_window: int, attn_hidden_size: int, attn_activation: str):
        super().__init__()
        self.attention_V = nn.Linear(int(d_window), int(attn_hidden_size))
        self.attention_U = nn.Linear(int(d_window), int(attn_hidden_size))
        self.attention_w = nn.Linear(int(attn_hidden_size), 1, bias=False)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if H.ndim != 2:
            raise ValueError(f"Expected [N_windows, D_window], got {H.shape}")
        if H.size(0) == 0:
            raise ValueError("N_windows must be > 0")

        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = A_V * A_U
        A = self.attention_w(A)
        a = torch.softmax(A, dim=0)

        z = torch.sum(a * H, dim=0)
        return z, a.squeeze(-1)


class MultiPathMILRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        self.task_type = data_cfg.task_type
        self.num_classes = data_cfg.num_classes
        self.loss_name = train_cfg.loss.lower()

        self.window_encoder = MultiPathWindowEncoder(cfg)
        self.mil_pool = AttentionMILPooling(
            cfg.d_window, cfg.attn_hidden_size, cfg.attn_activation
        )

        if self.task_type == "classification":
            if self.num_classes is None:
                raise ValueError("num_classes must be specified in DataConfig for classification task")
            if self.loss_name == "coral":
                from Zh.src.utils.losses import CoralLayer
                self.head = CoralLayer(size_in=cfg.d_window, num_classes=self.num_classes, preinit_bias=True)
            elif self.loss_name == "corn":
                self.head = MLP(cfg.d_window, self.num_classes - 1, cfg.reg_head_mlp)
            else:
                self.head = MLP(cfg.d_window, self.num_classes, cfg.reg_head_mlp)
        else:
            self.head = MLP(cfg.d_window, 1, cfg.reg_head_mlp)

    def forward(self, batch: List[Dict]) -> torch.Tensor:
        video_embeddings = []

        for video_i, video in enumerate(batch):
            emb_list: List[torch.Tensor] = video["embeddings"]
            blend_list: List[torch.Tensor] = video["blendshapes"]
            other_list: List[torch.Tensor] = video["others"]

            if not (len(emb_list) == len(blend_list) == len(other_list)):
                _log_block(
                    "MISMATCHED WINDOW LIST LENGTHS in MultiPathMILRegressor",
                    [
                        f"video_index_in_batch={video_i}",
                        f"len(embeddings)={len(emb_list)} len(blendshapes)={len(blend_list)} len(others)={len(other_list)}",
                        f"meta: {_fmt_kv(video.get('meta', {}))}",
                    ],
                )
                raise ValueError(
                    "Window lists must have same length. "
                    f"video_index_in_batch={video_i} "
                    f"len_emb={len(emb_list)} len_blend={len(blend_list)} len_other={len(other_list)} "
                    f"meta={_fmt_kv(video.get('meta', {}))}"
                )

            n_windows = len(emb_list)
            if n_windows == 0:
                _log_block(
                    "NO WINDOWS FOUND for video in MultiPathMILRegressor",
                    [
                        f"video_index_in_batch={video_i}",
                        f"meta: {_fmt_kv(video.get('meta', {}))}",
                    ],
                )
                raise ValueError(
                    f"Each video must have at least one window. video_index_in_batch={video_i} meta={_fmt_kv(video.get('meta', {}))}"
                )

            base_meta = dict(video.get("meta", {}))
            base_meta.update({"video_index_in_batch": video_i})

            windows_meta = video.get("windows_meta", None)
            window_embs = []

            for w_i in range(n_windows):
                e = emb_list[w_i]
                b = blend_list[w_i]
                o = other_list[w_i]

                meta = dict(base_meta)
                if isinstance(windows_meta, list) and w_i < len(windows_meta) and isinstance(windows_meta[w_i], dict):
                    meta.update(windows_meta[w_i])

                window_emb = self.window_encoder(e, b, o, meta=meta, window_index=w_i)
                window_embs.append(window_emb)

            H = torch.stack(window_embs, dim=0)
            z, _ = self.mil_pool(H)
            video_embeddings.append(z)

        video_embeddings = torch.stack(video_embeddings, dim=0)
        outputs = self.head(video_embeddings)

        if self.task_type == "regression":
            return outputs.squeeze(-1)
        return outputs