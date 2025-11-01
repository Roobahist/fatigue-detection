# embedders.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ---------- configuration ----------

@dataclass
class EmbedderConfig:
    repo_id: str
    l2_normalize: bool = True
    fp16_if_gpu: bool = True
    attn_impl: Optional[str] = "sdpa"  # None disables passing this kwarg

# ---------- base interface ----------

class BaseEmbedder(ABC):
    """
    Contract:
      - expose .embed(frame, kind) -> np.ndarray
      - provide methods embed_cls, embed_mean, embed_cls_mean, embed_tokens, embed_tokens_cls
      - subclasses load their own processor/model and implement token extraction
    """
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.processor = AutoImageProcessor.from_pretrained(cfg.repo_id)
        load_kwargs: Dict = {}
        if cfg.fp16_if_gpu and torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            if cfg.attn_impl is not None:
                load_kwargs["attn_implementation"] = cfg.attn_impl
        self.model = self._load_model(cfg.repo_id, **load_kwargs)
        self.model.eval()

    @abstractmethod
    def _load_model(self, repo_id: str, **kwargs) -> torch.nn.Module:
        ...

    # ---- public API ----
    def embed(self, frame: np.ndarray, kind: str) -> np.ndarray:
        fn = getattr(self, f"embed_{kind.replace('+','_')}", None)
        if fn is None:
            raise ValueError(f"Embedding kind '{kind}' not supported by {type(self).__name__}")
        return fn(frame)

    # ---- utilities ----
    def _finish(self, t: torch.Tensor) -> np.ndarray:
        x = t.squeeze(0).float()
        if self.cfg.l2_normalize:
            x = x / (x.norm(p=2) + 1e-8)
        return x.detach().cpu().numpy()

    def _prepare_inputs(self, frame: np.ndarray) -> Dict[str, torch.Tensor]:
        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt")
        # move to model device + dtype
        model_dtype = getattr(self.model, "dtype", None)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device, dtype=(model_dtype or v.dtype))
        return inputs

# ---------- ViT-family backbone (covers DINOv2, DeiT, ViT) ----------

class ViTBackboneEmbedder(BaseEmbedder):
    """
    Assumes token layout: [CLS] [patch tokens ...] [register tokens (optional)]
    DINOv2 exposes config.num_register_tokens > 0; others typically 0.
    """
    def _load_model(self, repo_id: str, **kwargs):
        return AutoModel.from_pretrained(repo_id, **kwargs)

    # --- core tokenization ---
    def _tokens(self, frame: np.ndarray):
        with torch.inference_mode():
            outputs = self.model(**self._prepare_inputs(frame))
        hs: torch.Tensor = outputs.last_hidden_state  # [B, N, D]
        B, N, D = hs.shape
        num_regs = int(getattr(self.model.config, "num_register_tokens", 0) or 0)

        cls_tok = hs[:, 0, :]  # [B, D]
        # exclude register tokens from "patch tokens"
        patches = hs[:, 1:(N - num_regs) if num_regs else None, :]  # [B, Np, D]
        return cls_tok, patches, hs, num_regs

    # --- embedding variants (public) ---
    def embed_cls(self, frame: np.ndarray) -> np.ndarray:
        cls_tok, _, _, _ = self._tokens(frame)
        return self._finish(cls_tok)

    def embed_mean(self, frame: np.ndarray) -> np.ndarray:
        _, patches, _, _ = self._tokens(frame)
        mean = patches.mean(dim=1)  # [B, D]
        return self._finish(mean)

    def embed_cls_mean(self, frame: np.ndarray) -> np.ndarray:
        cls_tok, patches, _, _ = self._tokens(frame)
        mean = patches.mean(dim=1)
        cls_mean = torch.cat([cls_tok, mean], dim=-1)  # [B, 2D]
        return self._finish(cls_mean)

    def embed_tokens(self, frame: np.ndarray) -> np.ndarray:
        _, patches, _, _ = self._tokens(frame)
        flat = patches.reshape(1, -1)  # [B, Np*D]
        return self._finish(flat)

    def embed_tokens_cls(self, frame: np.ndarray) -> np.ndarray:
        cls_tok, patches, _, _ = self._tokens(frame)
        # concat CLS + patches (still excluding registers)
        full = torch.cat([cls_tok.unsqueeze(1), patches], dim=1).reshape(1, -1)
        return self._finish(full)

# ---------- factory/registry ----------

# Simple heuristics: add patterns -> class
_REGISTRY = [
    # (predicate, class)
    (lambda repo: "dinov2" in repo or "vit" in repo or "deit" in repo, ViTBackboneEmbedder),
    # Future: ("openai/clip-vit-", CLIPImageEmbedder), ("facenet", FaceNetEmbedder), etc.
]

def make_embedder(repo_id: str, **kwargs) -> BaseEmbedder:
    for pred, cls in _REGISTRY:
        if pred(repo_id):
            return cls(EmbedderConfig(repo_id=repo_id, **kwargs))
    raise ValueError(f"No embedder registered for repo_id='{repo_id}'")