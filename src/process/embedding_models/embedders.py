# embedders.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)
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
      - expose .embed(frame(s), strategy) -> np.ndarray or List[np.ndarray]
      - expose .embed_batch(frames, strategy) -> List[np.ndarray]
      - provide methods embed_cls, embed_mean, embed_cls_mean, embed_tokens, embed_tokens_cls
      - subclasses load their own processor/model and implement token extraction
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.processor = AutoImageProcessor.from_pretrained(cfg.repo_id, use_fast=True)
        load_kwargs: Dict = {}
        if cfg.fp16_if_gpu and torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            if cfg.attn_impl is not None:
                load_kwargs["attn_implementation"] = cfg.attn_impl
        self.valid_strategies = []
        self.model = self._load_model(cfg.repo_id, **load_kwargs)
        first_param_device = next(self.model.parameters()).device
        logging.error(f"[DEBUG] Model loaded on: {first_param_device}")
        self.model.eval()

    @abstractmethod
    def _load_model(self, repo_id: str, **kwargs) -> torch.nn.Module: ...

    # ---- public API ----
    def embed(
        self, frames: Union[np.ndarray, List[np.ndarray]], strategy: str
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Embed single frame or batch of frames.
        
        Args:
            frames: Single frame (H, W, C) or list of frames
            strategy: Embedding strategy (cls, mean, cls_mean, tokens, tokens_cls)
            
        Returns:
            Single embedding array if single frame input, list of embeddings if batch input
        """
        # Detect if input is a batch
        is_batch = isinstance(frames, list)
        
        if is_batch:
            return self.embed_batch(frames, strategy)
        else:
            fn = getattr(self, f"embed_{strategy.replace('+','_')}", None)
            if fn is None:
                raise ValueError(
                    f"Embedding strategy '{strategy}' not supported by {type(self).__name__}"
                )
            return fn(frames)

    def embed_batch(self, frames: List[np.ndarray], strategy: str) -> List[np.ndarray]:
        """
        Embed a batch of frames efficiently.
        
        Args:
            frames: List of frames (each H, W, C)
            strategy: Embedding strategy
            
        Returns:
            List of embedding arrays, one per frame
        """
        fn = getattr(self, f"embed_{strategy.replace('+','_')}_batch", None)
        if fn is None:
            raise ValueError(
                f"Embedding strategy '{strategy}' not supported by {type(self).__name__}"
            )
        return fn(frames)

    # ---- utilities ----
    def _finish(self, t: torch.Tensor, batch_mode: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Finalize embeddings with optional L2 normalization.
        
        Args:
            t: Tensor of shape [B, ...] 
            batch_mode: If True, return list of arrays; if False, squeeze batch dim
        """
        if batch_mode:
            # Process each item in batch separately
            x = t.float()
            if self.cfg.l2_normalize:
                # Normalize each sample in the batch
                norms = x.norm(p=2, dim=tuple(range(1, x.ndim)), keepdim=True)
                x = x / (norms + 1e-8)
            # Convert to list of numpy arrays
            return [arr for arr in x.detach().cpu().numpy()]
        else:
            # Original single-frame behavior
            x = t.squeeze(0).float()
            if self.cfg.l2_normalize:
                x = x / (x.norm(p=2) + 1e-8)
            return x.detach().cpu().numpy()

    def _prepare_inputs(
        self, frames: Union[np.ndarray, List[np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for single frame or batch of frames.
        
        Args:
            frames: Single frame (H, W, C) or list of frames
            
        Returns:
            Dictionary of tensors ready for model input
        """
        # Convert to PIL images
        if isinstance(frames, list):
            images = [Image.fromarray(frame) for frame in frames]
        else:
            images = Image.fromarray(frames)
            
        inputs = self.processor(images=images, return_tensors="pt")
        
        # Move to model device + dtype
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

    def __init__(self, cfg: EmbedderConfig):
        super().__init__(cfg)
        self.valid_strategies = ["cls", "mean", "cls_mean", "tokens", "tokens_cls"]

    def _load_model(self, repo_id: str, **kwargs):
        return AutoModel.from_pretrained(repo_id, **kwargs)

    # --- core tokenization (single frame) ---
    def _tokens(self, frame: np.ndarray):
        with torch.inference_mode():
            outputs = self.model(**self._prepare_inputs(frame))
        hs: torch.Tensor = outputs.last_hidden_state  # [B, N, D]
        B, N, D = hs.shape
        num_regs = int(getattr(self.model.config, "num_register_tokens", 0) or 0)

        cls_tok = hs[:, 0, :]  # [B, D]
        # exclude register tokens from "patch tokens"
        patches = hs[:, 1 : (N - num_regs) if num_regs else None, :]  # [B, Np, D]
        return cls_tok, patches, hs, num_regs

    # --- core tokenization (batch) ---
    def _tokens_batch(self, frames: List[np.ndarray]):
        with torch.inference_mode():
            outputs = self.model(**self._prepare_inputs(frames))
        hs: torch.Tensor = outputs.last_hidden_state  # [B, N, D]
        B, N, D = hs.shape
        num_regs = int(getattr(self.model.config, "num_register_tokens", 0) or 0)

        cls_tok = hs[:, 0, :]  # [B, D]
        # exclude register tokens from "patch tokens"
        patches = hs[:, 1 : (N - num_regs) if num_regs else None, :]  # [B, Np, D]
        return cls_tok, patches, hs, num_regs

    # --- single frame embedding variants ---
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

    # --- batch embedding variants ---
    def embed_cls_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        cls_tok, _, _, _ = self._tokens_batch(frames)
        return self._finish(cls_tok, batch_mode=True)

    def embed_mean_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        _, patches, _, _ = self._tokens_batch(frames)
        mean = patches.mean(dim=1)  # [B, D]
        return self._finish(mean, batch_mode=True)

    def embed_cls_mean_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        cls_tok, patches, _, _ = self._tokens_batch(frames)
        mean = patches.mean(dim=1)
        cls_mean = torch.cat([cls_tok, mean], dim=-1)  # [B, 2D]
        return self._finish(cls_mean, batch_mode=True)

    def embed_tokens_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        _, patches, _, _ = self._tokens_batch(frames)
        B = patches.shape[0]
        flat = patches.reshape(B, -1)  # [B, Np*D]
        return self._finish(flat, batch_mode=True)

    def embed_tokens_cls_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        cls_tok, patches, _, _ = self._tokens_batch(frames)
        B = patches.shape[0]
        # concat CLS + patches (still excluding registers)
        full = torch.cat([cls_tok.unsqueeze(1), patches], dim=1).reshape(B, -1)
        return self._finish(full, batch_mode=True)


# ---------- factory/registry ----------

# Simple heuristics: add patterns -> class
_REGISTRY = [
    # (predicate, class)
    (
        lambda repo: "dinov2" in repo or "vit" in repo or "deit" in repo,
        ViTBackboneEmbedder,
    ),
    # Future: ("openai/clip-vit-", CLIPImageEmbedder), ("facenet", FaceNetEmbedder), etc.
]


def make_embedder(repo_id: str, **kwargs) -> BaseEmbedder:
    for pred, cls in _REGISTRY:
        if pred(repo_id):
            return cls(EmbedderConfig(repo_id=repo_id, **kwargs))
    raise ValueError(f"No embedder registered for repo_id='{repo_id}'")