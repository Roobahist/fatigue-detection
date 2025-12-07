import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Zh.experiments.e14.dataclasses import *
from Zh.experiments.e14.helpers import get_target_from_video_id, compute_training_picks
from Zh.src.utils.eventfulness import pick_eventful_segments
from Zh.src.utils.memmap_handler import MemmapHandler


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


class RealVideoDataset(Dataset):
    """
    Dataset scaffold for your real data stored in memmap files.

    Uses eventfulness scores to pick windows via `pick_eventful_segments`.

    mode:
      - "train": use epoch-dependent schedule (explore -> exploit)
      - "val" / "test": always use top-mean windows (pure exploitation)
    """

    def __init__(
        self,
        video_ids: List[str],
        data_cfg: DataConfig,
        selection_cfg: SelectionConfig,
        mode: str = "train",
    ):
        super().__init__()
        assert mode in ("train", "val", "test")
        self.video_ids = video_ids
        self.embeddings_dir = data_cfg.embeddings_dir
        self.features_dir = data_cfg.features_dir
        self.eventfulness_scores_dir = data_cfg.eventfulness_scores_dir
        self.dataset_name = data_cfg.dataset_name
        self.selection_cfg = selection_cfg
        self.converge_ratio = selection_cfg.converge_ratio
        self.mode = mode

        # These will be updated per epoch for training
        self._current_pick_mean = selection_cfg.pick_mean_top_begin
        self._current_pick_max = selection_cfg.pick_max_top_begin
        self._current_pick_random = selection_cfg.pick_random_begin

    def set_training_epoch(self, epoch: int, num_epochs: int) -> None:
        """
        Called once per epoch from the training loop to update
        (pick_mean_top, pick_max_top, pick_random) for training.
        """
        if self.mode != "train":
            return  # no-op for val/test
        m, x, r = compute_training_picks(
            self.selection_cfg, epoch, num_epochs, self.converge_ratio
        )

        self._current_pick_mean = m
        self._current_pick_max = x
        self._current_pick_random = r
        assert m + x + r == self.selection_cfg.n_windows

    def __len__(self) -> int:
        return len(self.video_ids)

    def _get_eventfulness_path(self, video_id: str) -> str:
        eventfulness_basename = (
            os.path.basename(video_id).split("__")[0] + "__features__eventfulness.dat"
        )
        return os.path.join(self.eventfulness_scores_dir, eventfulness_basename)

    def __getitem__(self, idx: int) -> Dict:
        video_id = self.video_ids[idx]

        # ---- 1) Decide pick counts depending on mode ----
        if self.mode == "train":
            pick_mean = self._current_pick_mean
            pick_max = self._current_pick_max
            pick_random = self._current_pick_random
        else:
            # val/test: only exploit – all windows via mean-top
            pick_mean = self.selection_cfg.n_windows
            pick_max = 0
            pick_random = 0

        # ---- 2) Use eventfulness scores to pick window indices ----
        eventfulness_path = self._get_eventfulness_path(video_id)

        # Returns a DataFrame with start/end and a list of chosen row indices
        eventfulness_df, picked_indices = pick_eventful_segments(
            eventfulness_scores_path=eventfulness_path,
            window_size=self.selection_cfg.window_size,
            stride=self.selection_cfg.stride,
            pick_mean_top=pick_mean,
            pick_max_top=pick_max,
            pick_random=pick_random,
        )

        # Each picked index corresponds to one window: we have start_frame/end_frame
        # for that window in eventfulness_df.
        # You now map those frame ranges into your memmaps for embeddings/blend/other.

        embeddings_list: List[torch.Tensor] = []
        blend_list: List[torch.Tensor] = []
        other_list: List[torch.Tensor] = []

        # ---- 3) PSEUDOCODE: read from memmaps using picked windows ----
        # Replace this with your actual memmap logic; here I keep it explicit
        # but non-functional so you can drop in your own handler.

        embed_memmap = MemmapHandler(path=video_id, mode="r")
        blend_other_memmap_basename = (
            os.path.basename(video_id).split("__")[0] + "__features.dat"
        )
        blend_other_memmap = MemmapHandler(
            path=os.path.join(self.features_dir, blend_other_memmap_basename), mode="r"
        )

        for idx_row in picked_indices:
            row = eventfulness_df.iloc[idx_row]
            start = int(row["start_frame"])
            end = int(row["end_frame"])

            # Replace these three with real slices from your memmaps:
            emb_window_np = embed_memmap.read(slice=(slice(start, end), slice(None)))
            blend_window_np = blend_other_memmap.read(
                slice=(slice(start, end), slice(7, None))
            )
            other_window_np = blend_other_memmap.read(
                slice=(slice(start, end), slice(1, 5))
            )

            # For now, placeholder random tensors with correct shapes so the
            # rest of the pipeline is runnable until you hook your memmaps:
            embeddings_list.append(torch.tensor(emb_window_np, dtype=torch.float32))
            blend_list.append(torch.tensor(blend_window_np, dtype=torch.float32))
            other_list.append(torch.tensor(other_window_np, dtype=torch.float32))

        # ---- 4) Load scalar target label for this video_id ----
        # Placeholder; you will replace this with your real label loader.
        # target = load_fatigue_label(video_id)
        target = get_target_from_video_id(video_id, dataset_name=self.dataset_name)

        return {
            "embeddings": embeddings_list,
            "blendshapes": blend_list,
            "others": other_list,
            "target": float(target),
        }
