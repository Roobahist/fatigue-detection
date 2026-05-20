# Zh/experiments/e15/datasets.py
# (or wherever your dataset lives)

import os
import random
import hashlib
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from Zh.experiments.e15.dataclasses import *
from Zh.src.utils.eventfulness import pick_eventful_segments
from Zh.src.utils.memmap_handler import MemmapHandler


def get_target_from_video_id(video_id: str, dataset_name: str) -> float:
    if dataset_name == "UTA":
        return float(os.path.basename(video_id).split("_")[2])
    elif dataset_name == "DROZY":
        subject_id = int(os.path.basename(video_id).split("_")[0].split("-")[0])
        session_id = int(os.path.basename(video_id).split("_")[0].split("-")[1])
        kss_score_path = "/work/long_lab/Mojtaba/data/DROZY/data/original/KSS.txt"
        with open(kss_score_path, "r") as f:
            kss_scores = list(
                map(lambda x: list(map(int, x.strip().split(" "))), f.readlines())
            )
        return float(kss_scores[subject_id - 1][session_id - 1])
    else:
        raise ValueError(f"Unknown dataset for target extraction: {dataset_name}")


def _stable_int_hash(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)

def compute_training_picks(
    sel_cfg: SelectionConfig,
    epoch: int,
    num_epochs: int,
    converge_ratio: float = 1.0,
) -> Tuple[int, int, int]:
    """
    Linearly interpolate (mean, max, random) from *_begin to *_end over epochs.

    Parameters
    ----------
    sel_cfg : SelectionConfig
        Holds pick_*_begin, pick_*_end, and n_windows.
    epoch : int
        0-based epoch index.
    num_epochs : int
        Total number of epochs.
    converge_ratio : float, default 1.0
        Fraction of training over which to converge from *_begin to *_end.
        - 1.0  → converge over the full training (original behavior).
        - 0.5  → converge by halfway, then stay at *_end.
        - 0.3  → converge by 30% of the epochs, then stay at *_end.
        - <= 0 → snap to *_end from the start.

    Ensures the three counts sum to n_windows (with minor rounding fixes).
    """
    if num_epochs <= 1:
        # Degenerate case: just use *_end
        mean_f = sel_cfg.pick_mean_top_end
        max_f = sel_cfg.pick_max_top_end
        rand_f = sel_cfg.pick_random_end
    else:
        # epoch is 0-based inside train loop, normalize to [0, 1]
        denom = num_epochs - 1
        raw_t = epoch / denom  # in [0, 1]

        if converge_ratio <= 0:
            # Immediately at *_end
            t = 1.0
        elif converge_ratio >= 1.0:
            # Original behavior: spread over full training
            t = raw_t
        else:
            # Stretch the [0, converge_ratio] part to [0, 1], then clamp
            t = raw_t / converge_ratio
            if t > 1.0:
                t = 1.0

        mean_f = sel_cfg.pick_mean_top_begin + t * (
            sel_cfg.pick_mean_top_end - sel_cfg.pick_mean_top_begin
        )
        max_f = sel_cfg.pick_max_top_begin + t * (
            sel_cfg.pick_max_top_end - sel_cfg.pick_max_top_begin
        )
        rand_f = sel_cfg.pick_random_begin + t * (
            sel_cfg.pick_random_end - sel_cfg.pick_random_begin
        )

    mean_i = int(round(mean_f))
    max_i = int(round(max_f))
    rand_i = int(round(rand_f))

    # Fix sum to be exactly n_windows
    total = mean_i + max_i + rand_i
    diff = sel_cfg.n_windows - total

    # Adjust random first (since it's the least “structural”)
    rand_i += diff
    if rand_i < 0:
        # Clamp random at 0, push remaining deficit into mean
        deficit = -rand_i
        rand_i = 0
        mean_i = max(0, mean_i - deficit)

    # Final safety clamp
    if mean_i < 0:
        mean_i = 0
    if max_i < 0:
        max_i = 0

    # If we still don’t sum correctly, force n_windows by tweaking mean
    total = mean_i + max_i + rand_i
    if total != sel_cfg.n_windows:
        mean_i = max(0, sel_cfg.n_windows - max_i - rand_i)

    assert mean_i + max_i + rand_i == sel_cfg.n_windows

    return mean_i, max_i, rand_i


def _safe_str(x: Any, max_len: int = 220) -> str:
    s = str(x)
    return s if len(s) <= max_len else (s[:max_len] + "...")


def _log_block(title: str, lines: List[str]) -> None:
    border = "=" * 88
    print("\n" + border)
    print(title)
    for ln in lines:
        print(ln)
    print(border + "\n")


def _shape(x: Any) -> str:
    try:
        return str(tuple(x.shape))
    except Exception:
        return "?"


class DummyVideoDataset(Dataset):
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

        target = random.random()
        return {
            "embeddings": embeddings_list,
            "blendshapes": blend_list,
            "others": other_list,
            "target": float(target),
            "meta": {"dataset": "dummy", "idx": idx},
            "windows_meta": [{"window_index": i} for i in range(n_windows)],
        }


class RealVideoDataset(Dataset):
    def __init__(
        self,
        video_ids: List[str],
        data_cfg: DataConfig,
        selection_cfg: SelectionConfig,
        mode: str = "train",
        base_seed: int = 0,
        outer_idx: int = 0,
        config_idx: int = 0,
        inner_idx: int = 0,
    ):
        super().__init__()
        assert mode in ("train", "val", "test")

        self.video_ids = video_ids
        self.data_cfg = data_cfg
        self.embeddings_dir = data_cfg.embeddings_dir
        self.features_dir = data_cfg.features_dir
        self.eventfulness_scores_dir = data_cfg.eventfulness_scores_dir
        self.dataset_name = data_cfg.dataset_name
        self.selection_cfg = selection_cfg
        self.converge_ratio = selection_cfg.converge_ratio
        self.mode = mode
        self.base_seed = int(base_seed)
        self.outer_idx = int(outer_idx)
        self.config_idx = int(config_idx)
        self.inner_idx = int(inner_idx)
        self._epoch = 0

        self._current_pick_mean = selection_cfg.pick_mean_top_begin
        self._current_pick_max = selection_cfg.pick_max_top_begin
        self._current_pick_random = selection_cfg.pick_random_begin

        self.remove_augmented_samples()

    def remove_augmented_samples(self) -> None:
        if self.mode != "train":
            filtered_video_ids = [
                vid for vid in self.video_ids if "__base__" in vid
            ]
            self.video_ids = filtered_video_ids

    def set_training_epoch(self, epoch: int, num_epochs: int) -> None:
        if self.mode != "train":
            return

        self._epoch = int(epoch)
        m, x, r = compute_training_picks(
            self.selection_cfg, epoch, num_epochs, self.converge_ratio
        )
        self._current_pick_mean = m
        self._current_pick_max = x
        self._current_pick_random = r
        assert m + x + r == self.selection_cfg.n_windows

    def _make_rng_for_video(self, video_id: str) -> np.random.Generator:
        vid_h = _stable_int_hash(video_id)
        seed = (
            self.base_seed
            + 1000003 * self._epoch
            + 1009 * self.outer_idx
            + 9176 * self.inner_idx
            + 313 * self.config_idx
            + (vid_h % 2_000_000_000)
        ) % 2_000_000_000
        return np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.video_ids)

    def _get_eventfulness_path(self, video_id: str) -> str:
        eventfulness_basename = (
            os.path.basename(video_id).split("__")[0] + "__features__eventfulness.dat"
        )
        return os.path.join(self.eventfulness_scores_dir, eventfulness_basename)

    def _features_memmap_path(self, video_id: str) -> str:
        blend_other_memmap_basename = (
            os.path.basename(video_id).split("__")[0] + "__features.dat"
        )
        return os.path.join(self.features_dir, blend_other_memmap_basename)

    def __getitem__(self, idx: int) -> Dict:
        video_id = self.video_ids[idx]

        if self.mode == "train":
            pick_mean = self._current_pick_mean
            pick_max = self._current_pick_max
            pick_random = self._current_pick_random
        else:
            pick_mean = self.selection_cfg.n_windows
            pick_max = 0
            pick_random = 0

        eventfulness_path = self._get_eventfulness_path(video_id)
        rng = self._make_rng_for_video(video_id)
        
        eventfulness_df, picked_indices = pick_eventful_segments(
            eventfulness_scores_path=eventfulness_path,
            window_size=self.selection_cfg.window_size,
            stride=self.selection_cfg.stride,
            pick_mean_top=pick_mean,
            pick_max_top=pick_max,
            pick_random=pick_random,
            rng=rng,
        )

        # If no windows picked, fail with identifiers immediately
        if picked_indices is None or len(picked_indices) == 0:
            _log_block(
                "NO WINDOWS PICKED in RealVideoDataset",
                [
                    f"idx={idx} video_id={_safe_str(video_id)} mode={self.mode}",
                    f"outer_idx={self.outer_idx} inner_idx={self.inner_idx} config_idx={self.config_idx} epoch={self._epoch}",
                    f"eventfulness_path={_safe_str(eventfulness_path)}",
                    f"pick_mean={pick_mean} pick_max={pick_max} pick_random={pick_random}",
                    f"eventfulness_df_rows={len(eventfulness_df)}",
                    "This usually means video shorter than window_size, or eventfulness file is empty/invalid.",
                ],
            )
            raise ValueError(
                "No windows picked for sample. "
                f"video_id={_safe_str(video_id)} eventfulness_path={_safe_str(eventfulness_path)} "
                f"mode={self.mode} outer={self.outer_idx} inner={self.inner_idx} cfg={self.config_idx} epoch={self._epoch}"
            )

        embed_memmap_path = video_id
        features_memmap_path = self._features_memmap_path(video_id)

        # Open memmaps
        embed_memmap = MemmapHandler(path=embed_memmap_path, mode="r")
        blend_other_memmap = MemmapHandler(path=features_memmap_path, mode="r")

        embeddings_list: List[torch.Tensor] = []
        blend_list: List[torch.Tensor] = []
        other_list: List[torch.Tensor] = []
        windows_meta: List[Dict[str, Any]] = []

        for w_i, idx_row in enumerate(picked_indices):
            row = eventfulness_df.iloc[int(idx_row)]
            start = int(row["start_frame"])
            end = int(row["end_frame"])

            # Basic sanity checks
            if end <= start:
                _log_block(
                    "INVALID WINDOW RANGE in RealVideoDataset",
                    [
                        f"idx={idx} video_id={_safe_str(video_id)} window_index={w_i}",
                        f"picked_row_index={idx_row} start={start} end={end}",
                        f"eventfulness_path={_safe_str(eventfulness_path)}",
                    ],
                )
                raise ValueError(
                    f"Invalid window range start={start} end={end} "
                    f"video_id={_safe_str(video_id)} row={idx_row} eventfulness_path={_safe_str(eventfulness_path)}"
                )

            emb_window_np = embed_memmap.read(slice=(slice(start, end), slice(0, 768)))
            blend_window_np = blend_other_memmap.read(slice=(slice(start, end), slice(7, 59)))
            other_window_np = blend_other_memmap.read(slice=(slice(start, end), slice(1, 5)))

            # Detect empty reads early and emit identifiers that point to the file
            if emb_window_np is None or getattr(emb_window_np, "shape", (0,))[0] == 0:
                _log_block(
                    "EMPTY EMBEDDING SLICE READ in RealVideoDataset",
                    [
                        f"idx={idx} video_id={_safe_str(video_id)} window_index={w_i}",
                        f"embed_memmap_path={_safe_str(embed_memmap_path)} embed_memmap_shape={_safe_str(getattr(embed_memmap, 'shape', '?'))}",
                        f"features_memmap_path={_safe_str(features_memmap_path)} features_memmap_shape={_safe_str(getattr(blend_other_memmap, 'shape', '?'))}",
                        f"start={start} end={end} (len={end-start}) picked_row_index={idx_row}",
                        f"eventfulness_path={_safe_str(eventfulness_path)}",
                        f"picked_indices={picked_indices}",
                    ],
                )
                raise ValueError(
                    "Empty embedding slice read. "
                    f"video_id={_safe_str(video_id)} embed_memmap_path={_safe_str(embed_memmap_path)} "
                    f"start={start} end={end} row={idx_row} eventfulness_path={_safe_str(eventfulness_path)}"
                )

            # Convert to tensors
            e_t = torch.tensor(emb_window_np, dtype=torch.float32)
            b_t = torch.tensor(blend_window_np, dtype=torch.float32)
            o_t = torch.tensor(other_window_np, dtype=torch.float32)

            # Log if any modality is empty or has unexpected time length
            if e_t.shape[0] == 0 or b_t.shape[0] == 0 or o_t.shape[0] == 0:
                _log_block(
                    "EMPTY MODALITY WINDOW after tensor conversion",
                    [
                        f"idx={idx} video_id={_safe_str(video_id)} window_index={w_i}",
                        f"start={start} end={end} picked_row_index={idx_row}",
                        f"embed_shape={tuple(e_t.shape)} blend_shape={tuple(b_t.shape)} other_shape={tuple(o_t.shape)}",
                        f"embed_memmap_path={_safe_str(embed_memmap_path)} features_memmap_path={_safe_str(features_memmap_path)}",
                        f"eventfulness_path={_safe_str(eventfulness_path)}",
                    ],
                )
                raise ValueError(
                    "Empty window tensor found. "
                    f"video_id={_safe_str(video_id)} window_index={w_i} start={start} end={end} "
                    f"embed_shape={tuple(e_t.shape)} blend_shape={tuple(b_t.shape)} other_shape={tuple(o_t.shape)}"
                )

            embeddings_list.append(e_t)
            blend_list.append(b_t)
            other_list.append(o_t)

            windows_meta.append(
                {
                    "window_index": w_i,
                    "picked_row_index": int(idx_row),
                    "window_start": start,
                    "window_end": end,
                    "window_len": int(end - start),
                    "embed_shape": tuple(e_t.shape),
                    "blend_shape": tuple(b_t.shape),
                    "other_shape": tuple(o_t.shape),
                }
            )

        raw_target = get_target_from_video_id(video_id, self.dataset_name)
        if self.data_cfg.label_mapping is not None:
            mapped_target = self.data_cfg.label_mapping[raw_target]
        else:
            mapped_target = raw_target

        if self.data_cfg.task_type == "classification":
            target = self.data_cfg.class_labels.index(mapped_target)
        else:
            target = float(mapped_target)

        meta = {
            "dataset": self.dataset_name,
            "mode": self.mode,
            "video_id": video_id,
            "embed_memmap_path": embed_memmap_path,
            "features_memmap_path": features_memmap_path,
            "eventfulness_path": eventfulness_path,
            "outer_fold": self.outer_idx,
            "inner_fold": self.inner_idx,
            "config_idx": self.config_idx,
            "epoch": self._epoch,
            "pick_mean": pick_mean,
            "pick_max": pick_max,
            "pick_random": pick_random,
            "picked_indices": list(map(int, picked_indices)),
            "eventfulness_df_rows": int(len(eventfulness_df)),
            "embed_memmap_shape": _safe_str(getattr(embed_memmap, "shape", None)),
            "features_memmap_shape": _safe_str(getattr(blend_other_memmap, "shape", None)),
        }

        # Optional: enable this if you want a single line per sample
        # print(f"[RealVideoDataset] idx={idx} video_id={_safe_str(video_id)} windows={len(embeddings_list)} mode={self.mode}")

        return {
            "embeddings": embeddings_list,
            "blendshapes": blend_list,
            "others": other_list,
            "target": target,
            "meta": meta,
            "windows_meta": windows_meta,
        }