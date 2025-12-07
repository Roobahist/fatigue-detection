import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from Zh.experiments.e11.dataclasses import *


def assign_class(value, thresholds, unique_classes):
    assert len(thresholds) + 1 == len(unique_classes)
    assert all(isinstance(i, (int, float)) for i in thresholds)
    unique_classes = [float(i) for i in unique_classes]
    return next(
        (unique_classes[i] for i, t in enumerate(thresholds) if value < t),
        unique_classes[-1],
    )


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

    return mean_i, max_i, rand_i


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
            "embeddings": [
                t.to(device, non_blocking=True) for t in video["embeddings"]
            ],
            "blendshapes": [
                t.to(device, non_blocking=True) for t in video["blendshapes"]
            ],
            "others": [t.to(device, non_blocking=True) for t in video["others"]],
            "target": video["target"],
        }
        new_batch.append(new_video)
    return new_batch


def get_subject_id_from_video_id(video_id: str, dataset_name: str) -> int:
    if dataset_name == "UTA":
        return int(os.path.basename(video_id).split("_")[1])
    elif dataset_name == "DROZY":
        return int(os.path.basename(video_id).split("-")[0])
    else:
        raise ValueError(f"Unknown dataset for subject ID extraction: {dataset_name}")


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


def kfold_split_indices(
    indices: List[int],
    n_splits: int,
    fold_idx: int,
    dataset_name: str = "",
    video_ids: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Simple K-fold split on indices.
    Returns (train_indices, val_indices) for a given fold_idx.
    """
    if dataset_name in ["UTA", "DROZY"]:
        subject_ids = {
            get_subject_id_from_video_id(video_id, dataset_name)
            for video_id in video_ids
        }

        train_subject_id_indices, val_subject_id_indices = kfold_split_indices(
            indices=list(range(len(subject_ids))),
            n_splits=n_splits,
            fold_idx=fold_idx,
        )

        train_subject_ids = {list(subject_ids)[i] for i in train_subject_id_indices}
        val_subject_ids = {list(subject_ids)[i] for i in val_subject_id_indices}

        train_indices = [
            i
            for i, video_id in enumerate(video_ids)
            if get_subject_id_from_video_id(video_id, dataset_name) in train_subject_ids
        ]
        val_indices = [
            i
            for i, video_id in enumerate(video_ids)
            if get_subject_id_from_video_id(video_id, dataset_name) in val_subject_ids
        ]

    else:
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


def collate_videos(batch: List[Dict]) -> List[Dict]:
    # Just return the list of video dicts, no padding.
    return batch


def safe_confusion_matrix(y_true, y_pred, all_labels):
    """
    Build a confusion matrix that includes all labels (even if missing)
    without raising any errors like sklearn does.

    Args:
        y_true (list or np.array): Ground truth labels
        y_pred (list or np.array): Predicted labels
        all_labels (list): Full list of labels (ordering preserved)

    Returns:
        np.ndarray: Confusion matrix of shape (L, L) where L=len(all_labels)
                    CM[i, j] = number of samples with true=all_labels[i]
                                and pred=all_labels[j]
    """
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    L = len(all_labels)
    cm = np.zeros((L, L), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1
        # if t or p not in all_labels, silently ignore

    return cm
