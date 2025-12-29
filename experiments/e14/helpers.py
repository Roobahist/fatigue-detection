import copy
import json
import os
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from Zh.experiments.e14.checkpoint_manager import CheckpointManager
from Zh.experiments.e14.dataclasses import *
from Zh.experiments.e14.datasets import *


def _format_location(outer_idx=None, config_idx=None, inner_idx=None):
    """Format location string for logging."""
    parts = []
    if outer_idx is not None:
        parts.append(f"Outer={outer_idx}")
    if config_idx is not None:
        parts.append(f"Config={config_idx}")
    if inner_idx is not None:
        parts.append(f"Inner={inner_idx}")
    return f"[{', '.join(parts)}]" if parts else ""


def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    thresholds: List[float],
    unique_classes: List[float],
    verbose: bool = True,
) -> Tuple[float, Tuple[List[float], List[float], List[float]]]:
    """Evaluate model on a dataloader."""

    model.eval()
    total_loss = 0.0
    num_samples = 0

    predicted_values = []
    true_classes = []
    predicted_classes = []

    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            preds = model(batch_on_device)
            targets = torch.tensor(
                [video["target"] for video in batch_on_device],
                dtype=torch.float32,
                device=device,
            )
            loss = loss_fn(preds, targets)
            total_loss += loss.item() * len(batch)
            num_samples += len(batch)

            predicted_values.extend(preds.tolist())
            true_classes.extend(targets.tolist())
            predicted_classes.extend(
                [
                    assign_class(
                        value=p, thresholds=thresholds, unique_classes=unique_classes
                    )
                    for p in preds.tolist()
                ]
            )
    mean_loss = total_loss / num_samples if num_samples > 0 else float("inf")

    if verbose:
        cr = classification_report(
            y_true=true_classes,
            y_pred=predicted_classes,
            labels=unique_classes,
            output_dict=True,
            zero_division=False,
        )

        cm = safe_confusion_matrix(
            y_true=true_classes, y_pred=predicted_classes, all_labels=unique_classes
        )

        print(
            "\n" + "=" * 80 + "\n"
            f"EVALUATION RESULTS\n"
            f"Mean Loss: {mean_loss:.4f}\n" + "-" * 80 + "\n"
            "Confusion Matrix:\n"
            f"{cm}\n" + "-" * 80 + "\n"
            "Classification Report:\n"
            f"{json.dumps(cr, cls=NumpyEncoder, indent=4)}\n" + "=" * 80
        )

    return mean_loss, (predicted_values, true_classes, predicted_classes)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def move_batch_to_device(batch: List[Dict], device: torch.device) -> List[Dict]:
    """Move batch of video dicts to device."""
    batch_on_device = []
    for video in batch:
        video_on_device = {
            "embeddings": [t.to(device) for t in video["embeddings"]],
            "blendshapes": [t.to(device) for t in video["blendshapes"]],
            "others": [t.to(device) for t in video["others"]],
            "target": video["target"],
        }
        batch_on_device.append(video_on_device)
    return batch_on_device


def train_one_split(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainingConfig,
    dataset_name: str = "UTA",
    verbose: bool = True,
    checkpoint_mgr: Optional[CheckpointManager] = None,
    outer_idx: Optional[int] = None,
    config_idx: Optional[int] = None,
    inner_idx: Optional[int] = None,
) -> Tuple[nn.Module, float]:
    """
    Train on a single train/val split with:
    - Early stopping
    - Training-level checkpointing
    - Time limit

    Returns best model and best validation loss.
    """
    loc = _format_location(outer_idx, config_idx, inner_idx)

    device_str = train_cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    thresholds, unique_classes = get_thresholds_unique_value_from_dataset_name(
        dataset_name
    )

    model.to(device)
    optimizer, scheduler = build_optimizer(model, train_cfg)
    loss_fn = get_loss_fn(train_cfg.loss)

    # Initialize training state
    start_epoch = 0
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    # Track training start time for time limit
    training_start_time = time.time()
    time_limit_exceeded = False

    # Try to resume from checkpoint
    if (
        checkpoint_mgr
        and outer_idx is not None
        and config_idx is not None
        and inner_idx is not None
    ):
        checkpoint = checkpoint_mgr.load_training_checkpoint(
            outer_idx, config_idx, inner_idx
        )

        if checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_loss = checkpoint["best_val_loss"]
            best_epoch = checkpoint["best_epoch"]
            best_state = checkpoint["best_model_state_dict"]
            epochs_without_improvement = checkpoint["epochs_without_improvement"]

            if verbose:
                print(
                    f"\n{'=' * 80}\n"
                    f"TRAINING RESUME {loc}\n"
                    f"{'=' * 80}\n"
                    f"  Resuming from epoch: {start_epoch}\n"
                    f"  Best val loss so far: {best_val_loss:.4f} (epoch {best_epoch})\n"
                    f"  Epochs without improvement: {epochs_without_improvement}\n"
                    f"{'=' * 80}\n"
                )

    # Training loop
    for epoch in range(start_epoch, train_cfg.num_epochs):
        # Check time limit
        if train_cfg.training_time_limit is not None:
            elapsed_time = time.time() - training_start_time
            if elapsed_time >= train_cfg.training_time_limit:
                if verbose:
                    print(
                        f"\n{'!' * 80}\n"
                        f"TIME LIMIT EXCEEDED {loc}\n"
                        f"{'!' * 80}\n"
                        f"  Elapsed: {elapsed_time:.1f}s >= Limit: {train_cfg.training_time_limit:.1f}s\n"
                        f"  Stopping training and returning best model\n"
                        f"{'!' * 80}\n"
                    )
                time_limit_exceeded = True
                break

        train_loader.dataset.set_training_epoch(epoch, train_cfg.num_epochs)

        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"{loc} Epoch {epoch+1}/{train_cfg.num_epochs}",
            disable=not verbose,
            leave=False,
        )

        for batch in progress_bar:
            batch_on_device = move_batch_to_device(batch, device)
            targets = torch.tensor(
                [video["target"] for video in batch_on_device],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            preds = model(batch_on_device)  # [B]
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_on_device)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = (
            train_loss_sum / train_samples if train_samples > 0 else float("inf")
        )

        # Validation phase
        val_loss, (predicted_values, true_classes, predicted_classes) = (
            evaluate_on_loader(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                thresholds=thresholds,
                unique_classes=unique_classes,
                verbose=False,
            )
        )

        if scheduler is not None:
            scheduler.step()

        # Update best model
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Verbose logging
        if verbose:
            cr = classification_report(
                y_true=true_classes,
                y_pred=predicted_classes,
                labels=unique_classes,
                output_dict=True,
                zero_division=False,
            )

            cm = safe_confusion_matrix(
                y_true=true_classes, y_pred=predicted_classes, all_labels=unique_classes
            )

            elapsed = time.time() - training_start_time
            time_remaining = (
                f"{train_cfg.training_time_limit - elapsed:.1f}s"
                if train_cfg.training_time_limit
                else "unlimited"
            )

            print(
                f"\n{'=' * 80}\n"
                f"EPOCH {epoch + 1}/{train_cfg.num_epochs} {loc}\n"
                f"{'=' * 80}\n"
                f"  Time - Elapsed: {elapsed:.1f}s | Remaining: {time_remaining}\n"
                f"  Loss - Train: {train_loss:.4f} | Val: {val_loss:.4f}\n"
                f"  Best - Val Loss: {best_val_loss:.4f} (epoch {best_epoch})\n"
                f"  Early Stop - Epochs w/o improvement: {epochs_without_improvement}/{train_cfg.early_stopping_patience}\n"
                + "-" * 80
                + "\n"
                "  Confusion Matrix:\n"
                f"  {cm}\n" + "-" * 80 + "\n"
                "  Classification Report:\n"
                f"  {json.dumps(cr, cls=NumpyEncoder, indent=4)}\n" + "-" * 80 + "\n"
                f"  Predictions: {[(p, c, t) for p, c, t in zip(predicted_values, predicted_classes, true_classes)]}\n"
                f"{'=' * 80}\n"
            )

        # Save checkpoint periodically (every 5 epochs or on improvement)
        should_checkpoint = (
            checkpoint_mgr
            and outer_idx is not None
            and config_idx is not None
            and inner_idx is not None
            and (epoch % 5 == 0 or epochs_without_improvement == 0)
        )

        if should_checkpoint:
            checkpoint_mgr.save_training_checkpoint(
                outer_idx=outer_idx,
                config_idx=config_idx,
                inner_idx=inner_idx,
                epoch=epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler else None,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                best_model_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
            )

        # Early stopping check
        if epochs_without_improvement >= train_cfg.early_stopping_patience:
            if verbose:
                print(
                    f"\n{'!' * 80}\n"
                    f"EARLY STOPPING TRIGGERED {loc}\n"
                    f"{'!' * 80}\n"
                    f"  Patience exhausted: {epochs_without_improvement}/{train_cfg.early_stopping_patience}\n"
                    f"  Best epoch: {best_epoch} with val loss: {best_val_loss:.4f}\n"
                    f"{'!' * 80}\n"
                )
            break

    # Load best model
    model.load_state_dict(best_state)
    model.best_epoch_ = best_epoch

    # Clean up training checkpoint after successful completion
    # (only if we didn't hit time limit - keep checkpoint if interrupted by time limit)
    if (
        checkpoint_mgr
        and outer_idx is not None
        and config_idx is not None
        and inner_idx is not None
        and not time_limit_exceeded
    ):
        checkpoint_mgr.clean_training_checkpoint(outer_idx, config_idx, inner_idx)

    if verbose:
        status = "TIME LIMITED" if time_limit_exceeded else "COMPLETED"
        print(
            f"\n{'=' * 80}\n"
            f"TRAINING {status} {loc}\n"
            f"{'=' * 80}\n"
            f"  Best epoch: {best_epoch}\n"
            f"  Best val loss: {best_val_loss:.4f}\n"
            f"{'=' * 80}\n"
        )

    return model, best_val_loss


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

    assert mean_i + max_i + rand_i == sel_cfg.n_windows

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
    elif name in ["l1", "mae"]:
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss()
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "bce":
        return nn.BCELoss()
    elif name == "bce_logit":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss: {name}")


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


def get_thresholds_unique_value_from_dataset_name(dataset_name: str):
    if dataset_name == "UTA":
        thresholds = [2.5, 7.5]
        unique_classes = [0.0, 5.0, 10]

    elif dataset_name == "DROZY":
        thresholds = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        unique_classes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return thresholds, unique_classes


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


def make_collate_fn(
    batch_transform: Optional[Callable[[List[Any]], List[Any]]] = None,
) -> Callable[[List[Any]], List[Any]]:
    """
    Dataloader collate: optionally apply a batch-level transform.
    Input batch is the raw list of dataset items.
    """

    def _collate(batch: List[Any]) -> List[Any]:
        if batch_transform is not None:
            return batch_transform(batch)
        return batch  # identity if no transform

    return _collate


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


def update_config_with_hparams(
    base_cfg: Any, hparams: Dict[str, Any], params_prefix: str
) -> Any:
    """Update config object with hyperparameters using dotted keys."""
    cfg = copy.deepcopy(base_cfg)

    for key, value in hparams.items():
        subkeys = key.split(".")
        current = cfg

        if subkeys[0] != params_prefix:
            continue

        # walk all but the last subkey
        for subkey in subkeys[
            1:-1
        ]:  # skip the first which is the config name and last which is to be set
            if not hasattr(current, subkey):
                raise AttributeError(
                    f"{current} has no attribute '{subkey}' for key '{key}' 1"
                )

            current = getattr(current, subkey)

        if current is None:
            raise AttributeError(
                f"{current} has no attribute '{subkeys[-1]}' for key '{key}' 2"
            )

        last = subkeys[-1]
        if hasattr(current, last):
            setattr(current, last, value)
        else:
            raise AttributeError(
                f"{current} has no attribute '{last}' for key '{key}' 3"
            )

    return cfg
