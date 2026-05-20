import copy

import json
import os
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from Zh.experiments.e15.checkpoint_manager import CheckpointManager
from Zh.experiments.e15.dataclasses import *
from Zh.experiments.e15.datasets import *
from Zh.src.utils.losses import *


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
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    thresholds: Optional[List[float]] = None,
    unique_classes: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[float, Tuple[List[float], List[float], List[float]]]:
    """
    Evaluate model on a dataloader.

    Updated to handle both regression and classification tasks via data_cfg.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    predicted_values = []  # Raw predictions or class probabilities
    true_classes = []
    predicted_classes = []

    task_type = data_cfg.task_type  # Get from single source of truth
    loss_name = train_cfg.loss.lower()

    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            outputs = model(batch_on_device)

            if task_type == "regression":
                # For regression: outputs are [B] scalars
                preds = outputs
                targets = torch.tensor(
                    [video["target"] for video in batch_on_device],
                    dtype=torch.float32,
                    device=device,
                )
                loss = loss_fn(preds, targets)

                # Store predictions and convert to classes for metrics
                predicted_values.extend(preds.tolist())
                true_classes.extend(targets.tolist())
                if thresholds and unique_classes:
                    predicted_classes.extend(
                        [
                            assign_class(p, thresholds, unique_classes)
                            for p in preds.tolist()
                        ]
                    )
                else:
                    predicted_classes.extend(targets.tolist())

            else:  # classification
                # For classification: outputs are [B, num_classes] logits
                logits = outputs
                targets = torch.tensor(
                    [video["target"] for video in batch_on_device],
                    dtype=torch.long,
                    device=device,
                )
                loss = loss_fn(logits, targets)

                # Get predictions
                preds = get_predictions_from_logits(logits, task_type, loss_name)

                # Get probabilities for reporting
                if loss_name == "coral":
                    probs = torch.sigmoid(logits)
                elif loss_name == "corn":
                    probs = torch.sigmoid(logits)
                    probs = torch.cumprod(probs, dim=1)
                else:
                    # Standard classification
                    probs = torch.softmax(logits, dim=1)

                predicted_values.extend(probs.tolist())
                true_classes.extend(targets.tolist())
                predicted_classes.extend(preds.tolist())

            total_loss += loss.item() * len(batch)
            num_samples += len(batch)

    mean_loss = total_loss / num_samples if num_samples > 0 else float("inf")

    if verbose:
        # For classification, map indices back to class labels
        if task_type == "classification" and unique_classes:
            true_labels = [unique_classes[int(c)] for c in true_classes]
            pred_labels = [unique_classes[int(c)] for c in predicted_classes]
        else:
            true_labels = true_classes
            pred_labels = predicted_classes

        cr = classification_report(
            y_true=true_labels,
            y_pred=pred_labels,
            labels=unique_classes if unique_classes else None,
            output_dict=True,
            zero_division=False,
        )

        cm = safe_confusion_matrix(
            y_true=true_labels,
            y_pred=pred_labels,
            all_labels=unique_classes if unique_classes else sorted(set(true_labels)),
        )

        loss_display = (
            loss_name.upper() if loss_name in ["coral", "corn"] else task_type.upper()
        )
        print(
            "\n" + "=" * 80 + "\n"
            f"EVALUATION RESULTS ({loss_display})\n"
            f"Mean Loss: {mean_loss:.4f}\n" + "-" * 80 + "\n"
            "Confusion Matrix:\n"
            f"{cm}\n" + "-" * 80 + "\n"
            "Classification Report:\n"
            f"{json.dumps(cr, cls=NumpyEncoder, indent=4)}\n" + "=" * 80
        )

    assert len(predicted_values) == len(true_classes) == len(predicted_classes)

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
    data_cfg: DataConfig,
    verbose: bool = True,
    checkpoint_mgr: Optional[CheckpointManager] = None,
    outer_idx: Optional[int] = None,
    config_idx: Optional[int] = None,
    inner_idx: Optional[int] = None,
) -> Tuple[nn.Module, float]:
    """
    Train on a single train/val split.
    Now supports both regression and classification via data_cfg.
    """
    loc = _format_location(outer_idx, config_idx, inner_idx)

    device_str = train_cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # Get task info from data_cfg (single source of truth)
    thresholds, unique_classes = get_task_info_from_dataset_name(
        data_cfg.dataset_name, data_cfg.task_type, data_cfg.label_mapping
    )
    assert data_cfg.class_labels == unique_classes

    model.to(device)
    optimizer, scheduler = build_optimizer(model, train_cfg)
    loss_fn = get_loss_fn(train_cfg.loss, data_cfg)

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

    true_classes_train = []

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
                dtype=(
                    torch.float32 if data_cfg.task_type == "regression" else torch.long
                ),
                device=device,
            )

            if epoch == start_epoch:
                true_classes_train.extend(targets.tolist())

            optimizer.zero_grad()
            preds = model(batch_on_device)
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

        # train_eval_loss, (
        #     predicted_values_train,
        #     true_classes_train,
        #     predicted_classes_train,
        # ) = evaluate_on_loader(
        #     model=model,
        #     loader=train_loader,
        #     loss_fn=loss_fn,
        #     device=device,
        #     data_cfg=data_cfg,
        #     train_cfg=train_cfg,
        #     thresholds=thresholds,
        #     unique_classes=unique_classes,
        #     verbose=False,
        # )

        # Validation phase
        val_loss, (predicted_values_val, true_classes_val, predicted_classes_val) = (
            evaluate_on_loader(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                data_cfg=data_cfg,
                train_cfg=train_cfg,
                thresholds=thresholds,
                unique_classes=unique_classes,
                verbose=False,
            )
        )

        true_train = torch.as_tensor(true_classes_train, dtype=torch.float32)
        true_val = torch.as_tensor(true_classes_val, dtype=torch.float32)

        baseline_train_loss = 0
        baseline_val_loss = 0

        baseline_response = (
            float(np.mean(true_classes_train))
            if data_cfg.task_type == "regression"
            else int(np.bincount(true_classes_train).argmax())
        )

        if data_cfg.task_type == "regression":
            # Build baseline tensors with the same shape
            baseline_train = torch.full_like(true_train, baseline_response)
            baseline_val = torch.full_like(true_val, baseline_response)

            baseline_train_loss = loss_fn(baseline_train, true_train)
            baseline_val_loss = loss_fn(baseline_val, true_val)

        else:
            baseline_train_loss = sum(
                1 for t in true_classes_train if t != baseline_response
            ) / len(true_classes_train)
            baseline_val_loss = sum(
                1 for t in true_classes_val if t != baseline_response
            ) / len(true_classes_val)

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

            if data_cfg.task_type == "classification" and unique_classes:
                true_labels_val = [unique_classes[int(c)] for c in true_classes_val]
                pred_labels_val = [
                    unique_classes[int(c)] for c in predicted_classes_val
                ]
                # true_labels_train = [unique_classes[int(c)] for c in true_classes_train]
                # pred_labels_train = [
                #     unique_classes[int(c)] for c in predicted_classes_train
                # ]
            else:
                true_labels_val = true_classes_val
                pred_labels_val = predicted_classes_val
                # true_labels_train = true_classes_train
                # pred_labels_train = predicted_classes_train

            cr_val = classification_report(
                y_true=true_labels_val,
                y_pred=pred_labels_val,
                labels=unique_classes if unique_classes else None,
                output_dict=True,
                zero_division=False,
            )
            cm_val = safe_confusion_matrix(
                y_true=true_labels_val,
                y_pred=pred_labels_val,
                all_labels=(
                    unique_classes if unique_classes else sorted(set(true_labels_val))
                ),
            )

            # cr_train = classification_report(
            #     y_true=true_labels_train,
            #     y_pred=pred_labels_train,
            #     labels=unique_classes if unique_classes else None,
            #     output_dict=True,
            #     zero_division=False,
            # )
            # cm_train = safe_confusion_matrix(
            #     y_true=true_labels_train,
            #     y_pred=pred_labels_train,
            #     all_labels=(
            #         unique_classes if unique_classes else sorted(set(true_labels_train))
            #     ),
            # )

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
                f"  Loss - Train: {train_loss:.4f} | Baseline: {baseline_train_loss:.4f}\n"
                f"  Loss - Val: {val_loss:.4f} | Baseline: {baseline_val_loss:.4f}\n"
                f"  Best - Val Loss: {best_val_loss:.4f} (epoch {best_epoch})\n"
                f"  Early Stop - Epochs w/o improvement: {epochs_without_improvement}/{train_cfg.early_stopping_patience}\n"
                + "-" * 80
                + "\n"
                "  Confusion Matrix - Val:\n"
                f"  {cm_val}\n" + "-" * 80 + "\n"
                # "  Classification Report - Val:\n"
                # f"  {json.dumps(cr_val, cls=NumpyEncoder, indent=4)}\n"
                # + "-" * 80
                # + "\n"
                # "  Confusion Matrix - Train:\n"
                # f"  {cm_train}\n" + "-" * 80 + "\n"
                # "  Classification Report - Train:\n"
                # f"  {json.dumps(cr_train, cls=NumpyEncoder, indent=4)}\n"
                # + "-" * 80
                # + "\n"
                f"  Predictions: {[(p, c, t) for p, c, t in zip(predicted_values_val, pred_labels_val, true_labels_val)]}\n"
                f"{'=' * 80}\n"
            )

        should_visualize = (
            checkpoint_mgr
            and outer_idx is not None
            and config_idx is not None
            and inner_idx is not None
        )

        if should_visualize:
            checkpoint_mgr.save_training_visualization(
                outer_idx=outer_idx,
                config_idx=config_idx,
                inner_idx=inner_idx,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                max_epochs=train_cfg.num_epochs,
            )

        # Save checkpoint periodically (every 5 epochs or on improvement)
        should_checkpoint = should_visualize and (
            epoch % 5 == 0 or epochs_without_improvement == 0
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
    # if (
    #     checkpoint_mgr
    #     and outer_idx is not None
    #     and config_idx is not None
    #     and inner_idx is not None
    #     and not time_limit_exceeded
    # ):
    #     checkpoint_mgr.clean_training_checkpoint(outer_idx, config_idx, inner_idx)

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


def get_loss_fn(name: str, data_cfg: DataConfig):
    """
    Get loss function based on task type.

    Args:
        name: Loss function name
        task_type: "regression" or "classification"
        num_classes: Number of classes (for classification)
    """
    name = name.lower()

    task_type = data_cfg.task_type
    num_classes = data_cfg.num_classes

    if task_type == "regression":
        if name == "mse":
            return nn.MSELoss()
        elif name == "smooth_l1":
            return nn.SmoothL1Loss()
        elif name in ["l1", "mae"]:
            return nn.L1Loss()
        elif name == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported regression loss: {name}")

    else:  # classification
        if num_classes is None:
            raise ValueError(
                "num_classes must be specified in DataConfig for classification"
            )
        if name == "coral":
            return CoralLoss(num_classes=num_classes)
        elif name == "corn":
            return CornLoss(num_classes=num_classes)
        elif name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif name == "emd":
            return EarthMoversDistanceLoss(num_classes=num_classes)
        elif name == "combined_emd_ce":
            # Combined EMD + CrossEntropy (50/50 by default)
            return CombinedEMDCrossEntropyLoss(num_classes=num_classes, alpha=0.5)
        elif name == "weighted_emd":
            # You can compute class weights from your dataset
            return WeightedEMDLoss(num_classes=num_classes)
        elif name == "focal_loss":
            raise NotImplementedError("Focal loss not implemented yet")
        elif name == "bce":
            if num_classes == 2:
                return nn.BCEWithLogitsLoss()
            else:
                raise ValueError("BCE loss only for binary classification")
        else:
            raise ValueError(f"Unsupported classification loss: {name}")


def get_subject_id_from_video_id(video_id: str, dataset_name: str) -> int:
    if dataset_name == "UTA":
        return int(os.path.basename(video_id).split("_")[1])
    elif dataset_name == "DROZY":
        return int(os.path.basename(video_id).split("-")[0])
    else:
        raise ValueError(f"Unknown dataset for subject ID extraction: {dataset_name}")


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


def get_task_info_from_dataset_name(
    dataset_name: str,
    task_type: str = "regression",
    label_mapping: Optional[Dict[float, float]] = None,
):
    """
    Get task-specific information for a dataset.

    Args:
        dataset_name: Name of the dataset
        task_type: "regression" or "classification"
        label_mapping: Optional mapping of original labels to new labels

    Returns:
        For regression: (thresholds, unique_classes) for binning
        For classification: (None, class_labels)
    """
    # If label_mapping is provided, derive everything from it
    if label_mapping is not None:
        unique_mapped_labels = sorted(set(label_mapping.values()))

        if task_type == "classification":
            return None, unique_mapped_labels
        else:  # regression with binning for metrics
            # Generate thresholds between consecutive unique labels
            if len(unique_mapped_labels) > 1:
                thresholds = [
                    (unique_mapped_labels[i] + unique_mapped_labels[i + 1]) / 2
                    for i in range(len(unique_mapped_labels) - 1)
                ]
            else:
                thresholds = []
            return thresholds, unique_mapped_labels

    # Default behavior when no mapping is provided
    if dataset_name == "UTA":
        if task_type == "classification":
            # Direct classification into 3 classes
            return None, [0.0, 5.0, 10.0]
        else:  # regression with binning for metrics
            return [2.5, 7.5], [0.0, 5.0, 10.0]

    elif dataset_name == "DROZY":
        if task_type == "classification":
            # Direct classification into 9 classes
            return None, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        else:  # regression with binning for metrics
            return [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_predictions_from_logits(logits, task_type, loss_name):
    """Convert logits to predictions based on loss type"""
    if task_type == "regression":
        return logits

    loss_name = loss_name.lower()
    if loss_name == "coral":
        from Zh.src.utils.losses import proba_to_label

        probas = torch.sigmoid(logits)
        return proba_to_label(probas)
    elif loss_name == "corn":
        from Zh.src.utils.losses import corn_label_from_logits

        return corn_label_from_logits(logits)
    else:
        return torch.argmax(logits, dim=1)
