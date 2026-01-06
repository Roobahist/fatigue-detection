import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Zh.experiments.e15.dataclasses import *
from Zh.experiments.e15.datasets import RealVideoDataset
from Zh.experiments.e15.helpers import *
from Zh.experiments.e15.model import MultiPathMILRegressor
from Zh.experiments.e15.nested_cv_engine import NestedCVEngine
from Zh.src.utils.normalization import (
    FileStats,
    aggregate_stats_for_files,
    stats_to_mean_std,
)

# ============================================================================
# Helper utilities specific to your data format
# ============================================================================


def _apply_normalization_to_sample(
    sample: Dict[str, Any],
    prepared: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    """
    In-place normalization of a single sample dict using prepared stats.

    - Supports:
        field: list[Tensor]  (e.g. list of windows, each (T, D) or (D,))
        field: Tensor        (e.g. stacked already)
    - Normalizes along the last dimension.
    """
    for field_name, stats in prepared.items():
        if field_name not in sample:
            continue

        mean_vec = stats["mean"]
        std_vec = stats["std"]

        field_value = sample[field_name]

        # a) list of tensors (your embeddings_list, blend_list, other_list)
        if isinstance(field_value, list) and field_value:
            device = field_value[0].device
            m = mean_vec.to(device)
            s = std_vec.to(device)

            for i, tensor in enumerate(field_value):
                # normalize along last dimension
                sample[field_name][i] = (tensor - m) / s

        # b) single tensor
        elif torch.is_tensor(field_value):
            device = field_value.device
            m = mean_vec.to(device)
            s = std_vec.to(device)
            sample[field_name] = (field_value - m) / s

        else:
            raise ValueError("Unsupported field type: " + str(type(field_value)))


def train_for_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    task_type: str,
    device: torch.device,
    verbose: bool = True,
) -> nn.Module:
    """Train model for fixed number of epochs without validation."""
    model.train()

    if verbose:
        print(
            f"\n{'-' * 80}\n"
            f"FIXED EPOCHS TRAINING\n"
            f"{'-' * 80}\n"
            f"  Total epochs: {num_epochs}\n"
            f"  Training samples: {len(train_loader.dataset)}\n"
            f"{'-' * 80}\n"
        )

    for epoch in range(num_epochs):
        if hasattr(train_loader.dataset, "set_training_epoch"):
            train_loader.dataset.set_training_epoch(epoch, num_epochs)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Show epoch progress with batch-level detail
        for batch_idx, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"Batches (Epoch {epoch+1}/{num_epochs})",
                disable=not verbose,
                leave=False,
            ),
            1,
        ):
            batch_on_device = move_batch_to_device(batch, device)
            preds = model(batch_on_device)
            targets = torch.tensor(
                [video["target"] for video in batch_on_device],
                dtype=torch.float32 if task_type == "regression" else torch.long,
                device=device,
            )
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        if verbose:
            print(
                f"  Epoch {epoch+1:3d}/{num_epochs}: "
                f"Avg Loss = {avg_epoch_loss:.4f} ({num_batches} batches)"
            )

    if verbose:
        print(f"{'-' * 80}\n" f"FIXED EPOCHS TRAINING COMPLETE\n" f"{'-' * 80}\n")

    return model


def make_batch_transform(
    train_video_ids: Optional[List[str]] = None,
    data_cfg: Optional[DataConfig] = None,
):
    if not train_video_ids or not data_cfg or not data_cfg.normalization_info_path:
        return None

    train_subject_ids = {
        get_subject_id_from_video_id(train_video_id, data_cfg.dataset_name)
        for train_video_id in train_video_ids
    }

    normalization_info = pd.read_csv(data_cfg.normalization_info_path)
    normalization_info = normalization_info[
        normalization_info["subject_id"].isin(train_subject_ids)
    ]

    grouped = normalization_info.groupby("modality")

    modality_stats = {}

    for modality, group in grouped:
        rows_to_iter = (
            FileStats(
                n=int(row["n"]),
                mean=np.asarray(
                    np.fromstring(row["mean"].strip("[]"), sep=" ", dtype=np.float64),
                    dtype=np.float64,
                ),
                M2=np.asarray(
                    np.fromstring(row["M2"].strip("[]"), sep=" ", dtype=np.float64),
                    dtype=np.float64,
                ),
            )
            for _, row in group.iterrows()
        )

        agg_stats = aggregate_stats_for_files(rows_to_iter)
        mean, std = stats_to_mean_std(agg_stats)
        modality_stats[modality] = {"mean": mean, "std": std}

    if not data_cfg.field_normalization:
        return None

    prepared: Dict[str, Dict[str, torch.Tensor]] = {}

    for field_name, cfg in data_cfg.field_normalization.items():
        modality = cfg["modality"]
        indices = cfg.get("indices", None)

        if modality not in modality_stats:
            logging.warning(
                f"Modality '{modality}' not found in modality_stats; "
                f"skipping field '{field_name}'."
            )
            continue

        mean_np = modality_stats[modality]["mean"]
        std_np = modality_stats[modality]["std"]

        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64)
            mean_np = mean_np[idx]
            std_np = std_np[idx]

        mean_t = torch.from_numpy(mean_np.astype(np.float32))
        std_t = torch.from_numpy(std_np.astype(np.float32))

        prepared[field_name] = {"mean": mean_t, "std": std_t}

    if not prepared:
        return None

    def batch_transform(batch):
        """
        batch is expected to be:
            - either a list of sample dicts,
            - or a collated dict with lists/tensors.
        We'll assume your current setup: list[dict] from collate_videos.
        """
        # Case 1: list of dicts
        if isinstance(batch, list):
            for sample in batch:
                _apply_normalization_to_sample(sample, prepared)
            return batch

        # Case 2: already collated dict {field: ...}
        if isinstance(batch, dict):
            # You can extend this branch later if you change collate behavior.
            _apply_normalization_to_sample(batch, prepared)
            return batch

        raise ValueError(f"Unsupported batch type: {type(batch)}")

    return batch_transform


# ============================================================================
# Adapter class that bridges generic engine with your specific code
# ============================================================================


class VideoMILAdapter:
    """
    Adapter that connects the generic NestedCVEngine to the
    Video MIL Regression implementation.
    """

    def __init__(
        self,
        base_model_cfg: ModelConfig,
        base_train_cfg: TrainingConfig,
        base_selection_cfg: SelectionConfig,
        data_cfg: DataConfig,
        device: torch.device,
    ):
        """
        Args:
            base_model_cfg: Base ModelConfig (will be updated with hyperparams)
            base_train_cfg: Base TrainingConfig (will be updated with hyperparams)
            base_selection_cfg: Base SelectionConfig (will be updated with hyperparams)
            data_cfg: DataConfig
            device: Device to use
        """
        self.base_model_cfg = base_model_cfg
        self.base_train_cfg = base_train_cfg
        self.base_selection_cfg = base_selection_cfg
        self.data_cfg = data_cfg
        self.device = device

        print(
            f"\n{'=' * 80}\n"
            f"VIDEO MIL ADAPTER INITIALIZED\n"
            f"{'=' * 80}\n"
            f"  Dataset: {data_cfg.dataset_name}\n"
            f"  Device: {device}\n"
            f"  Loss function: {self.base_train_cfg.loss}\n"
            f"  Base learning rate: {self.base_train_cfg.lr}\n"
            f"  Base batch size: {self.base_train_cfg.batch_size}\n"
            f"{'=' * 80}\n"
        )

    def create_dataloader(
        self,
        video_ids: List[str],
        mode: str,
        hparams: Dict[str, Any],
        train_video_ids: Optional[List[str]] = None,
        outer_idx: Optional[int] = 0,
        config_idx: Optional[int] = 0,
        inner_idx: Optional[int] = 0,
    ) -> DataLoader:
        """Create dataloader for given video IDs and mode."""
        selection_cfg = update_config_with_hparams(
            self.base_selection_cfg, hparams, params_prefix="sel"
        )
        train_cfg = update_config_with_hparams(
            self.base_train_cfg, hparams, params_prefix="train"
        )

        dataset = RealVideoDataset(
            video_ids=video_ids,
            data_cfg=self.data_cfg,
            selection_cfg=selection_cfg,
            mode=mode,
            base_seed=train_cfg.seed,
            outer_idx=outer_idx,
            config_idx=config_idx,
            inner_idx=inner_idx,
        )

        batch_transform = make_batch_transform(train_video_ids, self.data_cfg)
        collate_fn = make_collate_fn(batch_transform)

        g = torch.Generator()
        g.manual_seed(int(train_cfg.seed))

        return DataLoader(
            dataset,
            batch_size=int(train_cfg.batch_size),
            shuffle=(mode == "train"),
            collate_fn=collate_fn,
            num_workers=0,
            generator=g,
        )

    # ========================================================================
    # Callback implementations for NestedCVEngine
    # ========================================================================

    def model_builder(self, hparams: Dict[str, Any]) -> nn.Module:
        """Build model with hyperparameters."""
        model_cfg = update_config_with_hparams(
            self.base_model_cfg, hparams, params_prefix="model"
        )
        train_cfg = update_config_with_hparams(
            self.base_train_cfg, hparams, params_prefix="train"
        )
        return MultiPathMILRegressor(model_cfg, self.data_cfg, train_cfg)

    def train_fn(
        self,
        model: nn.Module,
        train_ids: List[str],
        val_ids: List[str],
        hparams: Dict[str, Any],
        checkpoint_mgr: Optional[CheckpointManager] = None,
        outer_idx: Optional[int] = 0,
        config_idx: Optional[int] = 0,
        inner_idx: Optional[int] = 0,
    ) -> Tuple[nn.Module, float, int]:
        """Train model with early stopping and training-level checkpointing."""
        train_loader = self.create_dataloader(
            video_ids=train_ids,
            mode="train",
            hparams=hparams,
            train_video_ids=train_ids,
            outer_idx=outer_idx,
            config_idx=config_idx,
            inner_idx=inner_idx,
        )
        val_loader = self.create_dataloader(
            video_ids=val_ids,
            mode="val",
            hparams=hparams,
            train_video_ids=train_ids,
            outer_idx=outer_idx,
            config_idx=config_idx,
            inner_idx=inner_idx,
        )

        train_cfg = update_config_with_hparams(
            self.base_train_cfg, hparams, params_prefix="train"
        )

        # Pass checkpoint parameters to train_one_split
        model, best_val_loss = train_one_split(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            data_cfg=self.data_cfg,
            checkpoint_mgr=checkpoint_mgr,
            outer_idx=outer_idx,
            config_idx=config_idx,
            inner_idx=inner_idx,
        )

        # Extract best epoch if stored, otherwise estimate from patience
        if hasattr(model, "best_epoch_"):
            best_epoch = model.best_epoch_
        else:
            best_epoch = train_cfg.num_epochs

        return model, best_val_loss, best_epoch

    def train_fixed_epochs_fn(
        self,
        model: nn.Module,
        train_ids: List[str],
        num_epochs: int,
        hparams: Dict[str, Any],
        task_type: str,
        outer_idx: Optional[int] = 0,
    ) -> nn.Module:
        """Train model for fixed epochs without validation."""
        train_loader = self.create_dataloader(
            video_ids=train_ids,
            mode="train",
            hparams=hparams,
            train_video_ids=train_ids,
            outer_idx=outer_idx,
        )

        # Create optimizer
        # todo : instead of updating the basecfg, store the complete config of the model so that
        #        if the initial config has different parameters, it stays consistent.
        train_cfg = update_config_with_hparams(
            self.base_train_cfg, hparams, params_prefix="train"
        )

        loss_fn = get_loss_fn(
            name=train_cfg.loss,
            data_cfg=self.data_cfg,
        )

        optimizer, scheduler = build_optimizer(model, train_cfg)

        return train_for_fixed_epochs(
            model=model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            task_type=task_type,
            device=self.device,
            verbose=True,
        )

    def evaluate_fn(
        self,
        model: nn.Module,
        data_ids: List[str],
        hparams: Dict[str, Any],
        train_ids: Optional[List[str]],
        outer_idx: Optional[int] = 0,
        config_idx: Optional[int] = 0,
        inner_idx: Optional[int] = 0,
        verbose: bool = False,
    ) -> float:
        """Evaluate model on data."""
        loader = self.create_dataloader(
            video_ids=data_ids,
            mode="test",
            hparams=hparams,
            train_video_ids=train_ids,
            outer_idx=outer_idx,
            config_idx=config_idx,
            inner_idx=inner_idx,
        )
        train_cfg = update_config_with_hparams(
            self.base_train_cfg, hparams, params_prefix="train"
        )
        loss_fn = get_loss_fn(
            name=train_cfg.loss,
            data_cfg=self.data_cfg,
        )
        thresholds, unique_classes = get_task_info_from_dataset_name(
            self.data_cfg.dataset_name,
            self.data_cfg.task_type,
            self.data_cfg.label_mapping,
        )
        return evaluate_on_loader(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            device=self.device,
            data_cfg=self.data_cfg,
            train_cfg=train_cfg,
            thresholds=thresholds,
            unique_classes=unique_classes,
            verbose=False,
        )[0]

    def split_fn(
        self, data_ids: List[str], num_folds: int, fold_idx: int
    ) -> Tuple[List[str], List[str]]:
        """Split data into train and test."""
        # Your kfold_split_indices expects indices, not data_ids
        indices = list(range(len(data_ids)))
        train_indices, test_indices = kfold_split_indices(
            indices=indices,
            n_splits=num_folds,
            fold_idx=fold_idx,
            dataset_name=self.data_cfg.dataset_name,
            video_ids=data_ids,
        )
        train_ids = [data_ids[i] for i in train_indices]
        test_ids = [data_ids[i] for i in test_indices]
        return train_ids, test_ids


# ============================================================================
# Convenience function for your specific use case
# ============================================================================


def nested_cv_video_mil(
    video_ids: List[str],
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    selection_cfg: SelectionConfig,
    data_cfg: DataConfig,
    cv_cfg: CrossValidationConfig,
    device: torch.device,
    search_space: Dict[str, Any] = None,
    output_dir: str = f"/home/mojtaba.kananisarche/Zh/outputs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}|E11",
    parallel_inner_cv: bool = False,
    max_workers: int = 4,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Run nested cross-validation for Video MIL Regression.

    Args:
        video_ids: List of video IDs
        model_cfg: Base ModelConfig
        train_cfg: Base TrainingConfig
        selection_cfg: Base SelectionConfig
        data_cfg: DataConfig
        cv_cfg: CrossValidationConfig
        search_space: Hyperparameter search space
        output_dir: Output directory
        device: Device to use
        resume: If True, attempt to resume from checkpoint if available.
                Set to False to start fresh even if checkpoints exist.

    Returns:
        Dictionary with nested CV results
    """
    # Default search space
    if search_space is None:
        print(
            f"\n{'!' * 80}\n" f"WARNING: Using default search space\n" f"{'!' * 80}\n"
        )
        search_space = {
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "weight_decay": ("log_uniform", 1e-6, 1e-3),
        }

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"\n{'#' * 80}\n"
        f"# NESTED CV VIDEO MIL REGRESSION\n"
        f"{'#' * 80}\n"
        f"  Total videos: {len(video_ids)}\n"
        f"  Outer folds: {cv_cfg.outer_folds}\n"
        f"  Inner folds: {cv_cfg.inner_folds}\n"
        f"  Search strategy: {cv_cfg.search_strategy}\n"
        f"  N trials: {cv_cfg.n_trials}\n"
        f"  Output directory: {output_dir}\n"
        f"  Random seed: {train_cfg.seed}\n"
        f"  Resume: {resume}\n"
        f"{'#' * 80}\n"
    )

    adapter = VideoMILAdapter(
        base_model_cfg=model_cfg,
        base_train_cfg=train_cfg,
        base_selection_cfg=selection_cfg,
        data_cfg=data_cfg,
        device=device,
    )

    # Create generic engine with adapter callbacks
    engine = NestedCVEngine(
        model_builder=adapter.model_builder,
        train_fn=adapter.train_fn,
        train_fixed_epochs_fn=adapter.train_fixed_epochs_fn,
        evaluate_fn=adapter.evaluate_fn,
        split_fn=adapter.split_fn,
        task_type=adapter.data_cfg.task_type,
        device=adapter.device,
        parallel_inner_cv=parallel_inner_cv,
        max_workers=max_workers,
    )

    # Run nested CV
    results = engine.run(
        data_ids=video_ids,
        cv_cfg=cv_cfg,
        search_space=search_space,
        output_dir=output_dir,
        random_seed=train_cfg.seed,
        verbose=True,
        resume=resume,
    )

    return results
