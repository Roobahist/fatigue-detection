import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Zh.experiments.e13.dataclasses import *
from Zh.experiments.e13.datasets import RealVideoDataset
from Zh.experiments.e13.helpers import *
from Zh.experiments.e13.model import MultiPathMILRegressor
from Zh.experiments.e13.nested_cv_engine import NestedCVEngine

# ============================================================================
# Helper utilities specific to your data format
# ============================================================================


def evaluate_on_loader(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    num_samples = 0

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

    return total_loss / num_samples if num_samples > 0 else float("inf")


def train_for_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> nn.Module:
    """Train model for fixed number of epochs without validation."""
    model.train()

    for epoch in range(num_epochs):
        if hasattr(train_loader.dataset, "set_training_epoch"):
            train_loader.dataset.set_training_epoch(epoch, num_epochs)

        for batch in train_loader:
            batch_on_device = move_batch_to_device(batch, device)
            preds = model(batch_on_device)
            targets = torch.tensor(
                [video["target"] for video in batch_on_device],
                dtype=torch.float32,
                device=device,
            )
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


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


# ============================================================================
# Adapter class that bridges generic engine with your specific code
# ============================================================================


class VideoMILAdapter:
    """
    Adapter that connects the generic NestedCVEngine to your specific
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
        self.loss_fn = get_loss_fn(self.base_train_cfg.loss)

    def create_dataloader(
        self, video_ids: List[str], mode: str, hparams: Dict[str, Any]
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
        )

        return DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=(mode == "train"),
            collate_fn=collate_videos,
            num_workers=0,
        )

    # ========================================================================
    # Callback implementations for NestedCVEngine
    # ========================================================================

    def model_builder(self, hparams: Dict[str, Any]) -> nn.Module:
        """Build model with hyperparameters."""
        model_cfg = update_config_with_hparams(
            self.base_model_cfg, hparams, params_prefix="model"
        )
        return MultiPathMILRegressor(model_cfg)

    def train_fn(
        self,
        model: nn.Module,
        train_ids: List[str],
        val_ids: List[str],
        hparams: Dict[str, Any],
    ) -> Tuple[nn.Module, float, int]:
        """Train model with early stopping."""
        train_loader = self.create_dataloader(
            video_ids=train_ids, mode="train", hparams=hparams
        )
        val_loader = self.create_dataloader(
            video_ids=val_ids, mode="val", hparams=hparams
        )

        train_cfg = update_config_with_hparams(
            self.base_train_cfg, hparams, params_prefix="train"
        )

        model, best_val_loss = train_one_split(
            model, train_loader, val_loader, train_cfg, self.data_cfg.dataset_name
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
    ) -> nn.Module:
        """Train model for fixed epochs without validation."""
        train_loader = self.create_dataloader(train_ids, "train", hparams)

        # Create optimizer
        train_cfg = update_config_with_hparams(
            self.base_train_cfg, hparams, params_prefix="train"
        )
        optimizer = build_optimizer(model, train_cfg)

        return train_for_fixed_epochs(
            model, train_loader, num_epochs, optimizer, self.loss_fn, self.device
        )

    def evaluate_fn(
        self, model: nn.Module, data_ids: List[str], hparams: Dict[str, Any]
    ) -> float:
        """Evaluate model on data."""
        loader = self.create_dataloader(data_ids, "test", hparams)
        return evaluate_on_loader(model, loader, self.loss_fn, self.device)

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
    random_seed: int = 42,
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
        random_seed: Random seed
        device: Device to use

    Returns:
        Dictionary with nested CV results
    """
    # Default search space
    if search_space is None:
        print("Using default search space.")
        search_space = {
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "weight_decay": ("log_uniform", 1e-6, 1e-3),
        }
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create adapter
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
        device=adapter.device,
    )

    # Run nested CV
    results = engine.run(
        data_ids=video_ids,
        cv_cfg=cv_cfg,
        search_space=search_space,
        output_dir=output_dir,
        random_seed=random_seed,
        verbose=True,
    )

    return results
