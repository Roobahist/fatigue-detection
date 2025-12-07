"""
Generic nested cross-validation + hyperparameter search with a thin
adapter for the UTA/DROZY MIL regression setup.

Most of this file is project-agnostic. The only project-specific part is
`nested_cv_with_hparam_search` at the bottom, which wires:

- MultiPathMILRegressor
- RealVideoDataset
- collate_videos
- kfold_split_indices
- train_one_split

into the generic driver.
"""

import copy
import json
import os
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Zh.experiments.e12.dataclasses import *
from Zh.experiments.e12.datasets import *
from Zh.experiments.e12.helpers import *
from Zh.experiments.e12.model import *
from Zh.experiments.e12.paramsearch import *

# ---------------------------------------------------------------------------
# Training for fixed epochs (generic)
# ---------------------------------------------------------------------------

def train_for_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    train_cfg: Any,
    num_epochs: int,
    device: torch.device,
    batch_to_device_fn: Callable[[Any, torch.device], Any],
    target_extractor_fn: Callable[[Any, torch.device], torch.Tensor],
) -> nn.Module:
    """
    Generic finite-epoch training loop (no early stopping).

    Assumptions:
        - `train_loader` yields batches in arbitrary structure.
        - `batch_to_device_fn(batch, device)` moves batch to device.
        - `target_extractor_fn(batch, device)` returns a Tensor[B] of targets.

    `model(batch)` must produce predictions matching targets.
    """
    model.to(device)

    loss_fn = getattr(train_cfg, "loss_fn", None)
    if loss_fn is None:
        loss_fn = nn.MSELoss(reduction="mean")

    opt_name = getattr(train_cfg, "optimizer", "adam").lower()
    lr = getattr(train_cfg, "learning_rate", getattr(train_cfg, "lr", 1e-3))
    weight_decay = getattr(train_cfg, "weight_decay", 0.0)

    if opt_name == "sgd":
        momentum = getattr(train_cfg, "momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    for epoch in range(num_epochs):
        model.train()

        # If dataset has an epoch-scheduling hook, call it.
        ds = getattr(train_loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_training_epoch"):
            try:
                ds.set_training_epoch(epoch, num_epochs)
            except TypeError:
                # Signature mismatch: ignore
                pass

        for batch in train_loader:
            batch = batch_to_device_fn(batch, device)
            targets = target_extractor_fn(batch, device)
            preds = model(batch)

            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Inner CV for a given hyperparameter config (generic)
# ---------------------------------------------------------------------------

def run_inner_cv_for_hparams(
    outer_fold_idx: int,
    hparams: Dict[str, Any],
    outer_train_ids: List[Any],
    outer_test_loader: DataLoader,
    base_model_cfg: Any,
    base_train_cfg: Any,
    base_selection_cfg: Any,
    inner_folds: int,
    split_indices_fn: Callable[[int, int], List[Tuple[List[int], List[int]]]],
    build_model_fn: Callable[[Any, Dict[str, Any], torch.device], nn.Module],
    build_dataloader_fn: Callable[[List[Any], str, Any, Any, Any], DataLoader],
    train_one_split_fn: Callable[..., Tuple],
    dataset_name: str,
    device: torch.device,
    batch_to_device_fn: Callable[[Any, torch.device], Any],
    target_extractor_fn: Callable[[Any, torch.device], torch.Tensor],
) -> Tuple[float, List[Dict[str, Any]], int]:
    """
    Run inner K-fold CV for a single hyperparameter configuration.

    Returns:
        mean_inner_val_loss,
        list of per-inner-fold result dicts,
        chosen_num_epochs (int, based on best_epoch across folds or max_epochs).
    """
    n_inner = inner_folds
    n_outer_train = len(outer_train_ids)
    if n_inner <= 1:
        raise ValueError("inner_folds must be >= 2 for nested CV.")

    inner_splits = split_indices_fn(n_outer_train, n_inner)

    eval_loss_fn = nn.MSELoss(reduction="mean")
    inner_val_losses: List[float] = []
    best_epochs: List[int] = []
    inner_results: List[Dict[str, Any]] = []

    for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
        inner_train_ids = [outer_train_ids[i] for i in inner_train_idx]
        inner_val_ids = [outer_train_ids[i] for i in inner_val_idx]

        # Clone base configs and apply hparams
        model_cfg, train_cfg, selection_cfg = clone_configs(
            base_model_cfg, base_train_cfg, base_selection_cfg
        )
        apply_hparams_to_configs(model_cfg, train_cfg, selection_cfg, hparams)

        # Build loaders
        train_loader = build_dataloader_fn(
            inner_train_ids, "train", train_cfg, model_cfg, selection_cfg
        )
        val_loader = build_dataloader_fn(
            inner_val_ids, "val", train_cfg, model_cfg, selection_cfg
        )

        # Build model and run early-stopping training via user-provided function
        model = build_model_fn(model_cfg, hparams, device)
        result = train_one_split_fn(
            model,
            train_loader,
            val_loader,
            train_cfg,
            dataset_name=dataset_name,
        )

        # Accept both 2-tuple and 3-tuple returns for train_one_split
        if isinstance(result, tuple) and len(result) == 3:
            model, best_val_loss, best_epoch = result  # type: ignore[assignment]
        else:
            model, best_val_loss = result  # type: ignore[misc]
            best_epoch = getattr(train_cfg, "max_epochs", 0)

        model.to(device)

        inner_train_loss = evaluate_on_loader(
            model,
            train_loader,
            loss_fn=eval_loss_fn,
            device=device,
            batch_to_device_fn=batch_to_device_fn,
            target_extractor_fn=target_extractor_fn,
        )
        inner_val_loss = evaluate_on_loader(
            model,
            val_loader,
            loss_fn=eval_loss_fn,
            device=device,
            batch_to_device_fn=batch_to_device_fn,
            target_extractor_fn=target_extractor_fn,
        )
        outer_test_loss_from_inner = evaluate_on_loader(
            model,
            outer_test_loader,
            loss_fn=eval_loss_fn,
            device=device,
            batch_to_device_fn=batch_to_device_fn,
            target_extractor_fn=target_extractor_fn,
        )

        inner_val_losses.append(inner_val_loss)
        best_epochs.append(int(best_epoch) if best_epoch is not None else 0)

        inner_results.append(
            {
                "outer_fold": outer_fold_idx,
                "inner_fold": inner_idx,
                "hparams": copy.deepcopy(hparams),
                "inner_train_loss": float(inner_train_loss),
                "inner_val_loss": float(inner_val_loss),
                "outer_test_loss_from_inner_model": float(
                    outer_test_loss_from_inner
                ),
                "best_epoch": int(best_epoch) if best_epoch is not None else 0,
            }
        )

    mean_inner_val_loss = float(statistics.mean(inner_val_losses))

    # Choose number of epochs based on best_epoch if available, else max_epochs
    if all(e > 0 for e in best_epochs):
        chosen_num_epochs = int(round(statistics.mean(best_epochs)))
    else:
        chosen_num_epochs = int(getattr(base_train_cfg, "max_epochs", 1))

    chosen_num_epochs = max(1, chosen_num_epochs)
    return mean_inner_val_loss, inner_results, chosen_num_epochs


# ---------------------------------------------------------------------------
# Full nested CV + hyperparameter search (generic core)
# ---------------------------------------------------------------------------

def run_nested_cv_with_hparam_search_generic(
    video_ids: List[Any],
    base_model_cfg: Any,
    base_train_cfg: Any,
    base_selection_cfg: Any,
    outer_folds: int,
    inner_folds: int,
    search_strategy: str,
    search_space: Dict[str, Any],
    n_trials: int,
    split_indices_fn: Callable[[int, int], List[Tuple[List[int], List[int]]]],
    build_model_fn: Callable[[Any, Dict[str, Any], torch.device], nn.Module],
    build_dataloader_fn: Callable[[List[Any], str, Any, Any, Any], DataLoader],
    train_one_split_fn: Callable[..., Tuple],
    batch_to_device_fn: Callable[[Any, torch.device], Any],
    target_extractor_fn: Callable[[Any, torch.device], torch.Tensor],
    output_dir: str,
    dataset_name: str = "dataset",
) -> Dict[str, Any]:
    """
    Generic nested CV + hyperparameter search.

    All project-specific details are pushed into callables.

    Args:
        video_ids: identifiers for all samples.
        base_model_cfg, base_train_cfg, base_selection_cfg: base config objects.
        outer_folds, inner_folds: number of outer/inner folds.
        search_strategy: 'grid', 'random', or 'bayesian'.
        search_space: hyperparameter space.
        n_trials: number of trials (for random/bayesian).
        split_indices_fn: function (n_samples, n_splits) -> List[(train_idx, val_idx)].
        build_model_fn: function to construct model from (model_cfg, hparams, device).
        build_dataloader_fn: function to construct DataLoader from (ids, mode, train_cfg, model_cfg, selection_cfg).
        train_one_split_fn: early-stopping training function for inner CV.
        batch_to_device_fn: function to move batch to device.
        target_extractor_fn: function to extract targets from batch.
        output_dir: where to save outer models and JSON summary.
        dataset_name: passed to train_one_split_fn for logging.

    Returns:
        JSON-serializable results dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = infer_device_from_train_cfg(base_train_cfg)

    n_samples = len(video_ids)
    if outer_folds < 2 or outer_folds > n_samples:
        raise ValueError("outer_folds must be between 2 and len(video_ids).")
    if inner_folds < 2:
        raise ValueError("inner_folds must be >= 2.")

    outer_splits = split_indices_fn(n_samples, outer_folds)

    search_strategy = search_strategy.lower()
    if search_strategy not in ("grid", "random", "bayesian"):
        raise ValueError("search_strategy must be 'grid', 'random', or 'bayesian'.")

    outer_results: List[Dict[str, Any]] = []
    outer_test_losses_final: List[float] = []
    serialized_search_space = search_space

    for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        outer_train_ids = [video_ids[i] for i in outer_train_idx]
        outer_test_ids = [video_ids[i] for i in outer_test_idx]

        # Build outer-test loader using base configs
        m_test, t_test, s_test = clone_configs(
            base_model_cfg, base_train_cfg, base_selection_cfg
        )
        outer_test_loader = build_dataloader_fn(
            outer_test_ids, "test", t_test, m_test, s_test
        )

        inner_results_all: List[Dict[str, Any]] = []
        best_hparams: Dict[str, Any] = {}
        best_mean_val_loss = float("inf")
        best_chosen_epochs = int(getattr(base_train_cfg, "max_epochs", 1))

        # --- Search strategies ---
        if search_strategy == "grid":
            configs = generate_grid_configs(search_space)
            if not configs:
                configs = [{}]

            for hparams in configs:
                mean_val_loss, inner_results, chosen_epochs = run_inner_cv_for_hparams(
                    outer_fold_idx=outer_idx,
                    hparams=hparams,
                    outer_train_ids=outer_train_ids,
                    outer_test_loader=outer_test_loader,
                    base_model_cfg=base_model_cfg,
                    base_train_cfg=base_train_cfg,
                    base_selection_cfg=base_selection_cfg,
                    inner_folds=inner_folds,
                    split_indices_fn=split_indices_fn,
                    build_model_fn=build_model_fn,
                    build_dataloader_fn=build_dataloader_fn,
                    train_one_split_fn=train_one_split_fn,
                    dataset_name=dataset_name,
                    device=device,
                    batch_to_device_fn=batch_to_device_fn,
                    target_extractor_fn=target_extractor_fn,
                )
                inner_results_all.extend(inner_results)
                if mean_val_loss < best_mean_val_loss:
                    best_mean_val_loss = mean_val_loss
                    best_hparams = copy.deepcopy(hparams)
                    best_chosen_epochs = chosen_epochs

        elif search_strategy == "random":
            configs = generate_random_configs(
                search_space,
                n_trials=n_trials,
                seed=getattr(base_train_cfg, "seed", 42),
            )
            if not configs:
                configs = [{}]

            for hparams in configs:
                mean_val_loss, inner_results, chosen_epochs = run_inner_cv_for_hparams(
                    outer_fold_idx=outer_idx,
                    hparams=hparams,
                    outer_train_ids=outer_train_ids,
                    outer_test_loader=outer_test_loader,
                    base_model_cfg=base_model_cfg,
                    base_train_cfg=base_train_cfg,
                    base_selection_cfg=base_selection_cfg,
                    inner_folds=inner_folds,
                    split_indices_fn=split_indices_fn,
                    build_model_fn=build_model_fn,
                    build_dataloader_fn=build_dataloader_fn,
                    train_one_split_fn=train_one_split_fn,
                    dataset_name=dataset_name,
                    device=device,
                    batch_to_device_fn=batch_to_device_fn,
                    target_extractor_fn=target_extractor_fn,
                )
                inner_results_all.extend(inner_results)
                if mean_val_loss < best_mean_val_loss:
                    best_mean_val_loss = mean_val_loss
                    best_hparams = copy.deepcopy(hparams)
                    best_chosen_epochs = chosen_epochs

        else:  # bayesian
            try:
                import optuna
            except ImportError as e:
                raise ImportError(
                    "search_strategy='bayesian' requires Optuna to be installed."
                ) from e

            def objective(trial: "optuna.trial.Trial") -> float:
                nonlocal best_mean_val_loss, best_hparams, best_chosen_epochs, inner_results_all

                hparams = suggest_hparams_from_trial(trial, search_space)
                mean_val_loss, inner_results, chosen_epochs = run_inner_cv_for_hparams(
                    outer_fold_idx=outer_idx,
                    hparams=hparams,
                    outer_train_ids=outer_train_ids,
                    outer_test_loader=outer_test_loader,
                    base_model_cfg=base_model_cfg,
                    base_train_cfg=base_train_cfg,
                    base_selection_cfg=base_selection_cfg,
                    inner_folds=inner_folds,
                    split_indices_fn=split_indices_fn,
                    build_model_fn=build_model_fn,
                    build_dataloader_fn=build_dataloader_fn,
                    train_one_split_fn=train_one_split_fn,
                    dataset_name=dataset_name,
                    device=device,
                    batch_to_device_fn=batch_to_device_fn,
                    target_extractor_fn=target_extractor_fn,
                )
                inner_results_all.extend(inner_results)
                if mean_val_loss < best_mean_val_loss:
                    best_mean_val_loss = mean_val_loss
                    best_hparams = copy.deepcopy(hparams)
                    best_chosen_epochs = chosen_epochs
                return mean_val_loss

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # --- Retrain on full outer-train with best hparams ---
        best_model_cfg, best_train_cfg, best_selection_cfg = clone_configs(
            base_model_cfg, base_train_cfg, base_selection_cfg
        )
        apply_hparams_to_configs(
            best_model_cfg, best_train_cfg, best_selection_cfg, best_hparams
        )

        outer_train_loader = build_dataloader_fn(
            outer_train_ids, "train", best_train_cfg, best_model_cfg, best_selection_cfg
        )
        outer_test_loader_final = build_dataloader_fn(
            outer_test_ids, "test", best_train_cfg, best_model_cfg, best_selection_cfg
        )

        final_model = build_model_fn(best_model_cfg, best_hparams, device)
        final_model = train_for_fixed_epochs(
            final_model,
            outer_train_loader,
            best_train_cfg,
            num_epochs=best_chosen_epochs,
            device=device,
            batch_to_device_fn=batch_to_device_fn,
            target_extractor_fn=target_extractor_fn,
        )

        eval_loss_fn = nn.MSELoss(reduction="mean")
        outer_train_loss_final = evaluate_on_loader(
            final_model,
            outer_train_loader,
            loss_fn=eval_loss_fn,
            device=device,
            batch_to_device_fn=batch_to_device_fn,
            target_extractor_fn=target_extractor_fn,
        )
        outer_test_loss_final = evaluate_on_loader(
            final_model,
            outer_test_loader_final,
            loss_fn=eval_loss_fn,
            device=device,
            batch_to_device_fn=batch_to_device_fn,
            target_extractor_fn=target_extractor_fn,
        )

        outer_model_path = os.path.join(
            output_dir, f"outer{outer_idx}_final_model.pt"
        )
        torch.save(final_model.state_dict(), outer_model_path)

        outer_test_losses_final.append(float(outer_test_loss_final))

        outer_results.append(
            {
                "outer_fold": outer_idx,
                "best_hparams": copy.deepcopy(best_hparams),
                "chosen_num_epochs": int(best_chosen_epochs),
                "inner_results": inner_results_all,
                "outer_train_loss_final": float(outer_train_loss_final),
                "outer_test_loss_final": float(outer_test_loss_final),
                "outer_model_path": outer_model_path,
            }
        )

    mean_outer = (
        float(statistics.mean(outer_test_losses_final))
        if outer_test_losses_final
        else float("nan")
    )
    std_outer = (
        float(statistics.stdev(outer_test_losses_final))
        if len(outer_test_losses_final) > 1
        else 0.0
    )

    results: Dict[str, Any] = {
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "search_strategy": search_strategy,
        "search_space": serialized_search_space,
        "n_trials": n_trials,
        "outer_results": outer_results,
        "mean_outer_test_loss_final": mean_outer,
        "std_outer_test_loss_final": std_outer,
    }

    json_path = os.path.join(output_dir, "nested_cv_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Project-specific adapter: UTA/DROZY MIL regression
# ============================================================================

# Only this block is specific to your current codebase. To reuse this module
# for another project, you just replace this block and keep everything above.

from Zh.experiments.e12.dataclasses import (
    ModelConfig,
    SelectionConfig,
    TrainingConfig,
)
from Zh.experiments.e12.datasets import (
    RealVideoDataset,
    collate_videos,
)
from Zh.experiments.e12.helpers import (
    kfold_split_indices,
    train_one_split,
)
from Zh.experiments.e11.model import MultiPathMILRegressor 


def nested_cv_with_hparam_search(
    video_ids: List[str],
    model_cfg: ModelConfig,
    base_train_cfg: TrainingConfig,
    selection_cfg: SelectionConfig,
    eventfulness_scores_root: str,
    embeddings_root: str,
    features_root: str,
    outer_folds: int,
    inner_folds: int,
    output_dir: str,
    dataset_name: str = "UTA",
    converge_ratio: float = 1.0,
    search_strategy: str = "random",  # "grid", "random", "bayesian"
    search_space: Optional[Dict[str, Any]] = None,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """
    UTA/DROZY-specific wrapper around the generic nested CV driver.

    This function keeps your original desired signature while delegating
    all logic to the generic core through adapter callables.
    """
    if search_space is None:
        search_space = {}

    device = infer_device_from_train_cfg(base_train_cfg)

    # How to build a model for this project
    def build_model_fn(
        cfg: ModelConfig,
        hparams: Dict[str, Any],
        dev: torch.device,
    ) -> nn.Module:
        model = MultiPathMILRegressor(cfg)
        model.to(dev)
        return model

    # How to build dataloaders for this project
    def build_dataloader_fn(
        split_ids: List[str],
        mode: str,
        train_cfg: TrainingConfig,
        model_cfg_local: ModelConfig,
        selection_cfg_local: SelectionConfig,
    ) -> DataLoader:
        ds = RealVideoDataset(
            video_ids=split_ids,
            embeddings_root=embeddings_root,
            features_root=features_root,
            eventfulness_scores_root=eventfulness_scores_root,
            model_cfg=model_cfg_local,
            selection_cfg=selection_cfg_local,
            mode=mode,
            dataset_name=dataset_name,
            converge_ratio=converge_ratio,
        )

        batch_size = int(getattr(train_cfg, "batch_size", 4))
        num_workers = int(getattr(train_cfg, "num_workers", 4))
        pin_memory = bool(getattr(train_cfg, "pin_memory", True))
        shuffle = mode == "train"

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_videos,
        )

    # How to move your batches to device
    def batch_to_device_fn(batch: List[Dict[str, Any]], dev: torch.device):
        # batch is a list of video dicts
        for video in batch:
            for key in ("embeddings", "blendshapes", "others"):
                if key not in video:
                    continue
                tensors = []
                for t in video[key]:
                    if isinstance(t, torch.Tensor):
                        tensors.append(t.to(dev, non_blocking=True))
                    else:
                        tensors.append(
                            torch.tensor(t, dtype=torch.float32, device=dev)
                        )
                video[key] = tensors
        return batch

    # How to extract targets from your batches
    def target_extractor_fn(batch: List[Dict[str, Any]], dev: torch.device) -> torch.Tensor:
        return torch.tensor(
            [v["target"] for v in batch],
            dtype=torch.float32,
            device=dev,
        )

    results = run_nested_cv_with_hparam_search_generic(
        video_ids=video_ids,
        base_model_cfg=model_cfg,
        base_train_cfg=base_train_cfg,
        base_selection_cfg=selection_cfg,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        search_strategy=search_strategy,
        search_space=search_space,
        n_trials=n_trials,
        split_indices_fn=kfold_split_indices,
        build_model_fn=build_model_fn,
        build_dataloader_fn=build_dataloader_fn,
        train_one_split_fn=train_one_split,
        batch_to_device_fn=batch_to_device_fn,
        target_extractor_fn=target_extractor_fn,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    return results