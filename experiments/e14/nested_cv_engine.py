import copy
import itertools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Import the checkpoint manager
from Zh.experiments.e14.checkpoint_manager import CheckpointManager
from Zh.experiments.e14.dataclasses import (
    CrossValidationConfig,
    InnerFoldResult,
    OuterFoldResult,
)
from Zh.experiments.e14.helpers import *  # if you need helpers like split functions etc.
from Zh.src.utils.common import NumpyEncoder, save_obj
from Zh.src.utils.file_folder import create_folders
from Zh.src.utils.serialized_parallel import process

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ============================================================================
# Hyperparameter Search
# ============================================================================


def _default_save_model(model: nn.Module, path: str) -> None:
    """Default model saving function."""
    torch.save(model.state_dict(), path)


class HyperparameterSearcher:
    """
    Generic hyperparameter search supporting grid, random, and Bayesian strategies.
    Completely agnostic to the problem domain.
    """

    def __init__(
        self,
        search_strategy: str,
        search_space: Union[Dict, Callable],
        n_trials: int = 20,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            search_strategy: One of "grid", "random", "bayesian"
            search_space:
                - For grid: Dict[str, List] of discrete values
                - For random: Dict with distributions or callable returning dict
                - For bayesian: Callable accepting Optuna trial
            n_trials: Number of trials for random/bayesian
            random_seed: Random seed for reproducibility
        """
        self.search_strategy = search_strategy.lower()
        self.search_space = search_space
        self.n_trials = n_trials
        self.random_seed = random_seed

        if self.random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if self.search_strategy not in ["grid", "random", "bayesian"]:
            raise ValueError(f"Unknown search strategy: {search_strategy}")

        if self.search_strategy == "bayesian" and not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian search")

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate hyperparameter configurations for grid/random search."""
        if self.search_strategy == "grid":
            return self._generate_grid_configs()
        elif self.search_strategy == "random":
            return self._generate_random_configs()
        else:
            return []  # Bayesian generates dynamically

    def _generate_grid_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        if not isinstance(self.search_space, dict):
            raise ValueError("Grid search requires search_space to be a dict")

        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]

        configs = []
        for combination in itertools.product(*values):
            configs.append(dict(zip(keys, combination)))

        return configs

    def _generate_random_configs(self) -> List[Dict[str, Any]]:
        """Generate random configurations."""
        if callable(self.search_space):
            return [self.search_space() for _ in range(self.n_trials)]
        elif isinstance(self.search_space, dict):
            configs = []
            for _ in range(self.n_trials):
                config = {}
                for key, value_spec in self.search_space.items():
                    config[key] = self._sample_from_spec(value_spec)
                configs.append(config)
            return configs
        else:
            raise ValueError("Random search requires callable or dict search_space")

    def _sample_from_spec(self, spec: Any) -> Any:
        """Sample a value from a specification."""
        if isinstance(spec, list):
            return np.random.choice(spec)
        elif isinstance(spec, tuple) and len(spec) == 3:
            dist_type, low, high = spec
            if dist_type == "uniform":
                return float(np.random.uniform(low, high))
            elif dist_type == "log_uniform":
                return float(np.exp(np.random.uniform(np.log(low), np.log(high))))
            elif dist_type == "int_uniform":
                return int(np.random.randint(low, high + 1))
            else:
                raise ValueError(f"Unknown distribution: {dist_type}")
        else:
            return spec

    def suggest_from_trial(self, trial, key: str, value_spec: Any) -> Any:
        """Suggest parameter value using Optuna trial."""
        if isinstance(value_spec, list):
            return trial.suggest_categorical(key, value_spec)
        elif isinstance(value_spec, tuple) and len(value_spec) == 3:
            dist_type, low, high = value_spec
            if dist_type in ["uniform", "log_uniform"]:
                return trial.suggest_float(
                    key, low, high, log=(dist_type == "log_uniform")
                )
            elif dist_type == "int_uniform":
                return trial.suggest_int(key, low, high)
        return value_spec


# ============================================================================
# Generic Nested CV Engine with Checkpointing
# ============================================================================


class NestedCVEngine:
    """
    Completely generic Nested Cross-Validation Engine for Deep Learning.
    Now with full checkpoint/resume support (outer-fold and inner-fold).
    """

    def __init__(
        self,
        model_builder: Callable[[Dict[str, Any]], nn.Module],
        train_fn: Callable[
            [nn.Module, Any, Any, Dict[str, Any]], Tuple[nn.Module, float, int]
        ],
        train_fixed_epochs_fn: Callable[
            [nn.Module, Any, int, Dict[str, Any]], nn.Module
        ],
        evaluate_fn: Callable[[nn.Module, Any, Dict[str, Any], Any], float],
        split_fn: Callable[[List[Any], int, int], Tuple[List[Any], List[Any]]],
        device: torch.device,
        save_model_fn: Callable[[nn.Module, str], None] = None,
        parallel_inner_cv: bool = False,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the generic nested CV engine with checkpoint support.

        All problem specifics are injected via callbacks:
        - model_builder
        - train_fn
        - train_fixed_epochs_fn
        - evaluate_fn
        - split_fn
        """
        self.model_builder = model_builder
        self.train_fn = train_fn
        self.train_fixed_epochs_fn = train_fixed_epochs_fn
        self.evaluate_fn = evaluate_fn
        self.split_fn = split_fn
        self.save_model_fn = save_model_fn or _default_save_model
        self.device = device
        self.parallel_inner_cv = parallel_inner_cv
        self.max_workers = max_workers or 0

        print(f"NestedCVEngine initialized on device: {self.device}")
        if parallel_inner_cv:
            print(f"Parallel inner CV enabled with max_workers={self.max_workers}")

    def run(
        self,
        data_ids: List[Any],
        cv_cfg: CrossValidationConfig,
        search_space: Union[Dict, Callable],
        output_dir: str,
        random_seed: Optional[int] = None,
        verbose: bool = True,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run nested cross-validation with hyperparameter search and checkpointing.

        Args:
            data_ids: List of data identifiers
            cv_cfg: CrossValidationConfig object
            search_space: Hyperparameter search space
            output_dir: Directory to save results and checkpoints
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
            resume: If True, attempt to resume from previous checkpoints

        Returns:
            Dictionary with complete nested CV results.
        """
        create_folders(output_dir)

        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(output_dir)

        # Check for existing checkpoints
        completed_folds: List[int] = []
        start_fold = 0

        if resume and checkpoint_mgr.can_resume():
            main_checkpoint = checkpoint_mgr.load_main_checkpoint()
            if main_checkpoint:
                completed_folds = main_checkpoint.get("completed_folds", [])
                start_fold = len(completed_folds)

                if verbose:
                    print(f"\n{'=' * 80}")
                    print("RESUMING FROM CHECKPOINT")
                    print(f"Completed folds: {completed_folds}")
                    print(f"Starting from fold: {start_fold}")
                    print(f"{'=' * 80}\n")

        outer_folds = cv_cfg.outer_folds
        inner_folds = cv_cfg.inner_folds
        n_trials = cv_cfg.n_trials
        search_strategy = cv_cfg.search_strategy

        searcher = HyperparameterSearcher(
            search_strategy=search_strategy,
            search_space=search_space,
            n_trials=n_trials,
            random_seed=random_seed,
        )

        outer_results: List[OuterFoldResult] = []
        valid_completed: List[int] = []

        # Load results from completed folds
        for fold_idx in completed_folds:
            fold_checkpoint = checkpoint_mgr.load_outer_fold_checkpoint(fold_idx)
            if fold_checkpoint and checkpoint_mgr.is_outer_fold_complete(fold_idx):
                outer_result = self._reconstruct_outer_result(
                    fold_idx, fold_checkpoint, output_dir
                )
                outer_results.append(outer_result)
                valid_completed.append(fold_idx)

                if verbose:
                    print(
                        f"Loaded completed fold {fold_idx}: "
                        f"test_metric={outer_result.outer_test_metric:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Skipping fold {fold_idx} (checkpoint incomplete or missing)"
                    )

        completed_folds = valid_completed
        start_fold = len(completed_folds)

        # Outer loop - start from where we left off
        for outer_idx in range(start_fold, outer_folds):
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"OUTER FOLD {outer_idx + 1}/{outer_folds}")
                print(f"{'=' * 80}")

            # Update main checkpoint to indicate current fold
            checkpoint_mgr.save_main_checkpoint(
                completed_folds=completed_folds,
                total_folds=outer_folds,
                current_fold=outer_idx,
            )

            # Split data
            outer_train_ids, outer_test_ids = self.split_fn(
                data_ids, outer_folds, outer_idx
            )

            # Check if we have a partial checkpoint for this fold
            fold_checkpoint = checkpoint_mgr.load_outer_fold_checkpoint(outer_idx)

            # Hyperparameter search (with resume support)
            best_hparams, chosen_epochs, inner_results = self._hyperparameter_search(
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,
                inner_folds=inner_folds,
                searcher=searcher,
                verbose=verbose,
                checkpoint_mgr=checkpoint_mgr,
                outer_idx=outer_idx,
                fold_checkpoint=fold_checkpoint,
            )

            if verbose:
                print(
                    f"\nBest hyperparameters: {json.dumps(best_hparams, indent=2, cls=NumpyEncoder)}"
                )
                print(f"Chosen epochs: {chosen_epochs}")

            # Save checkpoint after hyperparameter search
            checkpoint_mgr.save_outer_fold_checkpoint(
                outer_idx=outer_idx,
                best_hparams=best_hparams,
                chosen_epochs=chosen_epochs,
                inner_results=inner_results,
            )

            # Retrain on full outer train
            if verbose:
                print(
                    f"\nRetraining on full outer training set for {chosen_epochs} epochs..."
                )

            final_model = self._retrain_on_full_outer_train(
                outer_train_ids=outer_train_ids,
                hparams=best_hparams,
                num_epochs=chosen_epochs,
            )

            # Evaluate final model
            outer_train_metric = self.evaluate_fn(
                final_model, outer_train_ids, best_hparams, outer_train_ids
            )
            outer_test_metric = self.evaluate_fn(
                final_model, outer_test_ids, best_hparams, outer_train_ids
            )

            # Save model
            model_path = os.path.join(output_dir, f"outer_fold_{outer_idx}_model.pt")
            self.save_model_fn(final_model, model_path)

            if verbose:
                print(f"Final outer train metric: {outer_train_metric:.4f}")
                print(f"Final outer test metric: {outer_test_metric:.4f}")
                print(f"Model saved to: {model_path}")

            # Save complete checkpoint with final metrics
            checkpoint_mgr.save_outer_fold_checkpoint(
                outer_idx=outer_idx,
                best_hparams=best_hparams,
                chosen_epochs=chosen_epochs,
                inner_results=inner_results,
                outer_train_metric=outer_train_metric,
                outer_test_metric=outer_test_metric,
            )

            # Store results
            outer_result = OuterFoldResult(
                outer_fold=outer_idx,
                best_hparams=best_hparams,
                chosen_num_epochs=chosen_epochs,
                inner_results=inner_results,
                outer_train_metric=outer_train_metric,
                outer_test_metric=outer_test_metric,
                model_path=model_path,
            )
            outer_results.append(outer_result)

            # Mark this fold as completed
            completed_folds.append(outer_idx)
            checkpoint_mgr.save_main_checkpoint(
                completed_folds=completed_folds,
                total_folds=outer_folds,
                current_fold=None,
            )

        # Aggregate and save results
        final_results = self._aggregate_results(
            outer_results, outer_folds, inner_folds, search_strategy, search_space
        )

        results_path = os.path.join(output_dir, "nested_cv_results.json")
        save_obj(final_results, results_path, file_type="json")

        if verbose:
            print(f"\n{'=' * 80}")
            print("NESTED CV COMPLETE")
            print(f"{'=' * 80}")
            print(
                f"Mean outer test metric: "
                f"{final_results['mean_outer_test_metric']:.4f}"
            )
            print(
                f"Std outer test metric: "
                f"{final_results['std_outer_test_metric']:.4f}"
            )
            print(f"Results saved to: {results_path}")

        return final_results

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _reconstruct_outer_result(
        self,
        outer_idx: int,
        fold_checkpoint: Dict,
        output_dir: str,
    ) -> OuterFoldResult:
        """Reconstruct OuterFoldResult from a saved checkpoint."""
        model_path = os.path.join(output_dir, f"outer_fold_{outer_idx}_model.pt")

        return OuterFoldResult(
            outer_fold=outer_idx,
            best_hparams=fold_checkpoint["best_hparams"],
            chosen_num_epochs=fold_checkpoint["chosen_epochs"],
            inner_results=fold_checkpoint.get("inner_results", []),
            outer_train_metric=fold_checkpoint.get("outer_train_metric"),
            outer_test_metric=fold_checkpoint.get("outer_test_metric"),
            model_path=model_path,
        )

    def _hyperparameter_search(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        inner_folds: int,
        searcher: HyperparameterSearcher,
        verbose: bool,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        outer_idx: Optional[int] = None,
        fold_checkpoint: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], int, List[InnerFoldResult]]:
        """Dispatch to grid/random or Bayesian search with resume support."""
        if searcher.search_strategy == "bayesian":
            return self._bayesian_search(
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,
                inner_folds=inner_folds,
                searcher=searcher,
                verbose=verbose,
                checkpoint_mgr=checkpoint_mgr,
                outer_idx=outer_idx,
                fold_checkpoint=fold_checkpoint,
            )
        else:
            return self._grid_or_random_search(
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,
                inner_folds=inner_folds,
                searcher=searcher,
                verbose=verbose,
                checkpoint_mgr=checkpoint_mgr,
                outer_idx=outer_idx,
                fold_checkpoint=fold_checkpoint,
            )

    def _grid_or_random_search(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        inner_folds: int,
        searcher: HyperparameterSearcher,
        verbose: bool,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        outer_idx: Optional[int] = None,
        fold_checkpoint: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], int, List[InnerFoldResult]]:
        """
        Grid or random hyperparameter search with:
        - Outer-fold resume
        - Config-level resume
        - Inner-fold resume per config
        """
        configs = searcher.generate_configs()

        # Load completed configs (if any)
        completed_configs: List[int] = []
        config_results: List[Dict[str, Any]] = []

        if fold_checkpoint and "config_results" in fold_checkpoint:
            config_results = fold_checkpoint["config_results"]
            completed_configs = [r["config_idx"] for r in config_results]

            if verbose:
                print(
                    "Resuming hyperparameter search: "
                    f"{len(completed_configs)}/{len(configs)} configs completed"
                )

        if verbose:
            print(f"\nEvaluating {len(configs)} hyperparameter configurations...")
            if self.parallel_inner_cv:
                print("Running inner CV in parallel mode...")
            else:
                print("Running inner CV in serial mode...")

        # Prepare worker parameters only for incomplete configs
        worker_params_list = [
            {
                "outer_train_ids": outer_train_ids,
                "outer_test_ids": outer_test_ids,
                "config": config,
                "inner_folds": inner_folds,
                "config_idx": config_idx,
                "checkpoint_mgr": checkpoint_mgr,
                "outer_idx": outer_idx,
            }
            for config_idx, config in enumerate(configs)
            if config_idx not in completed_configs
        ]

        # Process remaining configs
        if worker_params_list:
            new_results = process(
                worker_function=self._evaluate_single_config_worker,
                worker_params_list=worker_params_list,
                parallel=self.parallel_inner_cv,
                max_workers=self.max_workers,
                use_tqdm=verbose,
                auto_slice=False,
            )
            config_results.extend(new_results)

            # Save checkpoint after processing configs
            if checkpoint_mgr and outer_idx is not None:
                checkpoint_mgr.save_outer_fold_checkpoint(
                    outer_idx=outer_idx,
                    config_results=config_results,
                )

        # Find best configuration
        best_hparams: Optional[Dict[str, Any]] = None
        best_mean_val_metric = float("inf")
        best_mean_epoch = 0
        all_inner_results: List[InnerFoldResult] = []

        for result in config_results:
            config_idx = result["config_idx"]
            inner_results = result["inner_results"]
            mean_val_metric = result["mean_val_metric"]
            mean_epoch = result["mean_epoch"]

            all_inner_results.extend(inner_results)

            if mean_val_metric < best_mean_val_metric:
                best_mean_val_metric = mean_val_metric
                best_hparams = configs[config_idx]
                best_mean_epoch = mean_epoch

        if best_hparams is None:
            raise RuntimeError("No valid hyperparameter configuration evaluated.")

        if verbose:
            print(
                f"\nBest config: {configs.index(best_hparams) + 1}/{len(configs)} "
                f"(mean val metric = {best_mean_val_metric:.4f})"
            )

        return best_hparams, best_mean_epoch, all_inner_results

    def _evaluate_single_config_worker(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        config: Dict[str, Any],
        inner_folds: int,
        config_idx: int,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        outer_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Worker function to evaluate a single config WITH inner-fold checkpointing.

        This is run in a separate process when using parallel mode.
        """
        # Check for partial progress for this config
        completed_inner_results: List[InnerFoldResult] = []
        start_inner_fold = 0

        if checkpoint_mgr and outer_idx is not None:
            partial = checkpoint_mgr.load_partial_config(outer_idx, config_idx)
            if partial:
                completed_inner_results = [
                    InnerFoldResult.from_dict(inner_result)
                    for inner_result in partial["inner_results"]
                ]
                start_inner_fold = len(completed_inner_results)
                print(
                    f"  [Config {config_idx}] Resuming from inner fold "
                    f"{start_inner_fold}/{inner_folds}"
                )

        # Run inner CV starting from where we left off
        inner_results = self._run_inner_cv_with_checkpointing(
            outer_train_ids=outer_train_ids,
            outer_test_ids=outer_test_ids,
            hparams=config,
            inner_folds=inner_folds,
            start_fold=start_inner_fold,
            completed_results=completed_inner_results,
            checkpoint_mgr=checkpoint_mgr,
            outer_idx=outer_idx,
            config_idx=config_idx,
            verbose=True,
        )
        # Calculate metrics
        mean_val_metric = np.mean([r.inner_val_metric for r in inner_results])
        mean_epoch = int(np.round(np.mean([r.best_epoch for r in inner_results])))

        return {
            "config_idx": config_idx,
            "config": config,
            "inner_results": inner_results,
            "mean_val_metric": mean_val_metric,
            "mean_epoch": mean_epoch,
        }

    def _bayesian_search(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        inner_folds: int,
        searcher: HyperparameterSearcher,
        verbose: bool,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        outer_idx: Optional[int] = None,
        fold_checkpoint: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], int, List[InnerFoldResult]]:
        """
        Bayesian hyperparameter search with checkpoint support (Optuna).
        Resume is at the trial level (via Optuna + search_state), not at inner-fold level.
        """
        if self.parallel_inner_cv:
            print("Warning: Parallel inner CV not supported for Bayesian search.")

        all_inner_results: List[InnerFoldResult] = []
        best_hparams_per_trial: Dict[int, Tuple[Dict[str, Any], int]] = {}
        completed_trials = 0

        # Check if we can resume Bayesian search state
        if fold_checkpoint and "search_state" in fold_checkpoint:
            state = fold_checkpoint["search_state"]
            all_inner_results = state.get("inner_results", [])
            best_hparams_per_trial = state.get("best_hparams_per_trial", {})
            completed_trials = state.get("completed_trials", 0)

            if verbose:
                print(
                    "Resuming Bayesian search: "
                    f"{completed_trials}/{searcher.n_trials} trials completed"
                )

        def objective(trial):
            # Generate hyperparameters
            if callable(searcher.search_space):
                hparams = searcher.search_space(trial)
            else:
                hparams = {
                    k: searcher.suggest_from_trial(trial, k, v)
                    for k, v in searcher.search_space.items()
                }

            if verbose:
                print(f"\n--- Trial {trial.number} ---")
                print(json.dumps(hparams, indent=2, cls=NumpyEncoder))

            # Run inner CV
            inner_results = self._run_inner_cv(
                outer_train_ids,
                outer_test_ids,
                hparams,
                inner_folds,
                verbose,
            )

            all_inner_results.extend(inner_results)
            mean_val_metric = np.mean([r.inner_val_metric for r in inner_results])
            mean_epoch = int(np.round(np.mean([r.best_epoch for r in inner_results])))

            best_hparams_per_trial[trial.number] = (hparams, mean_epoch)

            # Save checkpoint after each trial
            if checkpoint_mgr and outer_idx is not None:
                search_state = {
                    "inner_results": all_inner_results,
                    "best_hparams_per_trial": best_hparams_per_trial,
                    "completed_trials": trial.number + 1,
                }
                checkpoint_mgr.save_outer_fold_checkpoint(
                    outer_idx=outer_idx,
                    search_state=search_state,
                )

            return mean_val_metric

        sampler = optuna.samplers.TPESampler(seed=searcher.random_seed)

        # Create or load study with persistence
        study_name = f"outer_fold_{outer_idx}" if outer_idx is not None else "study"
        storage = None
        if checkpoint_mgr:
            storage = f"sqlite:///{checkpoint_mgr.output_dir}/optuna_study.db"

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,  # Resume if study exists
        )

        # Continue optimization from where we left off
        remaining_trials = searcher.n_trials - len(study.trials)
        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                show_progress_bar=verbose,
            )

        best_trial_number = study.best_trial.number
        best_hparams, best_mean_epoch = best_hparams_per_trial[best_trial_number]

        return best_hparams, best_mean_epoch, all_inner_results

    def _run_inner_cv_with_checkpointing(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        hparams: Dict[str, Any],
        inner_folds: int,
        start_fold: int = 0,
        completed_results: Optional[List[InnerFoldResult]] = None,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        outer_idx: Optional[int] = None,
        config_idx: Optional[int] = None,
        verbose: bool = True,
    ) -> List[InnerFoldResult]:
        """
        Run inner CV with checkpointing after each inner fold.

        Args:
            start_fold: Which inner fold to start from (for resume)
            completed_results: Already completed inner fold results
        """
        inner_results: List[InnerFoldResult] = (
            list(completed_results) if completed_results else []
        )

        for inner_idx in range(start_fold, inner_folds):
            if verbose:
                print(f"\n{'=-' * 40}")
                print(f"Inner fold {inner_idx + 1}/{inner_folds}")
                print(f"{'=-' * 40}")

            # Split outer_train into inner_train and inner_val
            inner_train_ids, inner_val_ids = self.split_fn(
                outer_train_ids, inner_folds, inner_idx
            )

            # Create and train model
            model = self.model_builder(hparams)
            model = model.to(self.device)

            model, best_val_metric, best_epoch = self.train_fn(
                model, inner_train_ids, inner_val_ids, hparams
            )

            if verbose:
                print(
                    f"| Best Val Metric: {best_val_metric:.4f} "
                    f"at Epoch: {best_epoch}",
                    end=" ",
                )

            # Evaluate on all sets
            train_metric = self.evaluate_fn(
                model, inner_train_ids, hparams, inner_train_ids
            )
            val_metric = self.evaluate_fn(
                model, inner_val_ids, hparams, inner_train_ids
            )
            outer_test_metric = self.evaluate_fn(
                model, outer_test_ids, hparams, inner_train_ids
            )

            if verbose:
                print(
                    f"| Train: {train_metric:.4f}, "
                    f"Val: {val_metric:.4f}, "
                    f"Outer Test: {outer_test_metric:.4f}"
                )

            result = InnerFoldResult(
                inner_fold=inner_idx,
                hparams=copy.deepcopy(hparams),
                inner_train_metric=train_metric,
                inner_val_metric=val_metric,
                outer_test_metric=outer_test_metric,
                best_epoch=best_epoch,
            )
            inner_results.append(result)

            # CHECKPOINT: Save progress after each inner fold
            if checkpoint_mgr and outer_idx is not None and config_idx is not None:
                checkpoint_mgr.save_partial_config(
                    outer_idx=outer_idx,
                    config_idx=config_idx,
                    config=hparams,
                    inner_results=inner_results,
                )

        return inner_results

    def _run_inner_cv(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        hparams: Dict[str, Any],
        inner_folds: int,
        verbose: bool = True,
    ) -> List[InnerFoldResult]:
        """Run inner cross-validation for a hyperparameter configuration (no checkpointing)."""
        inner_results: List[InnerFoldResult] = []

        for inner_idx in range(inner_folds):
            if verbose:
                print(f"\n{'=-' * 40}")
                print(f"Inner fold {inner_idx + 1}/{inner_folds}")
                print(f"{'=-' * 40}")

            # Split outer_train into inner_train and inner_val
            inner_train_ids, inner_val_ids = self.split_fn(
                outer_train_ids, inner_folds, inner_idx
            )

            # Create and train model
            model = self.model_builder(hparams)
            model = model.to(self.device)

            model, best_val_metric, best_epoch = self.train_fn(
                model, inner_train_ids, inner_val_ids, hparams
            )

            if verbose:
                print(
                    f"| Best Val Metric: {best_val_metric:.4f} "
                    f"at Epoch: {best_epoch}",
                    end=" ",
                )

            # Evaluate on all sets
            train_metric = self.evaluate_fn(
                model, inner_train_ids, hparams, inner_train_ids
            )
            val_metric = self.evaluate_fn(
                model, inner_val_ids, hparams, inner_train_ids
            )
            outer_test_metric = self.evaluate_fn(
                model, outer_test_ids, hparams, inner_train_ids
            )

            if verbose:
                print(
                    f"| Train: {train_metric:.4f}, "
                    f"Val: {val_metric:.4f}, "
                    f"Outer Test: {outer_test_metric:.4f}"
                )

            result = InnerFoldResult(
                inner_fold=inner_idx,
                hparams=copy.deepcopy(hparams),
                inner_train_metric=train_metric,
                inner_val_metric=val_metric,
                outer_test_metric=outer_test_metric,
                best_epoch=best_epoch,
            )
            inner_results.append(result)

        return inner_results

    def _retrain_on_full_outer_train(
        self,
        outer_train_ids: List[Any],
        hparams: Dict[str, Any],
        num_epochs: int,
    ) -> nn.Module:
        """Retrain model on full outer training set for a fixed number of epochs."""
        model = self.model_builder(hparams)
        model = model.to(self.device)

        model = self.train_fixed_epochs_fn(model, outer_train_ids, num_epochs, hparams)

        return model

    def _aggregate_results(
        self,
        outer_results: List[OuterFoldResult],
        outer_folds: int,
        inner_folds: int,
        search_strategy: str,
        search_space: Union[Dict, Callable],
    ) -> Dict[str, Any]:
        """Aggregate all results into a final nested CV summary dictionary."""
        outer_test_metrics = [r.outer_test_metric for r in outer_results]

        search_space_serialized: Union[str, Dict] = (
            "callable" if callable(search_space) else search_space
        )

        return {
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "search_strategy": search_strategy,
            "search_space": search_space_serialized,
            "outer_results": [
                {
                    "outer_fold": r.outer_fold,
                    "best_hparams": r.best_hparams,
                    "chosen_num_epochs": r.chosen_num_epochs,
                    "inner_results": [
                        {
                            "inner_fold": ir.inner_fold,
                            "hparams": ir.hparams,
                            "inner_train_metric": ir.inner_train_metric,
                            "inner_val_metric": ir.inner_val_metric,
                            "outer_test_metric": ir.outer_test_metric,
                            "best_epoch": ir.best_epoch,
                        }
                        for ir in r.inner_results
                    ],
                    "outer_train_metric": r.outer_train_metric,
                    "outer_test_metric": r.outer_test_metric,
                    "model_path": r.model_path,
                }
                for r in outer_results
            ],
            "mean_outer_test_metric": float(np.mean(outer_test_metrics)),
            "std_outer_test_metric": float(np.std(outer_test_metrics)),
            "all_outer_test_metrics": [float(m) for m in outer_test_metrics],
        }
