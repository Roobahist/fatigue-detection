import copy
import itertools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from Zh.experiments.e13.dataclasses import *
from Zh.experiments.e13.helpers import *
from Zh.src.utils.common import save_obj
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
# Generic Nested CV Engine
# ============================================================================


class NestedCVEngine:
    """
    Completely generic Nested Cross-Validation Engine for Deep Learning.

    This class is 100% agnostic to:
    - Model architecture
    - Data format
    - Dataset implementation
    - Training procedure
    - Evaluation metrics

    All specifics are injected via callback functions.
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
        evaluate_fn: Callable[[nn.Module, Any, Dict[str, Any]], float],
        split_fn: Callable[[List[Any], int, int], Tuple[List[Any], List[Any]]],
        device: torch.device,
        save_model_fn: Callable[[nn.Module, str], None] = None,
        parallel_inner_cv: bool = False,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the generic nested CV engine.

        Args:
            model_builder: Function(hparams) -> model
                Creates a fresh model instance with given hyperparameters.

            train_fn: Function(model, train_data, val_data, hparams) -> (model, best_val_metric, best_epoch)
                Trains model with early stopping and returns the best model state.

            train_fixed_epochs_fn: Function(model, train_data, num_epochs, hparams) -> model
                Trains model for exactly num_epochs without validation.

            evaluate_fn: Function(model, data, hparams) -> metric
                Evaluates model on data and returns a scalar metric (lower is better).

            split_fn: Function(data_ids, num_folds, fold_idx) -> (train_ids, test_ids)
                Splits data into train and test for a given fold.

            save_model_fn: Function(model, path) -> None
                Saves model to disk. If None, uses torch.save(model.state_dict(), path).

            device: Device to use (auto-detect if None)

            parallel_inner_cv: Whether to parallelize inner CV folds (only for grid/random search)

            max_workers: Maximum number of parallel workers for inner CV (None = auto-detect from SLURM)
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
    ) -> Dict[str, Any]:
        """
        Run nested cross-validation with hyperparameter search.

        Args:
            data_ids: List of data identifiers (can be any type)
            cv_cfg: CrossValidationConfig object containing CV settings
            search_space: Hyperparameter search space specification
            output_dir: Directory to save results and models
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress

        Returns:
            Dictionary with complete nested CV results
        """
        create_folders(output_dir)

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

        outer_results = []

        # Outer loop
        for outer_idx in range(outer_folds):
            if verbose:
                print(f"\n{'='*80}")
                print(f"OUTER FOLD {outer_idx + 1}/{outer_folds}")
                print(f"{'='*80}")

            # Split data
            outer_train_ids, outer_test_ids = self.split_fn(
                data_ids, outer_folds, outer_idx
            )

            # Hyperparameter search
            best_hparams, chosen_epochs, inner_results = self._hyperparameter_search(
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,  # to check the performance of the model trained on inner train set on outer test set without retraining using both of the train and validation set
                inner_folds=inner_folds,
                searcher=searcher,
                verbose=verbose,
            )

            if verbose:
                print(
                    f"\nBest hyperparameters: {json.dumps(best_hparams, indent=2, cls=NumpyEncoder)}"
                )
                print(f"Chosen epochs: {chosen_epochs}")

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

        # Aggregate and save results
        final_results = self._aggregate_results(
            outer_results, outer_folds, inner_folds, search_strategy, search_space
        )

        results_path = os.path.join(output_dir, "nested_cv_results.json")
        save_obj(final_results, results_path, file_type="json")

        if verbose:
            print(f"\n{'='*80}")
            print(f"NESTED CV COMPLETE")
            print(f"{'='*80}")
            print(
                f"Mean outer test metric: {final_results['mean_outer_test_metric']:.4f}"
            )
            print(
                f"Std outer test metric: {final_results['std_outer_test_metric']:.4f}"
            )
            print(f"Results saved to: {results_path}")

        return final_results

    def _hyperparameter_search(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        inner_folds: int,
        searcher: HyperparameterSearcher,
        verbose: bool,
    ) -> Tuple[Dict[str, Any], int, List[InnerFoldResult]]:
        """Perform hyperparameter search using inner CV."""
        if searcher.search_strategy == "bayesian":
            return self._bayesian_search(
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,
                inner_folds=inner_folds,
                searcher=searcher,
                verbose=verbose,
            )
        else:
            return self._grid_or_random_search(
                outer_train_ids=outer_train_ids,
                outer_test_ids=outer_test_ids,
                inner_folds=inner_folds,
                searcher=searcher,
                verbose=verbose,
            )

    def _grid_or_random_search(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        inner_folds: int,
        searcher: HyperparameterSearcher,
        verbose: bool,
    ) -> Tuple[Dict[str, Any], int, List[InnerFoldResult]]:
        """Grid or random hyperparameter search with optional parallelization."""
        configs = searcher.generate_configs()

        if verbose:
            print(f"\nEvaluating {len(configs)} hyperparameter configurations...")
            if self.parallel_inner_cv:
                print(
                    f"Running inner CV in parallel mode with {self.max_workers or 'auto'} workers..."
                )
            else:
                print("Running inner CV in serial mode...")

        # Prepare worker parameters for each config
        worker_params_list = [
            {
                "outer_train_ids": outer_train_ids,
                "outer_test_ids": outer_test_ids,
                "config": config,
                "inner_folds": inner_folds,
                "config_idx": config_idx,
            }
            for config_idx, config in enumerate(configs)
        ]

        # Process all configs (parallel or serial based on self.parallel_inner_cv)
        config_results = process(
            worker_function=self._evaluate_single_config_worker,
            worker_params_list=worker_params_list,
            parallel=self.parallel_inner_cv,
            max_workers=self.max_workers,
            use_tqdm=verbose,
            auto_slice=False,  # Don't use SLURM slicing for inner CV
        )

        # Find best configuration
        best_hparams = None
        best_mean_val_metric = float("inf")
        best_mean_epoch = 0
        all_inner_results = []

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

        if verbose:
            print(f"\nBest config: {configs.index(best_hparams) + 1}/{len(configs)}")
            print(f"Best mean validation metric: {best_mean_val_metric:.4f}")

        return best_hparams, best_mean_epoch, all_inner_results

    def _evaluate_single_config_worker(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        config: Dict[str, Any],
        inner_folds: int,
        config_idx: int,
    ) -> Dict[str, Any]:
        """
        Worker function to evaluate a single hyperparameter configuration.
        This runs in a separate process when using parallel mode.
        """
        # Run inner CV for this configuration
        inner_results = self._run_inner_cv(
            outer_train_ids, outer_test_ids, config, inner_folds, verbose=True
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
    ) -> Tuple[Dict[str, Any], int, List[InnerFoldResult]]:
        """Bayesian hyperparameter search using Optuna (sequential only)."""
        if self.parallel_inner_cv:
            print(
                "Warning: Parallel inner CV is not supported for Bayesian search. Running sequentially."
            )

        all_inner_results = []
        best_hparams_per_trial = {}

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

            return mean_val_metric

        sampler = optuna.samplers.TPESampler(seed=searcher.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=searcher.n_trials, show_progress_bar=verbose)

        best_trial_number = study.best_trial.number
        best_hparams, best_mean_epoch = best_hparams_per_trial[best_trial_number]

        return best_hparams, best_mean_epoch, all_inner_results

    def _run_inner_cv(
        self,
        outer_train_ids: List[Any],
        outer_test_ids: List[Any],
        hparams: Dict[str, Any],
        inner_folds: int,
        verbose: bool = True,
    ) -> List[InnerFoldResult]:
        """Run inner cross-validation for a hyperparameter configuration."""
        inner_results = []

        for inner_idx in range(inner_folds):
            if verbose:
                print(f"\n{'=-'*40}")
                print(f"Inner fold {inner_idx + 1}/{inner_folds}")
                print(f"{'=-'*40}")

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
                    f"| Best Val Metric: {best_val_metric:.4f} at Epoch: {best_epoch}",
                    end=" ",
                )

            # Evaluate on all sets
            train_metric = self.evaluate_fn(model, inner_train_ids, hparams, inner_train_ids)
            val_metric = self.evaluate_fn(model, inner_val_ids, hparams, inner_train_ids)
            outer_test_metric = self.evaluate_fn(model, outer_test_ids, hparams, inner_train_ids)

            if verbose:
                print(
                    f"| Train: {train_metric:.4f}, Val: {val_metric:.4f}, "
                    f"Outer Test: {outer_test_metric:.4f}, Epoch: {best_epoch}"
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
        self, outer_train_ids: List[Any], hparams: Dict[str, Any], num_epochs: int
    ) -> nn.Module:
        """Retrain model on full outer training set for fixed epochs."""
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
        """Aggregate all results into final dictionary."""
        outer_test_metrics = [r.outer_test_metric for r in outer_results]

        search_space_serialized = "callable" if callable(search_space) else search_space

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


# ============================================================================
# Example: Adapter for specific use case
# ============================================================================


# def create_adapter_example():

#     def model_builder(hparams):
#         """Build model with hyperparameters."""
#         # return MultiPathMILRegressor(create_config_from_hparams(hparams))
#         pass

#     def train_fn(model, train_ids, val_ids, hparams):
#         """Train with early stopping."""
#         # train_loader = create_dataloader(train_ids, "train", hparams)
#         # val_loader = create_dataloader(val_ids, "val", hparams)
#         # model, best_val_loss = your_train_one_split(model, train_loader, val_loader, ...)
#         # return model, best_val_loss, model.best_epoch_
#         pass

#     def train_fixed_epochs_fn(model, train_ids, num_epochs, hparams):
#         """Train for fixed epochs without validation."""
#         # train_loader = create_dataloader(train_ids, "train", hparams)
#         # return train_for_fixed_epochs(model, train_loader, num_epochs, ...)
#         pass

#     def evaluate_fn(model, data_ids, hparams):
#         """Evaluate model on data."""
#         # loader = create_dataloader(data_ids, "test", hparams)
#         # return evaluate_on_loader(model, loader, criterion, device)
#         pass

#     def split_fn(data_ids, num_folds, fold_idx):
#         """Split data into train and test."""
#         # train_indices, test_indices = kfold_split_indices(len(data_ids), num_folds, fold_idx)
#         # return [data_ids[i] for i in train_indices], [data_ids[i] for i in test_indices]
#         pass

#     # Create engine with your adapters
#     engine = NestedCVEngine(
#         model_builder=model_builder,
#         train_fn=train_fn,
#         train_fixed_epochs_fn=train_fixed_epochs_fn,
#         evaluate_fn=evaluate_fn,
#         split_fn=split_fn,
#         parallel_inner_cv=True,  # Enable parallel inner CV
#         max_workers=4,  # Or None for auto-detection from SLURM
#     )

#     # Run nested CV
#     results = engine.run(
#         data_ids=["video1", "video2", "..."],  # Your data IDs
#         cv_cfg=CrossValidationConfig(
#             outer_folds=5,
#             inner_folds=3,
#             search_strategy="random",  # or "grid"
#             n_trials=20,
#         ),
#         search_space=lambda: {
#             "learning_rate": np.random.uniform(1e-5, 1e-2),
#             "weight_decay": np.random.uniform(1e-6, 1e-3),
#         },
#         output_dir="./results",
#         random_seed=42,
#     )

#     return results
