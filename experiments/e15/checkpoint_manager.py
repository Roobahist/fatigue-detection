import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from Zh.src.utils.common import load_obj, save_obj
from Zh.src.utils.file_folder import create_folders


class CheckpointManager:
    """
    Enhanced checkpoint manager with training-level resume support.

    Directory layout:

        output_dir/
            nested_cv_checkpoint.json           # main run-level checkpoint
            search_space.json                   # hyperparameter search space
            args.json                           # full set of run arguments
            checkpoints/
                outer_fold_0/
                    checkpoint.json             # metadata for this outer fold
                    best_hparams.json           # best hyperparameters
                    inner_results.json          # all InnerFoldResult for this fold
                    config_results.json         # list of config-level results dicts
                    search_state.json           # Bayesian optimization state
                    partial_configs/
                        config_0/
                            config.json         # hyperparameter config
                            inner_results.json  # hyperparameter config
                            metadata.json       # num_inner_folds_completed, etc.
                            training_checkpoints/  # NEW
                                inner_fold_0/
                                    checkpoint.pt
                                    metadata.json
                                inner_fold_1/
                                    ...
                        config_1/
                            ...
                outer_fold_1/
                    ...
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.main_checkpoint_path = os.path.join(
            output_dir, "nested_cv_checkpoint.json"
        )
        self.search_space_path = os.path.join(output_dir, "search_space.json")
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        create_folders(self.checkpoint_dir)

    # -------------------------------------------------------------------------
    # Paths and directory helpers
    # -------------------------------------------------------------------------

    def _outer_fold_dir_path(self, outer_idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"outer_fold_{outer_idx}")

    def get_outer_fold_dir(self, outer_idx: int) -> str:
        """Return outer-fold directory and ensure it exists."""
        fold_dir = self._outer_fold_dir_path(outer_idx)
        os.makedirs(fold_dir, exist_ok=True)
        return fold_dir

    def _partial_config_dir_path(self, outer_idx: int, config_idx: int) -> str:
        fold_dir = self._outer_fold_dir_path(outer_idx)
        return os.path.join(fold_dir, "partial_configs", f"config_{config_idx}")

    def get_partial_config_dir(self, outer_idx: int, config_idx: int) -> str:
        """Return partial-config directory and ensure it exists."""
        partial_dir = self._partial_config_dir_path(outer_idx, config_idx)
        os.makedirs(partial_dir, exist_ok=True)
        return partial_dir

    # -------------------------------------------------------------------------
    # NEW: Training-level checkpointing
    # -------------------------------------------------------------------------

    def _training_checkpoint_dir_path(
        self, outer_idx: int, config_idx: int, inner_idx: int
    ) -> str:
        partial_dir = self._partial_config_dir_path(outer_idx, config_idx)
        return os.path.join(
            partial_dir, "training_checkpoints", f"inner_fold_{inner_idx}"
        )

    def get_training_checkpoint_dir(
        self, outer_idx: int, config_idx: int, inner_idx: int
    ) -> str:
        """Get directory for training checkpoints within an inner fold."""
        train_dir = self._training_checkpoint_dir_path(outer_idx, config_idx, inner_idx)
        os.makedirs(train_dir, exist_ok=True)
        return train_dir

    def save_training_checkpoint(
        self,
        outer_idx: int,
        config_idx: int,
        inner_idx: int,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        scheduler_state: Optional[Dict[str, Any]],
        best_val_loss: float,
        best_epoch: int,
        best_model_state: Dict[str, Any],
        epochs_without_improvement: int,
    ) -> None:
        """
        Save training state during train_one_split.

        This allows resuming training from a specific epoch if interrupted.
        """
        train_dir = self.get_training_checkpoint_dir(outer_idx, config_idx, inner_idx)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_model_state_dict": best_model_state,
            "epochs_without_improvement": epochs_without_improvement,
        }

        checkpoint_path = os.path.join(train_dir, "checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save metadata separately for quick access
        metadata = {
            "outer_idx": outer_idx,
            "config_idx": config_idx,
            "inner_idx": inner_idx,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        save_obj(
            obj=metadata,
            output_path=os.path.join(train_dir, "metadata.json"),
            file_type="json",
            atomic=True,
        )

    def load_training_checkpoint(
        self, outer_idx: int, config_idx: int, inner_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Load training checkpoint if it exists.

        Returns:
            dict with training state or None if no checkpoint exists
        """
        train_dir = self._training_checkpoint_dir_path(outer_idx, config_idx, inner_idx)
        checkpoint_path = os.path.join(train_dir, "checkpoint.pt")

        if not os.path.exists(checkpoint_path):
            return None

        try:
            checkpoint = torch.load(checkpoint_path)
            return checkpoint
        except Exception as e:
            logging.error(
                "Error loading training checkpoint at %s: %s", checkpoint_path, e
            )
            return None

    def clean_training_checkpoint(
        self, outer_idx: int, config_idx: int, inner_idx: int
    ) -> None:
        """Remove training checkpoint after inner fold completes successfully."""
        import shutil

        train_dir = self._training_checkpoint_dir_path(outer_idx, config_idx, inner_idx)
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)

    # -------------------------------------------------------------------------
    # Partial config (inner-CV) checkpointing
    # -------------------------------------------------------------------------

    def save_partial_config(
        self,
        outer_idx: int,
        config_idx: int,
        config: Dict[str, Any],
        inner_results: List[Any],
    ) -> None:
        """
        Save partial progress for a config being evaluated.

        This is called after each inner fold. The goal is that, if the job dies
        mid-config, we can resume from the last completed inner fold.

        Args:
            outer_idx: Outer fold index
            config_idx: Config index being evaluated
            config: Hyperparameter configuration
            inner_results: List of completed InnerFoldResult objects
        """
        partial_dir = self.get_partial_config_dir(outer_idx, config_idx)

        # Save the config
        save_obj(
            obj=config,
            output_path=os.path.join(partial_dir, "config.json"),
            file_type="json",
            atomic=True,
        )

        # Save completed inner results
        save_obj(
            obj=inner_results,
            output_path=os.path.join(partial_dir, "inner_results.json"),
            file_type="json",
            atomic=True,
        )

        # Save metadata
        metadata = {
            "config_idx": config_idx,
            "num_inner_folds_completed": len(inner_results),
        }
        save_obj(
            obj=metadata,
            output_path=os.path.join(partial_dir, "metadata.json"),
            file_type="json",
            atomic=True,
        )

    def load_partial_config(
        self,
        outer_idx: int,
        config_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Load partial progress for a config.

        Returns:
            dict with { "config", "inner_results", "num_completed" } if found,
            otherwise None.
        """
        partial_dir = self._partial_config_dir_path(outer_idx, config_idx)

        config_path = os.path.join(partial_dir, "config.json")
        inner_path = os.path.join(partial_dir, "inner_results.json")

        if not os.path.exists(config_path):
            return None

        try:
            config = load_obj(config_path, file_type="json")
        except json.JSONDecodeError:
            logging.error("Corrupted partial config JSON at %s", config_path)
            return None
        except Exception as e:
            logging.error("Error loading partial config JSON at %s: %s", config_path, e)
            return None

        inner_results: List[Any] = []
        if os.path.exists(inner_path):
            try:
                inner_results = load_obj(inner_path, file_type="json")
            except Exception as e:
                logging.error(
                    "Error loading partial inner_results at %s: %s", inner_path, e
                )
                inner_results = []

        return {
            "config": config,
            "inner_results": inner_results,
            "num_completed": len(inner_results),
        }

    def clean_partial_config(self, outer_idx: int, config_idx: int) -> None:
        """
        Remove the partial-config directory for a given config.

        Only safe to call once the outer-fold state has been fully persisted
        (e.g., after config_results.json and inner_results.json are written).
        """
        import shutil

        partial_dir = self._partial_config_dir_path(outer_idx, config_idx)
        if os.path.exists(partial_dir):
            shutil.rmtree(partial_dir)

    def list_partial_configs(self, outer_idx: int) -> List[int]:
        """
        List all config indices that have a partial-config directory for this
        outer fold.

        This does not distinguish between "fully completed but not yet cleaned"
        and "truly partial" – that decision belongs to the caller.
        """
        fold_dir = self._outer_fold_dir_path(outer_idx)
        partial_dir = os.path.join(fold_dir, "partial_configs")

        if not os.path.exists(partial_dir):
            return []

        config_indices: List[int] = []
        for dirname in os.listdir(partial_dir):
            full_dir = os.path.join(partial_dir, dirname)
            if not os.path.isdir(full_dir):
                continue
            if not dirname.startswith("config_"):
                continue

            config_path = os.path.join(full_dir, "config.json")
            if not os.path.exists(config_path):
                continue

            try:
                idx = int(dirname.split("_")[1])
                config_indices.append(idx)
            except (ValueError, IndexError):
                # Ignore unexpected directory names
                continue

        return sorted(config_indices)

    # -------------------------------------------------------------------------
    # Main run-level checkpointing (across outer folds)
    # -------------------------------------------------------------------------

    def save_main_checkpoint(
        self,
        completed_folds: List[int],
        total_folds: int,
        current_fold: Optional[int] = None,
    ) -> None:
        """
        Save top-level progress for the overall nested CV run.
        """
        checkpoint = {
            "completed_folds": completed_folds,
            "total_folds": total_folds,
            "current_fold": current_fold,
        }
        save_obj(
            obj=checkpoint,
            output_path=self.main_checkpoint_path,
            file_type="json",
            atomic=True,
        )

    def load_main_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the main nested-CV checkpoint, if it exists and is readable.
        """
        if not os.path.exists(self.main_checkpoint_path):
            return None

        try:
            return load_obj(self.main_checkpoint_path, file_type="json")
        except json.JSONDecodeError:
            logging.error(
                "Main checkpoint file is corrupted: %s", self.main_checkpoint_path
            )
            return None
        except Exception as e:
            logging.error(
                "Error loading main checkpoint at %s: %s",
                self.main_checkpoint_path,
                e,
            )
            return None

    # -------------------------------------------------------------------------
    # Outer-fold checkpointing (metadata + aggregate results)
    # -------------------------------------------------------------------------

    def save_outer_fold_checkpoint(
        self,
        outer_idx: int,
        best_hparams: Optional[Dict[str, Any]] = None,
        chosen_epochs: Optional[int] = None,
        inner_results: Optional[List[Any]] = None,
        config_results: Optional[List[Any]] = None,
        search_state: Optional[Dict[str, Any]] = None,
        outer_train_metric: Optional[float] = None,
        outer_test_metric: Optional[float] = None,
    ) -> None:
        """
        Save checkpoint for a specific outer fold.

        This writes:
            - checkpoint.json (metadata)
            - best_hparams.json (if provided)
            - inner_results.json (if provided)
            - config_results.json (if provided)
            - search_state.json (if provided)
        """
        fold_dir = self.get_outer_fold_dir(outer_idx)

        checkpoint: Dict[str, Any] = {
            "outer_fold": outer_idx,
            "best_hparams": best_hparams,
            "chosen_epochs": chosen_epochs,
            "outer_train_metric": outer_train_metric,
            "outer_test_metric": outer_test_metric,
            "has_inner_results": inner_results is not None,
            "has_config_results": config_results is not None,
            "has_search_state": search_state is not None,
        }

        # Save metadata
        save_obj(
            obj=checkpoint,
            output_path=os.path.join(fold_dir, "checkpoint.json"),
            file_type="json",
            atomic=True,
        )

        # Save best hyperparameters if available
        if best_hparams is not None:
            save_obj(
                obj=best_hparams,
                output_path=os.path.join(fold_dir, "best_hparams.json"),
                file_type="json",
                atomic=True,
            )

        # Save all inner fold results if available
        if inner_results is not None:
            save_obj(
                obj=inner_results,
                output_path=os.path.join(fold_dir, "inner_results.json"),
                file_type="json",
                atomic=True,
            )

        # Save per-config aggregated results if available
        if config_results is not None:
            save_obj(
                obj=config_results,
                output_path=os.path.join(fold_dir, "config_results.json"),
                file_type="json",
                atomic=True,
            )

        # Save search state for Bayesian optimization if available
        if search_state is not None:
            save_obj(
                obj=search_state,
                output_path=os.path.join(fold_dir, "search_state.json"),
                file_type="json",
                atomic=True,
            )

    def load_outer_fold_checkpoint(self, outer_idx: int) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a specific outer fold.

        Returns:
            dict with keys:
                - outer_fold
                - best_hparams
                - chosen_epochs
                - outer_train_metric
                - outer_test_metric
                - inner_results (if present)
                - config_results (if present)
                - search_state (if present)
                - has_inner_results
                - has_config_results
                - has_search_state
            or None if no checkpoint exists.
        """
        fold_dir = self._outer_fold_dir_path(outer_idx)
        checkpoint_path = os.path.join(fold_dir, "checkpoint.json")

        if not os.path.exists(checkpoint_path):
            return None

        try:
            checkpoint = load_obj(checkpoint_path, file_type="json")
        except json.JSONDecodeError:
            logging.error("Corrupted outer fold checkpoint at %s", checkpoint_path)
            return None
        except Exception as e:
            logging.error(
                "Error loading outer fold checkpoint at %s: %s", checkpoint_path, e
            )
            return None

        # Load best hyperparameters if available
        hparams_path = os.path.join(fold_dir, "best_hparams.json")
        if os.path.exists(hparams_path):
            try:
                checkpoint["best_hparams"] = load_obj(hparams_path, file_type="json")
            except Exception as e:
                logging.error("Error loading best_hparams at %s: %s", hparams_path, e)

        # Load inner results if available
        inner_path = os.path.join(fold_dir, "inner_results.json")
        if os.path.exists(inner_path):
            try:
                checkpoint["inner_results"] = load_obj(inner_path, file_type="json")
            except Exception as e:
                logging.error("Error loading inner_results at %s: %s", inner_path, e)

        # Load config results if available
        config_path = os.path.join(fold_dir, "config_results.json")
        if os.path.exists(config_path):
            try:
                checkpoint["config_results"] = load_obj(config_path, file_type="json")
            except Exception as e:
                logging.error("Error loading config_results at %s: %s", config_path, e)

        # Load search state if available
        search_path = os.path.join(fold_dir, "search_state.json")
        if os.path.exists(search_path):
            try:
                checkpoint["search_state"] = load_obj(search_path, file_type="json")
            except Exception as e:
                logging.error("Error loading search_state at %s: %s", search_path, e)

        checkpoint["has_inner_results"] = "inner_results" in checkpoint
        checkpoint["has_config_results"] = "config_results" in checkpoint
        checkpoint["has_search_state"] = "search_state" in checkpoint

        return checkpoint

    def is_outer_fold_complete(self, outer_idx: int) -> bool:
        """
        Check if an outer fold is completely finished.

        For now, the check is:
            - the final model file exists in output_dir
            - the outer-fold checkpoint.json exists under checkpoints/
        """
        model_path = os.path.join(self.output_dir, f"outer_fold_{outer_idx}_model.pt")
        fold_dir = self._outer_fold_dir_path(outer_idx)
        checkpoint_path = os.path.join(fold_dir, "checkpoint.json")
        return os.path.exists(model_path) and os.path.exists(checkpoint_path)

    def can_resume(self) -> bool:
        """
        Return True if there is a main checkpoint that we can attempt to resume from.
        """
        return os.path.exists(self.main_checkpoint_path)

    # -------------------------------------------------------------------------
    # High-level progress summary
    # -------------------------------------------------------------------------

    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current progress across the whole nested CV.

        Returns:
            {
                "status": "not_started" | "in_progress" | "completed",
                "completed_folds": List[int],
                "current_fold": Optional[int],
                "total_folds": Optional[int],
                "progress_percentage": float,
                "current_fold_progress": Optional[Dict[str, Any]],
            }
        """
        main_checkpoint = self.load_main_checkpoint()

        if not main_checkpoint:
            return {
                "status": "not_started",
                "completed_folds": [],
                "current_fold": None,
                "total_folds": None,
                "progress_percentage": 0.0,
                "current_fold_progress": None,
            }

        completed_folds: List[int] = main_checkpoint.get("completed_folds", [])
        current_fold = main_checkpoint.get("current_fold")
        total_folds = main_checkpoint.get("total_folds")

        # Details for current fold if any
        current_fold_progress: Optional[Dict[str, Any]] = None
        if current_fold is not None:
            fold_checkpoint = self.load_outer_fold_checkpoint(current_fold)
            if fold_checkpoint:
                current_fold_progress = {
                    "best_hparams": fold_checkpoint.get("best_hparams"),
                    "has_config_results": fold_checkpoint.get(
                        "has_config_results", False
                    ),
                    "has_inner_results": fold_checkpoint.get(
                        "has_inner_results", False
                    ),
                }
                if "config_results" in fold_checkpoint:
                    current_fold_progress["num_configs_evaluated"] = len(
                        fold_checkpoint["config_results"]
                    )

        progress_percentage = (
            (len(completed_folds) / total_folds * 100.0) if total_folds else 0.0
        )

        return {
            "status": "in_progress" if current_fold is not None else "completed",
            "completed_folds": completed_folds,
            "current_fold": current_fold,
            "total_folds": total_folds,
            "progress_percentage": progress_percentage,
            "current_fold_progress": current_fold_progress,
        }
    
    # -------------------------------------------------------------------------
    # Training visualizations
    # -------------------------------------------------------------------------

    def _get_training_viz_dir(
        self, outer_idx: int, config_idx: int, inner_idx: int
    ) -> str:
        """Get directory for training visualizations."""
        train_dir = self._training_checkpoint_dir_path(outer_idx, config_idx, inner_idx)
        viz_dir = os.path.join(train_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        return viz_dir

    def _get_outer_fold_viz_dir(self, outer_idx: int) -> str:
        """Get directory for outer fold visualizations."""
        fold_dir = self._outer_fold_dir_path(outer_idx)
        viz_dir = os.path.join(fold_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        return viz_dir

    def save_training_visualization(
        self,
        outer_idx: int,
        config_idx: int,
        inner_idx: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        max_epochs: Optional[int],
    ) -> None:
        """
        Save/update training loss visualization.

        Args:
            outer_idx: Outer fold index
            config_idx: Config index
            inner_idx: Inner fold index
            epoch: Training epoch number (1-based)
            train_loss: Training loss at this epoch
            val_loss: Validation loss at this epoch
            max_epochs: Maximum number of epochs (for scaling x-axis)
        """
        viz_dir = self._get_training_viz_dir(outer_idx, config_idx, inner_idx)
        data_path = os.path.join(viz_dir, "training_data.json")
        plot_path = os.path.join(viz_dir, "training_curves.png")

        # Load existing data or create new
        if os.path.exists(data_path):
            try:
                data = load_obj(data_path, file_type="json")
            except Exception as e:
                logging.error("Error loading training viz data: %s", e)
                data = {"epochs": [], "train_losses": [], "val_losses": []}
        else:
            data = {"epochs": [], "train_losses": [], "val_losses": []}

        # Append new entry
        data["epochs"].append(epoch)
        data["train_losses"].append(train_loss)
        data["val_losses"].append(val_loss)

        # Save updated data
        save_obj(obj=data, output_path=data_path, file_type="json", atomic=True)

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = data["epochs"]
        train_losses = data["train_losses"]
        val_losses = data["val_losses"]

        # Plot training and validation losses
        ax.plot(
            epochs, train_losses, "o-", color="orange", label="Training Loss", alpha=0.7
        )
        ax.plot(
            epochs, val_losses, "s-", color="blue", label="Validation Loss", alpha=0.7
        )

        # Find and mark best epoch (minimum validation loss)
        best_idx = np.argmin(val_losses)
        best_epoch = epochs[best_idx]

        ax.axvline(
            x=best_epoch,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best epoch: {best_epoch}",
        )

        ax.set_xlabel("Training epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            f"Training Progress - Outer {outer_idx}, Config {config_idx}, Inner {inner_idx}",
            fontsize=14,
        )
        if max_epochs is not None:
            ax.set_xlim(0.5, max_epochs + 0.5)
        else:
            ax.set_xlim(0.5, max(epochs) + 0.5)

        ax.set_xticks(range(1, (max_epochs or max(epochs)) + 1, 1))

        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_hyperparameter_visualizations(self, outer_idx: int) -> None:
        """
        Generate hyperparameter performance visualizations for an outer fold.

        This creates separate scatter plots for each hyperparameter dimension,
        showing how different values affect model performance across all configs
        and inner folds.

        Args:
            outer_idx: Outer fold index
        """
        fold_dir = self._outer_fold_dir_path(outer_idx)
        viz_dir = self._get_outer_fold_viz_dir(outer_idx)

        # Load search space
        if not os.path.exists(self.search_space_path):
            logging.warning("Search space file not found at %s", self.search_space_path)
            return

        try:
            search_space = load_obj(self.search_space_path, file_type="json")
        except Exception as e:
            logging.error("Error loading search space: %s", e)
            return

        # Collect all results from partial configs
        partial_configs_dir = os.path.join(fold_dir, "partial_configs")
        if not os.path.exists(partial_configs_dir):
            logging.warning("No partial configs directory found")
            return

        # Aggregate all hyperparameter results
        all_results = []

        for config_dir in os.listdir(partial_configs_dir):
            if not config_dir.startswith("config_"):
                continue

            inner_results_path = os.path.join(
                partial_configs_dir, config_dir, "inner_results.json"
            )

            if not os.path.exists(inner_results_path):
                continue

            try:
                inner_results = load_obj(inner_results_path, file_type="json")
                all_results.extend(inner_results)
            except Exception as e:
                logging.error("Error loading inner results from %s: %s", config_dir, e)
                continue

        if not all_results:
            logging.warning("No results found for visualization")
            return

        # Define marker styles for different inner folds
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

        # Define colors for different metrics
        metric_colors = {"train": "orange", "val": "blue", "test": "green"}

        # Create a plot for each hyperparameter
        hparam_names = list(search_space.keys())

        for hparam_name in hparam_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Collect data for this hyperparameter
            hparam_values = []
            train_metrics = []
            val_metrics = []
            test_metrics = []
            inner_fold_indices = []

            for result in all_results:
                if hparam_name not in result["hparams"]:
                    continue

                hparam_values.append(result["hparams"][hparam_name])
                train_metrics.append(result["inner_train_metric"])
                val_metrics.append(result["inner_val_metric"])
                test_metrics.append(result["outer_test_metric"])
                inner_fold_indices.append(result["inner_fold"])

            if not hparam_values:
                plt.close()
                continue

            # Get unique inner folds
            unique_folds = sorted(set(inner_fold_indices))

            # Plot each inner fold with different marker
            for fold_idx in unique_folds:
                marker = markers[fold_idx % len(markers)]

                # Filter data for this fold
                fold_mask = [
                    i for i, f in enumerate(inner_fold_indices) if f == fold_idx
                ]
                fold_hparams = [hparam_values[i] for i in fold_mask]
                fold_train = [train_metrics[i] for i in fold_mask]
                fold_val = [val_metrics[i] for i in fold_mask]
                fold_test = [test_metrics[i] for i in fold_mask]

                # Plot train, val, test with different colors
                ax.scatter(
                    fold_hparams,
                    fold_train,
                    marker=marker,
                    color=metric_colors["train"],
                    s=100,
                    alpha=0.6,
                    label=(
                        f"Fold {fold_idx} - Train"
                        if fold_idx == unique_folds[0]
                        else ""
                    ),
                )
                ax.scatter(
                    fold_hparams,
                    fold_val,
                    marker=marker,
                    color=metric_colors["val"],
                    s=100,
                    alpha=0.6,
                    label=(
                        f"Fold {fold_idx} - Val" if fold_idx == unique_folds[0] else ""
                    ),
                )
                ax.scatter(
                    fold_hparams,
                    fold_test,
                    marker=marker,
                    color=metric_colors["test"],
                    s=100,
                    alpha=0.6,
                    label=(
                        f"Fold {fold_idx} - Test" if fold_idx == unique_folds[0] else ""
                    ),
                )

            # # Set x-axis scale and limits based on search space type
            spec = search_space[hparam_name]
            print(spec, hparam_name, type(spec), len(spec))
            if isinstance(spec, list):
                if len(spec) == 1:
                    if isinstance(spec[0], str):
                        continue
                    elif isinstance(spec[0], (int, float)):
                        print("Setting fixed numeric limits for", hparam_name)
                        margin = abs(spec[0]) * 0.05 if spec[0] != 0 else 0.05
                        low = spec[0] - margin
                        high = spec[0] + margin
                        ax.set_xlim(low, high)
                        ax.set_xticks([spec[0]])
                        print("Set limits for", hparam_name, low, high)

                elif all(isinstance(v, (int, float)) for v in spec):
                    print("Setting categorical numeric limits for", hparam_name)
                    margin = (max(spec) - min(spec)) * 0.05
                    low = min(spec) - margin
                    high = max(spec) + margin
                    ax.set_xlim(low, high)
                    ax.set_xticks(spec)
                    print("Set limits for", hparam_name, low, high)

                elif all(isinstance(v, str) for v in spec):
                    print("Categorical string, skipping limits for", hparam_name)
                    pass

                elif len(spec) == 3 and isinstance(spec[0], str) and all(isinstance(v, (int, float)) for v in spec[1:]):
                    margin = (spec[2] - spec[1]) * 0.05
                    low = spec[1] - margin
                    high = spec[2] + margin
                    ax.set_xticks(np.linspace(spec[1], spec[2], num=5))
                    if spec[0] == "log_uniform":
                        print("Setting log scale for", hparam_name)
                        ax.set_xscale("log")
                        low = max(low, 1e-7)
                        ax.set_xticks(np.logspace(np.log10(spec[1]), np.log10(spec[2]), num=5))

                    ax.set_xlim(low, high)
                    
                    print("Set limits for", hparam_name, low, high)
                    pass
            else:
                raise ValueError("Unknown search space specification")

            ax.set_xlabel(hparam_name, fontsize=12)
            ax.set_ylabel("Metric Value", fontsize=12)
            ax.set_title(
                f"Hyperparameter Performance - {hparam_name} (Outer Fold {outer_idx})",
                fontsize=14,
            )

            # Create custom legend
            # Metric colors
            train_patch = mpatches.Patch(color=metric_colors["train"], label="Train")
            val_patch = mpatches.Patch(color=metric_colors["val"], label="Validation")
            test_patch = mpatches.Patch(color=metric_colors["test"], label="Test")

            # Inner fold markers
            fold_handles = []
            for fold_idx in unique_folds:
                marker = markers[fold_idx % len(markers)]
                fold_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color="gray",
                        linestyle="",
                        markersize=8,
                        label=f"Inner Fold {fold_idx}",
                    )
                )

            # Combine legends
            first_legend = ax.legend(
                handles=[train_patch, val_patch, test_patch],
                loc="upper left",
                title="Metric Type",
            )
            ax.add_artist(first_legend)

            ax.legend(
                handles=fold_handles,
                loc="upper right",
                title="Inner Fold",
                ncol=2 if len(unique_folds) > 5 else 1,
            )

            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_filename = f'hparam_{hparam_name.replace(".", "_")}.png'
            plot_path = os.path.join(viz_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        logging.info(
            "Generated hyperparameter visualizations for outer fold %d in %s",
            outer_idx,
            viz_dir,
        )

    def clean_training_visualizations(
        self, outer_idx: int, config_idx: int, inner_idx: int
    ) -> None:
        """Remove training visualizations after inner fold completes successfully."""
        import shutil

        viz_dir = os.path.join(
            self._training_checkpoint_dir_path(outer_idx, config_idx, inner_idx),
            "visualizations",
        )
        if os.path.exists(viz_dir):
            shutil.rmtree(viz_dir)
