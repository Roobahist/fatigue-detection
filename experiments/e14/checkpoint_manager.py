import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

from Zh.src.utils.common import load_obj, save_obj
from Zh.src.utils.file_folder import create_folders


class CheckpointManager:
    """
    Checkpoint manager with inner-fold-level resume support.

    Directory layout:

        output_dir/
            nested_cv_checkpoint.json          # main run-level checkpoint

            checkpoints/
                outer_fold_0/
                    checkpoint.json            # metadata for this outer fold
                    best_hparams.json          # best hyperparameters
                    inner_results.json          # all InnerFoldResult for this fold
                    config_results.json         # list of config-level results dicts
                    search_state.json           # Bayesian optimization state
                    partial_configs/
                        config_0/
                            config.json        # hyperparameter config
                            inner_results.json # completed inner folds for this config
                            metadata.json     # num_inner_folds_completed, etc.
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
