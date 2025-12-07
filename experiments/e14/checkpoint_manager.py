import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

from Zh.src.utils.common import save_obj, load_obj
from Zh.src.utils.file_folder import create_folders


class CheckpointManager:
    """
    Enhanced checkpoint manager with inner-fold-level resume support.

    Checkpoint structure:
    output_dir/
        checkpoints/
            outer_fold_0/
                checkpoint.json
                best_hparams.json
                inner_results.pkl           # All completed inner results
                config_results.pkl          # Completed config evaluations
                search_state.pkl            # Bayesian state
                partial_configs/            # NEW: Partial config evaluations
                    config_0/
                        config.json         # Config being evaluated
                        inner_results.pkl   # Completed inner folds for this config
                    config_5/
                        ...
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.main_checkpoint_path = os.path.join(
            output_dir, "nested_cv_checkpoint.json"
        )
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        create_folders(self.checkpoint_dir)

    def _outer_fold_dir_path(self, outer_idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"outer_fold_{outer_idx}")

    def get_outer_fold_dir(self, outer_idx: int) -> str:
        fold_dir = self._outer_fold_dir_path(outer_idx)
        os.makedirs(fold_dir, exist_ok=True)
        return fold_dir
    
    def _partial_config_dir_path(self, outer_idx: int, config_idx: int) -> str:
        fold_dir = self._outer_fold_dir_path(outer_idx)
        return os.path.join(fold_dir, "partial_configs", f"config_{config_idx}")

    def get_partial_config_dir(self, outer_idx: int, config_idx: int) -> str:
        # Only used by writers
        partial_dir = self._partial_config_dir_path(outer_idx, config_idx)
        os.makedirs(partial_dir, exist_ok=True)
        return partial_dir

    def save_partial_config(
        self,
        outer_idx: int,
        config_idx: int,
        config: Dict[str, Any],
        inner_results: List[Any],
    ):
        """
        Save partial progress for a config being evaluated.

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
            output_path=os.path.join(partial_dir, "inner_results.pkl"),
            file_type="pickle",
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


    def load_partial_config(self, outer_idx: int, config_idx: int) -> Optional[Dict[str, Any]]:
        partial_dir = self._partial_config_dir_path(outer_idx, config_idx)
        
        config_path = os.path.join(partial_dir, "config.json")
        inner_path = os.path.join(partial_dir, "inner_results.pkl")
        
        if not os.path.exists(config_path):
            return None

        try:
            config = load_obj(config_path, file_type="json")
        except json.JSONDecodeError:
            logging.error("Corrupted partial config JSON at %s", config_path)
            return None
        
        inner_results = []
        if os.path.exists(inner_path):
            with open(inner_path, "rb") as f:
                inner_results = pickle.load(f)
        
        return {
            "config": config,
            "inner_results": inner_results,
            "num_completed": len(inner_results),
        }

    def clean_partial_config(self, outer_idx: int, config_idx: int):
        import shutil
        partial_dir = self._partial_config_dir_path(outer_idx, config_idx)
        if os.path.exists(partial_dir):
            shutil.rmtree(partial_dir)

    def list_partial_configs(self, outer_idx: int) -> List[int]:
        """List all partial config indices for an outer fold."""
        fold_dir = self._outer_fold_dir_path(outer_idx)
        partial_dir = os.path.join(fold_dir, "partial_configs")

        if not os.path.exists(partial_dir):
            return []

        config_indices = []
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
                pass

        return sorted(config_indices)

    def save_main_checkpoint(
        self,
        completed_folds: list,
        total_folds: int,
        current_fold: Optional[int] = None,
    ):
        """Save main progress checkpoint."""
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

    def load_main_checkpoint(self) -> Optional[Dict]:
        if not os.path.exists(self.main_checkpoint_path):
            return None
        try:
            return load_obj(self.main_checkpoint_path, file_type="json")
        except json.JSONDecodeError:
            logging.error("Main checkpoint file is corrupted: %s", self.main_checkpoint_path)
            return None

    def save_outer_fold_checkpoint(
        self,
        outer_idx: int,
        best_hparams: Optional[Dict] = None,
        chosen_epochs: Optional[int] = None,
        inner_results: Optional[list] = None,
        config_results: Optional[list] = None,
        search_state: Optional[Dict] = None,
        outer_train_metric: Optional[float] = None,
        outer_test_metric: Optional[float] = None,
    ):
        """
        Save checkpoint for a specific outer fold.

        Args:
            outer_idx: Outer fold index
            best_hparams: Best hyperparameters found
            chosen_epochs: Number of epochs chosen for final training
            inner_results: List of InnerFoldResult objects
            config_results: Results from grid/random search
            search_state: State for Bayesian optimization
            outer_train_metric: Final training metric (after retraining)
            outer_test_metric: Final test metric (after retraining)
        """
        fold_dir = self.get_outer_fold_dir(outer_idx)

        checkpoint = {
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

        # Save inner results if available
        if inner_results is not None:
            save_obj(
                obj=inner_results,
                output_path=os.path.join(fold_dir, "inner_results.pkl"),
                file_type="pickle",
                atomic=True,
            )

        # Save config results if available
        if config_results is not None:
            save_obj(
                obj=config_results,
                output_path=os.path.join(fold_dir, "config_results.pkl"),
                file_type="pickle",
                atomic=True,
            )

        # Save search state (for Bayesian optimization)
        if search_state is not None:
            save_obj(
                obj=search_state,
                output_path=os.path.join(fold_dir, "search_state.pkl"),
                file_type="pickle",
                atomic=True,
            )

    def load_outer_fold_checkpoint(self, outer_idx: int) -> Optional[Dict]:
        """Load checkpoint for a specific outer fold."""
        fold_dir = self._outer_fold_dir_path(outer_idx)
        checkpoint_path = os.path.join(fold_dir, "checkpoint.json")

        if not os.path.exists(checkpoint_path):
            return None

        try:
            checkpoint = load_obj(checkpoint_path, file_type="json")
        except json.JSONDecodeError:
            logging.error("Corrupted outer fold checkpoint at %s", checkpoint_path)
            return None

        # Load best hyperparameters if available
        hparams_path = os.path.join(fold_dir, "best_hparams.json")
        if os.path.exists(hparams_path):
            checkpoint["best_hparams"] = load_obj(hparams_path, file_type="json")

        # Load inner results if available
        inner_path = os.path.join(fold_dir, "inner_results.pkl")
        if os.path.exists(inner_path):
            checkpoint["inner_results"] = load_obj(inner_path, file_type="pickle")

        # Load config results if available
        config_path = os.path.join(fold_dir, "config_results.pkl")
        if os.path.exists(config_path):
            checkpoint["config_results"] = load_obj(config_path, file_type="pickle")

        # Load search state if available
        search_path = os.path.join(fold_dir, "search_state.pkl")
        if os.path.exists(search_path):
            checkpoint["search_state"] = load_obj(search_path, file_type="pickle")

        checkpoint["has_inner_results"] = "inner_results" in checkpoint
        checkpoint["has_config_results"] = "config_results" in checkpoint
        checkpoint["has_search_state"] = "search_state" in checkpoint

        return checkpoint

    def is_outer_fold_complete(self, outer_idx: int) -> bool:
        """Check if an outer fold is completely finished."""
        model_path = os.path.join(self.output_dir, f"outer_fold_{outer_idx}_model.pt")
        fold_dir = self._outer_fold_dir_path(outer_idx)
        checkpoint_path = os.path.join(fold_dir, "checkpoint.json")
        return os.path.exists(model_path) and os.path.exists(checkpoint_path)

    def can_resume(self) -> bool:
        """Check if we can resume from a previous run."""
        return os.path.exists(self.main_checkpoint_path)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        main_checkpoint = self.load_main_checkpoint()

        if not main_checkpoint:
            return {
                "status": "not_started",
                "completed_folds": [],
                "current_fold": None,
                "total_folds": None,
            }

        completed_folds = main_checkpoint.get("completed_folds", [])
        current_fold = main_checkpoint.get("current_fold")
        total_folds = main_checkpoint.get("total_folds")

        # Get details for current fold if exists
        current_fold_progress = None
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

                # Count configs if available
                if "config_results" in fold_checkpoint:
                    current_fold_progress["num_configs_evaluated"] = len(
                        fold_checkpoint["config_results"]
                    )

        return {
            "status": "in_progress" if current_fold is not None else "completed",
            "completed_folds": completed_folds,
            "current_fold": current_fold,
            "total_folds": total_folds,
            "progress_percentage": (
                (len(completed_folds) / total_folds * 100) if total_folds else 0
            ),
            "current_fold_progress": current_fold_progress,
        }
