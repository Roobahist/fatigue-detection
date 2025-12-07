import json
import math
import os
import random
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Zh.experiments.e11.dataclasses import *
from Zh.experiments.e11.datasets import *
from Zh.experiments.e11.helpers import *
from Zh.experiments.e11.model import *
from Zh.src.utils.file_folder import *
from Zh.src.utils.file_folder import get_file_paths
from Zh.src.utils.memmap_handler import MemmapHandler


def train_one_split(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainingConfig,
    dataset_name: str = "UTA",
) -> Tuple[nn.Module, float]:
    """
    Train on a single train/val split with early stopping.
    Returns best model (loaded with best weights) and best validation loss.
    """
    device_str = train_cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    if dataset_name == "UTA":
        thresholds = [2.5, 7.5]
        unique_classes = [0.0, 5.0, 10]

    elif dataset_name == "DROZY":
        thresholds = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        unique_classes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model.to(device)
    optimizer, scheduler = build_optimizer(model, train_cfg)
    loss_fn = get_loss_fn(train_cfg.loss)

    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in tqdm(range(train_cfg.num_epochs)):
        # --- NEW: update selection schedule on training dataset, if applicable ---
        if isinstance(train_loader.dataset, RealVideoDataset):
            train_loader.dataset.set_training_epoch(epoch, train_cfg.num_epochs)

        # Training
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in tqdm(train_loader):
            # Move batch to device
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

        train_loss = train_loss_sum / max(train_samples, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        predictions = []
        target_classes = []
        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = move_batch_to_device(batch, device)
                targets = torch.tensor(
                    [video["target"] for video in batch_on_device],
                    dtype=torch.float32,
                    device=device,
                )
                target_classes.extend([video["target"] for video in batch_on_device])
                preds = model(batch_on_device)
                loss = loss_fn(preds, targets)
                predictions.extend(preds.tolist())

                batch_size = len(batch_on_device)
                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size

        val_loss = val_loss_sum / max(val_samples, 1)
        prediction_classes = [assign_class(p, thresholds, unique_classes) for p in predictions]
        cr = classification_report(
            y_true        = target_classes,
            y_pred        = prediction_classes,
            labels        = unique_classes,
            output_dict   = True,
            zero_division = False
        )

        cm = safe_confusion_matrix(
            y_true = target_classes, 
            y_pred = prediction_classes, 
            all_labels=unique_classes)

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{train_cfg.num_epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Best Val: {best_val_loss:.4f}"
            "\n"
            f"{cr}"
            "\n"
            f"{cm}"
            "\n"
            f"{[(c, p, t) for c, p, t in zip(prediction_classes, predictions, target_classes)]}"
        )

        if epochs_without_improvement >= train_cfg.early_stopping_patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    return model, best_val_loss


def nested_cv(
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
) -> Dict:
    """
    Nested cross-validation with logging + saving.

    Returns a dict with all results and also writes:
      - model checkpoints for each inner and outer fold
      - a JSON summary file in output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    all_indices = list(range(len(video_ids)))

    results: Dict = {
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "outer_results": [],
    }

    for outer_idx in range(outer_folds):
        print(f"\n====== Outer fold {outer_idx + 1}/{outer_folds} ======")
        outer_train_idx, outer_test_idx = kfold_split_indices(
            indices=all_indices,
            video_ids=video_ids,
            n_splits=outer_folds,
            fold_idx=outer_idx,
            dataset_name=dataset_name,
        )

        outer_train_ids = [video_ids[i] for i in outer_train_idx]
        outer_test_ids = [video_ids[i] for i in outer_test_idx]

        # INNER LOOP: model selection on outer_train_ids
        best_inner_val_loss = float("inf")
        best_inner_state = None
        best_inner_cfg = deepcopy(
            base_train_cfg
        )  # if you vary hyperparams, store them here

        inner_results = []  # store per-inner-fold metrics

        inner_all_idx = list(range(len(outer_train_ids)))

        for inner_idx in range(inner_folds):
            print(f"  -- Inner fold {inner_idx + 1}/{inner_folds}")
            inner_train_idx, inner_val_idx = kfold_split_indices(
                indices=inner_all_idx,
                video_ids=outer_train_ids,
                n_splits=inner_folds,
                fold_idx=inner_idx,
                dataset_name=dataset_name,
            )

            inner_train_ids = [outer_train_ids[i] for i in inner_train_idx]
            inner_val_ids = [outer_train_ids[i] for i in inner_val_idx]

            train_dataset = RealVideoDataset(
                video_ids=inner_train_ids,
                embeddings_root=embeddings_root,
                features_root=features_root,
                eventfulness_scores_root=eventfulness_scores_root,
                model_cfg=model_cfg,
                selection_cfg=selection_cfg,
                mode="train",
                dataset_name=dataset_name,
                converge_ratio=converge_ratio,
            )

            val_dataset = RealVideoDataset(
                video_ids=inner_val_ids,
                embeddings_root=embeddings_root,
                features_root=features_root,
                eventfulness_scores_root=eventfulness_scores_root,
                model_cfg=model_cfg,
                selection_cfg=selection_cfg,
                mode="val",
                dataset_name=dataset_name,
                converge_ratio=converge_ratio,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=base_train_cfg.batch_size,
                shuffle=True,
                collate_fn=collate_videos,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=base_train_cfg.batch_size,
                shuffle=False,
                collate_fn=collate_videos,
            )

            # Fresh model for each inner fold
            model = MultiPathMILRegressor(model_cfg)
            model, inner_val_loss = train_one_split(
                model, train_loader, val_loader, base_train_cfg, dataset_name=dataset_name
            )

            # Save this inner model
            inner_model_path = os.path.join(
                output_dir,
                f"outer{outer_idx}_inner{inner_idx}_model.pt",
            )
            torch.save(model.state_dict(), inner_model_path)

            inner_results.append(
                {
                    "outer_fold": outer_idx,
                    "inner_fold": inner_idx,
                    "val_loss": float(inner_val_loss),
                    "model_path": inner_model_path,
                    # optional: save training config snapshot
                    "train_cfg": asdict(base_train_cfg),
                }
            )

            if inner_val_loss < best_inner_val_loss:
                best_inner_val_loss = inner_val_loss
                best_inner_state = deepcopy(model.state_dict())
                # if you tune hyperparams per inner fold,
                # best_inner_cfg = cfg_for_this_inner_fold

        # Inner-fold summary
        inner_val_losses = [r["val_loss"] for r in inner_results]
        mean_inner_val_loss = sum(inner_val_losses) / len(inner_val_losses) if len(inner_val_losses) > 0 else math.nan

        print(
            f"  Best inner val loss for outer fold {outer_idx + 1}: {best_inner_val_loss:.4f}"
        )
        print(
            f"  Mean inner val loss for outer fold {outer_idx + 1}: {mean_inner_val_loss:.4f}"
        )

        # Retrain on full outer_train_ids with selected configuration
        print("  Retraining on full outer_train set with selected configuration...")

        outer_train_dataset = RealVideoDataset(
            video_ids=outer_train_ids,
            embeddings_root=embeddings_root,
            features_root=features_root,
            eventfulness_scores_root=eventfulness_scores_root,
            model_cfg=model_cfg,
            selection_cfg=selection_cfg,
            mode="train",
            dataset_name=dataset_name,
            converge_ratio=converge_ratio,
        )
        outer_test_dataset = RealVideoDataset(
            video_ids=outer_test_ids,
            embeddings_root=embeddings_root,
            features_root=features_root,
            eventfulness_scores_root=eventfulness_scores_root,
            model_cfg=model_cfg,
            selection_cfg=selection_cfg,
            mode="test",
            dataset_name=dataset_name,
            converge_ratio=converge_ratio,
        )

        outer_train_loader = DataLoader(
            outer_train_dataset,
            batch_size=base_train_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_videos,
        )
        outer_test_loader = DataLoader(
            outer_test_dataset,
            batch_size=base_train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_videos,
        )

        # New model for outer retrain. You *could* load best_inner_state here.
        model = MultiPathMILRegressor(model_cfg)
        # model.load_state_dict(best_inner_state)  # optional warm-start
        model, outer_test_loss = train_one_split(
            model,
            outer_train_loader,
            outer_test_loader,
            base_train_cfg,
            dataset_name=dataset_name,
        )

        outer_model_path = os.path.join(
            output_dir,
            f"outer{outer_idx}_final_model.pt",
        )
        torch.save(model.state_dict(), outer_model_path)

        print(f"  Outer fold {outer_idx + 1} test loss: {outer_test_loss:.4f}")
        print(f"====== End of outer fold {outer_idx + 1}/{outer_folds} ======")

        results["outer_results"].append(
            {
                "outer_fold": outer_idx,
                "inner_results": inner_results,
                "best_inner_val_loss": float(best_inner_val_loss),
                "mean_inner_val_loss": float(mean_inner_val_loss),
                "outer_test_loss": float(outer_test_loss),
                "outer_model_path": outer_model_path,
                # optional: which config used for retrain
                "train_cfg": asdict(base_train_cfg),
            }
        )

    # ---- Global summary across outer folds ----
    outer_losses = [o["outer_test_loss"] for o in results["outer_results"]]
    mean_outer_loss = sum(outer_losses) / len(outer_losses)
    results["mean_outer_test_loss"] = float(mean_outer_loss)

    print("\n===== Nested CV summary =====")
    print(f"Mean outer test loss across {outer_folds} folds: {mean_outer_loss:.4f}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "nested_cv_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved nested CV summary to: {summary_path}")

    return results


def main():
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    import argparse

    def build_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset_name", type=str, default="UTA",
                            choices=["UTA", "DROZY"])
        parser.add_argument("--outer_folds", type=int, default=5)
        parser.add_argument("--inner_folds", type=int, default=0)
        # ---------------------------
        # TemporalConfig
        # ---------------------------
        parser.add_argument("--temporal.backbone", type=str, default="gru",
                            choices=["gru", "lstm", "tcn", "transformer"])
        parser.add_argument("--temporal.hidden_size", type=int, default=128)
        parser.add_argument("--temporal.num_layers", type=int, default=1)
        parser.add_argument("--temporal.dropout", type=float, default=0.1)
        parser.add_argument("--temporal.activation", type=str, default="relu",
                            choices=["relu", "gelu", "tanh"])

        # ---------------------------
        # Window MLPConfig
        # ---------------------------
        parser.add_argument("--window_mlp.hidden_size", type=int, default=128)
        parser.add_argument("--window_mlp.num_layers", type=int, default=2)
        parser.add_argument("--window_mlp.dropout", type=float, default=0.1)
        parser.add_argument("--window_mlp.activation", type=str, default="relu",
                            choices=["relu", "gelu", "tanh"])

        # ---------------------------
        # Regression Head MLPConfig
        # ---------------------------
        parser.add_argument("--reg_head_mlp.hidden_size", type=int, default=64)
        parser.add_argument("--reg_head_mlp.num_layers", type=int, default=2)
        parser.add_argument("--reg_head_mlp.dropout", type=float, default=0.1)
        parser.add_argument("--reg_head_mlp.activation", type=str, default="relu",
                            choices=["relu", "gelu", "tanh"])

        # ---------------------------
        # ModelConfig
        # ---------------------------
        parser.add_argument("--model.d_embed", type=int, default=768)
        parser.add_argument("--model.d_blend", type=int, default=52)
        parser.add_argument("--model.d_other", type=int, default=4)

        parser.add_argument("--model.d_temporal_embed", type=int, default=256)
        parser.add_argument("--model.d_temporal_blend", type=int, default=64)
        parser.add_argument("--model.d_temporal_other", type=int, default=32)

        parser.add_argument("--model.attn_hidden_size", type=int, default=128)
        parser.add_argument("--model.attn_activation", type=str, default="tanh",
                            choices=["tanh", "relu", "gelu"])

        # ---------------------------
        # TrainingConfig
        # ---------------------------
        parser.add_argument("--train.num_epochs", type=int, default=100)
        parser.add_argument("--train.batch_size", type=int, default=64)
        parser.add_argument("--train.optimizer", type=str, default="adamw",
                            choices=["adam", "adamw", "sgd"])
        parser.add_argument("--train.lr", type=float, default=1e-3)
        parser.add_argument("--train.weight_decay", type=float, default=1e-4)
        parser.add_argument("--train.loss", type=str, default="mse",
                            choices=["mse", "mae", "huber"])
        parser.add_argument("--train.early_stopping_patience", type=int, default=40)
        parser.add_argument("--train.device", type=str, default="cuda",
                            choices=["cuda", "cpu"])
        parser.add_argument("--train.use_scheduler", action="store_true")

        # ---------------------------
        # SelectionConfig
        # ---------------------------
        parser.add_argument("--sel.window_size", type=int, default=64)
        parser.add_argument("--sel.stride", type=int, default=32)
        parser.add_argument("--sel.n_windows", type=int, default=16)

        parser.add_argument("--sel.pick_mean_top_begin", type=int, default=4)
        parser.add_argument("--sel.pick_max_top_begin", type=int, default=2)
        parser.add_argument("--sel.pick_random_begin", type=int, default=10)

        parser.add_argument("--sel.pick_mean_top_end", type=int, default=12)
        parser.add_argument("--sel.pick_max_top_end", type=int, default=2)
        parser.add_argument("--sel.pick_random_end", type=int, default=2)

        parser.add_argument("--sel.converge_ratio", type=float, default=0.5)

        return parser


    def build_configs_from_args(args):
        # ---------- TemporalConfig ----------
        temporal_cfg = TemporalConfig(
            backbone=vars(args)["temporal.backbone"],
            hidden_size=vars(args)["temporal.hidden_size"],
            num_layers=vars(args)["temporal.num_layers"],
            dropout=vars(args)["temporal.dropout"],
            activation=vars(args)["temporal.activation"],
        )

        # ---------- Window MLPConfig ----------
        window_mlp_cfg = MLPConfig(
            hidden_size=vars(args)["window_mlp.hidden_size"],
            num_layers=vars(args)["window_mlp.num_layers"],
            dropout=vars(args)["window_mlp.dropout"],
            activation=vars(args)["window_mlp.activation"],
        )

        # ---------- Regression Head MLPConfig ----------
        reg_head_mlp_cfg = MLPConfig(
            hidden_size=vars(args)["reg_head_mlp.hidden_size"],
            num_layers=vars(args)["reg_head_mlp.num_layers"],
            dropout=vars(args)["reg_head_mlp.dropout"],
            activation=vars(args)["reg_head_mlp.activation"],
        )

        # ---------- ModelConfig ----------
        model_cfg = ModelConfig(
            d_embed=vars(args)["model.d_embed"],
            d_blend=vars(args)["model.d_blend"],
            d_other=vars(args)["model.d_other"],

            d_temporal_embed=vars(args)["model.d_temporal_embed"],
            d_temporal_blend=vars(args)["model.d_temporal_blend"],
            d_temporal_other=vars(args)["model.d_temporal_other"],

            temporal=temporal_cfg,
            window_mlp=window_mlp_cfg,
            reg_head_mlp=reg_head_mlp_cfg,

            attn_hidden_size=vars(args)["model.attn_hidden_size"],
            attn_activation=vars(args)["model.attn_activation"],
        )

        # ---------- TrainingConfig ----------
        train_cfg = TrainingConfig(
            num_epochs=vars(args)["train.num_epochs"],
            batch_size=vars(args)["train.batch_size"],
            optimizer=vars(args)["train.optimizer"],
            lr=vars(args)["train.lr"],
            weight_decay=vars(args)["train.weight_decay"],
            loss=vars(args)["train.loss"],
            early_stopping_patience=vars(args)["train.early_stopping_patience"],
            device=vars(args)["train.device"],
            use_scheduler=vars(args)["train.use_scheduler"],
        )

        # ---------- SelectionConfig ----------
        selection_cfg = SelectionConfig(
            window_size=vars(args)["sel.window_size"],
            stride=vars(args)["sel.stride"],
            n_windows=vars(args)["sel.n_windows"],

            pick_mean_top_begin=vars(args)["sel.pick_mean_top_begin"],
            pick_max_top_begin=vars(args)["sel.pick_max_top_begin"],
            pick_random_begin=vars(args)["sel.pick_random_begin"],
            pick_mean_top_end=vars(args)["sel.pick_mean_top_end"],
            pick_max_top_end=vars(args)["sel.pick_max_top_end"],
            pick_random_end=vars(args)["sel.pick_random_end"],
        )

        return model_cfg, train_cfg, selection_cfg

    parser = build_parser()
    args = parser.parse_args()
    print(args)
    (
        model_cfg,
        train_cfg,
        selection_cfg,
    ) = build_configs_from_args(args)

    if args.dataset_name == "UTA":
        embeddings_root = (
            "/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI0"
        )
        features_root = "/work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05/features"
        eventfulness_scores_root = "/work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05/eventfulness"

    elif args.dataset_name == "DROZY":
        embeddings_root = (
            "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI0"
        )
        features_root = "/work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/features"
        eventfulness_scores_root = "/work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/eventfulness"

    else:    
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    video_ids = get_file_paths(
        input_path=embeddings_root, depth=3, EXT_LIST=".dat"
    )[:20] 

    output_dir = f"/home/mojtaba.kananisarche/Zh/outputs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}|E11"

    create_folders(output_dir)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    nested_cv(
        video_ids=video_ids,
        model_cfg=model_cfg,
        base_train_cfg=train_cfg,
        selection_cfg=selection_cfg,
        eventfulness_scores_root=eventfulness_scores_root,
        embeddings_root=embeddings_root,
        features_root=features_root,
        outer_folds=vars(args)["outer_folds"],
        inner_folds=vars(args)["inner_folds"],
        output_dir=output_dir,
        dataset_name=vars(args)["dataset_name"],
        converge_ratio=vars(args)["sel.converge_ratio"],
    )


if __name__ == "__main__":
    main()
