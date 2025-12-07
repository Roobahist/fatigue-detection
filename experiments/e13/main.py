from datetime import datetime

from Zh.experiments.e13.dataclasses import *
from Zh.experiments.e13.video_mil_adapter import nested_cv_video_mil
from Zh.src.utils.common import save_obj
from Zh.src.utils.cpu_gpu import setup_torch
from Zh.src.utils.file_folder import *


def main():
    # Reproducibility
    import argparse

    def build_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--dataset_name", type=str, default="DROZY", choices=["UTA", "DROZY"]
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Random seed for reproducibility"
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "cpu"],
            help="Device to use",
        )
        parser.add_argument(
            "--parallel_inner_cv",
            action="store_true",
            help="Run inner CV in parallel",
        )
        parser.add_argument(
            "--max_workers",
            type=int,
            default=6,
            help="Max workers for parallel inner CV",
        )
        # ---------------------------
        # TemporalConfig
        # ---------------------------
        parser.add_argument(
            "--model.temporal.backbone",
            type=str,
            default="gru",
            choices=["gru", "lstm", "tcn", "transformer"],
        )
        parser.add_argument("--model.temporal.hidden_size", type=int, default=128)
        parser.add_argument("--model.temporal.num_layers", type=int, default=1)
        parser.add_argument("--model.temporal.dropout", type=float, default=0.1)
        parser.add_argument(
            "--model.temporal.activation",
            type=str,
            default="relu",
            choices=["relu", "gelu", "tanh"],
        )

        # ---------------------------
        # Window MLPConfig
        # ---------------------------
        parser.add_argument("--model.window_mlp.hidden_size", type=int, default=128)
        parser.add_argument("--model.window_mlp.num_layers", type=int, default=2)
        parser.add_argument("--model.window_mlp.dropout", type=float, default=0.1)
        parser.add_argument(
            "--model.window_mlp.activation",
            type=str,
            default="relu",
            choices=["relu", "gelu", "tanh"],
        )

        # ---------------------------
        # Regression Head MLPConfig
        # ---------------------------
        parser.add_argument("--model.reg_head_mlp.hidden_size", type=int, default=64)
        parser.add_argument("--model.reg_head_mlp.num_layers", type=int, default=2)
        parser.add_argument("--model.reg_head_mlp.dropout", type=float, default=0.1)
        parser.add_argument(
            "--model.reg_head_mlp.activation",
            type=str,
            default="relu",
            choices=["relu", "gelu", "tanh"],
        )

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
        parser.add_argument(
            "--model.attn_activation",
            type=str,
            default="tanh",
            choices=["tanh", "relu", "gelu"],
        )

        # ---------------------------
        # TrainingConfig
        # ---------------------------
        parser.add_argument("--train.num_epochs", type=int, default=100)
        parser.add_argument("--train.batch_size", type=int, default=32)
        parser.add_argument(
            "--train.optimizer",
            type=str,
            default="adamw",
            choices=["adam", "adamw", "sgd"],
        )
        parser.add_argument("--train.lr", type=float, default=1e-3)
        parser.add_argument("--train.weight_decay", type=float, default=1e-4)
        parser.add_argument(
            "--train.loss", type=str, default="mse", choices=["mse", "mae", "huber"]
        )
        parser.add_argument("--train.early_stopping_patience", type=int, default=40)
        parser.add_argument(
            "--train.device", type=str, default="cuda", choices=["cuda", "cpu"]
        )
        parser.add_argument("--train.use_scheduler", action="store_true")
        parser.add_argument("--train.step_lr_step_size", type=int, default=30)
        parser.add_argument("--train.step_lr_gamma", type=float, default=0.1)
        parser.add_argument("--train.momentum", type=float, default=0.9)
        parser.add_argument("--train.betas", type=float, nargs=2, default=(0.9, 0.999))

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

        # ---------------------------
        # CrossValidationConfig
        # ---------------------------
        parser.add_argument(
            "--cv.search_strategy",
            type=str,
            default="random",
            choices=["grid", "random", "bayesian"],
            help="Hyperparameter search strategy",
        )
        parser.add_argument(
            "--cv.n_trials",
            type=int,
            default=12,
            help="Number of trials for random/bayesian search",
        )
        parser.add_argument(
            "--cv.outer_folds",
            type=int,
            default=5,
            help="Number of outer folds for cross-validation",
        )
        parser.add_argument(
            "--cv.inner_folds",
            type=int,
            default=0,
            help="Number of inner folds for cross-validation",
        )

        return parser

    def build_configs_from_args(args):
        # ---------- TemporalConfig ----------
        temporal_cfg = TemporalConfig(
            backbone=vars(args)["model.temporal.backbone"],
            hidden_size=vars(args)["model.temporal.hidden_size"],
            num_layers=vars(args)["model.temporal.num_layers"],
            dropout=vars(args)["model.temporal.dropout"],
            activation=vars(args)["model.temporal.activation"],
        )

        # ---------- Window MLPConfig ----------
        window_mlp_cfg = MLPConfig(
            hidden_size=vars(args)["model.window_mlp.hidden_size"],
            num_layers=vars(args)["model.window_mlp.num_layers"],
            dropout=vars(args)["model.window_mlp.dropout"],
            activation=vars(args)["model.window_mlp.activation"],
        )

        # ---------- Regression Head MLPConfig ----------
        reg_head_mlp_cfg = MLPConfig(
            hidden_size=vars(args)["model.reg_head_mlp.hidden_size"],
            num_layers=vars(args)["model.reg_head_mlp.num_layers"],
            dropout=vars(args)["model.reg_head_mlp.dropout"],
            activation=vars(args)["model.reg_head_mlp.activation"],
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
            step_lr_step_size=vars(args)["train.step_lr_step_size"],
            step_lr_gamma=vars(args)["train.step_lr_gamma"],
            momentum=vars(args)["train.momentum"],
            betas=tuple(vars(args)["train.betas"]),
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
            converge_ratio=vars(args)["sel.converge_ratio"],
        )

        # ---------- CrossValidationConfig ----------
        cv_cfg = CrossValidationConfig(
            search_strategy=vars(args)["cv.search_strategy"],
            n_trials=vars(args)["cv.n_trials"],
            outer_folds=vars(args)["cv.outer_folds"],
            inner_folds=vars(args)["cv.inner_folds"],
        )

        # ---------- DataConfig ----------
        if vars(args)["dataset_name"] == "UTA":
            data_cfg = DataConfig(
                dataset_name="UTA",
                embeddings_dir="/work/long_lab/Mojtaba/data/UTA/3.1-embedding/dinov2-base/cls/ROI0",
                features_dir="/work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05/features",
                eventfulness_scores_dir="/work/long_lab/Mojtaba/data/UTA/3-augmented_roi/masked_aligned_cropped0.05/eventfulness",
                normalization_info_path = "/work/long_lab/Mojtaba/data/UTA/3.2.1-subjectwise_normalization_info_aggregated/normalization_info.csv"
                
            )
        elif vars(args)["dataset_name"] == "DROZY":
            data_cfg = DataConfig(
                dataset_name="DROZY",
                embeddings_dir="/work/long_lab/Mojtaba/data/DROZY/data/videos/3.1-embedding/dinov2-base/cls/ROI0",
                features_dir="/work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/features",
                eventfulness_scores_dir="/work/long_lab/Mojtaba/data/DROZY/data/videos/3-augmented_roi/masked_aligned_cropped0.05/eventfulness",
                normalization_info_path = "/work/long_lab/Mojtaba/data/DROZY/data/videos/3.2.1-subjectwise_normalization_info_aggregated/normalization_info.csv"
            )

        data_cfg.field_normalization = {
            "embeddings": {
                "modality": "embeddings_ROI0",  # or ROI1/ROI2/ROI3, depending on which you use
                "indices": None,                # use full vector
            },
            "blendshapes": {
                "modality": "features",        # features modality contains several kinds
                "indices": list(range(7, 59))  # first 52 dims are blendshapes
            },
            "others": {
                "modality": "features",
                "indices": [1, 2, 3, 4],  # example: pick only some feature dims
            },
        }

        return model_cfg, train_cfg, selection_cfg, cv_cfg, data_cfg

    args = build_parser().parse_args()
    (model_cfg, train_cfg, selection_cfg, cv_cfg, data_cfg) = build_configs_from_args(
        args
    )

    video_ids = get_file_paths(
        input_path=data_cfg.embeddings_dir, depth=3, EXT_LIST=".dat"
    )

    search_space = {
        "train.lr": ("log_uniform", 1e-5, 1e-2),
        # "train.weight_decay": ("log_uniform", 1e-6, 1e-3),
        "train.batch_size": [32, 64],
        "model.temporal.hidden_size": [64, 128],
        "model.temporal.num_layers": [1, 2, 3],
        "model.temporal.dropout": ("uniform", 0.1, 0.5),
        "sel.window_size": [32, 64]

        # "model.temporal.backbone": ["gru", "lstm", "transformer"],
    }

    output_dir = f"/home/mojtaba.kananisarche/Zh/outputs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}|E13"
    create_folders(output_dir)
    save_obj(vars(args), os.path.join(output_dir, "args.json"), "json")
    save_obj(search_space, os.path.join(output_dir, "search_space.json"), "json")

    results = nested_cv_video_mil(
        video_ids=video_ids,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        selection_cfg=selection_cfg,
        data_cfg=data_cfg,
        cv_cfg=cv_cfg,
        search_space=search_space,
        output_dir=output_dir,
        random_seed=vars(args)["seed"],
        device=setup_torch(vars(args)["train.device"]),
        parallel_inner_cv=vars(args)["parallel_inner_cv"],
        max_workers=vars(args)["max_workers"] if vars(args)["parallel_inner_cv"] else 1,
    )
    save_obj(results, os.path.join(output_dir, "results.json"), "json")

    """
    from Zh.experiments.e11.dataclasses import ModelConfig, TrainingConfig, SelectionConfig
    
    # Define search space for Bayesian optimization
    def search_space(trial):
        return {
            "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "temporal_hidden_size": trial.suggest_categorical("hidden", [64, 128, 256]),
            "temporal_num_layers": trial.suggest_int("layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5)
        }
    
    results = nested_cv_video_mil(
        video_ids=my_video_ids,
        model_cfg=my_model_cfg,
        train_cfg=my_train_cfg,
        selection_cfg=my_selection_cfg,
        embeddings_root="/path/to/embeddings",
        features_root="/path/to/features",
        eventfulness_scores_root="/path/to/eventfulness",
        outer_folds=5,
        inner_folds=3,
        output_dir="./nested_cv_results",
        search_strategy="bayesian",
        search_space=search_space,
        n_trials=30,
        random_seed=42
    )
    """

    # Example 2: Random search
    """
    results = nested_cv_video_mil(
        video_ids=my_video_ids,
        model_cfg=my_model_cfg,
        train_cfg=my_train_cfg,
        selection_cfg=my_selection_cfg,
        embeddings_root="/path/to/embeddings",
        features_root="/path/to/features",
        eventfulness_scores_root="/path/to/eventfulness",
        outer_folds=5,
        inner_folds=3,
        output_dir="./nested_cv_results",
        search_strategy="random",
        search_space={
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "weight_decay": ("log_uniform", 1e-6, 1e-3),
            "temporal_hidden_size": [64, 128, 256]
        },
        n_trials=20,
        random_seed=42
    )
    """

    # Example 3: Grid search
    """
    results = nested_cv_video_mil(
        video_ids=my_video_ids,
        model_cfg=my_model_cfg,
        train_cfg=my_train_cfg,
        selection_cfg=my_selection_cfg,
        embeddings_root="/path/to/embeddings",
        features_root="/path/to/features",
        eventfulness_scores_root="/path/to/eventfulness",
        outer_folds=5,
        inner_folds=3,
        output_dir="./nested_cv_results",
        search_strategy="grid",
        search_space={
            "learning_rate": [1e-4, 1e-3],
            "weight_decay": [0.0, 1e-5],
            "temporal_hidden_size": [128, 256]
        }
    )
    """


if __name__ == "__main__":
    main()
