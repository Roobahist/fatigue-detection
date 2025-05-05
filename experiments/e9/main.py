import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import Callback, EarlyStopping, History, ModelCheckpoint

from Zh.src.utils.datagenerator import MyDataGenerator
from Zh.src.utils.file_folder import create_folders, get_file_paths
from Zh.src.utils.model_tuner import *
from Zh.src.utils.report import report
from Zh.src.utils.serialized_parallel import process

# from keras import backend as K
# from keras import layers, losses
# from keras.src.layers import Lambda
# from keras.src.models.sequential import Sequential
# from sklearn.decomposition import PCA
# from sklearn.utils.class_weight import compute_sample_weight
# from tensorflow import keras


logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return round(float(obj), 2)
        elif isinstance(obj, (np.ndarray)):
            return [
                round(float(x), 2) if isinstance(x, (np.floating, float)) else x
                for x in obj.tolist()
            ]
        return super(NumpyEncoder, self).default(obj)


# fmt: off
class EpochEndLoggingCallback(Callback):
    def __init__(
        self,
        mapping                : Dict,
        index                  : int,
        X_train                : Union[np.ndarray, MyDataGenerator],
        X_val                  : Union[np.ndarray, MyDataGenerator],
        X_test                 : Union[np.ndarray, MyDataGenerator],
        y_train                : Union[np.ndarray, None] = None,
        y_val                  : Union[np.ndarray, None] = None,
        y_test                 : Union[np.ndarray, None] = None,
        group_identifiers_train: Union[np.ndarray, None] = None,
        group_identifiers_val  : Union[np.ndarray, None] = None,
        group_identifiers_test : Union[np.ndarray, None] = None,
    ):
        """
        Callback to log metrics at the end of each epoch using data generators.

        Args:
            train_generator: Generator for training data
            val_generator: Generator for validation data
            test_generator: Generator for test data
            mapping: Dictionary mapping class indices to class names
            index: Index identifier for logging
            steps_per_epoch: Number of batches per epoch for training (optional)
            validation_steps: Number of batches for validation (optional)
            test_steps: Number of batches for testing (optional)
        """
        super().__init__()

        self.mapping = mapping
        self.index   = index

        self.X_train                 = X_train
        self.y_train                 = y_train
        self.group_identifiers_train = group_identifiers_train
        
        self.X_val                   = X_val
        self.y_val                   = y_val
        self.group_identifiers_val   = group_identifiers_val

        self.X_test                  = X_test
        self.y_test                  = y_test
        self.group_identifiers_test  = group_identifiers_test

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch to log metrics for each dataset."""
        _, train_report = report(
            model             = self.model,
            x                 = self.X_train,
            y_true            = self.y_train,
            group_identifiers = self.group_identifiers_train,
            unique_classes    = list(self.mapping.values()),
        )
        _, val_report = report(
            model             = self.model,
            x                 = self.X_val,
            y_true            = self.y_val,
            group_identifiers = self.group_identifiers_val,
            unique_classes    = list(self.mapping.values()),
        )
        _, test_report = report(
            model             = self.model,
            x                 = self.X_test,
            y_true            = self.y_test,
            group_identifiers = self.group_identifiers_test,
            unique_classes    = list(self.mapping.values()),
        )

        logging.info(
            "\n"
            "============================\n"
            f"{self.index} - Epoch {epoch + 1}\n"
            f"Train Report: {train_report['slice']['accuracy']}\n"
            f"Validation Report: {val_report['slice']['accuracy']}\n"
            f"Test Report: {test_report['slice']['accuracy']}\n"
            f"logs: {logs}\n"
            "============================"
        )


class Model:
    def __init__(
        self,
        batch_size,
        epochs,
        patience,
        slice_size,
        step_size,
        discard_threshold_ratio,
        mapping,
    ):
        self.batch_size              = batch_size
        self.epochs                  = epochs
        self.patience                = patience
        self.slice_size              = slice_size
        self.step_size               = step_size
        self.discard_threshold_ratio = discard_threshold_ratio
        self.mapping                 = mapping

    # fmt: on
    def _save_result(
        self,
        result,
        output_folder_path: Optional[str] = "/home/mojtaba.kananisarche/Zh/UTA/Output/",
    ):
        logging.info(
            f"{result['split_info']['index']} - ======================= Saving ======================="
        )
        result_file_path = os.path.join(
            output_folder_path, f"{result['split_info']['index']}.json"
        )
        with open(result_file_path, "w") as file:
            json.dump(result, file, cls=NumpyEncoder, indent=4)

        logging.info(
            f"{result['split_info']['index']} - {result_file_path}\n"
            f"{result['split_info']['index']} - ======================= Saved ======================="
        )

    # fmt: off
    def _train_single_fold(
        self,
        split_info                : Dict[str, Dict[str, List[str]]],
        split_paths               : Dict[str, List[str]],
        video_info_csv_path       : str,
        column_names              : Union[list[str], None] = None,
        output_folder_path        : str = '.',
        loss_function             : Union[str, tf.keras.losses.Loss] = "mse",
        metrics                   : List[Union[str, tf.keras.metrics.Metric]] = ["mae"],
        hp_config                 : Optional[Dict[str, Dict[str, Any]]] = None,
        columns_to_drop           : Optional[List[str]] = [],
        data_gen_workers          : Optional[int] = 1,
        tuner_model_name          : str = ALL_MODELS[0],
        tuner_max_trials          : int = 15,
        tuner_executions_per_trial: int = 1,
        tuner_epochs              : int = 50,
        tuner_patience            : int = 10,
        tuner_type                : str = "bayesian",
        index                     : Optional[int] = 0,
    ):
        """Helper function for parallel processing in test_model.

        Args:
            args: Tuple containing (model_instance, test_idx, val_idx, data)

        Returns:
            Dictionary containing model, history, and reports
        """
        logging.info(
            f"{index} - ======================= Starting =======================\n"
            f"{index} - {split_info}"
        )

        groupby_column, sortby_column, target_column = "file_name", "frame", "target"
        
        train_data_gen = MyDataGenerator(
            paths                   = split_paths["train"],
            video_info_csv_path     = video_info_csv_path,
            mapping                 = self.mapping,
            column_names            = column_names,
            groupby_column          = groupby_column,
            sortby_column           = sortby_column,
            target_column           = target_column,
            columns_to_drop         = columns_to_drop,
            batch_size              = self.batch_size,
            slice_size              = self.slice_size,
            step_size               = self.step_size,
            discard_threshold_ratio = self.discard_threshold_ratio,
            shuffle                 = False,                                   # Not connected currently. will be deleted probably as it is not needed
            workers                 = data_gen_workers,
            use_multiprocessing     = True if data_gen_workers > 1 else False,
        )

        val_data_gen = MyDataGenerator(
            paths                   = split_paths["val"],
            video_info_csv_path     = video_info_csv_path,
            mapping                 = self.mapping,
            column_names            = column_names,
            groupby_column          = groupby_column,
            sortby_column           = sortby_column,
            target_column           = target_column,
            columns_to_drop         = columns_to_drop,
            batch_size              = self.batch_size,
            slice_size              = self.slice_size,
            step_size               = self.step_size,
            discard_threshold_ratio = self.discard_threshold_ratio,
            class_weights           = train_data_gen.class_weights, 
            shuffle                 = False,                                   # Not connected currently. will be deleted probably as it is not needed
            workers                 = data_gen_workers,
            use_multiprocessing     = True if data_gen_workers > 1 else False,
        )

        test_data_gen = MyDataGenerator(
            paths                   = split_paths["test"],
            video_info_csv_path     = video_info_csv_path,
            mapping                 = self.mapping,
            column_names            = column_names,
            groupby_column          = groupby_column,
            sortby_column           = sortby_column,
            target_column           = target_column,
            columns_to_drop         = columns_to_drop,
            batch_size              = self.batch_size,
            slice_size              = self.slice_size,
            step_size               = self.step_size,
            discard_threshold_ratio = self.discard_threshold_ratio,
            class_weights           = train_data_gen.class_weights, 
            shuffle                 = False,                                   # Not connected currently. will be deleted probably as it is not needed
            workers                 = data_gen_workers,
            use_multiprocessing     = True if data_gen_workers > 1 else False,
        )

        # sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        logging.info(f"{index} - building model")

        if tuner_model_name not in ALL_MODELS:
            logging.info(f"{index} - Model Default Used {ALL_MODELS[0]}")
            tuner_model_name = ALL_MODELS[0]

        model = ModelTuner(
            model_class   = eval(tuner_model_name),
            input_shape   = (self.slice_size, len(train_data_gen.feature_columns)),
            loss_function = loss_function,
            metrics       = metrics,
            hp_config     = hp_config,
        )

        logging.info(f"{index} - Finding Best HP")

        best_hps, tuner = model.tune_model(
            train_data           = train_data_gen,
            val_data             = val_data_gen,
            directory            = output_folder_path,
            project_name         = str(index),
            max_trials           = tuner_max_trials,
            executions_per_trial = tuner_executions_per_trial,
            epochs               = tuner_epochs,               # using lower number of epochs for finding the best hp
            batch_size           = self.batch_size,
            patience             = tuner_patience,
            tuner_type           = tuner_type,
            overwrite            = False,
        )

        best_model = model.get_best_model(tuner)

        # epoch_end_logging = EpochEndLoggingCallback(
        #     X_train        = train_data_gen,
        #     X_val          = val_data_gen,
        #     X_test         = test_data_gen,
        #     mapping        = self.mapping,
        #     index          = index,
        # )

        early_stop = EarlyStopping(
            monitor              = "val_loss",
            patience             = self.patience,
            restore_best_weights = True,
        )

        checkpoint = ModelCheckpoint(
            os.path.join(output_folder_path, f"{index}_best_model.keras"),
            monitor           = "val_loss",
            save_best_only    = True,
            save_weights_only = False,
        )

        logging.info(f"{index} - Fitting Best HP {best_hps}")

        history = best_model.fit(
            x               = train_data_gen,
            validation_data = val_data_gen,
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            callbacks       = [early_stop, checkpoint],
            verbose         = 1,
        )

        logging.info(f"{index} - getting reports")

        y_train_pred, train_report = report(
            model             = best_model,
            x                 = train_data_gen,
            unique_classes    = list(self.mapping.values()),
        )
        y_val_pred, val_report = report(
            model             = best_model,
            x                 = val_data_gen,
            unique_classes    = list(self.mapping.values()),
        )
        y_test_pred, test_report = report(
            model             = best_model,
            x                 = test_data_gen,
            unique_classes    = list(self.mapping.values()),
        )

        # Store results to return
        local_results = {
            "split_info": {
                "index": index,
                "test" : split_info["test"],
                "val"  : split_info["val"],
                "train": split_info["train"],
            },
            "evaluation": {
                "train": {
                    "metrics"          : train_report,
                    "group_identifiers": train_data_gen.true_labels["g"],
                    "y_true"           : train_data_gen.true_labels["y"],
                    "y_pred"           : y_train_pred,
                },
                "val": {
                    "metrics"          : val_report,
                    "group_identifiers": val_data_gen.true_labels["g"],
                    "y_true"           : train_data_gen.true_labels["y"],
                    "y_pred"           : y_val_pred,
                },
                "test": {
                    "metrics"          : test_report,
                    "group_identifiers": test_data_gen.true_labels["g"],
                    "y_true"           : train_data_gen.true_labels["y"],
                    "y_pred"           : y_test_pred,
                },
            },
            "training_history": {
                "history": history.history,
            },
        }

        logging.info(
            f"epochs : {len(history.history['val_loss'])}\n"
            f"{index} - {split_info}"
            f"{index} - ======================= Ending =======================\n"
        )

        return {
            "result": local_results,
            "output_folder_path": output_folder_path,
        }

    def test_model(
        self,
        input_directory_path         : str,
        split_info_csv_path          : str,
        column_names_json_path       : str = None,
        columns_to_drop              : Optional[List[str]] = [],
        parallel                     : bool = False,
        max_workers                  : Optional[int] = None,
        maxtasksperchild             : Optional[int] = 1,
        use_tqdm                     : bool = False,
        output_parent_dir            : str = "/home/mojtaba.kananisarche/Zh/UTA/Output/",
        fold_by_primary_column_name  : str = None,
        fold_by_secondary_column_name: str = None,
        fold_start_index             : Optional[int] = None,
        fold_end_index               : Optional[int] = None,
        fold_indices                 : Optional[List[int]] = None,
        data_gen_workers             : Optional[int] = 1,
        tuner_model_name             : str = ALL_MODELS[0],
        tuner_max_trials             : int = 15,
        tuner_executions_per_trial   : int = 1,
        tuner_epochs                 : int = 50,
        tuner_patience               : int = 10,
        tuner_type                   : str = "bayesian",
        loss_function                : Union[str, tf.keras.losses.Loss] = "mse",
        metrics                      : List[Union[str, tf.keras.metrics.Metric]] = ["mae"],
        hp_config                    : Optional[Dict[str, Dict[str, Any]]] = None,
    ):

        output_folder_name = datetime.now().strftime(
            "%Y-%m-%d|%H:%M:%S|RNN_FACENET_5CV"
        )
        output_folder_path = os.path.join(output_parent_dir, output_folder_name)
        create_folders(output_folder_path)

        with open(os.path.join(output_folder_path, "results.json"), "w") as file:
            json.dump(
                {
                    "general": {
                        "batch_size"                   : self.batch_size,
                        "epochs"                       : self.epochs,
                        "slice_size"                   : self.slice_size,
                        "step_size"                    : self.step_size,
                        "mapping"                      : self.mapping,
                        "discard_threshold_ratio"      : self.discard_threshold_ratio,
                        "input_directory_path"         : input_directory_path,
                        "split_info_csv_path"          : split_info_csv_path,
                        "column_names_json_path"       : column_names_json_path,
                        "columns_to_drop"              : columns_to_drop,
                        "output_folder_name"           : output_folder_name,
                        "fold_by_primary_column_name"  : fold_by_primary_column_name,
                        "fold_by_secondary_column_name": fold_by_secondary_column_name,
                        "fold_start_index"             : fold_start_index,
                        "fold_end_index"               : fold_end_index,
                        "fold_indices"                 : fold_indices,
                        "tuner_model_name"             : tuner_model_name,
                        "tuner_max_trials"             : tuner_max_trials,
                        "tuner_executions_per_trial"   : tuner_executions_per_trial,
                        "tuner_epochs"                 : tuner_epochs,
                        "tuner_patience"               : tuner_patience,
                        "tuner_type"                   : tuner_type,
                        "hp"                           : hp_config,
                        "loss_function"                : loss_function,
                        "metrics"                      : metrics,
                    }
                },
                file,
            )
        
        if split_info_csv_path is not None and os.path.isfile(split_info_csv_path):
            with open(split_info_csv_path, "r") as json_file:
                train_val_test_indices = json.load(json_file)
        else:
            raise ValueError("split_info_csv_path is not a file")
        
        if column_names_json_path is not None and os.path.isfile(column_names_json_path):
            with open(column_names_json_path, "r") as json_file:
                column_names = json.load(json_file)
        else:
            logging.warning("column_names_json_path is not a file")
            column_names = None

        print(f"Column Names - {column_names}")

        args_list = [
            {
                "split_info"                : split["info"],
                "split_paths"               : split["output_paths"],
                "video_info_csv_path"       : os.path.join(
                    os.path.commonpath(
                        split["output_paths"]["train"] +
                        split["output_paths"]["test"] +
                        split["output_paths"]["val"]
                    ),
                    "video_info.csv",
                ),
                "column_names"              : column_names,
                "output_folder_path"        : output_folder_path,
                "columns_to_drop"           : columns_to_drop,
                "data_gen_workers"          : data_gen_workers,
                "tuner_model_name"          : tuner_model_name,
                "tuner_max_trials"          : tuner_max_trials,
                "tuner_executions_per_trial": tuner_executions_per_trial,
                "tuner_epochs"              : tuner_epochs,
                "tuner_patience"            : tuner_patience,
                "tuner_type"                : tuner_type,
                "loss_function"             : loss_function,
                "metrics"                   : metrics,
                "hp_config"                 : hp_config,
                "index"                     : split["index"],
            }
            for split in train_val_test_indices
        ]

        logging.info(f"Found {len(args_list)} valid test-validation pairs.")
        logging.info(
            "\n".join([f"{i['index']} - {i['split_info']}" for i in args_list])
        )

        random.shuffle(args_list)

        args_list = (
            [args_list[i] for i in fold_indices]
            if fold_indices is not None
            else args_list[fold_start_index:fold_end_index]
        )

        logging.info(f"Processing {len(args_list)} selected test-validation pairs.")
        logging.info(
            "\n".join([f"{i['index']} - {i['split_info']}" for i in args_list])
        )

        results = process(
            worker_function    = self._train_single_fold,
            handler_function   = self._save_result,
            worker_params_list = args_list,
            parallel           = parallel,
            max_workers        = max_workers,
            maxtasksperchild   = maxtasksperchild,
            use_tqdm           = use_tqdm,
        )

        logging.info(f"Completed all {len(args_list)} test-validation pairs.")
