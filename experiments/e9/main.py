import json
import logging
import os
import pickle
import random
from datetime import datetime
from locale import strcoll
from typing import Dict, List, Optional, Tuple, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import layers, losses
from keras.src.callbacks import Callback, EarlyStopping, History, ModelCheckpoint
from keras.src.layers import Lambda
from keras.src.models.sequential import Sequential
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow import keras

from Zh.src.preprocess.reshape import count_slices_per_video, reshape_data_rnn
from Zh.src.utils.datagenerator import MyDataGenerator
from Zh.src.utils.file_folder import create_folders, get_file_paths
from Zh.src.utils.report import report
from Zh.src.utils.serialized_parallel import process

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


class EpochEndLoggingCallback(Callback):
    def __init__(
        self,
        X_train,
        y_train,
        group_identifiers_train,
        X_val,
        y_val,
        group_identifiers_val,
        X_test,
        y_test,
        group_identifiers_test,
        mapping,
        index,
    ):

        super().__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.group_identifiers_train = group_identifiers_train

        self.X_val = X_val
        self.y_val = y_val
        self.group_identifiers_val = group_identifiers_val

        self.X_test = X_test
        self.y_test = y_test
        self.group_identifiers_test = group_identifiers_test

        self.mapping = mapping
        self.index = index

    def on_epoch_end(self, epoch, logs=None):
        # print(f"Epoch {epoch+1} ended. Logs: {logs}")

        _, train_report = report(
            model=self.model,
            x=self.X_train,
            y_true=self.y_train,
            group_identifiers=self.group_identifiers_train,
            unique_classes=list(self.mapping.values()),
        )
        _, val_report = report(
            model=self.model,
            x=self.X_val,
            y_true=self.y_val,
            group_identifiers=self.group_identifiers_val,
            unique_classes=list(self.mapping.values()),
        )
        _, test_report = report(
            model=self.model,
            x=self.X_test,
            y_true=self.y_test,
            group_identifiers=self.group_identifiers_test,
            unique_classes=list(self.mapping.values()),
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
        units,
        dropout_ratio,
        kernel_regularizer_l2,
        kernel_regularizer_l1,
        batch_size,
        learning_rate,
        weight_decay,
        beta_1,
        beta_2,
        epsilon,
        epochs,
        patience,
        slice_size,
        step_size,
        discard_threshold_ratio,
        mapping,
    ):
        self.units = units
        self.dropout_ratio = dropout_ratio
        self.kernel_regularizer_l2 = kernel_regularizer_l2
        self.kernel_regularizer_l1 = kernel_regularizer_l1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.epochs = epochs
        self.patience = patience
        self.slice_size = slice_size
        self.step_size = step_size
        self.discard_threshold_ratio = discard_threshold_ratio
        self.mapping = mapping

    def _build_model(self, input_shape) -> Sequential:
        model = Sequential()
        model.add(layers.Input(shape=(input_shape[0], input_shape[1])))

        # Add LSTM layers
        for u in self.units[:-1]:
            model.add(
                layers.LSTM(
                    u,
                    activation="tanh",
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.kernel_regularizer_l2
                    ),
                )
            )
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_ratio))

        # Final LSTM layer without return_sequences
        model.add(
            layers.LSTM(
                self.units[-1],
                activation="tanh",
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_l2),
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.dropout_ratio))

        # Final output layer
        model.add(layers.Dense(1, activation="linear"))  # Use sigmoid activation

        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                # beta_1=self.beta_1,
                # beta_2=self.beta_2,
                # epsilon=self.epsilon
            ),
            loss="mse",
            metrics=["mae"],
        )
        return model

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

    def _get_train_val_test_file_paths(
        input_path: str,
        primary_column_name: str,
        secondary_column_name: Optional[str] = None,
    ) -> List[Dict]:

        file_paths = get_file_paths(folder_path=input_path, depth=2, EXT_LIST=[".csv"])
        data = pd.DataFrame()
        data["path"] = file_paths
        data["fold"] = (
            data["path"]
            .apply(lambda x: os.path.basename(x).split("_")[0])
            .astype(int)
            .astype(str)
        )
        data["subject_id"] = (
            data["path"]
            .apply(lambda x: os.path.basename(x).split("_")[1])
            .astype(int)
            .astype(str)
        )
        data = data.sort_values(["fold", "subject_id"]).reset_index(drop=True)

        columns_to_fold_by = data[
            [primary_column_name]
            + ([secondary_column_name] if secondary_column_name else [])
        ]

        train_val_test_file_paths = []
        unique_main_column_values = sorted(
            columns_to_fold_by[primary_column_name].unique()
        )

        if primary_column_name and secondary_column_name:
            for main_column_value in unique_main_column_values:
                corresponding_secondary_value = columns_to_fold_by[
                    columns_to_fold_by[primary_column_name] == main_column_value
                ][secondary_column_name].unique()
                assert (
                    len(corresponding_secondary_value) == 1
                ), f"Multiple Unique Secondary Values Found For Main Column Value {main_column_value} - {corresponding_secondary_value}"
                corresponding_secondary_value = corresponding_secondary_value[0]

                fold_train_val_test_file_paths = {"info": {}, "paths": {}}

                test_split = columns_to_fold_by[
                    columns_to_fold_by[primary_column_name] == main_column_value
                ]
                fold_train_val_test_file_paths["paths"]["test"] = data.loc[
                    test_split.index.to_list()
                ]["path"].to_list()
                fold_train_val_test_file_paths["info"]["test"] = {
                    f"{primary_column_name}": sorted(
                        test_split[primary_column_name].unique().tolist()
                    )
                }

                val_split = columns_to_fold_by[
                    (columns_to_fold_by[primary_column_name] != main_column_value)
                    & (
                        columns_to_fold_by[secondary_column_name]
                        == corresponding_secondary_value
                    )
                ]
                fold_train_val_test_file_paths["paths"]["val"] = data.loc[
                    val_split.index.to_list()
                ]["path"].to_list()
                fold_train_val_test_file_paths["info"]["val"] = {
                    f"{primary_column_name}": sorted(
                        val_split[primary_column_name].unique().tolist()
                    )
                }

                train_split = columns_to_fold_by[
                    columns_to_fold_by[secondary_column_name]
                    != corresponding_secondary_value
                ]
                fold_train_val_test_file_paths["paths"]["train"] = data.loc[
                    train_split.index.to_list()
                ]["path"].to_list()
                fold_train_val_test_file_paths["info"]["train"] = {
                    f"{primary_column_name}": sorted(
                        train_split[primary_column_name].unique().tolist()
                    )
                }
                train_val_test_file_paths.append(fold_train_val_test_file_paths)

        elif primary_column_name:
            for main_column_value_i in unique_main_column_values:
                for main_column_value_j in unique_main_column_values:
                    if main_column_value_i != main_column_value_j:
                        fold_train_val_test_file_paths = {"info": {}, "paths": {}}

                        test_split = columns_to_fold_by[
                            columns_to_fold_by[primary_column_name]
                            == main_column_value_i
                        ]
                        fold_train_val_test_file_paths["paths"]["test"] = data.loc[
                            test_split.index.to_list()
                        ]["path"].to_list()
                        fold_train_val_test_file_paths["info"]["test"] = {
                            f"{primary_column_name}": sorted(
                                test_split[primary_column_name].unique().tolist()
                            )
                        }

                        val_split = columns_to_fold_by[
                            columns_to_fold_by[primary_column_name]
                            == main_column_value_j
                        ]
                        fold_train_val_test_file_paths["paths"]["val"] = data.loc[
                            val_split.index.to_list()
                        ]["path"].to_list()
                        fold_train_val_test_file_paths["info"]["val"] = {
                            f"{primary_column_name}": sorted(
                                val_split[primary_column_name].unique().tolist()
                            )
                        }

                        train_split = columns_to_fold_by[
                            ~columns_to_fold_by[primary_column_name].isin(
                                [main_column_value_i, main_column_value_j]
                            )
                        ]
                        fold_train_val_test_file_paths["paths"]["train"] = data.loc[
                            train_split.index.to_list()
                        ]["path"].to_list()
                        fold_train_val_test_file_paths["info"]["train"] = {
                            f"{primary_column_name}": sorted(
                                train_split[primary_column_name].unique().tolist()
                            )
                        }

                        train_val_test_file_paths.append(fold_train_val_test_file_paths)

        return train_val_test_file_paths

    def _train_single_fold(
        self,
        split_info: Dict[str, Dict[str, List[str]]],
        split_paths: Dict[str, List[str]],
        video_info_csv_path: str,
        output_folder_path: str,
        columns_to_drop: Optional[List[str]] = [],
        workers: Optional[int] = 1,
        index: Optional[int] = 0,
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
            paths=split_paths["train"],
            video_info_csv_path=video_info_csv_path,
            mapping=self.mapping,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
            columns_to_drop=columns_to_drop,
            batch_size=self.batch_size,
            slice_size=self.slice_size,
            step_size=self.step_size,
            discard_threshold_ratio=self.discard_threshold_ratio,
            shuffle=False, #Not connected currently. will be deleted probably as it is not needed
            workers=workers,
            use_multiprocessing=True,
        )

        val_data_gen = MyDataGenerator(
            paths=split_paths["val"],
            video_info_csv_path=video_info_csv_path,
            mapping=self.mapping,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
            columns_to_drop=columns_to_drop,
            batch_size=self.batch_size,
            slice_size=self.slice_size,
            step_size=self.step_size,
            discard_threshold_ratio=self.discard_threshold_ratio,
            shuffle=False, #Not connected currently. will be deleted probably as it is not needed
            workers=workers,
            use_multiprocessing=True,
        )

        test_data_gen = MyDataGenerator(
            paths=split_paths["test"],
            video_info_csv_path=video_info_csv_path,
            mapping=self.mapping,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
            columns_to_drop=columns_to_drop,
            batch_size=self.batch_size,
            slice_size=self.slice_size,
            step_size=self.step_size,
            discard_threshold_ratio=self.discard_threshold_ratio,
            shuffle=False, #Not connected currently. will be deleted probably as it is not needed
            workers=workers,
            use_multiprocessing=True,
        )

        # sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        logging.info(f"{index} - building model")
        model = self._build_model(
            (self.slice_size, len(train_data_gen.feature_columns))
        )

        logging.info(f"{index} - fitting model")
        # Train the model

        # epoch_end_logging = EpochEndLoggingCallback(
        #     X_train=X_train,
        #     y_train=y_train,
        #     group_identifiers_train=group_identifiers_train,
        #     X_val=X_val,
        #     y_val=y_val,
        #     group_identifiers_val=group_identifiers_val,
        #     X_test=X_test,
        #     y_test=y_test,
        #     group_identifiers_test=group_identifiers_test,
        #     mapping=self.mapping,
        #     index=index,
        # )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )

        checkpoint = ModelCheckpoint(
            os.path.join(output_folder_path, f"{index}_best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        )

        history = model.fit(
            x=train_data_gen,
            validation_data=val_data_gen,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, checkpoint],  # TODO Add epoch_end_logging
            verbose=1,
        )

        logging.info(f"{index} - getting reports")

        y_train_pred, train_report = report(
            model=model,
            x=train_data_gen,
            y=train_data_gen.true_labels['y'],
            group_identifiers=train_data_gen.true_labels['g'],
            unique_classes=list(self.mapping.values()),
        )
        y_val_pred, val_report = report(
            model=model,
            x=val_data_gen,
            y=val_data_gen.true_labels['y'],
            group_identifiers=val_data_gen.true_labels['g'],
            unique_classes=list(self.mapping.values()),
        )
        y_test_pred, test_report = report(
            model=model,
            x=test_data_gen,
            y=test_data_gen.true_labels['y'],
            group_identifiers=test_data_gen.true_labels['g'],
            unique_classes=list(self.mapping.values()),
        )

        # Store results to return
        local_results = {
            "split_info": {
                "index": index,
                "test": split_info["test"],
                "val": split_info["val"],
                "train": split_info["train"],
            },
            "evaluation": {
                "train": {
                    "metrics": train_report,
                    "group_identifiers": train_data_gen.true_labels["g"],
                    "y_true": train_data_gen.true_labels["y"],
                    "y_pred": y_train_pred,
                },
                "val": {
                    "metrics": val_report,
                    "group_identifiers": val_data_gen.true_labels["g"],
                    "y_true": train_data_gen.true_labels["y"],
                    "y_pred": y_val_pred,
                },
                "test": {
                    "metrics": test_report,
                    "group_identifiers": test_data_gen.true_labels["g"],
                    "y_true": train_data_gen.true_labels["y"],
                    "y_pred": y_test_pred,
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
        input_directory_path: str,
        split_info_csv_path: str,
        columns_to_drop: Optional[List[str]] = [],
        parallel: bool = False,
        max_workers: Optional[int] = None,
        maxtasksperchild: Optional[int] = 1,
        use_tqdm: bool = False,
        output_parent_dir: str = "/home/mojtaba.kananisarche/Zh/UTA/Output/",
        fold_by_primary_column_name: str = None,
        fold_by_secondary_column_name: str = None,
        fold_start_index: Optional[int] = None,
        fold_end_index: Optional[int] = None,
        fold_indices: Optional[List[int]] = None,
        workers: Optional[int] = 1,
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
                        "units": self.units,
                        "dropout_ratio": self.dropout_ratio,
                        "batch_size": self.batch_size,
                        "learning_rate": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "epochs": self.epochs,
                        "slice_size": self.slice_size,
                        "step_size": self.step_size,
                        "mapping": self.mapping,
                        "discard_threshold_ratio": self.discard_threshold_ratio,
                        "input_directory_path": input_directory_path,
                        "video_info_csv_path": video_info_csv_path,
                        "columns_to_drop": columns_to_drop,
                        "output_folder_name": output_folder_name,
                        "fold_by_primary_column_name": fold_by_primary_column_name,
                        "fold_by_secondary_column_name": fold_by_secondary_column_name,
                        "fold_start_index": fold_start_index,
                        "fold_end_index": fold_end_index,
                    }
                },
                file,
            )
        with open(split_info_csv_path, 'r') as json_file:
            train_val_test_indices = json.load(json_file)

        # train_val_test_indices = self._get_train_val_test_file_paths(
        #     input_path=input_directory_path,
        #     primary_column_name=fold_by_primary_column_name,
        #     secondary_column_name=fold_by_secondary_column_name,
        # )

        args_list = [
            {
                "split_info": split["info"],
                "split_paths": split["paths"],
                "video_info_csv_path": os.path.join(os.path.commonpath(split['paths']['train'] + 
                                                                       split['paths']['test'] + 
                                                                       split['paths']['val']), 
                                                    split['index']),
                "output_folder_path": output_folder_path,
                "columns_to_drop": columns_to_drop,
                "workers": workers,
                "index": split["index"],
            }
            for split in train_val_test_indices
        ]

        logging.info(f"Found {len(args_list)} valid test-validation pairs.")
        logging.info(
            "\n".join([f"{i['index']} - {i['split_info']}" for i in args_list])
        )

        random.shuffle(args_list)

        selected_indices = fold_indices or slice(fold_start_index, fold_end_index)
        args_list = args_list[selected_indices]
        logging.info(f"Processing {len(args_list)} selected test-validation pairs.")
        logging.info(
            "\n".join([f"{i['index']} - {i['split_info']}" for i in args_list])
        )

        results = process(
            worker_function=self._train_single_fold,
            handler_function=self._save_result,
            worker_params_list=args_list,
            parallel=parallel,
            max_workers=max_workers,
            maxtasksperchild=maxtasksperchild,
            use_tqdm=use_tqdm,
        )

        logging.info(f"Completed all {len(args_list)} test-validation pairs.")
