import json
import logging
import os
import pickle
import random
from datetime import datetime
from typing import List, Optional, Tuple, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import layers, losses
from keras.layers import Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.src.callbacks import History
from sklearn.decomposition import PCA
from sklearn.metrics import (  # , root_mean_squared_error
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from Zh.src.utils.file_folder import create_folders
from Zh.src.utils.serialized_parallel import *

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(asctime)s][%(levelname)s] %(message)s",
)


# Function to set the random seed
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


def reshape_data_rnn(
    df: pd.DataFrame,
    feature_columns: List[str],
    slice_size: int,
    step_size: Optional[int],
    groupby_column: str = "file_name",
    sortby_column: str = "frame",
    target_column: str = "target",
    discard_threshold_ratio: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data for RNN input by grouping, slicing, and padding sequences.

    Args:
        df (pd.DataFrame)              : Input dataframe.
        feature_columns (List[str])    : List of feature column names.
        slice_size (int)               : Maximum number of slices in each chunk.
        step_size (int)                : Step size for moving to the next chunk.
        groupby_column (str)           : Column to group by (e.g., video or file name).
        sortby_column (str)            : Column to sort within each group (e.g., frame index).
        target_column (str)            : Column containing target labels.
        discard_threshold_ratio (float): Last slice in each video will be discarded if the number of its frames is less than
                                         discard_threshold_ratio*max_slice and will be post padded with zero and kept otherwise.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature sequences (X) and corresponding labels (y).
    """
    step_size = step_size or slice_size
    X, y = [], []

    for _, group in df.groupby(groupby_column):
        # Sort by the specified column to maintain temporal order
        group = group.sort_values(by=sortby_column, ascending=True)

        y_group = group[target_column].unique()
        assert (
            len(y_group) == 1
        ), f"Expected one unique target value, but found {len(y_group)}"
        y_group = y_group[0]

        # Extract features as a list of lists
        slices = group[feature_columns].values.tolist()
        # Create chunks with the specified step size
        for start_idx in range(0, len(slices) - slice_size + 1, step_size):
            chunk = slices[start_idx : start_idx + slice_size]
            X.append(chunk)
            y.append(y_group)

        discard_threshold_frame = int(discard_threshold_ratio * slice_size)
        if len(slices) - (start_idx + slice_size) > discard_threshold_frame:
            chunk = slices[start_idx + slice_size : len(slices)]
            X.append(chunk)
            y.append(y_group)

    # Pad sequences to ensure uniform length
    X_padded = pad_sequences(X, maxlen=slice_size, padding="post", dtype="float32")

    return np.array(X_padded), np.array(y)


def report(model, x, y_true, unique_classes):
    y_true = np.array(y_true)
    y_pred = np.array(model.predict(x, verbose=0)).flatten()
    results = {}
    thresholds = []

    # Sort unique classes and calculate thresholds
    unique_classes = sorted(unique_classes)
    for i in range(len(unique_classes) - 1):
        thresholds.append((unique_classes[i] + unique_classes[i + 1]) / 2)

    # Classify predictions
    y_pred_classified = np.digitize(y_pred, thresholds, right=False)
    y_pred_classified = np.array([unique_classes[i] for i in y_pred_classified])

    class_results = {}
    for cls in unique_classes:
        idx = y_true == cls
        if np.all(idx == False):
            class_results[cls] = {"mae": -1, "accuracy": -1}
            continue
        mae = mean_absolute_error(y_true[idx], y_pred[idx])
        accuracy = np.mean(y_pred_classified[idx] == y_true[idx])
        class_results[cls] = {"mae": mae, "accuracy": accuracy}

    results["classes"] = class_results
    # Calculate overall metrics
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_accuracy = np.mean(y_pred_classified == y_true)
    results["overall"] = {"mae": overall_mae, "accuracy": overall_accuracy}

    return y_pred, results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return round(float(obj), 1)
        elif isinstance(obj, (np.ndarray)):
            return [
                round(float(x), 1) if isinstance(x, (np.floating, float)) else x
                for x in obj.tolist()
            ]
        return super(NumpyEncoder, self).default(obj)


class EpochEndLoggingCallback(Callback):
    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        mapping,
        index,
    ):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.mapping = mapping
        self.index = index

    def on_epoch_end(self, epoch, logs=None):
        # print(f"Epoch {epoch+1} ended. Logs: {logs}")
        trained_model = self.model

        _, train_report = report(
            model=trained_model,
            x=self.X_train,
            y_true=self.y_train,
            unique_classes=list(self.mapping.values()),
        )
        _, val_report = report(
            model=trained_model,
            x=self.X_val,
            y_true=self.y_val,
            unique_classes=list(self.mapping.values()),
        )
        _, test_report = report(
            model=trained_model,
            x=self.X_test,
            y_true=self.y_test,
            unique_classes=list(self.mapping.values()),
        )

        logging.info(
            "\n"
            "============================\n"
            f"{self.index} - Epoch {epoch + 1}\n"
            f"Train Report: {train_report}\n"
            f"Validation Report: {val_report}\n"
            f"Test Report: {test_report}\n"
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

    def _build_model(self, input_shape):
        model = keras.Sequential()
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
            f"{result['index']} - ======================= Saving ======================="
        )
        test_fold_id = result['test'][list(result['test'].keys())[0]]
        result_file_path = os.path.join(output_folder_path, f'{test_fold_id}.json')
        with open(result_file_path, "w") as file:
            json.dump(result, file, cls=NumpyEncoder, indent=4)

        logging.info(
            f"{result['index']} - {result_file_path}\n"
            f"{result['index']} - ======================= Saved ======================="
        )

    def _get_train_val_test_indices(
        self,
        columns_to_fold_by: pd.DataFrame,
        primary_column_name: str,
        secondary_column_name: Optional[str] = None,
    ) -> List[Dict]:

        train_val_test_indices = []
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

                fold_train_val_test_indices = {"info": {}, "indices": {}}

                test_split = columns_to_fold_by[
                    columns_to_fold_by[primary_column_name] == main_column_value
                ]
                fold_train_val_test_indices["indices"][
                    "test"
                ] = test_split.index.to_list()
                fold_train_val_test_indices["info"]["test"] = {
                    f"{primary_column_name}": sorted(
                        test_split[primary_column_name].unique()
                    )
                }

                val_split = columns_to_fold_by[
                    (columns_to_fold_by[primary_column_name] != main_column_value)
                    & (
                        columns_to_fold_by[secondary_column_name]
                        == corresponding_secondary_value
                    )
                ]
                fold_train_val_test_indices["indices"][
                    "val"
                ] = val_split.index.to_list()
                fold_train_val_test_indices["info"]["val"] = {
                    f"{primary_column_name}": sorted(
                        val_split[primary_column_name].unique()
                    )
                }

                train_split = columns_to_fold_by[
                    columns_to_fold_by[secondary_column_name]
                    != corresponding_secondary_value
                ]
                fold_train_val_test_indices["indices"][
                    "train"
                ] = train_split.index.to_list()
                fold_train_val_test_indices["info"]["train"] = {
                    f"{primary_column_name}": sorted(
                        train_split[primary_column_name].unique()
                    )
                }
                train_val_test_indices.append(fold_train_val_test_indices)

        elif primary_column_name:
            for main_column_value_i in unique_main_column_values:
                for main_column_value_j in unique_main_column_values:
                    if main_column_value_i != main_column_value_j:
                        fold_train_val_test_indices = {"info": {}, "indices": {}}

                        test_split = columns_to_fold_by[
                            columns_to_fold_by[primary_column_name]
                            == main_column_value_i
                        ]
                        fold_train_val_test_indices["indices"][
                            "test"
                        ] = test_split.index.to_list()
                        fold_train_val_test_indices["info"]["test"] = {
                            f"{primary_column_name}": sorted(
                                test_split[primary_column_name].unique()
                            )
                        }

                        val_split = columns_to_fold_by[
                            columns_to_fold_by[primary_column_name]
                            == main_column_value_j
                        ]
                        fold_train_val_test_indices["indices"][
                            "val"
                        ] = val_split.index.to_list()
                        fold_train_val_test_indices["info"]["val"] = {
                            f"{primary_column_name}": sorted(
                                val_split[primary_column_name].unique()
                            )
                        }

                        train_split = columns_to_fold_by[
                            ~columns_to_fold_by[primary_column_name].isin(
                                [main_column_value_i, main_column_value_j]
                            )
                        ]
                        fold_train_val_test_indices["indices"][
                            "train"
                        ] = train_split.index.to_list()
                        fold_train_val_test_indices["info"]["train"] = {
                            f"{primary_column_name}": sorted(
                                train_split[primary_column_name].unique()
                            )
                        }

                        train_val_test_indices.append(fold_train_val_test_indices)

        return train_val_test_indices

    def _train_single_fold(
        self,
        data: pd.DataFrame,
        split_info: Dict[str, str],
        split_indices: Dict[str, List],
        output_folder_path: str,
        columns_to_drop: Optional[List[str]] = [],
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

        train = data.loc[split_indices["train"]]
        val = data.loc[split_indices["val"]]
        test = data.loc[split_indices["test"]]

        logging.info(
            f"{index} - train {train.shape} | val {val.shape} | test {test.shape}"
        )

        groupby_column, sortby_column, target_column = "file_name", "frame", "target"
        columns_to_drop_1 = [
            column_name
            for column_name in columns_to_drop
            if column_name not in [groupby_column, sortby_column, target_column]
        ]
        train = train.drop(columns=columns_to_drop_1)
        val = val.drop(columns=columns_to_drop_1)
        test = test.drop(columns=columns_to_drop_1)

        feature_columns = [
            column_name
            for column_name in train.columns
            if column_name not in columns_to_drop
        ]

        X_train, y_train = reshape_data_rnn(
            df=train,
            feature_columns=feature_columns,
            slice_size=self.slice_size,
            step_size=self.step_size,
            discard_threshold_ratio=self.discard_threshold_ratio,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
        )
        X_val, y_val = reshape_data_rnn(
            df=val,
            feature_columns=feature_columns,
            slice_size=self.slice_size,
            step_size=self.step_size,
            discard_threshold_ratio=self.discard_threshold_ratio,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
        )
        X_test, y_test = reshape_data_rnn(
            df=test,
            feature_columns=feature_columns,
            slice_size=self.slice_size,
            step_size=self.step_size,
            discard_threshold_ratio=self.discard_threshold_ratio,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
        )
        logging.info(
            f"{index} - X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}"
        )

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        logging.info(f"{index} - building model")
        model = self._build_model((X_train.shape[1], X_train.shape[2]))

        logging.info(f"{index} - fitting model")
        # Train the model

        epoch_end_logging = EpochEndLoggingCallback(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            mapping=self.mapping,
            index=index,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        test_fold_id = split_info['test'][list(split_info['test'].keys())[0]]
        checkpoint = ModelCheckpoint(
            os.path.join(output_folder_path, f"{test_fold_id}_best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        )

        history: History = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, epoch_end_logging, checkpoint],
            verbose=0,
            sample_weight=sample_weight,
        )

        logging.info(f"{index} - getting reports")
        y_train_pred, train_report = report(
            model=model,
            x=X_train,
            y_true=y_train,
            unique_classes=list(self.mapping.values()),
        )
        y_val_pred, val_report = report(
            model=model,
            x=X_val,
            y_true=y_val,
            unique_classes=list(self.mapping.values()),
        )
        y_test_pred, test_report = report(
            model=model,
            x=X_test,
            y_true=y_test,
            unique_classes=list(self.mapping.values()),
        )

        # Store results to return
        local_results = {
            "index": index,
            "test": split_info["test"],
            "val": split_info["val"],
            "train": split_info["train"],
            "history": history.history,
            "train_report": train_report,
            "y_train": y_train,
            "y_train_pred": y_train_pred,
            "val_report": val_report,
            "y_val": y_val,
            "y_val_pred": y_val_pred,
            "test_report": test_report,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
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
        data: pd.DataFrame,
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
                        "discard_threshold_ratio" : self.discard_threshold_ratio,
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

        train_val_test_indices = self._get_train_val_test_indices(
            columns_to_fold_by=data[
                    [fold_by_primary_column_name]
                    + (
                        [fold_by_secondary_column_name]
                        if fold_by_secondary_column_name
                        else []
                    )
            ],
            primary_column_name=fold_by_primary_column_name,
            secondary_column_name=fold_by_secondary_column_name,
        )

        args_list = [
            {
                "data": data,
                "split_info": test_fold_value["info"],
                "split_indices": test_fold_value["indices"],
                "output_folder_path": output_folder_path,
                "columns_to_drop": columns_to_drop,
                "index": idx,
            }
            for idx, test_fold_value in enumerate(train_val_test_indices)
        ]

        logging.info(f"Found {len(args_list)} valid test-validation pairs.")
        logging.info(
            "\n".join([f"{i['index']} - {i['split_info']}" for i in args_list])
        )

        random.shuffle(args_list)

        selected_indices = fold_indices or slice(fold_start_index, fold_end_index)
        args_list = args_list[selected_indices]
        logging.info(f"Processing {len(args_list)} valid test-validation pairs.")
        logging.info(
            "\n".join([f"{i['index']} - {i['split_info']}" for i in args_list])
        )

        results = process(
            handler_function=self._save_result,
            worker_function=self._train_single_fold,
            worker_params_list=args_list,
            parallel=parallel,
            max_workers=max_workers,
            maxtasksperchild=maxtasksperchild,
            use_tqdm=use_tqdm,
        )

        logging.info(f"Completed all {len(args_list)} test-validation pairs.")
