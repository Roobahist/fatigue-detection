import json
import logging
import os
import pickle
import random
from datetime import datetime
from typing import List, Optional, Tuple

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
from tensorflow.keras.callbacks import Callback

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
    feature_columns=[],
    max_slices=30,
    groupby_column="file_name",
    sortby_column="frame",
    target_column="target",
):
    X, y = [], []

    grouped = df.groupby(groupby_column)

    for _, group in grouped:
        # Sort by slice_index to maintain temporal order
        group = group.sort_values(by=sortby_column, ascending=True)

        # Extract features
        slices = group[feature_columns].values.tolist()

        # Split into chunks of max_slices
        for i in range(0, len(slices), max_slices):
            chunk = slices[i : i + max_slices]
            X.append(chunk)
            y.append(
                group[target_column].iloc[0]
            )  # Same label for all chunks from the same video

    # Pad sequences to the max_slices length
    X_padded = pad_sequences(X, maxlen=max_slices, padding="post", dtype="float32")

    # Convert labels to numpy array
    y = np.array(y)

    return X_padded, y


def report(model, x, y_true, unique_classes):
    y_true = np.array(y_true)
    y_pred = np.array(model.predict(x, verbose=0)).flatten()
    results = {}
    for cls in unique_classes:
        idx = y_true == cls
        if np.all(idx == False):
            results[cls] = -1
            continue
        mae = mean_absolute_error(y_true[idx], y_pred[idx])
        results[cls] = mae

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

        logging.info("=============")
        logging.info(
            f"Epoch {epoch + 1} - Fold {self.index}\n"
            f"Train Report: {train_report}\n"
            f"Validation Report: {val_report}\n"
            f"Test Report: {test_report}\n"
            f"logs: {logs}"
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
        max_slices,
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
        self.max_slices = max_slices
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

        # Use a custom output layer to scale the output to 1-9
        # model.add(layers.Lambda(lambda x: x * 8 + 1))
        # model.add(layers.Lambda(lambda x: x * 10))

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
        model,
        output_folder_path: Optional[str] = "/home/mojtaba.kananisarche/Zh/UTA/Output/",
    ):
        result_file_path = os.path.join(output_folder_path, f'{result["test"]}.json')
        with open(result_file_path, "w") as file:
            json.dump(result, file, cls=NumpyEncoder, indent=4)
            logging.info(
                f"Saved results for Test subject {result['test']} to {result_file_path}"
            )
        # Save the trained model as a pickle file
        model_file_path = os.path.join(
            output_folder_path, f'{result["test"]}_model.pkl'
        )
        with open(model_file_path, "wb") as model_file:
            pickle.dump(model, model_file)
            logging.info(
                f"Saved model for Test subject {result['test']} to {model_file_path}"
            )

    # First, define the train_single_fold function outside of the class
    def _train_single_fold(
        self,
        data: pd.DataFrame,
        test_subject_id: str,
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
            f"{index} ============= Processing Subject - {test_subject_id} ============="
        )

        local_results = {}

        test_subject_fold = data[data["subject_id"] == test_subject_id]["fold"].values[
            0
        ]

        validation_subjects_id = (
            data[
                (data["subject_id"] != test_subject_id)
                & (data["fold"] == test_subject_fold)
            ]["subject_id"]
            .unique()
            .tolist()
        )

        logging.info(
            f"{index} - Starting -> Test ID : {test_subject_id} & Validation IDs : {validation_subjects_id}"
        )

        train = data[
            ~data["subject_id"].isin(validation_subjects_id + [test_subject_id])
        ]
        val = data[data["subject_id"].isin(validation_subjects_id)]
        test = data[data["subject_id"] == test_subject_id]

        logging.info(
            f"{index} - train {train.shape} & val {val.shape} & test {test.shape}"
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
            max_slices=self.max_slices,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
        )
        X_val, y_val = reshape_data_rnn(
            df=val,
            feature_columns=feature_columns,
            max_slices=self.max_slices,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
        )
        X_test, y_test = reshape_data_rnn(
            df=test,
            feature_columns=feature_columns,
            max_slices=self.max_slices,
            groupby_column=groupby_column,
            sortby_column=sortby_column,
            target_column=target_column,
        )
        logging.info(
            f"{index} - X_train {X_train.shape} & X_val {X_val.shape} & X_test {X_test.shape}"
        )

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        logging.info(f"{index} - building model")
        model = self._build_model((X_train.shape[1], X_train.shape[2]))

        logging.info(f"{index} - fitting model")
        # Train the model

        epoch_end_logging_callback = EpochEndLoggingCallback(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            mapping=self.mapping,
            index=index,
        )

        history: History = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                ),
                epoch_end_logging_callback,
            ],
            verbose=0,
            sample_weight=sample_weight,
        )
        history: History = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            ],
            verbose=2,  # Reduced verbosity for parallel execution
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
            "test": test_subject_id,
            "val": validation_subjects_id,
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
            f"{index} - Ended -> Test ID : {test_subject_id} & Validation IDs : {validation_subjects_id}\n"
            f"epochs : {len(history.history['val_loss'])}\n"
            f"========================="
        )

        return {
            "result": local_results,
            "model": model,
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
        limit_folds: Optional[int] = None,
    ):

        output_folder_name = datetime.now().strftime("%Y-%m-%d|%H:%M:%S|RNN_FACENET")
        output_folder_path = os.path.join(output_parent_dir, output_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

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
                        "max_slices": self.max_slices,
                    }
                },
                file,
            )

        unique_subject_id = data["subject_id"].unique()
        logging.info(f"Found {len(unique_subject_id)} valid test-validation pairs.")

        # Prepare arguments for parallel processing
        args_list = [
            {
                "data": data,
                "test_subject_id": unique_subject_id[i],
                "output_folder_path": output_folder_path,
                "columns_to_drop": columns_to_drop,
                "index": i,
            }
            for i in range(min(len(unique_subject_id), max(0, limit_folds)))
        ]
        random.shuffle(args_list)

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
