import subprocess
import sys

# # List of required packages
# required_packages = [
#     'opencv-python',
#     'mediapipe',
#     'pandas',
#     'numpy',
#     'tqdm',
#     'scikit-learn',
#     'mord',
#     'tensorflow',
#     'matplotlib',
#     'tensorflow-addons',
#     'imbalanced-learn'
# ]

# # Function to install packages if not already installed
# def install_package(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # Check and install missing packages
# for package in required_packages:
#     try:
#         __import__(package)
#     except ImportError:
#         print(f"{package} not found, installing...")
#         install_package(package)

# Now, import the libraries

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mord import LogisticAT  # Ordinal logistic regression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, root_mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras import backend as K
from keras import losses
from keras.layers import Lambda
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from keras.preprocessing.sequence import pad_sequences
import random
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_video.log"),
        logging.StreamHandler()
    ]
)


# Function to set the random seed
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


def reshape_data_rnn(df, feature_columns=[], max_slices=30, sortby_column='slice_index', target_column='target'):
    X, y = [], []
    
    grouped = df.groupby('filename')
    
    for video, group in grouped:
        # Sort by slice_index to maintain temporal order
        group = group.sort_values(by=sortby_column)

        # Extract features
        slices = group[feature_columns].values.tolist()

        # Split into chunks of max_slices
        for i in range(0, len(slices), max_slices):
            chunk = slices[i:i + max_slices]
            X.append(chunk)
            y.append(group[target_column].iloc[0])  # Same label for all chunks from the same video

    # Pad sequences to the max_slices length
    X_padded = pad_sequences(X, maxlen=max_slices, padding='post', dtype='float32')

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
        mae = root_mean_squared_error(y_true[idx], y_pred[idx])
        results[cls] = mae

    return y_pred, results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return round(float(obj), 1)
        elif isinstance(obj, (np.ndarray)):
            return [round(float(x), 1) if isinstance(x, (np.floating, float)) else x for x in obj.tolist()]
        return super(NumpyEncoder, self).default(obj)


class Model:
    def __init__(self, units, dropout_ratio, kernel_regularizer_l2, kernel_regularizer_l1, batch_size, num_classes, learning_rate, weight_decay, beta_1, beta_2, epsilon, epochs, patience, max_slices, dataset_name):
        self.units = units
        self.dropout_ratio = dropout_ratio
        self.kernel_regularizer_l2 = kernel_regularizer_l2
        self.kernel_regularizer_l1 = kernel_regularizer_l1
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.epochs = epochs
        self.patience = patience
        self.max_slices = max_slices
        self.dataset_name = dataset_name
        self.dataset_id = self.dataset_name.split("Blendshapes_Summary_")[1].split(".csv")[0]
        # self.models = []
        # self.histories = []
        # self.reports = []
        
        
    def build_model(self, input_shape):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_shape[0], input_shape[1])))

        # Add LSTM layers
        for u in self.units[:-1]:
            model.add(layers.LSTM(u, activation='tanh', return_sequences=True, 
                                  kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_l2)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_ratio))

        # Final LSTM layer without return_sequences
        model.add(layers.LSTM(self.units[-1], activation='tanh', 
                              kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_l2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.dropout_ratio))

        # Final output layer
        model.add(layers.Dense(1, activation='sigmoid'))  # Use sigmoid activation

        # Use a custom output layer to scale the output to 1-9
        model.add(layers.Lambda(lambda x: x * 8 + 1)) 

        model.compile(
            optimizer = keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon
            ),
            loss='mse',
            metrics=['mae']
        )

        return model



    # First, define the train_single_fold function outside of the class
    def _train_single_fold(self, args):
        """Helper function for parallel processing in test_model.

        Args:
            args: Tuple containing (model_instance, test_idx, val_idx, data)

        Returns:
            Dictionary containing model, history, and reports
        """
        self, test_subject_idx, data = args
        local_results = {}

        subject_fold = data[data['subject_id']==test_subject_idx]['fold'].values[0]
        validation_subjects_idx = data[(data['subject_id']!=test_subject_idx) & (data['fold']==subject_fold)]['subject_id'].unique().tolist()
        logging.info(f"Starting Test {test_subject_idx} & Validation {validation_subjects_idx}")

        train = data[~data['subject_id'].isin(validation_subjects_idx + [test_subject_idx])]
        val   = data[data['subject_id'].isin(validation_subjects_idx)]
        test  = data[data['subject_id']==test_subject_idx]
        logging.info(f"train {train.shape} & val {val.shape} & test {test.shape}")

        # List of columns to exclude from scaling
        exclude_cols = ['slice_index', 'filename', 'start_frame', 'end_frame', 'target', 'fold', 'subject_id']
        feature_columns = data.columns.difference(exclude_cols)

        # Fit the scaler on the training data
        scaler = StandardScaler().fit(train[feature_columns])

        # Transform the training, validation, and test data
        train_scaled = scaler.transform(train[feature_columns])
        val_scaled   = scaler.transform(val[feature_columns]  )
        test_scaled  = scaler.transform(test[feature_columns] )

        # Convert scaled arrays back to DataFrames and add the excluded columns
        train_scaled_df = pd.DataFrame(train_scaled, columns=feature_columns, index=train.index)
        val_scaled_df   = pd.DataFrame(val_scaled,   columns=feature_columns, index=val.index  )
        test_scaled_df  = pd.DataFrame(test_scaled,  columns=feature_columns, index=test.index )

        # Concatenate the scaled features with the excluded columns
        train = pd.concat([train_scaled_df, train[exclude_cols]], axis=1)
        val   = pd.concat([val_scaled_df,   val[exclude_cols]],   axis=1)
        test  = pd.concat([test_scaled_df,  test[exclude_cols]],  axis=1)

        X_train, y_train = reshape_data_rnn(train, feature_columns, max_slices=self.max_slices, sortby_column='slice_index', target_column='target')
        X_val,   y_val   = reshape_data_rnn(val,   feature_columns, max_slices=self.max_slices, sortby_column='slice_index', target_column='target')
        X_test,  y_test  = reshape_data_rnn(test,  feature_columns, max_slices=self.max_slices, sortby_column='slice_index', target_column='target')

        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

        model = self.build_model((X_train.shape[1], X_train.shape[2]))

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)],
            verbose=0,  # Reduced verbosity for parallel execution
            sample_weight=sample_weight
        )

        y_train_pred, train_report = report(model=model, x=X_train, y_true=y_train, unique_classes=[2, 6.5, 8.5])
        y_val_pred,   val_report   = report(model=model, x=X_val,   y_true=y_val,   unique_classes=[2, 6.5, 8.5])
        y_test_pred,  test_report  = report(model=model, x=X_test,  y_true=y_test,  unique_classes=[2, 6.5, 8.5])

        # Store results to return
        local_results = {
            'test': test_subject_idx,
            'val': validation_subjects_idx,
            'history': history.history,
            'train_report': train_report,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'val_report': val_report,
            'y_val': y_val,
            'y_val_pred': y_val_pred,
            'test_report': test_report,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
        }
        
        logging.info(f"epoch : {len(history.history['val_loss'])}")
        logging.info(f"Completed Test {test_subject_idx} & Validation {validation_subjects_idx}")
        logging.info('=======================================')

        return local_results
            

    def test_model(self, data):
        unique_subject_id = data['subject_id'].unique()

        logging.info(f"Found {len(unique_subject_id)} valid test-validation pairs.")

        # Prepare arguments for parallel processing
        args_list = [(self, i, data) for i in unique_subject_id]
        random.shuffle(args_list)

        output_folder_name = datetime.now().strftime("%Y-%m-%d|%H:%M:%S|RNN")
        output_parent_dir = "/home/mojtaba.kananisarche/Zh/UTA/Output/"
        output_folder_path = os.path.join(output_parent_dir, output_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        with open(os.path.join(output_folder_path, 'results.json'), 'w') as file:
            json.dump({
                'general':{
                    'units':self.units,
                    'dropout_ratio':self.dropout_ratio,
                    'batch_size':self.batch_size,
                    'learning_rate':self.learning_rate,
                    'weight_decay':self.weight_decay,
                    'epochs':self.epochs,
                    'dataset_id':self.dataset_id,
                    'max_slices':self.max_slices
                }
            }, file)

        # Run training in parallel and process results as they come
        all_results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self._train_single_fold, args) for args in args_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Training folds"):
                result = future.result()
                file_path = os.path.join(output_folder_path, f'{result["test"]}.json')
                with open(file_path, 'w') as file:
                    json.dump(result, file, cls=NumpyEncoder, indent=4)
                    logging.info(f"Saved results for Test subject {result['test']} to {file_path}")
                    
                    
        logging.info(f"Completed all {len(unique_subject_id)} test-validation pairs.")




