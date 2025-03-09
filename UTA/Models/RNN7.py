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
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras import backend as K
from keras import losses
from tensorflow.keras.layers import Lambda
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random



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

def replace_target(value, mapping):
    return mapping[str(value)]

def step(val, steps, outputs):
    for i in range(len(steps)-1):
        low = steps[i]
        high = steps[i+1]
        if val >= low and val <= high:
            return outputs[i]
        
def report(model, x, y, output_dict):
    y_pred = [step(v,[0,4.25,7.5,10],[0,1,2]) for v in model.predict(x, verbose=0)]
    y = [int(replace_target(i, {'2.0': 0, '6.5':1, '8.5':2})) for i in y]
    if output_dict:
        cr = classification_report(list(y), y_pred, output_dict=True)
        cr['confusion_matrix'] = list(confusion_matrix(list(y), y_pred))
        return cr

def reshape_data_rnn(df, feature_columns=[], groupby_column='filename', max_slices=30, sortby_column='slice_index', target_column='target'):
    X, y = [], []
    grouped = df.groupby(groupby_column)
    
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

class Model:
    def __init__(self, units, dropout_ratio, kernel_regularizer_l2, kernel_regularizer_l1, batch_size, num_classes, learning_rate, weight_decay, beta_1, beta_2, epsilon, cv_on, epochs, patience):
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
        self.cv_on = cv_on #subject_id, fold
        self.epochs = epochs
        self.patience = patience
        self.models = []
        self.histories = []
        self.reports = []
        
        
    def build_model(self, input_shape):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_shape[0], input_shape[1])))  # (30, 218)

        # Add LSTM layers
        for u in self.units:
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

        # Use a custom output layer to scale the output to 0-10
        model.add(layers.Lambda(lambda x: x * 10))  # Multiply by 10

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

    
    def get_reports(self):
        records = []

        for item in self.reports:
            base_info = {'test': item['test'], 'val': item['val']}
            for report_type in ['train_report', 'val_report', 'test_report']:
                report = item.get(report_type, {})
                for category, metrics in report.items():
                    if category == 'confusion_matrix':
                        cm_flat = np.array(metrics).flatten()
                        for i, value in enumerate(cm_flat):
                            records.append({**base_info, 'report_type': report_type, 'category': f'conf_matrix', 'metric_name':i, 'metric_value': value})
                    elif isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            records.append({**base_info, 'report_type': report_type, 'category': category, 'metric_name': metric, 'metric_value': value})
                    else:
                        records.append({**base_info, 'report_type': report_type, 'category': 'macro avg', 'metric_name': category, 'metric_value': metrics})

        return pd.DataFrame(records)

    # First, define the train_single_fold function outside of the class
    def _train_single_fold(self, args):
        """Helper function for parallel processing in test_model.

        Args:
            args: Tuple containing (model_instance, test_idx, val_idx, data)

        Returns:
            Dictionary containing model, history, and reports
        """
        self, i, j, data = args
        local_results = {}

        print(f"Starting Test {i} & Validation {j}")

        train = data[~data[self.cv_on].isin([i, j])].reset_index(drop=True)
        val = data[data[self.cv_on] == j].reset_index(drop=True)
        test = data[data[self.cv_on] == i].reset_index(drop=True)

        # List of columns to exclude from scaling
        exclude_cols = ['slice_index', 'filename', 'start_frame', 'end_frame', 'target', 'fold', 'subject_id']
        feature_columns = data.columns.difference(exclude_cols)

        # Fit the scaler on the training data
        scaler = StandardScaler().fit(train[feature_columns])

        # Transform the training, validation, and test data
        train_scaled = scaler.transform(train[feature_columns])
        val_scaled = scaler.transform(val[feature_columns])
        test_scaled = scaler.transform(test[feature_columns])

        # Convert scaled arrays back to DataFrames and add the excluded columns
        train_scaled_df = pd.DataFrame(train_scaled, columns=feature_columns, index=train.index)
        val_scaled_df = pd.DataFrame(val_scaled, columns=feature_columns, index=val.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=feature_columns, index=test.index)

        # Concatenate the scaled features with the excluded columns
        train = pd.concat([train_scaled_df, train[exclude_cols]], axis=1)
        val = pd.concat([val_scaled_df, val[exclude_cols]], axis=1)
        test = pd.concat([test_scaled_df, test[exclude_cols]], axis=1)

        X_train, y_train = reshape_data_rnn(train, feature_columns, groupby_column='filename', max_slices=30, sortby_column='slice_index', target_column='target')
        X_val, y_val = reshape_data_rnn(val, feature_columns, groupby_column='filename', max_slices=30, sortby_column='slice_index', target_column='target')
        X_test, y_test = reshape_data_rnn(test, feature_columns, groupby_column='filename', max_slices=30, sortby_column='slice_index', target_column='target')

        classes = np.array([2, 6.5, 8.5])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {}
        for c, w in zip(classes, class_weights):
            class_weight_dict[c] = w

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
            class_weight=class_weight_dict
        )

        train_report = report(model, X_train, y_train, True)
        val_report = report(model, X_val, y_val, True)
        test_report = report(model, X_test, y_test, True)

        # Store results to return
        local_results = {
            'test': i,
            'val': j,
            'model': model,
            'history': history,
            'train_report': train_report,
            'val_report': val_report,
            'test_report': test_report
        }

        print(f"Completed Test {i} & Validation {j}")
        return local_results
            
    # Now modify the test_model method
    def test_model(self, data):
        import concurrent.futures
        from tqdm import tqdm

        unique = data[self.cv_on].unique()
        test_val_pairs = [(i, j) for i in unique for j in unique if i!=j]
       
        logging.info(f"Found {len(test_val_pairs)} valid test-validation pairs.")
        
        # Prepare arguments for parallel processing
        args_list = [(self, i, j, data) for i, j in test_val_pairs]

        # Run training in parallel
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(test_val_pairs), 40)) as executor:
            futures = [executor.submit(self._train_single_fold, args) for args in args_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Training folds"):
                all_results.append(future.result())

        # Process results
        for result in all_results:
            self.models.append(result['model'])
            self.histories.append(result['history'])
            self.reports.append({
                'test': result['test'],
                'val': result['val'],
                'train_report': result['train_report'],
                'val_report': result['val_report'],
                'test_report': result['test_report']
            })
        logging.info(f"Completed all {len(test_val_pairs)} test-validation pairs.")
        
        
if __name__ == "__main__":
    logging.info(f"Starting")
    # Call the function to set the seed
    set_seed(42)
    
    dataset_name = "Blendshapes_Summary_1800_100.csv"
    
    r = pd.read_csv(f"/home/mojtaba.kananisarche/Zh/UTA/Blendshapes/Summary/{dataset_name}", index_col=0)
    r['target'] = r['filename'].apply(lambda x: replace_target(x.split('_')[2], {'10': 8.5, '0': 2.0, '5': 6.5}))
    r['fold'] = r['filename'].apply(lambda x: int(x.split('_')[0]))
    r['subject_id'] = r['filename'].apply(lambda x: int(x.split('_')[1]))
    m = Model(units = [50,50,50,50,50,50,50], 
          dropout_ratio = 0.2, 
          kernel_regularizer_l2 = 0, 
          kernel_regularizer_l1 = 0, 
          batch_size = 32, 
          num_classes = 3, 
          learning_rate = 1e-3, 
          weight_decay = 1e-4, 
          beta_1 = 0.9, 
          beta_2 = 0.999, 
          epsilon = 1e-7, 
          cv_on = 'fold', 
          epochs = 1000, 
          patience = 200)

    m.test_model(r)
    reports = m.get_reports()
    
    # Extract dataset identifier
    dataset_id = dataset_name.split('_')[-2] + "_" + dataset_name.split('_')[-1].split('.')[0]

    # Generate a unique filename based on model parameters
    file_name = f"report_units-{m.units}_dropout-{m.dropout_ratio}_bs-{m.batch_size}_lr-{m.learning_rate}_wd-{m.weight_decay}_epochs-{m.epochs}_cv-{m.cv_on}_data-{dataset_id}.csv"

    # Define the full file path
    output_dir = "/home/mojtaba.kananisarche/Zh/UTA/Output/"
    file_path = os.path.join(output_dir, file_name)

    # Ensure uniqueness if file already exists
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{file_name[:-4]}_{counter}.csv")
        counter += 1

    # Save the new CSV file
    reports.to_csv(file_path, index=False)

    print(f"Saved file as: {file_path}")

    logging.info(f"Saved file as: {file_path}")

    