import tensorflow as tf
import keras
from keras import layers
from keras.src.models.sequential import Sequential
import keras_tuner as kt
from typing import Dict, List, Tuple, Union, Any, Optional

class LSTMModelTuner(kt.HyperModel):
    def __init__(self, 
                 input_shape: Tuple[int, int], 
                 loss_function: Union[str, tf.keras.losses.Loss], 
                 metrics: List[Union[str, tf.keras.metrics.Metric]],
                 hp_config: Dict[str, Dict[str, Any]]):
        """
        Initialize the LSTMModelTuner with input shape, compilation parameters, and hyperparameter config.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            loss_function (str or tf.keras.losses): Loss function for model compilation
            metrics (list): Metrics for model evaluation
            hp_config (dict): Dictionary containing hyperparameter search space configurations
        """
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.metrics = metrics
        self.hp_config = hp_config
    
    def build(self, hp):
        """
        Build and return a model with hyperparameters to be tuned.
        
        Args:
            hp (keras_tuner.HyperParameters): Hyperparameter manager
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
        model.add(layers.Input(shape=(self.input_shape[0], self.input_shape[1])))
        
        # Number of LSTM layers
        layer_config = self.hp_config.get('num_lstm_layers', {})
        num_lstm_layers = hp.Int(
            'num_lstm_layers',
            min_value=layer_config.get('min_value', 1),
            max_value=layer_config.get('max_value', 3),
            step=layer_config.get('step', 1)
        )
        
        # Define units for each layer using hyperparameters
        for i in range(num_lstm_layers):
            # Units per layer
            units_config = self.hp_config.get('units', {})
            units = hp.Int(
                f'units_{i}', 
                min_value=units_config.get('min_value', 32),
                max_value=units_config.get('max_value', 256),
                step=units_config.get('step', 32)
            )
            
            # Return sequences for all layers except the last one
            return_sequences = i < num_lstm_layers - 1
            
            # L1 regularization
            l1_config = self.hp_config.get('l1_reg', {})
            l1_reg = hp.Float(
                f'l1_reg_{i}',
                min_value=l1_config.get('min_value', 1e-6),
                max_value=l1_config.get('max_value', 1e-2),
                sampling=l1_config.get('sampling', 'log')
            )
            
            # L2 regularization
            l2_config = self.hp_config.get('l2_reg', {})
            l2_reg = hp.Float(
                f'l2_reg_{i}',
                min_value=l2_config.get('min_value', 1e-6),
                max_value=l2_config.get('max_value', 1e-2),
                sampling=l2_config.get('sampling', 'log')
            )
            
            # Add LSTM layer
            model.add(
                layers.LSTM(
                    units=units,
                    activation=self.hp_config.get('activation', 'tanh'),
                    return_sequences=return_sequences,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(
                        l1=l1_reg,
                        l2=l2_reg
                    ),
                )
            )
            
            # Add Batch Normalization if enabled
            if self.hp_config.get('use_batch_norm', True):
                model.add(layers.BatchNormalization())
            
            # Add Dropout
            dropout_config = self.hp_config.get('dropout', {})
            model.add(layers.Dropout(
                hp.Float(
                    f'dropout_{i}',
                    min_value=dropout_config.get('min_value', 0.1),
                    max_value=dropout_config.get('max_value', 0.5),
                    step=dropout_config.get('step', 0.1)
                )
            ))
        
        # Output configuration
        output_config = self.hp_config.get('output', {})
        output_units = output_config.get('units', 1)
        output_activation = output_config.get('activation', 'linear')
        
        # Final output layer
        model.add(layers.Dense(output_units, activation=output_activation))
        
        # Optimizer hyperparameters
        lr_config = self.hp_config.get('learning_rate', {})
        learning_rate = hp.Float(
            'learning_rate',
            min_value=lr_config.get('min_value', 1e-4),
            max_value=lr_config.get('max_value', 1e-2),
            sampling=lr_config.get('sampling', 'log')
        )
        
        wd_config = self.hp_config.get('weight_decay', {})
        weight_decay = hp.Float(
            'weight_decay',
            min_value=wd_config.get('min_value', 1e-6),
            max_value=wd_config.get('max_value', 1e-2),
            sampling=wd_config.get('sampling', 'log')
        )
        
        b1_config = self.hp_config.get('beta_1', {})
        beta_1 = hp.Float(
            'beta_1',
            min_value=b1_config.get('min_value', 0.8),
            max_value=b1_config.get('max_value', 0.99),
            step=b1_config.get('step', 0.01)
        )
        
        b2_config = self.hp_config.get('beta_2', {})
        beta_2 = hp.Float(
            'beta_2',
            min_value=b2_config.get('min_value', 0.9),
            max_value=b2_config.get('max_value', 0.999),
            step=b2_config.get('step', 0.001)
        )
        
        # Get epsilon from config
        epsilon = self.hp_config.get('epsilon', 1e-7)
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                amsgrad=self.hp_config.get('amsgrad', True),
            ),
            loss=self.loss_function,
            metrics=self.metrics,
        )
        
        return model


class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int], 
                 loss_function: Union[str, tf.keras.losses.Loss] = 'mse', 
                 metrics: List[Union[str, tf.keras.metrics.Metric]] = ['mae'],
                 hp_config: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the LSTM model with input shape, compilation parameters, and optional hyperparameter config.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            loss_function (str or loss object): Loss function name or object
            metrics (list): List of metrics for evaluation
            hp_config (dict, optional): Dictionary containing hyperparameter search space configurations
        """
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.metrics = metrics
        
        # Default hyperparameter configuration if none is provided
        self.hp_config = hp_config if hp_config is not None else {
            'num_lstm_layers': {'min_value': 1, 'max_value': 3, 'step': 1},
            'units': {'min_value': 32, 'max_value': 256, 'step': 32},
            'activation': 'tanh',
            'l1_reg': {'min_value': 1e-6, 'max_value': 1e-2, 'sampling': 'log'},
            'l2_reg': {'min_value': 1e-6, 'max_value': 1e-2, 'sampling': 'log'},
            'dropout': {'min_value': 0.1, 'max_value': 0.5, 'step': 0.1},
            'use_batch_norm': True,
            'output': {'units': 1, 'activation': 'linear'},
            'learning_rate': {'min_value': 1e-4, 'max_value': 1e-2, 'sampling': 'log'},
            'weight_decay': {'min_value': 1e-6, 'max_value': 1e-2, 'sampling': 'log'},
            'beta_1': {'min_value': 0.8, 'max_value': 0.99, 'step': 0.01},
            'beta_2': {'min_value': 0.9, 'max_value': 0.999, 'step': 0.001},
            'epsilon': 1e-7,
            'amsgrad': True
        }
    
    def update_hp_config(self, new_config: Dict[str, Dict[str, Any]]):
        """
        Update the hyperparameter configuration.
        
        Args:
            new_config (dict): New hyperparameter configuration to merge with existing
        """
        # Merge new_config with self.hp_config (new_config takes precedence)
        for key, value in new_config.items():
            if key in self.hp_config and isinstance(self.hp_config[key], dict) and isinstance(value, dict):
                self.hp_config[key].update(value)
            else:
                self.hp_config[key] = value
        
    def tune_model(self, 
                  train_data, 
                  val_data,
                  project_name: str = 'lstm_tuning', 
                  max_trials: int = 10, 
                  executions_per_trial: int = 2, 
                  directory: str = 'my_dir',
                  epochs: int = 50,
                  batch_size: Optional[int] = None,
                  patience: int = 5,
                  tuner_type: str = 'bayesian',
                  checkpoint_path: Optional[str] = None,
                  save_best_only: bool = True,
                  save_weights_only: bool = False,
                  monitor: str = 'val_loss',
                  mode: str = 'min',
                  overwrite: bool = False):
        """
        Use Keras Tuner to find optimal hyperparameters for the LSTM model.
        
        Args:
            train_data: Either a tuple (x_train, y_train) of numpy arrays or a tf.data.Dataset
            val_data: Either a tuple (x_val, y_val) of numpy arrays or a tf.data.Dataset
            project_name (str): Name for the tuning project
            max_trials (int): Maximum number of hyperparameter combinations to try
            executions_per_trial (int): Number of models to train per trial
            directory (str): Directory to store tuning results
            epochs (int): Maximum number of epochs to train
            batch_size (int, optional): Batch size for training when using numpy arrays
            patience (int): Patience for early stopping
            tuner_type (str): Type of tuner ('bayesian', 'random', 'hyperband')
            checkpoint_path (str, optional): Path to save model checkpoints. If None, no checkpoints are saved.
                                            Can include formatting options like {epoch:02d}
            save_best_only (bool): Only save the best model according to the monitored metric
            save_weights_only (bool): If True, only save model weights, not the entire model
            monitor (str): Metric to monitor for early stopping and model checkpointing
            mode (str): One of {'min', 'max'}. For 'min', checkpoint saves when metric is minimized
            overwrite (bool): If True, overwrite previous tuning results. If False, continue from previous search
            
        Returns:
            Best hyperparameters and the tuner object
        """
        # Create the hypermodel with current hp_config
        hypermodel = LSTMModelTuner(
            self.input_shape, 
            self.loss_function, 
            self.metrics,
            self.hp_config
        )
        
        # Create tuner directory if it doesn't exist
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Remove directory if overwrite is True and directory exists
        if overwrite and os.path.exists(os.path.join(directory, project_name)):
            import shutil
            shutil.rmtree(os.path.join(directory, project_name))
        
        # Select tuner type
        if tuner_type.lower() == 'random':
            tuner = kt.RandomSearch(
                hypermodel,
                objective=monitor,
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=directory,
                project_name=project_name,
                overwrite=overwrite
            )
        elif tuner_type.lower() == 'hyperband':
            tuner = kt.Hyperband(
                hypermodel,
                objective=monitor,
                max_epochs=epochs,
                factor=3,
                directory=directory,
                project_name=project_name,
                overwrite=overwrite
            )
        else:  # Default: Bayesian Optimization
            tuner = kt.BayesianOptimization(
                hypermodel,
                objective=monitor,
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=directory,
                project_name=project_name,
                overwrite=overwrite
            )
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping callback
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True
        )
        callbacks.append(stop_early)
        
        # ModelCheckpoint callback (if checkpoint path is provided)
        if checkpoint_path is not None:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=save_best_only,
                save_weights_only=save_weights_only,
                monitor=monitor,
                mode=mode
            )
            callbacks.append(checkpoint_callback)
        
        # Prepare fit arguments based on data type
        fit_args = {
            'epochs': epochs,
            'callbacks': callbacks
        }
        
        # Check if input is a tuple of numpy arrays or a tf.data.Dataset
        if isinstance(train_data, (tuple, list)) and len(train_data) == 2:
            # For numpy arrays: x_train, y_train
            x_train, y_train = train_data
            x_val, y_val = val_data
            
            fit_args['x'] = x_train
            fit_args['y'] = y_train
            fit_args['validation_data'] = (x_val, y_val)
            
            if batch_size is not None:
                fit_args['batch_size'] = batch_size
        else:
            # For tf.data.Dataset or any other data generator
            fit_args['x'] = train_data
            fit_args['validation_data'] = val_data
        
        # Start the search
        tuner.search(**fit_args)
        
        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        return best_hps, tuner
    
    def get_best_model(self, tuner):
        """
        Get the best model from the tuner.
        
        Args:
            tuner: Keras Tuner object that has completed search
            
        Returns:
            The best model found by the tuner
        """
        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model
    
    def build_model_from_hps(self, best_hps):
        """
        Build a model using the best hyperparameters.
        
        Args:
            best_hps: Best hyperparameters from the tuner
            
        Returns:
            Compiled model with the best hyperparameters
        """
        hypermodel = LSTMModelTuner(
            self.input_shape, 
            self.loss_function, 
            self.metrics,
            self.hp_config
        )
        return hypermodel.build(best_hps)


# Example usage:

# 1. Create the model instance
# input_shape = (30, 10)  # 30 timesteps, 10 features
# model = LSTMModel(input_shape)

# 2. Tune the model
# best_hps, tuner = model.tune_model(
#     x_train, y_train, 
#     x_val, y_val, 
#     max_trials=20
# )

# 3. Get and train the best model
# best_model = model.get_best_model(tuner)
# or
# best_model = model.build_model_from_hps(best_hps)
# best_model.fit(
#     x_train, y_train,
#     epochs=100,
#     validation_data=(x_val, y_val),
#     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
# )

# 4. Display hyperparameters
# print("Best hyperparameters:", best_hps.values)