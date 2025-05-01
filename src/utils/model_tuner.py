from typing import Any, Dict, List, Optional, Tuple, Union

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src.models import Sequential
from tensorflow import keras

ALL_MODELS = ['LSTMModel', 'TransformerModel', 'ConvLSTMModel', 'TCNModel', 'BiLSTMAttentionModel']

# fmt: off
class TransformerModel(kt.HyperModel):
    def __init__(self, 
                input_shape: Tuple[int, int], 
                loss_function: Union[str, tf.keras.losses.Loss], 
                metrics: List[Union[str, tf.keras.metrics.Metric]],
                hp_config: Dict[str, Dict[str, Any]]):
        """
        Transformer-based model for time series prediction.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            loss_function: Loss function for model compilation
            metrics: Metrics for model evaluation
            hp_config: Dictionary containing hyperparameter search space configurations
        """
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.metrics = metrics
        self.hp_config = hp_config
    
    def build(self, hp):
        # Get default values from config
        head_size = hp.Int(
            'head_size',
            min_value=self.hp_config.get('head_size', {}).get('min_value', 32),
            max_value=self.hp_config.get('head_size', {}).get('max_value', 128),
            step=self.hp_config.get('head_size', {}).get('step', 32)
        )
        
        num_heads = hp.Int(
            'num_heads',
            min_value=self.hp_config.get('num_heads', {}).get('min_value', 1),
            max_value=self.hp_config.get('num_heads', {}).get('max_value', 8),
            step=self.hp_config.get('num_heads', {}).get('step', 1)
        )
        
        ff_dim = hp.Int(
            'ff_dim',
            min_value=self.hp_config.get('ff_dim', {}).get('min_value', 32),
            max_value=self.hp_config.get('ff_dim', {}).get('max_value', 256),
            step=self.hp_config.get('ff_dim', {}).get('step', 32)
        )
        
        num_transformer_blocks = hp.Int(
            'num_transformer_blocks',
            min_value=self.hp_config.get('num_transformer_blocks', {}).get('min_value', 1),
            max_value=self.hp_config.get('num_transformer_blocks', {}).get('max_value', 4),
            step=self.hp_config.get('num_transformer_blocks', {}).get('step', 1)
        )
        
        mlp_units = hp.Int(
            'mlp_units',
            min_value=self.hp_config.get('mlp_units', {}).get('min_value', 32),
            max_value=self.hp_config.get('mlp_units', {}).get('max_value', 128),
            step=self.hp_config.get('mlp_units', {}).get('step', 32)
        )
        
        mlp_dropout = hp.Float(
            'mlp_dropout',
            min_value=self.hp_config.get('mlp_dropout', {}).get('min_value', 0.0),
            max_value=self.hp_config.get('mlp_dropout', {}).get('max_value', 0.5),
            step=self.hp_config.get('mlp_dropout', {}).get('step', 0.1)
        )
        
        dropout = hp.Float(
            'dropout',
            min_value=self.hp_config.get('dropout', {}).get('min_value', 0.0),
            max_value=self.hp_config.get('dropout', {}).get('max_value', 0.5),
            step=self.hp_config.get('dropout', {}).get('step', 0.1)
        )
        
        # Define model
        inputs = keras.Input(shape=(self.input_shape[0], self.input_shape[1]))
        x = inputs
        
        # Positional encoding
        position_embedding = layers.Embedding(
            input_dim=self.input_shape[0], 
            output_dim=self.input_shape[1]
        )
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        positions = position_embedding(positions)
        x = x + positions
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=head_size,
                dropout=dropout
            )(x, x)
            
            # Skip connection 1
            x = layers.Add()([attention_output, x])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed-forward network
            ffn = keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(self.input_shape[1]),
            ])
            
            # Skip connection 2
            ffn_output = ffn(x)
            x = layers.Add()([ffn_output, x])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final MLP
        for _ in range(hp.Int('mlp_layers', 1, 2)):
            x = layers.Dense(mlp_units, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        
        # Output layer
        output_config = self.hp_config.get('output', {})
        output_units = output_config.get('units', 1)
        output_activation = output_config.get('activation', 'linear')
        outputs = layers.Dense(output_units, activation=output_activation)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Optimizer hyperparameters
        lr_config = self.hp_config.get('learning_rate', {})
        learning_rate = hp.Float(
            'learning_rate',
            min_value=lr_config.get('min_value', 1e-4),
            max_value=lr_config.get('max_value', 1e-2),
            sampling=lr_config.get('sampling', 'log')
        )
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=hp.Float(
                    'weight_decay',
                    min_value=1e-6,
                    max_value=1e-3,
                    sampling='log'
                )
            ),
            loss=self.loss_function,
            metrics=self.metrics
        )
        
        return model


class ConvLSTMModel(kt.HyperModel):
    def __init__(self, 
                input_shape: Tuple[int, int], 
                loss_function: Union[str, tf.keras.losses.Loss], 
                metrics: List[Union[str, tf.keras.metrics.Metric]],
                hp_config: Dict[str, Dict[str, Any]]):
        """
        Convolutional LSTM model for time series prediction.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            loss_function: Loss function for model compilation
            metrics: Metrics for model evaluation
            hp_config: Dictionary containing hyperparameter search space configurations
        """
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.metrics = metrics
        self.hp_config = hp_config
    
    def build(self, hp):
        model = Sequential()
        model.add(layers.Input(shape=(self.input_shape[0], self.input_shape[1])))
        
        # Reshape for Conv1D
        model.add(layers.Reshape((self.input_shape[0], self.input_shape[1], 1)))
        
        # Conv layers
        num_conv_blocks = hp.Int(
            'num_conv_blocks',
            min_value=self.hp_config.get('num_conv_blocks', {}).get('min_value', 1),
            max_value=self.hp_config.get('num_conv_blocks', {}).get('max_value', 3),
            step=self.hp_config.get('num_conv_blocks', {}).get('step', 1)
        )
        
        for i in range(num_conv_blocks):
            filters = hp.Int(
                f'conv_filters_{i}',
                min_value=self.hp_config.get('conv_filters', {}).get('min_value', 16),
                max_value=self.hp_config.get('conv_filters', {}).get('max_value', 128),
                step=self.hp_config.get('conv_filters', {}).get('step', 16)
            )
            
            kernel_size = hp.Int(
                f'kernel_size_{i}',
                min_value=self.hp_config.get('kernel_size', {}).get('min_value', 2),
                max_value=self.hp_config.get('kernel_size', {}).get('max_value', 5),
                step=self.hp_config.get('kernel_size', {}).get('step', 1)
            )
            
            dilation_rate = hp.Int(
                f'dilation_rate_{i}',
                min_value=self.hp_config.get('dilation_rate', {}).get('min_value', 1),
                max_value=self.hp_config.get('dilation_rate', {}).get('max_value', 4),
                step=self.hp_config.get('dilation_rate', {}).get('step', 1)
            )
            
            model.add(layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_size, 1),
                padding='same',
                dilation_rate=(dilation_rate, 1),
                activation='relu'
            ))
            
            if hp.Boolean(f'use_batch_norm_{i}', default=True):
                model.add(layers.BatchNormalization())
                
            dropout_rate = hp.Float(
                f'conv_dropout_{i}',
                min_value=self.hp_config.get('conv_dropout', {}).get('min_value', 0.0),
                max_value=self.hp_config.get('conv_dropout', {}).get('max_value', 0.3),
                step=self.hp_config.get('conv_dropout', {}).get('step', 0.1)
            )
            model.add(layers.Dropout(dropout_rate))
        
        # Reshape back for ConvLSTM1D
        new_shape = model.output_shape
        model.add(layers.Reshape((new_shape[1], new_shape[2] * new_shape[3])))
        
        # LSTM layers
        num_lstm_layers = hp.Int(
            'num_lstm_layers',
            min_value=self.hp_config.get('num_lstm_layers', {}).get('min_value', 1),
            max_value=self.hp_config.get('num_lstm_layers', {}).get('max_value', 3),
            step=self.hp_config.get('num_lstm_layers', {}).get('step', 1)
        )
        
        for i in range(num_lstm_layers):
            lstm_units = hp.Int(
                f'lstm_units_{i}',
                min_value=self.hp_config.get('lstm_units', {}).get('min_value', 32),
                max_value=self.hp_config.get('lstm_units', {}).get('max_value', 256),
                step=self.hp_config.get('lstm_units', {}).get('step', 32)
            )
            
            return_sequences = i < num_lstm_layers - 1
            
            model.add(layers.LSTM(
                units=lstm_units,
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_dropout=hp.Float(
                    f'recurrent_dropout_{i}',
                    min_value=self.hp_config.get('recurrent_dropout', {}).get('min_value', 0.0),
                    max_value=self.hp_config.get('recurrent_dropout', {}).get('max_value', 0.3),
                    step=self.hp_config.get('recurrent_dropout', {}).get('step', 0.1)
                )
            ))
            
            if hp.Boolean(f'lstm_batch_norm_{i}', default=True):
                model.add(layers.BatchNormalization())
                
            model.add(layers.Dropout(hp.Float(
                f'lstm_dropout_{i}',
                min_value=self.hp_config.get('lstm_dropout', {}).get('min_value', 0.0),
                max_value=self.hp_config.get('lstm_dropout', {}).get('max_value', 0.5),
                step=self.hp_config.get('lstm_dropout', {}).get('step', 0.1)
            )))
        
        # Dense output layer
        output_config = self.hp_config.get('output', {})
        output_units = output_config.get('units', 1)
        output_activation = output_config.get('activation', 'linear')
        
        model.add(layers.Dense(output_units, activation=output_activation))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=hp.Float(
                    'learning_rate',
                    min_value=self.hp_config.get('learning_rate', {}).get('min_value', 1e-4),
                    max_value=self.hp_config.get('learning_rate', {}).get('max_value', 1e-2),
                    sampling='log'
                )
            ),
            loss=self.loss_function,
            metrics=self.metrics
        )
        
        return model


class TCNModel(kt.HyperModel):
    def __init__(self, 
                input_shape: Tuple[int, int], 
                loss_function: Union[str, tf.keras.losses.Loss], 
                metrics: List[Union[str, tf.keras.metrics.Metric]],
                hp_config: Dict[str, Dict[str, Any]]):
        """
        Temporal Convolutional Network model for time series prediction.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            loss_function: Loss function for model compilation
            metrics: Metrics for model evaluation
            hp_config: Dictionary containing hyperparameter search space configurations
        """
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.metrics = metrics
        self.hp_config = hp_config
    
    def _residual_block(self, x, filters, kernel_size, dilation_rate, hp, block_id):
        """Create a residual block for the TCN"""
        # Skip connection
        skip = x
        
        # If dimensions don't match, adapt skip connection
        if x.shape[-1] != filters:
            skip = layers.Conv1D(filters, 1, padding='same')(skip)
        
        # First Conv1D layer
        conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='linear'
        )(x)
        
        # Optional BatchNormalization
        if hp.Boolean(f'use_batch_norm_{block_id}', default=True):
            conv1 = layers.BatchNormalization()(conv1)
        
        # Activation and Dropout
        conv1 = layers.Activation('relu')(conv1)
        conv1 = layers.Dropout(hp.Float(
            f'dropout_{block_id}_1',
            min_value=self.hp_config.get('dropout', {}).get('min_value', 0.0),
            max_value=self.hp_config.get('dropout', {}).get('max_value', 0.5),
            step=self.hp_config.get('dropout', {}).get('step', 0.1)
        ))(conv1)
        
        # Second Conv1D layer
        conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='linear'
        )(conv1)
        
        # Optional BatchNormalization
        if hp.Boolean(f'use_batch_norm_{block_id}_2', default=True):
            conv2 = layers.BatchNormalization()(conv2)
        
        # Activation and Dropout
        conv2 = layers.Activation('relu')(conv2)
        conv2 = layers.Dropout(hp.Float(
            f'dropout_{block_id}_2',
            min_value=self.hp_config.get('dropout', {}).get('min_value', 0.0),
            max_value=self.hp_config.get('dropout', {}).get('max_value', 0.5),
            step=self.hp_config.get('dropout', {}).get('step', 0.1)
        ))(conv2)
        
        # Add skip connection
        output = layers.Add()([conv2, skip])
        
        return output
    
    def build(self, hp):
        inputs = keras.Input(shape=(self.input_shape[0], self.input_shape[1]))
        x = inputs
        
        # Initial Conv1D layer
        initial_filters = hp.Int(
            'initial_filters',
            min_value=self.hp_config.get('initial_filters', {}).get('min_value', 32),
            max_value=self.hp_config.get('initial_filters', {}).get('max_value', 128),
            step=self.hp_config.get('initial_filters', {}).get('step', 32)
        )
        
        kernel_size = hp.Int(
            'kernel_size',
            min_value=self.hp_config.get('kernel_size', {}).get('min_value', 2),
            max_value=self.hp_config.get('kernel_size', {}).get('max_value', 5),
            step=self.hp_config.get('kernel_size', {}).get('step', 1)
        )
        
        x = layers.Conv1D(filters=initial_filters, kernel_size=1, padding='causal')(x)
        
        # Number of TCN blocks
        num_blocks = hp.Int(
            'num_blocks',
            min_value=self.hp_config.get('num_blocks', {}).get('min_value', 2),
            max_value=self.hp_config.get('num_blocks', {}).get('max_value', 6),
            step=self.hp_config.get('num_blocks', {}).get('step', 1)
        )
        
        # Build TCN dilated residual blocks
        for i in range(num_blocks):
            filters = hp.Int(
                f'filters_block_{i}',
                min_value=self.hp_config.get('filters', {}).get('min_value', 32),
                max_value=self.hp_config.get('filters', {}).get('max_value', 256),
                step=self.hp_config.get('filters', {}).get('step', 32)
            )
            
            # Exponentially increasing dilation rate: 2^i (1, 2, 4, 8, 16, ...)
            dilation_rate = 2 ** i
            
            x = self._residual_block(
                x=x,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                hp=hp,
                block_id=i
            )
        
        # Final layers
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        dense_layers = hp.Int(
            'dense_layers',
            min_value=self.hp_config.get('dense_layers', {}).get('min_value', 0),
            max_value=self.hp_config.get('dense_layers', {}).get('max_value', 2),
            step=self.hp_config.get('dense_layers', {}).get('step', 1)
        )
        
        for i in range(dense_layers):
            units = hp.Int(
                f'dense_units_{i}',
                min_value=self.hp_config.get('dense_units', {}).get('min_value', 32),
                max_value=self.hp_config.get('dense_units', {}).get('max_value', 128),
                step=self.hp_config.get('dense_units', {}).get('step', 32)
            )
            
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(hp.Float(
                f'dense_dropout_{i}',
                min_value=self.hp_config.get('dense_dropout', {}).get('min_value', 0.0),
                max_value=self.hp_config.get('dense_dropout', {}).get('max_value', 0.5),
                step=self.hp_config.get('dense_dropout', {}).get('step', 0.1)
            ))(x)
        
        # Output layer
        output_config = self.hp_config.get('output', {})
        output_units = output_config.get('units', 1)
        output_activation = output_config.get('activation', 'linear')
        
        outputs = layers.Dense(output_units, activation=output_activation)(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=hp.Float(
                    'learning_rate',
                    min_value=self.hp_config.get('learning_rate', {}).get('min_value', 1e-4),
                    max_value=self.hp_config.get('learning_rate', {}).get('max_value', 1e-2),
                    sampling='log'
                )
            ),
            loss=self.loss_function,
            metrics=self.metrics
        )
        
        return model


class BiLSTMAttentionModel(kt.HyperModel):
    def __init__(self, 
                input_shape: Tuple[int, int], 
                loss_function: Union[str, tf.keras.losses.Loss], 
                metrics: List[Union[str, tf.keras.metrics.Metric]],
                hp_config: Dict[str, Dict[str, Any]]):
        """
        Bidirectional LSTM with Attention for time series prediction.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            loss_function: Loss function for model compilation
            metrics: Metrics for model evaluation
            hp_config: Dictionary containing hyperparameter search space configurations
        """
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.metrics = metrics
        self.hp_config = hp_config
    
    def attention_layer(self, inputs, attention_size):
        """
        Attention mechanism implementation.
        
        Args:
            inputs: Input tensor
            attention_size: Size of the attention layer
            
        Returns:
            Context vector and attention weights
        """
        # Compute a time-wise hidden representation
        hidden_layer = layers.Dense(attention_size, activation='tanh')(inputs)
        # Compute weights
        attention_weights = layers.Dense(1, activation='softmax', use_bias=False)(hidden_layer)
        # Reshape to match the input shape
        attention_weights = tf.reshape(attention_weights, (-1, tf.shape(inputs)[1]))
        # Create attention weights
        attention_weights = tf.transpose(attention_weights, [0, 1])
        # Compute the context vector
        context_vector = tf.matmul(tf.transpose(inputs, [0, 2, 1]), 
                                  tf.expand_dims(attention_weights, 2))
        context_vector = tf.reshape(context_vector, (-1, inputs.shape[2]))
        
        return context_vector, attention_weights
    
    def build(self, hp):
        # Input layer
        inputs = keras.Input(shape=(self.input_shape[0], self.input_shape[1]))
        
        # Number of BiLSTM layers
        num_bilstm_layers = hp.Int(
            'num_bilstm_layers',
            min_value=self.hp_config.get('num_bilstm_layers', {}).get('min_value', 1),
            max_value=self.hp_config.get('num_bilstm_layers', {}).get('max_value', 3),
            step=self.hp_config.get('num_bilstm_layers', {}).get('step', 1)
        )
        
        x = inputs
        
        # Build BiLSTM layers
        for i in range(num_bilstm_layers):
            units = hp.Int(
                f'bilstm_units_{i}',
                min_value=self.hp_config.get('bilstm_units', {}).get('min_value', 32),
                max_value=self.hp_config.get('bilstm_units', {}).get('max_value', 256),
                step=self.hp_config.get('bilstm_units', {}).get('step', 32)
            )
            
            # Return sequences for all layers (needed for attention)
            return_sequences = True if i < num_bilstm_layers - 1 else hp.Boolean(
                'return_sequences_last', default=True
            )
            
            # Bidirectional LSTM layer
            x = layers.Bidirectional(
                layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=hp.Float(
                        f'dropout_{i}',
                        min_value=self.hp_config.get('dropout', {}).get('min_value', 0.0),
                        max_value=self.hp_config.get('dropout', {}).get('max_value', 0.5),
                        step=self.hp_config.get('dropout', {}).get('step', 0.1)
                    ),
                    recurrent_dropout=hp.Float(
                        f'recurrent_dropout_{i}',
                        min_value=self.hp_config.get('recurrent_dropout', {}).get('min_value', 0.0),
                        max_value=self.hp_config.get('recurrent_dropout', {}).get('max_value', 0.3),
                        step=self.hp_config.get('recurrent_dropout', {}).get('step', 0.1)
                    )
                )
            )(x)
            
            # Optional BatchNormalization
            if hp.Boolean(f'batch_norm_{i}', default=True):
                x = layers.BatchNormalization()(x)
        
        # Apply attention if return_sequences is True
        if hp.Boolean('return_sequences_last', default=True):
            attention_size = hp.Int(
                'attention_size',
                min_value=self.hp_config.get('attention_size', {}).get('min_value', 32),
                max_value=self.hp_config.get('attention_size', {}).get('max_value', 128),
                step=self.hp_config.get('attention_size', {}).get('step', 32)
            )
            
            context_vector, _ = self.attention_layer(x, attention_size)
            x = context_vector
        else:
            # Global average pooling if no attention
            x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        num_dense_layers = hp.Int(
            'num_dense_layers',
            min_value=self.hp_config.get('num_dense_layers', {}).get('min_value', 0),
            max_value=self.hp_config.get('num_dense_layers', {}).get('max_value', 2),
            step=self.hp_config.get('num_dense_layers', {}).get('step', 1)
        )
        
        for i in range(num_dense_layers):
            dense_units = hp.Int(
                f'dense_units_{i}',
                min_value=self.hp_config.get('dense_units', {}).get('min_value', 32),
                max_value=self.hp_config.get('dense_units', {}).get('max_value', 128),
                step=self.hp_config.get('dense_units', {}).get('step', 32)
            )
            
            x = layers.Dense(dense_units, activation='relu')(x)
            
            x = layers.Dropout(hp.Float(
                f'dense_dropout_{i}',
                min_value=self.hp_config.get('dense_dropout', {}).get('min_value', 0.0),
                max_value=self.hp_config.get('dense_dropout', {}).get('max_value', 0.5),
                step=self.hp_config.get('dense_dropout', {}).get('step', 0.1)
            ))(x)
        
        # Output layer
        output_config = self.hp_config.get('output', {})
        output_units = output_config.get('units', 1)
        output_activation = output_config.get('activation', 'linear')
        
        outputs = layers.Dense(output_units, activation=output_activation)(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        learning_rate = hp.Float(
            'learning_rate',
            min_value=self.hp_config.get('learning_rate', {}).get('min_value', 1e-4),
            max_value=self.hp_config.get('learning_rate', {}).get('max_value', 1e-2),
            sampling='log'
        )
        
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
            loss=self.loss_function,
            metrics=self.metrics
        )
        
        return model


class LSTMModel(kt.HyperModel):
    def __init__(self, 
                 input_shape: Tuple[int, int], 
                 loss_function: Union[str, tf.keras.losses.Loss], 
                 metrics: List[Union[str, tf.keras.metrics.Metric]],
                 hp_config: Dict[str, Dict[str, Any]]):
        """
        Initialize the LSTMModel with input shape, compilation parameters, and hyperparameter config.
        
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


class ModelTuner:
    def __init__(self, 
                 model_class,
                 input_shape: Union[Tuple, List], 
                 loss_function: Union[str, tf.keras.losses.Loss] = 'mse', 
                 metrics: List[Union[str, tf.keras.metrics.Metric]] = ['mae'],
                 hp_config: Optional[Dict[str, Dict[str, Any]]] = {}):
        """
        Initialize a generic model tuner that can work with any model class that inherits from kt.HyperModel.
        
        Args:
            model_class: The model class to use (must be a kt.HyperModel subclass)
            input_shape (tuple/list): Shape of input data
            loss_function (str or loss object): Loss function name or object
            metrics (list): List of metrics for evaluation
            hp_config (dict, optional): Dictionary containing hyperparameter search space configurations
        """
        self.model_class = model_class
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.metrics = metrics
        self.hp_config = hp_config
    
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
                  project_name: str = 'model_tuning', 
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
        Use Keras Tuner to find optimal hyperparameters for the model.
        
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
        # Create the hypermodel instance with current hp_config
        hypermodel = self.model_class(
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
        hypermodel = self.model_class(
            self.input_shape, 
            self.loss_function, 
            self.metrics,
            self.hp_config
        )
        return hypermodel.build(best_hps)


# Example usage:

# # 1. Create the model instance using LSTMModel
# input_shape = (30, 10)  # 30 timesteps, 10 features
# tuner = ModelTuner(
#     model_class=LSTMModel,
#     input_shape=input_shape
# )

# # 2. Tune the model
# best_hps, tuner_instance = tuner.tune_model(
#     train_data=(x_train, y_train), 
#     val_data=(x_val, y_val), 
#     max_trials=20
# )

# # 3. Get and train the best model
# best_model = tuner.get_best_model(tuner_instance)
# # or
# best_model = tuner.build_model_from_hps(best_hps)
# best_model.fit(
#     x_train, y_train,
#     epochs=100,
#     validation_data=(x_val, y_val),
#     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
# )