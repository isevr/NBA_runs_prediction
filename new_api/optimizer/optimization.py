import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from joblib import load


class SequenceOptimization:
    def __init__(self, model):
        self.model = load_model(model)
        
    def loss(self, x_partial, x_missing):
        full_input = tf.concat([x_partial, x_missing], axis=0)  
        full_input = tf.expand_dims(full_input, axis=0)  # batch dim

        # model predictions
        predictions = self.model(full_input)

        # probability of the 'run' class
        run_probability = predictions[0, 1]

        # negative log probability 
        return -tf.math.log(run_probability + 1e-8)  # epsilon to avoid log(0)

    def confidence_metric(self, loss_value, length, max_len=10, alpha=1.0):
        normalized_length = length / max_len
        
        confidence = (1 / (1 + alpha * loss_value)) * normalized_length
        return confidence

    def opt_loop(self, custom_row, missing_value=np.nan, steps=500):
        """
    Process a single basketball game sequence row (1x110).
    
    Args:
    - custom_row: A 1D NumPy array (shape 110,) representing one sequence.
    - model: The trained TensorFlow model.
    - missing_value: Placeholder for missing values (NaN, -1, etc.).
    - steps: Number of optimization steps.
    
    Returns:
    - full_filled: Completed sequence with missing values filled.
    - prediction: Model prediction for the completed array.
    - confidence: Confidence score.
    """
    
        # Reshape to (10,11) for processing
        reshaped_array = custom_row.reshape(10, 11).astype(float)

        # Identify missing positions (NaN or placeholder)
        missing_mask = np.isnan(reshaped_array) | (reshaped_array == missing_value)
        
        # Convert known values to TensorFlow tensor
        known_values = tf.convert_to_tensor(np.nan_to_num(reshaped_array), dtype=tf.float32)

        # Initialize missing values as trainable variables
        x_missing = tf.Variable(known_values, dtype=tf.float32)

        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        # Optimization loop
        for step in range(steps):
            with tf.GradientTape() as tape:
                loss_value = self.loss(x_missing, x_missing)

            grads = tape.gradient(loss_value, [x_missing])
            
            if grads[0] is not None:  # Ensure gradients exist
                optimizer.apply_gradients(zip(grads, [x_missing]))

        # Convert optimized tensor back to NumPy
        optimized_array = x_missing.numpy()

        # Fill only missing parts in the original array
        full_filled = reshaped_array.copy()
        full_filled[missing_mask] = np.clip(np.round(optimized_array[missing_mask]), 0, 3)

        # Prepare for model prediction
        exp_full = np.expand_dims(full_filled, axis=0)
        prediction = np.argmax(self.model.predict(exp_full))

        # Compute confidence
        confidence = self.confidence_metric(loss_value.numpy(), len(reshaped_array))

        return full_filled, prediction, confidence
