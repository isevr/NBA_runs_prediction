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

    def opt_loop(self, df):
        df = pd.read_csv(df)
        b = df.values.reshape(-1,10,11)

        ranges = {}
        confidences = {}

        # iteratively removing a length of events
        for i in range(1, 10):
            gen_preds = []
            confs = []

            # number of event arrays to use for this optimization
            for arr in b[:2]:
                rand_len = i
                
                arr_partial = arr[:rand_len, :].reshape(rand_len, 11, 1)

                x_missing = tf.Variable(np.random.rand(10 - rand_len, 11, 1), dtype=tf.float32)

                optimizer = tf.optimizers.Adam(learning_rate=0.01)

                loss_value = None  

                # steps per iteration
                for step in range(2):
                    with tf.GradientTape() as tape:
                        loss_value = self.loss(arr_partial, x_missing)

                    grads = tape.gradient(loss_value, [x_missing])

                    optimizer.apply_gradients(zip(grads, [x_missing]))

                missing = np.clip(np.round(x_missing.numpy()), 0, 3)
                
                full = np.concatenate([arr_partial, missing], axis=0)
                
                exp_full = np.expand_dims(full, axis=0)
                
                pred = np.argmax(self.model.predict(exp_full))
                
                gen_preds.append(pred)
                
                confidence = self.confidence_metric(loss_value.numpy(), rand_len)
                confs.append(confidence)

            ranges[i] = gen_preds
            confidences[i] = confs

        for i in range(1,10):
            print(np.mean(ranges[i]), np.mean(confidences[i]))

        predicted_full = pd.DataFrame(full.reshape(-1,110)).astype(int)
        predicted_full.columns = df.columns

        encoders = []
        encoder_dir = os.path.join('model', 'encoders')
        for file_name in os.listdir(encoder_dir):
            file_path = os.path.join(encoder_dir, file_name)
            
            if os.path.isfile(file_path):
                encoders.append(load(file_path))

        rules = []
        for encoder, column in zip(encoders, predicted_full.columns):
            try:
                rules.append(encoder.inverse_transform(predicted_full[column])[0])
            except ValueError:
                print(encoder, column)

        generated_rules = pd.DataFrame(rules).T

        return generated_rules
