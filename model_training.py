# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras as keras
import tensorflow as tf
tf.data.experimental.enable_debug_mode()

X = pd.read_csv('data.csv')
del X['Unnamed: 0']
X = X.values.reshape(-1,20,10)
X = X / 255
y = pd.read_csv('labels.csv')
del y['Unnamed: 0']
y = keras.utils.to_categorical(y, 2)

#Model training
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(50, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(20,10,1)))   
model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
model.add(keras.layers.Conv2D(25,(3,3), strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=512, activation="relu"))
model.add(keras.layers.Dense(units=2, activation="softmax"))

model.compile(loss="binary_crossentropy", metrics=["accuracy"],optimizer='rmsprop')

history = model.fit(X, y, epochs=20, verbose=1)

# Saves model
model.save('runs_predictor.h5')