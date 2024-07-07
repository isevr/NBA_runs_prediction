# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
tf.data.experimental.enable_debug_mode()


X = pd.read_csv('data.csv', index_col=0)
X = X.values.reshape(-1,20,10)
X = X / 255
y = pd.read_csv('labels.csv', index_col=0)
y = keras.utils.to_categorical(y, 2)

#Model training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(50, (3, 3), strides=1, padding="same", activation="relu", input_shape=(20,10,1)))   
model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
model.add(keras.layers.Conv2D(25,(3,3), strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), strides=2, padding="same"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=512, activation="relu"))
model.add(keras.layers.Dense(units=2, activation="softmax"))

model.compile(loss="binary_crossentropy", metrics=['accuracy','precision','recall','f1_score'], optimizer='rmsprop')

history = model.fit(X, y, epochs=20, verbose=1, validation_split=0.2, callbacks=[early_stop])

# plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

plt.figure(figsize=(12,5))

plt.subplot(3,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xticks([i for i in range(0,20)])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','validation'],loc='upper right')

plt.subplot(3,2,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','validation'],loc='lower right')
plt.xticks([i for i in range(0,20)])

plt.subplot(3,2,3)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.xlabel('epochs')
plt.ylabel('precision')
plt.legend(['train','validation'],loc='lower right')
plt.xticks([i for i in range(0,20)])
plt.show()

plt.subplot(3,2,4)
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.xlabel('epochs')
plt.ylabel('recall')
plt.legend(['train','validation'],loc='lower right')
plt.xticks([i for i in range(0,20)])

plt.subplot(3,2,5)
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.xlabel('epochs')
plt.ylabel('f1_score')
plt.legend(['train','validation'],loc='lower right')
plt.xticks([i for i in range(0,20)])

# Saves model
model.save('runs_predictor.keras')