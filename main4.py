import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical


model = Sequential()

from keras.layers import Dense

from audio_data import load_data

cmds = load_data('/home/nk/Development/scratch/speech_commands')

y_train = to_categorical(cmds.train.y)
x_train = cmds.train.x
num_classes = y_train.shape[1]

model.add(Dense(num_classes, input_shape=(3920,)))
model.add(Dense(num_classes, input_shape=(3920,)))
model.add(Dense(num_classes, input_shape=(3920,)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(x_train.shape)
batch_size = 100

# model.train_on_batch(x_batch, y_batch)

model.fit(x_train, y_train, epochs=500, batch_size=batch_size)
