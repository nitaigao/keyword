import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.utils import to_categorical


model = Sequential()

from keras.layers import Dense, Conv2D
from audio_data import load_data


def generator(path, batch_size):
  cmds = load_data(path)
  offset = 0
  while 1:
    x, y = cmds.train.fetch_batch(offset, batch_size)
    yield (x, y)
    offset += 1

model.add(Dense(31, input_shape=(3920,)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100

model.fit_generator(generator('/home/nk/Development/scratch/speech_commands', batch_size), steps_per_epoch=10000, epochs=10)
