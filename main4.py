import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.utils import to_categorical


model = Sequential()

from keras.layers import Dense

cmds = load_data('/home/nk/Development/scratch/speech_commands')

batch_size = 200

def generator(path):
  while 1:
    x, y = process_line(line)
    yield (img, y)


model.add(Dense(30, input_shape=(3920,)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(generator('/home/nk/Development/scratch/speech_commands'), samples_per_epoch=10000, nb_epoch=10)

# for step in range(int(cmds.train.num_examples / batch_size)):
#   x_batch, y_batch = cmds.train.fetch_batch(step, batch_size)
#   y_batch_one_hot = to_categorical(y_batch)
#   loss, acc = model.train_on_batch(x_batch, y_batch_one_hot)
#   print(model.metrics_names[0], loss, model.metrics_names[1], acc)

