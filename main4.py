import tensorflow as tf
from keras.models import Sequential

model = Sequential()

from keras.layers import Dense

from audio_data import load_data

cmds = load_data('/home/nk/Development/scratch/speech_commands')

model.add(Dense(31, input_shape=(3920,)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cmds = load_data('/home/nk/Development/scratch/speech_commands')
batch_size = 100
step = 0

with tf.Session() as sess:
  x_train, y_train = cmds.train.next_batch(step, batch_size, sess)

  model.fit(x_train, y_train, epochs=5, batch_size=batch_size)
