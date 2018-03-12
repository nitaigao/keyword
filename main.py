from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils

from audio_data import load_data

model = Sequential()

num_classes = 5

def generator(path, batch_size, steps):
  cmds = load_data(path)
  step = 0
  while 1:
    if steps == step:
      step = 0

    x, y = cmds.train.fetch_batch(step, batch_size)
    x = x.reshape(-1, 98, 40, 1)
    y = np_utils.to_categorical(y, num_classes)

    yield (x, y)

    step += 1

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(98, 40, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
steps = 1000 / batch_size

model.fit_generator(generator('/home/nk/Development/scratch/speech_commands', batch_size, steps), steps_per_epoch=steps, epochs=5000)
