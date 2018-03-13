from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from audio_data import load_data

model = Sequential()

num_classes = 15

path = '/home/nk/Development/scratch/speech_commands'
cmds = load_data(path)

def train_generator(batch_size, steps):
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
steps = 20000 / batch_size

model.fit_generator(train_generator(batch_size, steps), steps_per_epoch=steps, epochs=10)

x, y = cmds.test.fetch_batch(0, 100)
x_test = x.reshape(-1, 98, 40, 1)
y_test = np_utils.to_categorical(y, num_classes)

predictions = model.predict(x_test, batch_size=100)
classes = np.argmax(predictions, axis=1)

print(confusion_matrix(y, classes))
