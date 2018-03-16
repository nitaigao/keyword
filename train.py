from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import numpy as np

from audio_data import load_data

NUM_CLASSES = 15

CMDS = load_data('/home/nk/Development/scratch/speech_commands')

class Callbacks(Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs):
        x, y = CMDS.test.fetch_batch(0, 100)
        x_test = x.reshape(-1, 98, 40, 1)
        y_test = np_utils.to_categorical(y, NUM_CLASSES)

        predictions = self.model.predict(x_test, batch_size=100)
        classes = np.argmax(predictions, axis=1)

        print(confusion_matrix(y, classes))
        return

def train_generator(batch_size, steps):
    step = 0
    while 1:
        if steps == step:
            step = 0

        x, y = CMDS.train.fetch_batch(step, batch_size)
        x = x.reshape(-1, 98, 40, 1)
        y = np_utils.to_categorical(y, NUM_CLASSES)
        yield (x, y)
        step += 1

def main():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(98, 40, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 200
    batch_size = 100
    steps = 20000 / batch_size
    callbacks = Callbacks(model)
    model.fit_generator(train_generator(batch_size, steps), steps_per_epoch=steps,
                                                            epochs=epochs,
                                                            callbacks=[callbacks])

    x, y = CMDS.test.fetch_batch(0, 100)
    x_test = x.reshape(-1, 98, 40, 1)
    y_test = np_utils.to_categorical(y, NUM_CLASSES)

    predictions = model.predict(x_test, batch_size=100)
    classes = np.argmax(predictions, axis=1)

    print(confusion_matrix(y, classes))
