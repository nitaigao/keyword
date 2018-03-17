from os import makedirs, path
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, Activation
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import numpy as np
import time

from audio_data import load_data

NUM_CLASSES = 15
EPOCHS = 150
BATCH_SIZE = 128

CMDS = load_data('./data')


class Callbacks(Callback):
    def __init__(self, model, session):
        super().__init__()
        self.model = model
        self.session = session

    def on_epoch_end(self, epoch, logs):
        x_test = CMDS.test.x.reshape(-1, 98, 40, 1)
        y_test = np_utils.to_categorical(CMDS.test.y, NUM_CLASSES)

        predictions = self.model.predict(x_test, batch_size=BATCH_SIZE)
        classes = np.argmax(predictions, axis=1)

        print(confusion_matrix(CMDS.test.y, classes))

        model_filename = path.join(self.session, f"model-{epoch}.h5")
        self.model.save(model_filename)

def main():
    timestamp = int(time.time())
    session_directory = f"./tmp/{timestamp}"
    makedirs(session_directory)

    model = Sequential()
    model.add(Conv2D(128, 5, border_mode='valid', input_shape=(98, 40, 1)))
    model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

    sgd = SGD(lr = 0.02)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = Callbacks(model, session_directory)
    train_x = CMDS.train.x.reshape(-1, 98, 40, 1)
    train_y = np_utils.to_categorical(CMDS.train.y, NUM_CLASSES)
    model.fit(train_x, train_y, epochs=EPOCHS, callbacks=[callbacks], batch_size=BATCH_SIZE)

main()
