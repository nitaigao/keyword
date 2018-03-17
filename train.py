from os import makedirs, path
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import numpy as np
import time

from audio_data import load_data

NUM_CLASSES = 15
EPOCHS = 200
BATCH_SIZE = 100

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

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(98, 40, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = Callbacks(model, session_directory)
    train_x = CMDS.train.x.reshape(-1, 98, 40, 1)
    train_y = np_utils.to_categorical(CMDS.train.y, NUM_CLASSES)
    model.fit(train_x, train_y, epochs=EPOCHS, callbacks=[callbacks], batch_size=BATCH_SIZE)

main()
