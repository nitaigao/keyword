import numpy as np
from keras.models import load_model
from audio_data import load_clip, files_and_labels

model = load_model('./tmp/1521265089/model-99.h5')
_, _, _, int_to_label = files_and_labels('./data')

x = load_clip('./data/nine/6a700f9d_nohash_0.wav.pkl')
x = x.reshape(-1, 98, 40, 1)

for i in range(100):
    predictions = model.predict(x)
    classes = np.argmax(predictions, axis=1)

    for prediction in classes:
        label = int_to_label[prediction]
        print(label)
