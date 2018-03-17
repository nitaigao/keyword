from keras.models import load_model
import numpy as np
from audio_data import load_clip, files_and_labels

model = load_model('./tmp/model-199.h5')
_, _, _, int_to_label = files_and_labels('./data')

x = load_clip('./data/up/686d030b_nohash_2.wav.pkl')
x = x.reshape(-1, 98, 40, 1)

predictions = model.predict(x)
classes = np.argmax(predictions, axis=1)

for prediction in classes:
    label = int_to_label[prediction]
    print(label)
