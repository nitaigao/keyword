import pyaudio
import tensorflow as tf
import numpy as np
from keras.models import load_model

from audio_data import encode_data, files_and_labels
from ring_buffer import RingBuffer

FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 320

def main():
    model = load_model('./tmp/1521265089/model-99.h5')
    _, _, _, int_to_label = files_and_labels('./data')

    ring_buffer = RingBuffer(SAMPLE_RATE)

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    output = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        frames_per_buffer=CHUNK,
                        output=True)

    stream.start_stream()
    output.start_stream()

    silence = chr(0) * CHUNK * CHANNELS * 2

    audio_data = tf.placeholder(tf.float32, [SAMPLE_RATE, 1])
    mfcc = encode_data(audio_data, SAMPLE_RATE)

    while True:
        available = stream.get_read_available()
        if available > 0:
            for _ in range(int(available / CHUNK)):
                data = stream.read(CHUNK)
                ring_buffer.append(data)
                # output.write(data)
        else:
            output.write(silence)

            raw_audio_data = np.array(ring_buffer.get())

            to_audio_data = raw_audio_data.reshape([SAMPLE_RATE, 1])

            feed_dict = {
                audio_data: to_audio_data
            }

            with tf.Session() as sess:
                mfcc_result = sess.run(mfcc, feed_dict)

            x = mfcc_result.reshape(-1, 98, 40, 1)
            predictions = model.predict(x)

            classes = np.argmax(predictions, axis=1)
            for prediction in classes:
              label = int_to_label[prediction]
              print(label)

    stream.stop_stream()
    stream.close()

    audio.terminate()

main()
