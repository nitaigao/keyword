import glob
import os
import re
import hashlib
import pickle
from os import path

from multiprocessing import Pool
import numpy as np

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

desired_channels = 1
sample_rate = 16000
clip_duration_ms = 1000
window_size_ms = 30.0
window_stride_ms = 10.0
dct_coefficient_count = 40

desired_samples = int(sample_rate * clip_duration_ms / 1000)
window_size_samples = int(sample_rate * window_size_ms / 1000)
window_stride_samples = int(sample_rate * window_stride_ms / 1000)
length_minus_window = (desired_samples - window_size_samples)

if length_minus_window < 0:
    spectrogram_length = 0
else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

fingerprint_size = dct_coefficient_count * spectrogram_length

input_time_size = spectrogram_length
input_frequency_size = dct_coefficient_count


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1 # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()

    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))

    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'

    return result

def encode_data(audio_data, sample_rate):
    spectrogram = contrib_audio.audio_spectrogram(
        audio_data,
        window_size_samples,
        window_stride_samples
    )

    print(spectrogram.shape)

    mfcc = contrib_audio.mfcc(
        spectrogram,
        sample_rate=sample_rate,
        dct_coefficient_count=dct_coefficient_count
    )

    return mfcc

def encode_image(image_filepath):
    with tf.Session() as sess:
        input_filename = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(input_filename)

        wav_decoder = contrib_audio.decode_wav(
            wav_loader,
            desired_channels=desired_channels,
            desired_samples=sample_rate
        )

        mfcc = encode_data(wav_decoder.audio, wav_decoder.sample_rate)

        feed_dict = {
            input_filename: image_filepath
        }

        encoded_data = sess.run(mfcc, feed_dict).flatten()
        return encoded_data

class Datasets:
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

class Dataset:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.num_examples = len(x)
        self.y = y

    def fetch_batch(self, offset, batch_size):
        start = offset * batch_size
        end = start + batch_size

        x_batch = np.array(self.x[start:end])
        y_batch = self.y[start:end]
        return [x_batch, y_batch]


def filenames(path, pattern):
    files_path = os.path.join(path, pattern)
    files = glob.glob(files_path)
    return files

def pickle_data(file):
    mfcc_filename = file + '.pkl'
    if not path.exists(mfcc_filename):
        encoded_data = encode_image(file)
        print(f"Caching {file}")
        with open(mfcc_filename, 'wb') as mfcc_file:
            pickle.dump(encoded_data, mfcc_file)
    return mfcc_filename

def cache_data(files):
    pool = Pool(processes=4)
    pool.map(pickle_data, files)

def load_clip(clip_path):
    with open(clip_path, 'rb') as mfcc_file:
        encoded_data = pickle.load(mfcc_file)
        encoded_x = np.array(encoded_data)
        return encoded_x

def files_and_labels(samples_path):
    files = []

    files.extend(filenames(samples_path, 'one/*.wav'))
    files.extend(filenames(samples_path, 'two/*.wav'))
    files.extend(filenames(samples_path, 'three/*.wav'))
    files.extend(filenames(samples_path, 'four/*.wav'))
    files.extend(filenames(samples_path, 'five/*.wav'))
    files.extend(filenames(samples_path, 'six/*.wav'))
    files.extend(filenames(samples_path, 'seven/*.wav'))
    files.extend(filenames(samples_path, 'eight/*.wav'))
    files.extend(filenames(samples_path, 'nine/*.wav'))
    files.extend(filenames(samples_path, 'zero/*.wav'))
    files.extend(filenames(samples_path, 'left/*.wav'))
    files.extend(filenames(samples_path, 'right/*.wav'))
    files.extend(filenames(samples_path, 'up/*.wav'))
    files.extend(filenames(samples_path, 'down/*.wav'))
    files.extend(filenames(samples_path, 'wow/*.wav'))

    cache_data(files)

    label_buckets = {}
    for file in files:
        label = file.split(os.sep)[-2]
        label_buckets[label] = 1

    labels = list(label_buckets.keys())

    label_to_int = dict((c, i) for i, c in enumerate(labels))
    int_to_label = dict((i, c) for i, c in enumerate(labels))

    np.random.shuffle(files)

    return [files, labels, label_to_int, int_to_label]

def load_data(samples_path):
    files, labels, label_to_int, int_to_label = files_and_labels(samples_path)

    training_x = []
    training_y = []

    testing_x = []
    testing_y = []

    for file in files:
        category = which_set(file, 20, 20)
        label = file.split(os.sep)[-2]

        label_index = label_to_int[label]

        mfcc_filename = file + '.pkl'

        encoded_x = load_clip(mfcc_filename)

        if category == 'training':
            training_x.append(encoded_x)
            training_y.append(label_index)

        if category == 'testing':
            testing_x.append(encoded_x)
            testing_y.append(label_index)

    train_y = np.array(training_y, dtype=int)
    training = Dataset(training_x, train_y)
    print(f"{len(training_x)} available training samples")

    test_y = np.array(testing_y, dtype=int)
    testing = Dataset(testing_x, test_y)

    return Datasets(training, None, testing)
