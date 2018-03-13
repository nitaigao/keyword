import numpy as np
import glob
import os
import re
import hashlib

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

# print(input_time_size, input_frequency_size)


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

def encode_image(image_filepath):
  # print("Encoding", image_filepath)
  with tf.Session() as sess:
    input_filename = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(input_filename)

    wav_decoder = contrib_audio.decode_wav(
      wav_loader,
      desired_channels=desired_channels,
      desired_samples=sample_rate
    )

    spectrogram = contrib_audio.audio_spectrogram(
      wav_decoder.audio,
      window_size_samples,
      window_stride_samples
    )

    mfcc = contrib_audio.mfcc(
      spectrogram,
      sample_rate=wav_decoder.sample_rate,
      dct_coefficient_count=dct_coefficient_count
    )

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
    self.x = x
    self.num_examples = len(x)
    self.y = y
    self.cache = {}

  def fetch_batch(self, offset, batch_size):
    start = offset * batch_size
    end = start + batch_size
    # print(f"\n{start} => {end} / {self.num_examples}")

    if offset in self.cache:
      x_batch = self.cache[offset]
    else:
      x_files = self.x[start:end]

      x_encoded_data = []
      for file in x_files:
        encoded_data = encode_image(file)
        x_encoded_data.append(encoded_data)

      x_batch = np.array(x_encoded_data)
      self.cache[offset] = x_batch

    y_batch = self.y[start:end]
    return [x_batch, y_batch]


def filenames(path, pattern):
  files_path = os.path.join(path, pattern)
  files = glob.glob(files_path)
  return files

def load_data(path):
  files = []

  files.extend(filenames(path, 'one/*.wav'))
  files.extend(filenames(path, 'two/*.wav'))
  files.extend(filenames(path, 'three/*.wav'))
  files.extend(filenames(path, 'four/*.wav'))
  files.extend(filenames(path, 'five/*.wav'))
  files.extend(filenames(path, 'six/*.wav'))
  files.extend(filenames(path, 'seven/*.wav'))
  files.extend(filenames(path, 'eight/*.wav'))
  files.extend(filenames(path, 'nine/*.wav'))
  files.extend(filenames(path, 'zero/*.wav'))
  files.extend(filenames(path, 'left/*.wav'))
  files.extend(filenames(path, 'right/*.wav'))
  files.extend(filenames(path, 'up/*.wav'))
  files.extend(filenames(path, 'down/*.wav'))
  files.extend(filenames(path, 'wow/*.wav'))

  np.random.shuffle(files)

  label_buckets = {}
  for file in files:
    label = file.split(os.sep)[-2]
    label_buckets[label] = 1

  labels = list(label_buckets.keys())

  label_to_int = dict((c, i) for i, c in enumerate(labels))
  int_to_label = dict((i, c) for i, c in enumerate(labels))

  one_hot = list(i for i, c in enumerate(labels))
  int_to_onehot = np.eye(len(one_hot), dtype=int)

  training_x = []
  training_y = []

  testing_x = []
  testing_y = []

  for file in files:
    set = which_set(file, 20, 20)
    label = file.split(os.sep)[-2]

    label_index = label_to_int[label]
    label_one_hot = int_to_onehot[label_index]

    if set == 'training':
      training_x.append(file)
      training_y.append(label_index)

    if set == 'testing':
      testing_x.append(file)
      testing_y.append(label_index)

  train_y = np.array(training_y, dtype=int)
  training = Dataset(training_x, train_y)
  print(f"{len(training_x)} available training samples")

  test_y = np.array(testing_y, dtype=int)
  testing = Dataset(testing_x, test_y)

  return Datasets(training, None, testing)
