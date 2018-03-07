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

class Datasets:
  def __init__(self, train, valid, test):
    self.train = train
    self.valid = valid
    self.test = test

class Dataset:
  def __init__(self, files):
    self.files = files
    self.num_examples = len(files)

  def next_batch(self, offset, batch_size, sess):
    data = np.zeros((batch_size, fingerprint_size))
    labels = np.zeros((batch_size, 31))

    # print(batch_size * offset, batch_size * offset + batch_size)

    for i, j in enumerate(range(batch_size * offset, batch_size * offset + batch_size)):
      input_filename = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(input_filename)
      file = self.files[j]

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
        input_filename: file['path']
      }

      data[i, :] = sess.run(mfcc, feed_dict).flatten()

      # We might not want to straight up return this
      labels[i, :] = file['label']

    return [
      data,
      labels
    ]


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

def load_data(path):
  files_path = os.path.join(path, '**/*.wav')
  files = glob.glob(files_path)
  np.random.shuffle(files)

  buckets = {
    'training':   [],
    'validation': [],
    'testing':    []
  }

  label_buckets = {}
  for file in files:
    label = file.split(os.sep)[-2]
    label_buckets[label] = 1

  labels = list(label_buckets.keys())

  label_to_int = dict((c, i) for i, c in enumerate(labels))
  int_to_label = dict((i, c) for i, c in enumerate(labels))

  one_hot = list(i for i, c in enumerate(labels))
  int_to_onehot = np.eye(len(one_hot))

  for file in files:
    set = which_set(file, 20, 20)
    label = file.split(os.sep)[-2]

    label_index = label_to_int[label]
    label_one_hot = int_to_onehot[label_index]

    sample = { 'label': label_one_hot, 'path': file }
    buckets[set].append(sample)

  training   = Dataset(buckets['training'][:300])
  validation = Dataset(buckets['validation'][:300])
  testing    = Dataset(buckets['testing'][:300])

  return Datasets(training, validation, testing)
