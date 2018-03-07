import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

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

batch_size = 100

def main():
  with tf.Session() as sess:
    input_filename = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(input_filename)
    wav_decoder = contrib_audio.decode_wav(
      wav_loader,
      desired_channels=1,
      desired_samples=16000
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
      input_filename: '/tmp/speech_dataset/left/004ae714_nohash_0.wav'
    }

    result = sess.run(mfcc, feed_dict).flatten()
    print(result.shape)

    data = np.zeros((batch_size, fingerprint_size))
    print(data.shape)
    data[0, :] = result

main()
