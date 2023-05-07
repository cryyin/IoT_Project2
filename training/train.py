# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras import models
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
print(tf.__version__)

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
# from tqdm import tqdm # replace with this if moving out of notebook

import os
import pathlib

from datetime import datetime as dt

from IPython import display


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

i16min = -2**15
i16max = 2**15-1
fsamp = 16000
wave_length_ms = 1000
wave_length_samps = int(wave_length_ms*fsamp/1000)
window_size_ms=60
window_step_ms=40
num_filters = 32
use_microfrontend = True
# dataset = 'mini-speech'
# dataset = 'full-speech-ds' # use the full speech commands as a pre-built TF dataset
dataset = 'full-speech-files' # use the full speech commands stored as files

silence_str = "_silence"
spice_str = "_spice"
unknown_str = "_unknown"
no_noise_mode=1
EPOCHS = 25

commands = ['left', 'right']
if dataset == 'mini-speech':
  data_dir = pathlib.Path(os.path.join(os.getcwd(), 'data/mini_speech_commands'))
  if not data_dir.exists():
    tf.keras.utils.get_file('mini_speech_commands.zip',
          origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
          extract=True, cache_dir='.', cache_subdir='data')
  # commands = np.array(tf.io.gfile.listdir(str(data_dir))) # if you want to use all the command words
  # commands = commands[commands != 'README.md']
elif dataset == 'full-speech-files':
  # data_dir = '/dfs/org/Holleman-Coursework/data/speech_dataset'
  data_dir = pathlib.Path(os.path.join(os.getcwd(), 'audio\dataset\speech_commands_v0.01'))

elif dataset == 'full-speech-ds':
    raise RuntimeError("full-speech-ds is not really supported yet")

label_list = commands.copy()
label_list.insert(0, silence_str)
label_list.insert(1, unknown_str)
print('label_list:', label_list)

if dataset == 'mini-speech' or dataset == 'full-speech-files':
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
    # with the next commented-out line, you can choose only files for words in label_list
    # filenames = tf.concat([tf.io.gfile.glob(str(data_dir) + '/' + cmd + '/*') for cmd in label_list], 0)
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    # print('Number of examples per label:',
    #       len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])
for i in range(10):
    print(filenames[i].numpy().decode('utf8'))

if dataset == 'mini-speech':
    print('Using mini-speech')
    num_train_files = int(0.8 * num_samples)
    num_val_files = int(0.1 * num_samples)
    num_test_files = num_samples - num_train_files - num_val_files
    train_files = filenames[:num_train_files]
    val_files = filenames[num_train_files: num_train_files + num_val_files]
    test_files = filenames[-num_test_files:]
elif dataset == 'full-speech-files':
    # the full speech-commands set lists which files are to be used
    # as test and validation data; train with everything else
    fname_val_files = os.path.join(data_dir, 'validation_list.txt')
    with open(fname_val_files) as fpi_val:
        val_files = fpi_val.read().splitlines()
    # validation_list.txt only lists partial paths
    val_files = [os.path.join(data_dir, fn) for fn in val_files]
    fname_test_files = os.path.join(data_dir, 'testing_list.txt')

    with open(fname_test_files) as fpi_tst:
        test_files = fpi_tst.read().splitlines()
    # testing_list.txt only lists partial paths
    test_files = [os.path.join(data_dir, fn).rstrip() for fn in test_files]

    # convert the TF tensor filenames into an array of strings so we can use basic python constructs
    train_files = [f.decode('utf8') for f in filenames.numpy()]

    # for f in train_files:
    #  f.split('\\')
    #  print(f)

    # don't train with the _background_noise_ files; exclude when directory name starts with '_'
    train_files = [f for f in train_files if f.split('\\')[-2][0] != '_']
    # validation and test files are listed explicitly in *_list.txt; train with everything else
    train_files = list(set(train_files) - set(test_files) - set(val_files))
    # now convert back into a TF tensor so we can use the tf.dataset pipeline
    train_files = tf.constant(train_files)
    print("full-speech-files is in progress.  Good luck!")
elif dataset == 'full-speech-ds':
    print("Using full-speech-ds. This is in progress.  Good luck!")
else:
    raise ValueError("dataset must be either full-speech-files, full-speech-ds or mini-speech")
print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)


# @tf.function
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    in_set = tf.reduce_any(parts[-2] == label_list)
    label = tf.cond(in_set, lambda: parts[-2], lambda: tf.constant(unknown_str))

    print(label)
    # print(f"parts[-2] = {parts[-2]}, in_set = {in_set}, label = {label.numpy()}")
    # print(f"parts[-2] = {parts[-2]}, in_set = {in_set}, label = {label}")
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return label  # parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  print("===========================================")
  print(label)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Concatenate audio with padding so that all audio clips will be of the
  # same length (16000 samples)
  zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.int16)
  waveform = tf.cast(0.5*waveform*(i16max-i16min), tf.int16)  # scale float [-1,+1]=>INT16
  equal_length = tf.concat([waveform, zero_padding], 0)
  ## Make sure these labels correspond to those used in micro_features_micro_features_generator.cpp
  spectrogram = frontend_op.audio_microfrontend(equal_length, sample_rate=fsamp, num_channels=num_filters,
                                    window_size=window_size_ms, window_step=window_step_ms)
  return spectrogram

def create_silence_dataset(num_waves, samples_per_wave, rms_noise_range=[0.01,0.2], silent_label=silence_str):
    # create num_waves waveforms of white gaussian noise, with rms level drawn from rms_noise_range
    # to act as the "silence" dataset
    rng = np.random.default_rng()
    rms_noise_levels = rng.uniform(low=rms_noise_range[0], high=rms_noise_range[1], size=num_waves)
    rand_waves = np.zeros((num_waves, samples_per_wave), dtype=np.float32) # pre-allocate memory
    for i in range(num_waves):
        rand_waves[i,:] = rms_noise_levels[i]*rng.standard_normal(samples_per_wave)
    labels = [silent_label]*num_waves
    return tf.data.Dataset.from_tensor_slices((rand_waves, labels))

def wavds2specds(waveform_ds, verbose=True):
  wav, label = next(waveform_ds.as_numpy_iterator())
  one_spec = get_spectrogram(wav)
  one_spec = tf.expand_dims(one_spec, axis=0)  # add a 'batch' dimension at the front
  one_spec = tf.expand_dims(one_spec, axis=-1) # add a singleton 'channel' dimension at the back

  num_waves = 0 # count the waveforms so we can allocate the memory
  for wav, label in waveform_ds:
    num_waves += 1
  print(f"About to create spectrograms from {num_waves} waves")
  spec_shape = (num_waves,) + one_spec.shape[1:]
  spec_grams = np.nan * np.zeros(spec_shape)  # allocate memory
  labels = np.nan * np.zeros(num_waves)
  idx = 0
  for wav, label in waveform_ds:
    if verbose and idx % 250 == 0:
      print(f"\r {idx} wavs processed", end='')
    spectrogram = get_spectrogram(wav)
    # TF conv layer expect inputs structured as 4D (batch_size, height, width, channels)
    # the microfrontend returns 2D tensors (freq, time), so we need to
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # add a 'batch' dimension at the front
    spectrogram = tf.expand_dims(spectrogram, axis=-1) # add a singleton 'channel' dimension at the back
    spec_grams[idx, ...] = spectrogram
    new_label = label.numpy().decode('utf8')
    new_label_id = np.argmax(new_label == np.array(label_list))
    labels[idx] = new_label_id # for numeric labels
    # labels.append(new_label) # for string labels
    idx += 1
  labels = np.array(labels, dtype=int)
  output_ds = tf.data.Dataset.from_tensor_slices((spec_grams, labels))
  return output_ds

AUTOTUNE = tf.data.experimental.AUTOTUNE
num_train_files = len(train_files)
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
train_ds = wavds2specds(waveform_ds)

def copy_with_noise(ds_input, rms_level=0.25):
  rng = tf.random.Generator.from_seed(1234)
  wave_shape = tf.constant((wave_length_samps,))
  def add_noise(waveform, label):
    noise = rms_level*rng.normal(shape=wave_shape)
    zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    noisy_wave = waveform + noise
    return noisy_wave, label

  return ds_input.map(add_noise)

# waveform_ds = augment_with_noise(waveform_ds)
count = 0
for w,l in waveform_ds:
  if w.shape != (16000,):
    print(f"element {count} has shape {w.shape}")
    break
  count += 1
print(count)

def pad_16000(waveform, label):
    zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    return waveform, label

def count_labels(dataset):
    counts = {}
    for _, lbl in dataset:
        if lbl.dtype == tf.string:
            label = lbl.numpy().decode('utf-8')
        else:
            label = lbl.numpy()
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts

# Collect what we did to generate the training dataset into a
# function, so we can repeat with the validation and test sets.
def preprocess_dataset(files, num_silent=None, noisy_reps_of_known=None):
  # if noisy_reps_of_known is not None, it should be a list of rms noise levels
  # For every target word in the data set, 1 copy will be created with each level
  # of noise added to it.  So [0.1, 0.2] will add 2x noisy copies of the target words
  if num_silent is None:
    num_silent = int(0.2*len(files))+1
  print(f"Processing {len(files)} files")
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  waveform_ds = files_ds.map(get_waveform_and_label)
  if noisy_reps_of_known is not None:
    # create a few copies of only the target words to balance the distribution
    # create a tmp dataset with only the target words
    ds_only_cmds = waveform_ds.filter(lambda w,l: tf.reduce_any(l == commands))
    for noise_level in noisy_reps_of_known:
       waveform_ds = waveform_ds.concatenate(copy_with_noise(ds_only_cmds, rms_level=noise_level))
  if num_silent > 0:
    silent_wave_ds = create_silence_dataset(num_silent, wave_length_samps,
                                            rms_noise_range=[0.01,0.2],
                                            silent_label=silence_str)
    waveform_ds = waveform_ds.concatenate(silent_wave_ds)
  print(f"Added {num_silent} silent wavs and ?? noisy wavs")
  num_waves = 0
  output_ds = wavds2specds(waveform_ds)
  return output_ds

print(f"We have {len(train_files)}/{len(val_files)}/{len(test_files)} training/validation/test files")

# print(train_files[:20])
print(label_list)
train_files[:20]
#print(train_files[:20])

tmp_ds = preprocess_dataset(train_files[:20])
print(count_labels(tmp_ds))

with tf.device('/GPU:0'): # needed on M1 mac
    tmp_ds = preprocess_dataset(train_files[:20], noisy_reps_of_known=[0.05,0.1])
    print(count_labels(tmp_ds))

# print(val_files[:20])
# print(train_files[:20])
val_ds = preprocess_dataset(val_files)
print(type(val_ds))
# print(next(iter(val_ds)))
# train_ds is already done
with tf.device('/GPU:0'):  # needed on M1 mac
    train_ds = preprocess_dataset(train_files, noisy_reps_of_known=[0.05, 0.1, 0.15, 0.2, 0.25])

# print(count_labels(val_files))
val_ds = preprocess_dataset(val_files)
d = tf.data.Dataset.from_tensor_slices(val_ds)
print(next(iter(d)).numpy())

## You can also use loops as follows to traverse the full set one item at a time
for elem in d:
    print(elem)
test_ds = preprocess_dataset(test_files)