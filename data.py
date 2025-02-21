# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data."""

import functools
import random

from fitvid.randaug import randaugment
from flax import jax_utils
import jax
import numpy as np
import tensorflow as tf  # tf
import tensorflow_datasets as tfds
import something_something  # Register something_something
import something_something_DVD  # Register something_something_DVD
import something_something_DVD_subgoal  # Register something_something_DVD_subgoal


def rand_crop(seeds, video, width, height, wiggle):
  """Random crop of a video. Assuming height < width."""
  x_wiggle = wiggle
  crop_width = height - wiggle
  y_wiggle = width - crop_width
  xx = tf.random.stateless_uniform(
      [], seed=seeds[0], minval=0, maxval=x_wiggle, dtype=tf.int32)
  yy = tf.random.stateless_uniform(
      [], seed=seeds[1], minval=0, maxval=y_wiggle, dtype=tf.int32)
  return video[:, xx:xx+crop_width, yy:yy+crop_width, :]


def rand_crop_fitvid(seeds, video, width, height, min_height_ratio):
  """Random crop of a video. As described in the fitvid paper
  Args:
    width: width of video before cropping
    height: height of video before cropping
    min_height_ratio: Crop height minimum ratio C in the paper
  """
  min_height = int(min_height_ratio * height)
  crop_width = tf.random.stateless_uniform(
    [], seed=seeds[2], minval=min_height, maxval=height, dtype=tf.int32
  )
  xx_max = height - crop_width
  yy_max = width - crop_width
  xx = tf.random.stateless_uniform(
      [], seed=seeds[0], minval=0, maxval=xx_max, dtype=tf.int32
  )
  yy = tf.random.stateless_uniform(
      [], seed=seeds[1], minval=0, maxval=yy_max, dtype=tf.int32
  )
  return video[:, xx:xx+crop_width, yy:yy+crop_width, :]


def rand_aug(seeds, video, num_layers, magnitude):
  """RandAug for video with the same random seed for all frames."""
  image_aug = lambda a, x: randaugment(x, num_layers, magnitude, seeds)
  return tf.scan(image_aug, video)


def augment_dataset(dataset, augmentations):
  """Augment dataset with a list of augmentations."""
  def augment(seeds, features):
    video = tf.cast(features['video'], tf.uint8)
    for aug_fn in augmentations:
      video = aug_fn(seeds, video)
    video = tf.image.resize(video, (64, 64))
    features['video'] = video
    return features

  randds = tf.data.experimental.RandomDataset(1).batch(2).batch(4)
  # randds = tf.data.Dataset.random(1).batch(2).batch(4)
  dataset = tf.data.Dataset.zip((randds, dataset))
  dataset = dataset.map(
      augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def normalize_video(features):
  features['video'] = tf.cast(features['video'], tf.float32) / 255.0
  return features


def get_iterator(dataset, batch_size, is_train):
  """"Returns a performance optimized iterator from dataset."""
  local_device_count = jax.local_device_count()
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  # options.threading.private_threadpool_size = 48
  # options.threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)
  dataset = dataset.map(normalize_video)
  dataset = dataset.repeat()
  if is_train:
    dataset = dataset.shuffle(batch_size * 64, seed=0)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(32)

  def prepare_tf_data(xs):
    def _prepare(x):
      x = x._numpy()
      return x.reshape((local_device_count, -1) + x.shape[1:])
    return jax.tree_map(_prepare, xs)

  iterator = map(prepare_tf_data, dataset)
  iterator = jax_utils.prefetch_to_device(iterator, 2)
  return iterator




def load_dataset_robonet(batch_size, video_len, is_train):
  """"Load RoboNet dataset."""

  def extract_features_robonet(features):
    dtype = tf.float32
    video = tf.cast(features['video'], dtype)
    actions = tf.cast(features['actions'], dtype)
    video /= 255.0
    return {
        'video': tf.identity(video[:video_len]),
        'actions': tf.identity(actions[:video_len-1]),
    }

  def robonet_filter_by_filename(features, filenames, white=True):
    in_list = tf.reduce_any(tf.math.equal(features['filename'], filenames))
    return in_list if white else tf.math.logical_not(in_list)

  def get_robonet_test_filenames():
    testfiles = None
    if testfiles is None:
      with tf.io.gfile.GFile('robonet_testset_filenames.txt', 'r') as f:
        testfiles = f.read()
    testfiles = ([x.encode('ascii') for x in testfiles.split('\n') if x])
    return testfiles

  dataset_builder = tfds.builder('robonet/robonet_64')
  dataset_builder.download_and_prepare()
  num_examples = dataset_builder.info.splits['train'].num_examples
  split_size = num_examples // jax.host_count()
  start = jax.host_id() * split_size
  split = 'train[{}:{}]'.format(start, start + split_size)
  dataset = dataset_builder.as_dataset(split=split)
  options = tf.data.Options()
  # options.experimental_threading.private_threadpool_size = 48
  # options.experimental_threading.max_intra_op_parallelism = 1
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)

  test_filenames = get_robonet_test_filenames()
  train_filter = functools.partial(
      robonet_filter_by_filename, filenames=test_filenames, white=not is_train)

  dataset = dataset.filter(train_filter)
  dataset = dataset.map(
      extract_features_robonet,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return get_iterator(dataset, batch_size, is_train)

def load_dataset_robonet_sample(batch_size, video_len):
  """Load RoboNet Sample dataset."""

  def extract_features_robonet(features):
    dtype = tf.float32
    video = tf.cast(features['video'], dtype)
    actions = tf.cast(features['actions'], dtype)
    return {
        'video': tf.identity(video[:video_len]),
        'actions': tf.identity(actions[:video_len-1]),
    }

  dataset_builder = tfds.builder('robonet/robonet_sample_64')
  dataset_builder.download_and_prepare()
  # num_examples = dataset_builder.info.splits['train'].num_examples

  # calculate split percentage and generate dataset
  valid_start = random.randint(0, 80)
  valid_end = valid_start + 10
  valid_split = f'train[{valid_start}%:{valid_end}%]'
  train_split = f'train[0%:{valid_start}%]+train[{valid_end}%:90%]'

  train_dataset = dataset_builder.as_dataset(split=train_split)
  valid_dataset = dataset_builder.as_dataset(split=valid_split)
  test_dataset  = dataset_builder.as_dataset(split='train[90%:]')

  # set options and extract features
  options = tf.data.Options()
  # options.experimental_threading.private_threadpool_size = 48
  # options.experimental_threading.max_intra_op_parallelism = 1
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  train_dataset = train_dataset.with_options(options)
  train_dataset = train_dataset.map(
    extract_features_robonet,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  valid_dataset = valid_dataset.with_options(options)
  valid_dataset = valid_dataset.map(
    extract_features_robonet,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  test_dataset = test_dataset.with_options(options)
  test_dataset = test_dataset.map(
    extract_features_robonet,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  itrs = [
    get_iterator(train_dataset, batch_size, True),
    get_iterator(valid_dataset, batch_size, False),
    get_iterator(test_dataset, batch_size, False)
  ]
  return itrs


def load_dataset_something_something(batch_size, video_len, augment_train_data=True):
  """Load something_something dataset."""

  def extract_features_smth_smth(features):
    dtype = tf.float32
    first_frame = features['first_frame']
    last_frame = features['last_frame']
    video = tf.stack([first_frame, last_frame])
    actions = np.zeros((video_len - 1, 5))  # action not used, just need to match dimension
    video = tf.cast(video, dtype)
    actions = tf.cast(actions, dtype)
    return {
        'video': tf.identity(video),
        'actions': tf.identity(actions),
    }

  dataset_builder = tfds.builder('something_something')
  dataset_builder.download_and_prepare()
  train_dataset = dataset_builder.as_dataset(split='train')
  valid_dataset = dataset_builder.as_dataset(split='valid')

  # set options and extract features
  options = tf.data.Options()
  # options.experimental_threading.private_threadpool_size = 48
  # options.experimental_threading.max_intra_op_parallelism = 1
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  train_dataset = train_dataset.with_options(options)
  train_dataset = train_dataset.map(
    extract_features_smth_smth,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if augment_train_data:
    rand_crop_func = functools.partial(
      rand_crop_fitvid, width=64, height=64, min_height_ratio=0.8
    )
    rand_aug_func = functools.partial(
      rand_aug, num_layers=1, magnitude=5
    )
    train_dataset = augment_dataset(train_dataset, [rand_crop_func, rand_aug_func])

  valid_dataset = valid_dataset.with_options(options)
  valid_dataset = valid_dataset.map(
    extract_features_smth_smth,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  itrs = [
    get_iterator(train_dataset, batch_size, True),
    get_iterator(valid_dataset, batch_size, True),
  ]
  return itrs


def load_dataset_something_something_dvd(batch_size, augment_train_data=True):
  """Load something_something dataset."""

  def extract_features_smth_smth(features):
    dtype = tf.float32
    first_frame = features['first_frame']
    label = features['label']
    last_frame = features['last_frame']
    video = tf.stack([first_frame, last_frame])
    actions = tf.expand_dims(label, axis=0)
    video = tf.cast(video, dtype)
    actions = tf.cast(actions, dtype)
    return {
        'video': tf.identity(video),
        'actions': tf.identity(actions),
    }

  dataset_builder = tfds.builder('something_something_DVD')
  dataset_builder.download_and_prepare()
  train_dataset = dataset_builder.as_dataset(split='train')
  valid_dataset = dataset_builder.as_dataset(split='valid')

  # set options and extract features
  options = tf.data.Options()
  # options.experimental_threading.private_threadpool_size = 48
  # options.experimental_threading.max_intra_op_parallelism = 1
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  train_dataset = train_dataset.with_options(options)
  train_dataset = train_dataset.map(
    extract_features_smth_smth,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if augment_train_data:
    rand_crop_func = functools.partial(
      rand_crop_fitvid, width=64, height=64, min_height_ratio=0.8
    )
    rand_aug_func = functools.partial(
      rand_aug, num_layers=1, magnitude=5
    )
    train_dataset = augment_dataset(train_dataset, [rand_crop_func, rand_aug_func])

  valid_dataset = valid_dataset.with_options(options)
  valid_dataset = valid_dataset.map(
    extract_features_smth_smth,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  itrs = [
    get_iterator(train_dataset, batch_size, True),
    get_iterator(valid_dataset, batch_size, True),
  ]
  return itrs


def load_dataset_something_something_dvd_subgoal(batch_size, augment_train_data=True):
  """Load something_something dataset."""

  def extract_features_smth_smth(features):
    dtype = tf.float32
    current_frame = features['current_frame']
    goal_frame = features['goal_frame']
    label = features['label']
    subgoal_frame = features['subgoal_frame']
    video = tf.stack([current_frame, goal_frame, subgoal_frame])
    video_len = video.shape[0]
    actions = tf.repeat(tf.expand_dims(label, axis=0), video_len, axis=0)
    video = tf.cast(video, dtype)
    actions = tf.cast(actions, dtype)
    return {
        'video': tf.identity(video),
        'actions': tf.identity(actions),
    }

  dataset_builder = tfds.builder('something_something_DVD_subgoal')
  dataset_builder.download_and_prepare()
  train_dataset = dataset_builder.as_dataset(split='train')
  valid_dataset = dataset_builder.as_dataset(split='valid')

  # set options and extract features
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  # options.threading.private_threadpool_size = 48
  # options.threading.max_intra_op_parallelism = 1

  train_dataset = train_dataset.with_options(options)
  train_dataset = train_dataset.map(
    extract_features_smth_smth,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if augment_train_data:
    rand_crop_func = functools.partial(
      rand_crop_fitvid, width=64, height=64, min_height_ratio=0.8
    )
    rand_aug_func = functools.partial(
      rand_aug, num_layers=1, magnitude=5
    )
    train_dataset = augment_dataset(train_dataset, [rand_crop_func, rand_aug_func])

  valid_dataset = valid_dataset.with_options(options)
  valid_dataset = valid_dataset.map(
    extract_features_smth_smth,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  itrs = [
    get_iterator(train_dataset, batch_size, True),
    get_iterator(valid_dataset, batch_size, True),
  ]
  return itrs


if __name__ == '__main__':
  # test random crop
  video = tf.cast(
    tf.random.uniform(shape=(2, 64, 64, 3), minval=0, maxval=255, dtype=tf.int32),
    dtype=tf.uint8
  )
  seeds = [[random.randint(0, 100), random.randint(0, 100)] for _ in range(4)]
  rand_crop_fitvid_func = functools.partial(
    rand_crop_fitvid, width=64, height=64, min_height_ratio=0.8
  )
  video_cropped = rand_crop_fitvid_func(seeds, video)
  print(f'Cropped video shape: {video_cropped.shape}')

  # test resize
  video_resized = tf.image.resize(video_cropped, (64, 64))
  print(f'Resized video shape: {video_resized.shape}')

  # test random augment
  rand_aug_func = functools.partial(
    rand_aug, num_layers=1, magnitude=5
  )
  video_augmented = rand_aug_func(seeds, video)
  print(f'Cropped video shape: {video_augmented.shape}')

  # test resize
  video_resized = tf.image.resize(video_augmented, (64, 64))
  print(f'Resized video shape: {video_resized.shape}')
