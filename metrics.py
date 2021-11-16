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

"""Metrics."""

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import fid
import lpips
import torch
from torchvision import transforms


i3d_model = None
lpips_model = None


def flatten_video(video):
  return np.reshape(video, (-1,) + video.shape[2:])


def psnr(video_1, video_2, sample_size=100, topk=10):
  """
  Returns:
    Single sample mode:
      PSNR, None
    Else:
      the max PSNR among samples, indices of samples corresponding to top k PSNR
  """
  if sample_size == 1:
    video_1 = flatten_video(video_1)
    video_2 = flatten_video(video_2)
    dist = tf.image.psnr(video_1, video_2, max_val=1.0)
    return tf.reduce_mean(dist), None
  else:
    dist = tf.squeeze(tf.image.psnr(video_1, video_2, max_val=1.0))
    # topk_inds: (bsize, topk)
    topk_inds = tf.math.top_k(tf.transpose(dist, [1, 0]), topk).indices
    # vectors: (topk*bsize)
    batch_size = video_1.shape[1]
    batch_vector = tf.repeat(tf.range(batch_size), [topk])
    topk_vector = tf.reshape(topk_inds, [-1])
    # topk_inds: (topk*bsize, 2)
    topk_inds = tf.stack([topk_vector, batch_vector], axis=1)
    return tf.reduce_mean(tf.reduce_max(dist, axis=0)).numpy(), topk_inds


def ssim(video_1, video_2, sample_size=100, topk=10):
  if sample_size == 1:
    video_1 = flatten_video(video_1)
    video_2 = flatten_video(video_2)
    dist = tf.image.ssim(video_1, video_2, max_val=1.0)
    return np.mean(dist.numpy())
  else:
    dist = tf.squeeze(tf.image.ssim(video_1, video_2, max_val=1.0))
    # topk_inds: (bsize, topk)
    topk_inds = tf.math.top_k(tf.transpose(dist, [1, 0]), topk).indices
    # vectors: (topk*bsize)
    batch_size = video_1.shape[1]
    batch_vector = tf.repeat(tf.range(batch_size), [topk])
    topk_vector = tf.reshape(topk_inds, [-1])
    # topk_inds: (topk*bsize, 2)
    topk_inds = tf.stack([topk_vector, batch_vector], axis=1)
    return tf.reduce_mean(tf.reduce_max(dist, axis=0)).numpy(), topk_inds


def psnr_image(target_image, out_image):
  dist = tf.image.psnr(target_image, out_image, max_val=1.0)
  return np.mean(dist.numpy())


def psnr_per_frame(target_video, out_video):
  max_val = 1.0
  mse = np.mean(np.square(out_video - target_video), axis=(2, 3, 4))
  return 20 * np.log10(max_val) - 10.0 * np.log10(mse)


def lpips_image(generated_image, real_image):
  global lpips_model
  result = tf.convert_to_tensor(0.0)
  return result


# def lpips(video_1, video_2):
#   video_1 = flatten_video(video_1)
#   video_2 = flatten_video(video_2)
#   dist = lpips_image(video_1, video_2)
#   return np.mean(dist.numpy())


def fvd_preprocess(videos, target_resolution):
  videos = tf.convert_to_tensor(videos * 255.0, dtype=tf.float32)
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def create_id3_embedding(videos):
  """Get id3 embeddings."""
  global i3d_model
  module_spec = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'

  if not i3d_model:
    base_model = hub.load(module_spec)
    input_tensor = base_model.graph.get_tensor_by_name('input_frames:0')
    i3d_model = base_model.prune(input_tensor, 'RGB/inception_i3d/Mean:0')

  output = i3d_model(videos)
  return output


def calculate_fvd(real_activations, generated_activations):
  return tfgan.eval.frechet_classifier_distance_from_activations(
      real_activations, generated_activations)


def fvd(video_1, video_2):
  video_1 = fvd_preprocess(video_1, (224, 224))
  video_2 = fvd_preprocess(video_2, (224, 224))
  x = create_id3_embedding(video_1)
  y = create_id3_embedding(video_2)
  result = calculate_fvd(x, y)
  return result.numpy()


def inception_score(images):
  return tfgan.eval.inception_score(images)


def calculate_fid(image_1, image_2):
  """
  image_1, image_2: (B, L, H, W, C)
  """
  # convert to 0 to 255
  image_1 = tf.cast(image_1 * 255, dtype=tf.uint8).numpy()
  image_2 = tf.cast(image_2 * 255, dtype=tf.uint8).numpy()

  # change shape to [N, 3, HEIGHT, WIDTH]
  image_1, image_2 = image_1[:, 0], image_2[:, 0]
  N, H, W, C = image_1.shape
  image_1, image_2 = np.reshape(image_1, [N, C, H, W]), np.reshape(image_2, [N, C, H, W])

  return fid.get_fid(image_1, image_2).numpy()


def calculate_lpips(video_1, video_2, sample_size=100, topk=10):
  """
  video_1, video_2: (sample, B, L, H, W, C)
  """
  video_1, video_2 = video_1[:, :, 0], video_2[:, :, 0]  # (sample, B, H, W, C)
  S, B, H, W, C = video_1.shape
  # reshape to N, 3, H, W
  video_1, video_2 = tf.reshape(video_1, [S * B, C, H, W]), tf.reshape(video_2, [S * B, C, H, W])
  video_1, video_2 = torch.from_numpy(video_1.numpy()), torch.from_numpy(video_2.numpy())
  # normalize to (-1, 1)
  video_1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(video_1)
  video_2 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(video_2)
  loss_fn_alex = lpips.LPIPS(net='alex')
  if sample_size == 1:
    raise NotImplementedError
  else:
    dist = loss_fn_alex(video_1, video_2).squeeze().detach().numpy()
    dist = tf.convert_to_tensor(dist)
    dist = tf.reshape(dist, [sample_size, B])
    # topk_inds: (bsize, topk)
    topk_inds = tf.math.top_k(tf.transpose(dist, [1, 0]), topk).indices
    # vectors: (topk*bsize)
    batch_vector = tf.repeat(tf.range(B), [topk])
    topk_vector = tf.reshape(topk_inds, [-1])
    # topk_inds: (topk*bsize, 2)
    topk_inds = tf.stack([topk_vector, batch_vector], axis=1)
    return tf.reduce_mean(tf.reduce_max(dist, axis=0)).numpy(), topk_inds


if __name__ == '__main__':
  gt = tf.random.uniform((8, 1, 64, 64, 3))
  out = tf.random.uniform((8, 1, 64, 64, 3))
  gt_repeat = tf.repeat(gt, 9, axis=1)
  out_repeat = tf.repeat(out, 9, axis=1)
  gt_samples = tf.random.uniform((5, 8, 1, 64, 64, 3))
  out_samples = tf.random.uniform((5, 8, 1, 64, 64, 3))
  lpips_score, _ = calculate_lpips(gt_samples, out_samples, sample_size=5, topk=3)
  fvd_score = fvd(gt_repeat, out_repeat)
  fid_score = calculate_fid(gt, out)
  print(f'fvd score: {fvd_score} | fid score: {fid_score} | lpips score: {lpips_score}')
