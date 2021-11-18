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

"""Trainer binary."""

# Description:
#
# Training script for something-something dataset,
# no action condition,
# predict last frame from first frame


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import random

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import lax
import jax.numpy as jnp
from fitvid import data
from fitvid import models
from fitvid import utils
from fitvid.metrics import fvd
from fitvid.metrics import psnr
from fitvid.metrics import psnr_per_frame
from fitvid.metrics import ssim
from fitvid.metrics import calculate_fid, calculate_lpips
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

FLAGS = flags.FLAGS

# flags.DEFINE_string('output_dir', None, 'Path to model checkpoints/summaries.')
flags.DEFINE_string('datetime', None, 'YYYYMMDDHHMM')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('n_past', 2, 'Number of past frames.')
flags.DEFINE_integer('n_future', 10, 'Number of future frames.')
flags.DEFINE_integer('training_steps', 10000000, 'Number of training steps.')
flags.DEFINE_integer('log_every', 1000, 'How frequently log.')
flags.DEFINE_integer('sample_size', 100, 'Number of samples generated per input for metrics')
flags.DEFINE_integer('topk', 10, 'Top K samples will be logged to tensorboard')
flags.DEFINE_integer('fps', 5, 'FPS for gif logged to tensorboard')
flags.DEFINE_integer('log_n_per_batch', 10, 'log n out of batch_size of each batch')
flags.DEFINE_string('dataset', 'smth', 'smth or smth_dvd or smth_dvd_subgoal')
flags.DEFINE_string('aug', 'True', 'whether or not to augment the training dataset; True or False')
flags.DEFINE_float('beta', 1.0, 'beta term in the loss function')


MODEL_CLS = models.FitVid


def additional_metrics(metrics, model, state, batch,
                       gt, prior_out_video=None, post_out_video=None,
                       sample_size=100, topk=10):
  """
  Return:
    metrics, topk videos (topk psnr, ssim, lpips for prior and posterior)
  """
  # gt/out_video: (B, L, H, W, C)
  B, L, H, W, C = gt.shape
  topk_prior_psnr = tf.random.uniform(shape=[topk, B, L, H, W, C])
  topk_prior_ssim = tf.random.uniform(shape=[topk, B, L, H, W, C])
  topk_prior_lpips = tf.random.uniform(shape=[topk, B, L, H, W, C])
  topk_post_psnr = tf.random.uniform(shape=[topk, B, L, H, W, C])
  topk_post_ssim = tf.random.uniform(shape=[topk, B, L, H, W, C])
  topk_post_lpips = tf.random.uniform(shape=[topk, B, L, H, W, C])
  if sample_size == 1:
    if prior_out_video is not None:
      metrics['metrics/prior_psnr'], _ = psnr(gt, prior_out_video)
      metrics['metrics/prior_ssim'],  = ssim(gt, prior_out_video)
    if post_out_video is not None:
      metrics['metrics/post_psnr'], _ = psnr(gt, post_out_video)
      metrics['metrics/post_ssim'],  = ssim(gt, post_out_video)
  else:
    # sample 100 future trajectories per video
    # and pick the best one as the final score (acc. to paper)
    gt_samples, prior_out_samples, post_out_samples = [], [], []
    for _ in trange(sample_size, desc='Generating samples for metrics'):
      # generate random rng key for model sampling
      rng = jax.random.PRNGKey(random.randint(0, sample_size * 10))
      rng = jax.random.split(rng, jax.local_device_count())
      output = eval_step(model, batch, state, rng)
      gt_sample, prior_out, post_out, _ = output
      gt_samples.append(tf.squeeze(gt_sample, axis=0))
      prior_out_samples.append(tf.squeeze(prior_out, axis=0))
      post_out_samples.append(tf.squeeze(post_out, axis=0))
    gt_samples = tf.stack(gt_samples)  # (sample, B, L, H, W, C)
    prior_out_samples = tf.stack(prior_out_samples)  # (sample, B, L, H, W, C)
    post_out_samples = tf.stack(post_out_samples)  # (sample, B, L, H, W, C)

    if prior_out_video is not None:
      # feed this into the different metrics functions
      metrics['metrics/prior_psnr'], topk_prior_psnr_inds = psnr(gt_samples, prior_out_samples, sample_size, topk)
      topk_prior_psnr = tf.gather_nd(prior_out_samples, topk_prior_psnr_inds)  # (topk*bsize, L, H, W, C)
      _, L, H, W, C = topk_prior_psnr.shape
      B = gt_samples.shape[1]
      topk_prior_psnr = tf.reshape(topk_prior_psnr, [topk, B, L, H, W, C])  # (topk, bsize, L, H, W, C)

      metrics['metrics/prior_ssim'], topk_prior_ssim_inds = ssim(gt_samples, prior_out_samples, sample_size, topk)
      topk_prior_ssim = tf.gather_nd(prior_out_samples, topk_prior_ssim_inds)  # (topk*bsize, L, H, W, C)
      topk_prior_ssim = tf.reshape(topk_prior_ssim, [topk, B, L, H, W, C])  # (topk, bsize, L, H, W, C)

      metrics['metrics/prior_lpips'], topk_prior_lpips_inds = calculate_lpips(gt_samples, prior_out_samples,
                                                                              sample_size, topk)
      topk_prior_lpips = tf.gather_nd(prior_out_samples, topk_prior_lpips_inds)  # (topk*bsize, L, H, W, C)
      topk_prior_lpips = tf.reshape(topk_prior_lpips, [topk, B, L, H, W, C])  # (topk, bsize, L, H, W, C)

      # FVD needs video length >= 9, so repeating the image 9 times
      gt_repeat = tf.repeat(gt, 9, axis=1)
      prior_out_repeat = tf.repeat(prior_out_video, 9, axis=1)
      metrics['metrics/prior_fvd'] = fvd(gt_repeat, prior_out_repeat)
      metrics['metrics/prior_fid'] = calculate_fid(gt, prior_out_video)

    if post_out_video is not None:
      # feed this into the different metrics functions
      metrics['metrics/post_psnr'], topk_post_psnr_inds = psnr(gt_samples, post_out_samples, sample_size, topk)
      topk_post_psnr = tf.gather_nd(post_out_samples, topk_post_psnr_inds)  # (topk*bsize, L, H, W, C)
      _, L, H, W, C = topk_post_psnr.shape
      B = gt_samples.shape[1]
      topk_post_psnr = tf.reshape(topk_post_psnr, [topk, B, L, H, W, C])  # (topk, bsize, L, H, W, C)

      metrics['metrics/post_ssim'], topk_post_ssim_inds = ssim(gt_samples, post_out_samples, sample_size, topk)
      topk_post_ssim = tf.gather_nd(post_out_samples, topk_post_ssim_inds)  # (topk*bsize, L, H, W, C)
      topk_post_ssim = tf.reshape(topk_post_ssim, [topk, B, L, H, W, C])  # (topk, bsize, L, H, W, C)

      metrics['metrics/post_lpips'], topk_post_lpips_inds = calculate_lpips(gt_samples, post_out_samples,
                                                                              sample_size, topk)
      topk_post_lpips = tf.gather_nd(post_out_samples, topk_post_lpips_inds)  # (topk*bsize, L, H, W, C)
      topk_post_lpips = tf.reshape(topk_post_lpips, [topk, B, L, H, W, C])  # (topk, bsize, L, H, W, C)

      # FVD needs video length >= 9, so repeating the image 9 times
      gt_repeat = tf.repeat(gt, 9, axis=1)
      post_out_repeat = tf.repeat(post_out_video, 9, axis=1)
      metrics['metrics/post_fvd'] = fvd(gt_repeat, post_out_repeat)
      metrics['metrics/post_fid'] = calculate_fid(gt, post_out_video)

  topk_videos = {
    'prior': [topk_prior_psnr, topk_prior_ssim, topk_prior_lpips],
    'post': [topk_post_psnr, topk_post_ssim, topk_post_lpips]
  }
  return metrics, topk_videos


def write_summaries(summary_writer, metrics, step,
                    vid_past, vid_out, gt,
                    topk_psnr_videos, topk_ssim_videos, topk_lpips_videos,
                    prior=True):
  """"Writes TensorBoard summaries.
  Args:
    prior: True for prior, False for posterior
  """
  # Scalar summaries
  for key, val in metrics.items():
    tag = key
    if key == 'graphs/prior_psnr' or key == 'graphs/post_psnr':
      image = utils.plot_1d_signals([np.mean(val, axis=0)], [''])
      summary_writer.image(tag=tag, image=image, step=step)
    elif key.startswith('hist'):
      summary_writer.histogram(tag, val, step)
    else:
      summary_writer.scalar(tag, val, step)

  # GIFs and side-by-side
  video_summary, psnr_video_summary, ssim_video_summary, lpips_video_summary = generate_video_summary(
    vid_past, vid_out, gt, topk_psnr_videos, topk_ssim_videos, topk_lpips_videos
  )
  utils.write_video_summaries(
    summary_writer, video_summary, FLAGS.log_n_per_batch, step,
    tag_name='one', fps=FLAGS.fps, gif=FLAGS.n_future != 1, prior=prior
  )
  utils.write_video_summaries(
    summary_writer, psnr_video_summary, FLAGS.log_n_per_batch, step,
    tag_name='psnr', fps=FLAGS.fps, gif=FLAGS.n_future != 1, prior=prior
  )
  utils.write_video_summaries(
    summary_writer, ssim_video_summary, FLAGS.log_n_per_batch, step,
    tag_name='ssim', fps=FLAGS.fps, gif=FLAGS.n_future != 1, prior=prior
  )
  utils.write_video_summaries(
    summary_writer, lpips_video_summary, FLAGS.log_n_per_batch, step,
    tag_name='lpips', fps=FLAGS.fps, gif=FLAGS.n_future != 1, prior=prior
  )
  summary_writer.flush()


def generate_video_summary(vid_past, vid_out, gt, topk_psnr, topk_ssim, topk_lpips):
  # concatenate past and future gt frames
  gt_full = np.concatenate([vid_past, gt], axis=1)

  # add borders and concatenat past and future output frames
  vid_past, vid_out = vid_past.copy(), vid_out.copy()
  # vid: (B, L, W, H, C)
  vid_past[:, :, :, [0, -1]] = vid_past[:, :, [0, -1]] = np.array([1., 0., 0.])  # change border to red
  vid_out[:, :, :, [0, -1]] = vid_out[:, :, [0, -1]] = np.array([0., 1., 0.])  # change border to green
  vid_out_full = np.concatenate([vid_past, vid_out], axis=1)

  # handle psnr and ssim
  topk, B, L, W, H, C = topk_psnr.shape
  topk_psnr, topk_ssim, topk_lpips = topk_psnr.numpy(), topk_ssim.numpy(), topk_lpips.numpy()

  topk_psnr[:, :, :, :, [0, -1]] = topk_psnr[:, :, :, [0, -1]] = np.array([0., 0., 1.])  # change border to blue
  topk_psnr = tf.reshape(topk_psnr, [B, L * topk, W, H, C])
  topk_psnr = np.concatenate([vid_past, topk_psnr], axis=1)
  topk_ssim[:, :, :, :, [0, -1]] = topk_ssim[:, :, :, [0, -1]] = np.array([1., 1., 0.])  # change border to yellow
  topk_ssim = tf.reshape(topk_ssim, [B, L * topk, W, H, C])
  topk_ssim = np.concatenate([vid_past, topk_ssim], axis=1)
  topk_lpips[:, :, :, :, [0, -1]] = topk_lpips[:, :, :, [0, -1]] = np.array([1., 0., 1.])  # change border to purple
  topk_lpips = tf.reshape(topk_lpips, [B, L * topk, W, H, C])
  topk_lpips = np.concatenate([vid_past, topk_lpips], axis=1)

  gt_k = tf.repeat(gt, topk, axis=1)
  gt_k_full = np.concatenate([vid_past, gt_k], axis=1)

  video_summary = np.concatenate([gt_full, vid_out_full], axis=3)
  psnr_video_summary = np.concatenate([gt_k_full, topk_psnr], axis=3)
  ssim_video_summary = np.concatenate([gt_k_full, topk_ssim], axis=3)
  lpips_video_summary = np.concatenate([gt_k_full, topk_lpips], axis=3)
  return video_summary, psnr_video_summary, ssim_video_summary, lpips_video_summary


@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=0)
def eval_step(model, batch, state, rng):
  """A single evaluation step."""
  variables = {'params': state.optimizer.target, **state.model_state}
  (_, prior_out_video, post_out_video, metrics), _ = model.apply(
      variables,
      video=batch['video'],
      actions=batch['actions'],
      rngs=utils.generate_rng_dict(rng),
      step=state.step,
      mutable=['batch_stats'])
  n_past = FLAGS.n_past

  # # gather from all replicas (doesn't seem useful for GPU training)
  # out_video = jax.lax.all_gather(out_video[:, n_past-1:], axis_name='batch')
  # gt = jax.lax.all_gather(batch['video'][:, n_past:], axis_name='batch')
  # metrics = jax.lax.all_gather(metrics, axis_name='batch')

  # select last n_future frames
  prior_out_video = prior_out_video[:, n_past-1:]
  post_out_video = post_out_video[:, n_past - 1:]
  gt = batch['video'][:, n_past:]

  return gt, prior_out_video, post_out_video, metrics


@functools.partial(
    jax.pmap, axis_name='batch',
    static_broadcasted_argnums=0, donate_argnums=(2,))
def train_step(model, batch, state, rng):
  """A single training step."""
  def loss(params):
    variables = {'params': params, **state.model_state}
    (loss, prior_out_video, post_out_video, metrics), new_model_state = model.apply(
        variables,
        video=batch['video'],
        actions=batch['actions'],
        rngs=utils.generate_rng_dict(rng),
        step=state.step,
        mutable=['batch_stats'])
    return loss, (new_model_state, prior_out_video, post_out_video, metrics)

  optimizer = state.optimizer
  grad_fn = jax.value_and_grad(loss, has_aux=True)
  aux, grads = grad_fn(optimizer.target)
  new_model_state, prior_out_video, post_out_video, metrics = aux[1]
  grads = lax.pmean(grads, axis_name='batch')
  # metrics = jax.lax.pmean(metrics, axis_name='batch')
  grads_clipped = utils.clip_grads(grads, 100.0)
  new_optimizer = optimizer.apply_gradient(grads_clipped)
  # Apply update if the new optimizer state is all finite
  ok = jnp.all(jnp.asarray([
      jnp.all(jnp.isfinite(p)) for p in jax.tree_leaves(new_optimizer)]))
  new_state_with_update = state.replace(
      step=state.step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state)
  new_state_no_update = state.replace(
      step=state.step + 1)
  new_state = jax.tree_multimap(
      lambda a, b: jnp.where(ok, a, b),
      new_state_with_update, new_state_no_update)
  rng = jax.random.split(rng)[1]
  new_state = state.replace(
      step=state.step + 1, optimizer=new_optimizer, model_state=new_model_state)

  # select last n_future frames
  n_past = FLAGS.n_past
  prior_out_video = prior_out_video[:, n_past - 1:]
  post_out_video = post_out_video[:, n_past - 1:]
  gt = batch['video'][:, n_past:]

  return new_state, rng, metrics, gt, prior_out_video, post_out_video


def get_log_directories():
  output_dir = f'output/{FLAGS.dataset}_batch={FLAGS.batch_size}_steps={FLAGS.training_steps}' \
               f'_log={FLAGS.log_every}_sample={FLAGS.sample_size}' \
               f'_topk={FLAGS.topk}_fps={FLAGS.fps}_aug={FLAGS.aug}_beta={FLAGS.beta}' \
               f'_time={FLAGS.datetime}'
  model_dir = os.path.join(output_dir, 'model')
  train_log_dir = os.path.join(output_dir, 'train')
  valid_log_dir = os.path.join(output_dir, 'valid')
  fixed_log_dir = os.path.join(output_dir, 'fixed')
  train_summary_writer = tensorboard.SummaryWriter(train_log_dir)
  valid_summary_writer = tensorboard.SummaryWriter(valid_log_dir)
  fixed_summary_writer = tensorboard.SummaryWriter(fixed_log_dir)
  return model_dir, train_summary_writer, valid_summary_writer, fixed_summary_writer


def get_data(dataset, augment_train_data=True):
  video_len = FLAGS.n_past + FLAGS.n_future
  # local_batch_size = FLAGS.batch_size // jax.host_count()
  local_batch_size = FLAGS.batch_size // jax.process_count()
  if dataset == 'smth':
    return data.load_dataset_something_something(local_batch_size, video_len, augment_train_data)
  elif dataset == 'smth_dvd':
    return data.load_dataset_something_something_dvd(local_batch_size, augment_train_data)
  elif dataset == 'smth_dvd_subgoal':
    return data.load_dataset_something_something_dvd_subgoal(local_batch_size, augment_train_data)
  else:
    raise ValueError(f'Unrecognized dataset name: {dataset}')


def init_model_state(rng_key, model, sample):
  """Initialize the model state."""
  variables = model.init(
      rngs=utils.generate_rng_dict(rng_key),
      video=sample['video'],
      actions=sample['actions'],
      step=0)
  model_state, params = variables.pop('params')

  optimizer_def = optim.Adam(learning_rate=1e-3)
  optimizer = optimizer_def.create(params)
  utils.print_model_size(params, 'Model Size')
  return utils.TrainState(step=0, optimizer=optimizer, model_state=model_state)


def evaluate(rng_key, state, model, data_itr, eval_steps):
  """Evaluates the model on the entire dataset."""
  all_metrics = []
  for _ in range(eval_steps):
    batch = next(data_itr)
    gt, out_video, _, metrics = eval_step(model, batch, state, rng_key)

    def get_all(x):
      return utils.get_all_devices(jax_utils.unreplicate(x))

    metrics = utils.get_average_across_devices(metrics)
    out_video = utils.get_all_devices(out_video)
    gt = utils.get_all_devices(gt)
    metrics, *_ = additional_metrics(
      metrics, gt, out_video,
      model, state, batch,
      sample_size=FLAGS.sample_size, topk=FLAGS.topk
    )
    all_metrics.append(metrics)

  if jax.host_id() == 0:
    metrics = {
        k: np.mean([dic[k] for dic in all_metrics]) for k in all_metrics[0]}
    if FLAGS.n_future != 1:
      metrics['graphs/psnr'] = psnr_per_frame(gt, out_video)
  return metrics, gt, out_video


def train():
  """Main training loop."""
  training_steps = FLAGS.training_steps
  log_every = FLAGS.log_every
  n_past, n_future = FLAGS.n_past, FLAGS.n_future
  sample_size = FLAGS.sample_size
  topk = FLAGS.topk
  log_n_per_batch = FLAGS.log_n_per_batch
  batch_size = FLAGS.batch_size
  dataset = FLAGS.dataset
  augment_train_data = True if FLAGS.aug == 'True' else False
  beta = FLAGS.beta

  assert dataset == 'smth' or dataset == 'smth_dvd' or dataset == 'smth_dvd_subgoal'
  if dataset == 'smth' or dataset == 'smth_dvd':
    assert n_past == 1
  if dataset == 'smth_dvd_subgoal':
    assert n_past == 2
  assert n_future == 1
  assert sample_size != 0 and topk != 0 and log_n_per_batch != 0
  assert topk <= sample_size
  assert log_n_per_batch <= batch_size

  rng_key = jax.random.PRNGKey(0)
  model_dir, train_summary_writer, valid_summary_writer, fixed_summary_writer = get_log_directories()
  train_itr, valid_itr = get_data(dataset=dataset, augment_train_data=augment_train_data)

  train_batch = next(train_itr)
  sample = utils.get_first_device(train_batch)

  # use this to see model progress on a fixed batch of validation video
  fixed_batch = next(valid_itr)
  action_conditioned = True if dataset == 'smth_dvd' or dataset == 'smth_dvd_subgoal' else False
  model = MODEL_CLS(n_past=n_past, training=True, action_conditioned=action_conditioned, beta=beta)

  state = init_model_state(rng_key, model, sample)
  state = checkpoints.restore_checkpoint(model_dir, state)
  start_step = int(state.step)
  state = jax_utils.replicate(state)

  rng_key = jax.random.split(rng_key, jax.local_device_count())
  t_loop_start = time.time()

  # variables for early stopping
  best_valid_loss = np.inf
  last_improvement = 0
  patience = np.inf

  for step in trange(start_step, training_steps):
    try:
      output = train_step(model, train_batch, state, rng_key)
      state, rng_key, metrics, train_gt, train_prior_out, train_post_out = output

      if step % log_every == 0:
        # process and log training info
        synced_state = utils.sync_batch_stats(state)
        steps_per_sec = log_every / (time.time() - t_loop_start)
        t_loop_start = time.time()
        if jax.host_id() == 0:
          train_metrics = utils.get_average_across_devices(metrics)
          state_ = jax_utils.unreplicate(synced_state)
          checkpoints.save_checkpoint(model_dir, state_, step, keep=5, keep_every_n_steps=10000)
          train_summary_writer.scalar('info/steps-per-second', steps_per_sec, step)
          train_prior_out = utils.get_all_devices(train_prior_out)
          train_post_out = utils.get_all_devices(train_post_out)
          train_gt = utils.get_all_devices(train_gt)
          train_metrics, train_topk_videos = additional_metrics(
            train_metrics, model, state, train_batch,
            train_gt, train_prior_out, train_post_out,
            sample_size=sample_size, topk=topk
          )
          train_topk_prior_psnr, train_topk_prior_ssim, train_topk_prior_lpips = train_topk_videos['prior']
          train_topk_post_psnr, train_topk_post_ssim, train_topk_post_lpips = train_topk_videos['post']
          if n_future != 1:
            train_metrics['graphs/prior_psnr'] = psnr_per_frame(train_gt, train_prior_out)
            train_metrics['graphs/post_psnr'] = psnr_per_frame(train_gt, train_post_out)
          train_past_video = utils.get_all_devices(train_batch['video'][:, :, :n_past])
          write_summaries(
            train_summary_writer, train_metrics, step,
            train_past_video, train_prior_out, train_gt,
            train_topk_prior_psnr, train_topk_prior_ssim, train_topk_prior_lpips,
            prior=True
          )
          write_summaries(
            train_summary_writer, train_metrics, step,
            train_past_video, train_post_out, train_gt,
            train_topk_post_psnr, train_topk_post_ssim, train_topk_post_lpips,
            prior=False
          )
          logging.info(f">>> Step: {step} "
                       f"Train Prior Loss: {train_metrics['loss/prior_all']:.4f}"
                       f"Train Posterior Loss: {train_metrics['loss/post_all']:.4f}")
          del train_gt
          del train_prior_out
          del train_post_out
          del train_past_video
          del train_topk_prior_psnr
          del train_topk_prior_ssim
          del train_topk_prior_lpips
          del train_topk_post_psnr
          del train_topk_post_ssim
          del train_topk_post_lpips

        # run model on validation set
        valid_batch = next(valid_itr)
        valid_output = eval_step(model, valid_batch, state, rng_key)
        valid_gt, valid_out_video, _, valid_metrics = valid_output

        # run model on fixed validation video for progress tracking
        fixed_output = eval_step(model, fixed_batch, state, rng_key)
        fixed_gt, fixed_out, _, fixed_metrics = fixed_output

        # process and log validation info
        if jax.host_id() == 0:
          # validation set
          valid_metrics = utils.get_average_across_devices(valid_metrics)
          valid_out_video = utils.get_all_devices(valid_out_video)
          valid_gt = utils.get_all_devices(valid_gt)
          valid_metrics, valid_topk_videos = additional_metrics(
            valid_metrics, model, state, valid_batch,
            valid_gt, valid_out_video, post_out_video=None,
            sample_size=sample_size, topk=topk
          )
          valid_topk_prior_psnr, valid_topk_prior_ssim, valid_topk_prior_lpips = valid_topk_videos['prior']
          if n_future != 1:
            valid_metrics['graphs/prior_psnr'] = psnr_per_frame(valid_gt, valid_out_video)
          valid_past_video = utils.get_all_devices(valid_batch['video'][:, :, :n_past])
          write_summaries(
            valid_summary_writer, valid_metrics, step,
            valid_past_video, valid_out_video, valid_gt,
            valid_topk_prior_psnr, valid_topk_prior_ssim, valid_topk_prior_lpips,
            prior=True
          )
          logging.info(f">>> Step: {step} "
                       f"Valid Prior Loss: {valid_metrics['loss/prior_all']:.4f}")
          del valid_gt
          del valid_out_video
          del valid_past_video
          del valid_topk_prior_psnr
          del valid_topk_prior_ssim
          del valid_topk_prior_lpips

          # fixed validation video
          fixed_metrics = utils.get_average_across_devices(fixed_metrics)
          fixed_out = utils.get_all_devices(fixed_out)
          fixed_gt = utils.get_all_devices(fixed_gt)

          fixed_metrics, fixed_topk_videos = additional_metrics(
            fixed_metrics, model, state, valid_batch,
            fixed_gt, fixed_out, post_out_video=None,
            sample_size=sample_size, topk=topk
          )
          fixed_topk_prior_psnr, fixed_topk_prior_ssim, fixed_topk_prior_lpips = fixed_topk_videos['prior']
          if n_future != 1:
            fixed_metrics['graphs/prior_psnr'] = psnr_per_frame(fixed_gt, fixed_out)
          fixed_past_video = utils.get_all_devices(fixed_batch['video'][:, :, :n_past])
          write_summaries(
            fixed_summary_writer, fixed_metrics, step,
            fixed_past_video, fixed_out, fixed_gt,
            fixed_topk_prior_psnr, fixed_topk_prior_ssim, fixed_topk_prior_lpips,
            prior=True
          )
          logging.info(f">>> Step: {step} "
                       f"Fixed Prior Loss: {fixed_metrics['loss/prior_all']:.4f}")
          del fixed_gt
          del fixed_out
          del fixed_past_video
          del fixed_topk_prior_psnr
          del fixed_topk_prior_ssim
          del fixed_topk_prior_lpips

        # early stopping
        valid_loss = valid_metrics['loss/prior_all']
        if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          last_improvement = 0
        else:
          last_improvement += 1
        if last_improvement > patience:
          logging.info('No improvement for the past %d steps. Stopping the training...', patience)
          break
        logging.info('>>> Step: %d Best Valid Loss: %.4f Last Improvement: %d',
                     step, best_valid_loss, last_improvement)

      train_batch = next(train_itr)
    except KeyboardInterrupt:
      logging.info('Manual keyboard interrupt occured.')
      logging.info(f'Done training for %d steps', step)
      output = train_step(model, train_batch, state, rng_key)
      state, rng_key, metrics, train_gt, train_prior_out, train_post_out = output
      synced_state = utils.sync_batch_stats(state)
      steps_per_sec = log_every / (time.time() - t_loop_start)
      t_loop_start = time.time()
      if jax.host_id() == 0:
        train_metrics = utils.get_average_across_devices(metrics)
        state_ = jax_utils.unreplicate(synced_state)
        checkpoints.save_checkpoint(model_dir, state_, step, keep=5, keep_every_n_steps=10000)
        train_summary_writer.scalar('info/steps-per-second', steps_per_sec, step)
        train_prior_out = utils.get_all_devices(train_prior_out)
        train_post_out = utils.get_all_devices(train_post_out)
        train_gt = utils.get_all_devices(train_gt)
        train_metrics, train_topk_videos = additional_metrics(
          train_metrics, model, state, train_batch,
          train_gt, train_prior_out, train_post_out,
          sample_size=sample_size, topk=topk
        )
        train_topk_prior_psnr, train_topk_prior_ssim, train_topk_prior_lpips = train_topk_videos['prior']
        train_topk_post_psnr, train_topk_post_ssim, train_topk_post_lpips = train_topk_videos['post']
        if n_future != 1:
          train_metrics['graphs/prior_psnr'] = psnr_per_frame(train_gt, train_prior_out)
          train_metrics['graphs/post_psnr'] = psnr_per_frame(train_gt, train_post_out)
        past_video = utils.get_all_devices(train_batch['video'][:, :, :n_past])
        write_summaries(
          train_summary_writer, train_metrics, step,
          past_video, train_prior_out, train_gt,
          train_topk_prior_psnr, train_topk_prior_ssim, train_topk_prior_lpips,
          prior=True
        )
        write_summaries(
          train_summary_writer, train_metrics, step,
          past_video, train_post_out, train_gt,
          train_topk_post_psnr, train_topk_post_ssim, train_topk_post_lpips,
          prior=False
        )
        logging.info(f">>> Step: {step} "
                     f"Train Prior Loss: {train_metrics['loss/prior_all']:.4f}"
                     f"Train Posterior Loss: {train_metrics['loss/post_all']:.4f}")
      logging.info('Saved model checkpoint. Ending process.')

def main(argv):
  del argv  # Unused
  # tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')
  train()
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == '__main__':
  flags.mark_flags_as_required(['datetime'])
  app.run(main)
