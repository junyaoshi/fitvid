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
from fitvid.metrics import lpips
from fitvid.metrics import psnr
from fitvid.metrics import psnr_per_frame
from fitvid.metrics import ssim
import numpy as np
import tensorflow.compat.v2 as tf
from tqdm import trange


# limit GPU memory

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
#   # tf.config.experimental.set_virtual_device_configuration(
#   #   gpu,
#   #   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
#   # )
#   print('limited gpus!')


FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Path to model checkpoints/summaries.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('n_past', 2, 'Number of past frames.')
flags.DEFINE_integer('n_future', 10, 'Number of future frames.')
flags.DEFINE_integer('training_steps', 10000000, 'Number of training steps.')
flags.DEFINE_integer('log_every', 1000, 'How frequently log.')
flags.DEFINE_string('dataset', 'robonet', 'name of dataset.')


MODEL_CLS = models.FitVid


def additional_metrics(metrics, gt, out_video):
  metrics['metrics/psnr'] = psnr(gt, out_video)
  metrics['metrics/ssim'] = ssim(gt, out_video)
  metrics['metrics/fvd'] = fvd(gt, out_video)
  metrics['metrics/lpips'] = lpips(gt, out_video)
  return metrics


def write_summaries(summary_writer, metrics, step, vid_past, vid_out, gt):
  """"Writes TensorBoard summaries."""
  # Scalar summaries
  for key, val in metrics.items():
    tag = key
    if key == 'graphs/psnr':
      image = utils.plot_1d_signals([np.mean(val, axis=0)], [''])
      summary_writer.image(tag=tag, image=image, step=step)
    elif key.startswith('hist'):
      summary_writer.histogram(tag, val, step)
    else:
      summary_writer.scalar(tag, val, step)

  # GIFs
  video_summary = generate_video_summary(vid_past, vid_out, gt)
  utils.write_video_summaries(summary_writer, video_summary, 1, step)
  summary_writer.flush()


def generate_video_summary(vid_past, vid_out, gt):
  # concatenate past and future gt frames
  gt = np.concatenate([vid_past, gt], axis=1)

  # add borders and concatenat past and future output frames
  vid_past, vid_out = vid_past.copy(), vid_out.copy()
  vid_past[:, :, :, [0, -1]] = vid_past[:, :, [0, -1]] = np.array([1., 0., 0.])  # change border to red
  vid_out[:, :, :, [0, -1]] = vid_out[:, :, [0, -1]] = np.array([0., 1., 0.])  # change border to green
  vid_out = np.concatenate([vid_past, vid_out], axis=1)

  video_summary = np.concatenate([gt, vid_out], axis=3)
  return video_summary


@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=0)
def eval_step(model, batch, state, rng):
  """A single evaluation step."""
  variables = {'params': state.optimizer.target, **state.model_state}
  (_, out_video, metrics), _ = model.apply(
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
  out_video = out_video[:, n_past-1:]
  gt = batch['video'][:, n_past:]

  return gt, out_video, metrics


@functools.partial(
    jax.pmap, axis_name='batch',
    static_broadcasted_argnums=0, donate_argnums=(2,))
def train_step(model, batch, state, rng):
  """A single training step."""
  def loss(params):
    variables = {'params': params, **state.model_state}
    (loss, out_video, metrics), new_model_state = model.apply(
        variables,
        video=batch['video'],
        actions=batch['actions'],
        rngs=utils.generate_rng_dict(rng),
        step=state.step,
        mutable=['batch_stats'])
    return loss, (new_model_state, out_video, metrics)

  optimizer = state.optimizer
  grad_fn = jax.value_and_grad(loss, has_aux=True)
  aux, grads = grad_fn(optimizer.target)
  new_model_state, out_video, metrics = aux[1]
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
  out_video = out_video[:, n_past - 1:]

  return new_state, rng, metrics, out_video


def get_log_directories():
  output_dir = FLAGS.output_dir
  model_dir = os.path.join(output_dir, 'model')
  train_log_dir = os.path.join(output_dir, 'train')
  valid_log_dir = os.path.join(output_dir, 'valid')
  train_summary_writer = tensorboard.SummaryWriter(train_log_dir)
  valid_summary_writer = tensorboard.SummaryWriter(valid_log_dir)
  return model_dir, train_summary_writer, valid_summary_writer


def get_data():
  video_len = FLAGS.n_past + FLAGS.n_future
  local_batch_size = FLAGS.batch_size // jax.host_count()
  if FLAGS.dataset == 'robonet_sample':
    return data.load_dataset_robonet_sample(local_batch_size, video_len)
  elif FLAGS.dataset == 'robonet':
    raise NotImplementedError
  else:
    raise ValueError(f'Unrecognized dataset: {FLAGS.dataset}')



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
    gt, out_video, metrics = eval_step(model, batch, state, rng_key)

    def get_all(x):
      return utils.get_all_devices(jax_utils.unreplicate(x))
    out_video = get_all(out_video)
    gt = get_all(gt)
    metrics = jax.tree_map(get_all, metrics)
    metrics = additional_metrics(metrics, gt, out_video)
    all_metrics.append(metrics)

  if jax.host_id() == 0:
    metrics = {
        k: np.mean([dic[k] for dic in all_metrics]) for k in all_metrics[0]}
    metrics['graphs/psnr'] = psnr_per_frame(gt, out_video)
  return metrics, gt, out_video


def train():
  """Main training loop."""
  rng_key = jax.random.PRNGKey(0)
  training_steps = FLAGS.training_steps
  log_every = FLAGS.log_every

  model_dir, train_summary_writer, valid_summary_writer = get_log_directories()
  train_itr, valid_itr, test_itr = get_data()

  train_batch = next(train_itr)
  sample = utils.get_first_device(train_batch)

  # use this to see model progress on a single validation video
  single_valid_vid_idx = random.randint(0, FLAGS.batch_size)
  single_valid_batch = next(valid_itr)

  model = MODEL_CLS(n_past=FLAGS.n_past, training=True)

  state = init_model_state(rng_key, model, sample)
  state = checkpoints.restore_checkpoint(model_dir, state)
  start_step = int(state.step)
  state = jax_utils.replicate(state)

  rng_key = jax.random.split(rng_key, jax.local_device_count())
  t_loop_start = time.time()

  # variables for early stopping
  best_valid_loss = np.inf
  last_improvement = 0
  patience = 10

  for step in trange(start_step, training_steps):
    try:
      output = train_step(model, train_batch, state, rng_key)
      state, rng_key, metrics, out_video = output

      if step % log_every == 0:
        # process and log training info
        synced_state = utils.sync_batch_stats(state)
        steps_per_sec = log_every / (time.time() - t_loop_start)
        t_loop_start = time.time()
        if jax.host_id() == 0:
          train_metrics = utils.get_average_across_devices(metrics)
          state_ = jax_utils.unreplicate(synced_state)
          checkpoints.save_checkpoint(model_dir, state_, step, keep=patience)
          train_summary_writer.scalar('info/steps-per-second', steps_per_sec, step)
          out_video = utils.get_all_devices(out_video)
          gt = utils.get_all_devices(train_batch['video'])[:, FLAGS.n_past:]
          train_metrics = additional_metrics(train_metrics, gt, out_video)
          train_metrics['graphs/psnr'] = psnr_per_frame(gt, out_video)
          past_video = utils.get_all_devices(train_batch['video'][:, :, :FLAGS.n_past])
          write_summaries(train_summary_writer, train_metrics, step, past_video, out_video, gt)
          logging.info('>>> Step: %d Train Loss: %.4f', step, train_metrics['loss/all'])

        # run model on validation set
        valid_batch = next(valid_itr)
        valid_output = eval_step(model, valid_batch, state, rng_key)
        valid_gt, valid_out_video, valid_metrics = valid_output

        # run model on single validation video for progress tracking
        single_valid_output = eval_step(model, single_valid_batch, state, rng_key)
        single_valid_gt, single_valid_out, _ = single_valid_output

        # process and log validation info
        if jax.host_id() == 0:
          # validation set
          valid_metrics = utils.get_average_across_devices(valid_metrics)
          valid_out_video = utils.get_all_devices(valid_out_video)
          valid_gt = utils.get_all_devices(valid_gt)
          valid_metrics = additional_metrics(valid_metrics, valid_gt, valid_out_video)
          valid_metrics['graphs/psnr'] = psnr_per_frame(valid_gt, valid_out_video)
          valid_past_video = utils.get_all_devices(valid_batch['video'][:, :, :FLAGS.n_past])
          write_summaries(valid_summary_writer, valid_metrics, step, valid_past_video, valid_out_video, valid_gt)
          logging.info('>>> Step: %d Valid Loss: %.4f', step, valid_metrics['loss/all'])

          # single validation video
          single_valid_out = utils.get_all_devices(single_valid_out)
          single_valid_gt = utils.get_all_devices(single_valid_gt)
          single_valid_past = utils.get_all_devices(single_valid_batch['video'][:, :, :FLAGS.n_past])
          video_summary = generate_video_summary(single_valid_past, single_valid_out, single_valid_gt)
          utils.write_single_video_summaries(valid_summary_writer, video_summary, single_valid_vid_idx, step)
          valid_summary_writer.flush()

        # early stopping
        valid_loss = valid_metrics['loss/all']
        if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          last_improvement = 0
        else:
          last_improvement += 1
        if last_improvement > patience:
          logging.info('No improvement for the past %d steps. Stopping the training...', patience)
          break
        logging.info('>>> Step: %d Best Valid Loss: %.4f Last Improvement: %d',
                     patience, best_valid_loss, last_improvement)

      train_batch = next(train_itr)
    except KeyboardInterrupt:
      logging.info('Manual keyboard interrupt occured.')
      logging.info(f'Done training for %d steps', step)
      output = train_step(model, train_batch, state, rng_key)
      state, rng_key, metrics, out_video = output
      synced_state = utils.sync_batch_stats(state)
      steps_per_sec = log_every / (time.time() - t_loop_start)
      t_loop_start = time.time()
      if jax.host_id() == 0:
        train_metrics = utils.get_average_across_devices(metrics)
        state_ = jax_utils.unreplicate(synced_state)
        checkpoints.save_checkpoint(model_dir, state_, step, keep=patience)
        train_summary_writer.scalar('info/steps-per-second', steps_per_sec, step)
        out_video = utils.get_all_devices(out_video)
        gt = utils.get_all_devices(train_batch['video'])[:, FLAGS.n_past:]
        train_metrics = additional_metrics(train_metrics, gt, out_video)
        train_metrics['graphs/psnr'] = psnr_per_frame(gt, out_video)
        past_video = train_batch[:, :FLAGS.n_past]
        write_summaries(train_summary_writer, train_metrics, step, past_video, out_video, gt)
        logging.info('>>> Step: %d Train Loss: %.4f', step, train_metrics['loss/all'])
      logging.info('Saved model checkpoint. Ending process.')

def main(argv):
  del argv  # Unused
  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')
  train()
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
