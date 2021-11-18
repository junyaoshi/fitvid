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

"""Models."""

# pytype: skip-file

import functools

from flax import linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp

from fitvid import nvae
from fitvid import utils


class MultiGaussianLSTM(nn.Module):
  """Multi layer lstm with Gaussian output."""
  num_layers: int = 2
  hidden_size: int = 10
  output_size: int = 10
  dtype: int = jnp.float32

  def setup(self):
    self.embed = nn.Dense(self.hidden_size)
    self.mean = nn.Dense(self.output_size)
    self.logvar = nn.Dense(self.output_size)
    self.layers = [nn.recurrent.LSTMCell() for _ in range(self.num_layers)]

  def init_states(self, batch_size):
    init_fn = functools.partial(initializers.zeros, dtype=self.dtype)
    states = [None] * self.num_layers
    for i in range(self.num_layers):
      states[i] = nn.recurrent.LSTMCell.initialize_carry(
          self.make_rng('rng'),
          (batch_size,),
          self.hidden_size,
          init_fn=init_fn)
    return states

  def reparameterize(self, mu, logvar):
    var = jnp.exp(0.5 * logvar)
    epsilon = jax.random.normal(self.make_rng('rng'), var.shape)
    return mu + var * epsilon

  def __call__(self, x, states):
    x = self.embed(x)
    for i in range(self.num_layers):
      states[i], x = self.layers[i](states[i], x)
    mean = self.mean(x)
    logvar = self.logvar(x)
    z = self.reparameterize(mean, logvar)
    return states, (z, mean, logvar)




class FitVid(nn.Module):
  """FitVid video predictor."""
  training: bool
  stochastic: bool = True
  action_conditioned: bool = True
  z_dim: int = 10
  g_dim: int = 128
  rnn_size: int = 256
  n_past: int = 2
  beta: float = 1e-4
  dtype: int = jnp.float32

  def setup(self):
    self.encoder = nvae.NVAE_ENCODER_VIDEO(
        training=self.training,
        stage_sizes=[2, 2, 2, 2],
        num_classes=self.g_dim)
    self.decoder = nvae.NVAE_DECODER_VIDEO(
        training=self.training,
        stage_sizes=[2, 2, 2, 2],
        first_block_shape=(8, 8, 512),
        skip_type='residual')
    self.frame_predictor = MultiGaussianLSTM(
        hidden_size=self.rnn_size, output_size=self.g_dim, num_layers=2)
    self.posterior = MultiGaussianLSTM(
        hidden_size=self.rnn_size, output_size=self.z_dim, num_layers=1)
    self.prior = MultiGaussianLSTM(
        hidden_size=self.rnn_size, output_size=self.z_dim, num_layers=1)

  def get_input(self, hidden, action, z):
    inp = [hidden]
    if self.action_conditioned:
      inp += [action]
    if self.stochastic:
      inp += [z]
    return jnp.concatenate(inp, axis=1)

  def __call__(self, video, actions, step):
    batch_size, video_len = video.shape[0], video.shape[1]
    pred_s = self.frame_predictor.init_states(batch_size)
    post_s = self.posterior.init_states(batch_size)
    prior_s = self.prior.init_states(batch_size)
    kl = functools.partial(utils.kl_divergence, batch_size=batch_size)

    # encode frames
    hidden, skips = self.encoder(video)
    # Keep the last available skip only
    skips = {k: skips[k][:, self.n_past-1] for k in skips.keys()}

    kld, post_means, post_logvars, prior_means, prior_logvars = 0.0, [], [], [], []

    # if self.training:
    prior_preds, prior_x_pred = [], None
    post_h_preds = []
    for i in range(1, video_len):
      prior_h, prior_h_target = hidden[:, i-1], hidden[:, i]
      post_h, post_h_target = hidden[:, i-1], hidden[:, i]
      if i > self.n_past:
        prior_h = self.encoder(jnp.expand_dims(prior_x_pred, 1))[0][:, 0]
      post_s, (post_z_t, post_mu, post_logvar) = self.posterior(post_h_target, post_s)
      prior_s, (prior_z_t, prior_mu, prior_logvar) = self.prior(prior_h, prior_s)

      # forward prior
      prior_inp = self.get_input(prior_h, actions[:, i-1], prior_z_t)
      prior_pred_s, (_, prior_h_pred, _) = self.frame_predictor(prior_inp, pred_s)
      prior_h_pred = nn.sigmoid(prior_h_pred)
      prior_x_pred = self.decoder(jnp.expand_dims(prior_h_pred, 1), skips)[:, 0]
      prior_preds.append(prior_x_pred)
      prior_means.append(prior_mu)
      prior_logvars.append(prior_logvar)

      # forward posterior
      post_inp = self.get_input(post_h, actions[:, i - 1], post_z_t)
      post_pred_s, (_, post_h_pred, _) = self.frame_predictor(post_inp, pred_s)
      post_h_pred = nn.sigmoid(post_h_pred)
      post_h_preds.append(post_h_pred)
      post_means.append(post_mu)
      post_logvars.append(post_logvar)

      # KL divergence
      kld += kl(post_mu, post_logvar, prior_mu, prior_logvar)

    # prior predictions
    prior_preds = jnp.stack(prior_preds, axis=1)

    # posterior predictions
    post_h_preds = jnp.stack(post_h_preds, axis=1)
    post_preds = self.decoder(post_h_preds, skips)

    # else:  # eval
    #   preds, x_pred = [], None
    #   for i in range(1, video_len):
    #     h, h_target = hidden[:, i-1], hidden[:, i]
    #     if i > self.n_past:
    #       h = self.encoder(jnp.expand_dims(x_pred, 1))[0][:, 0]
    #
    #     post_s, (_, mu, logvar) = self.posterior(h_target, post_s)
    #     prior_s, (z_t, prior_mu, prior_logvar) = self.prior(h, prior_s)
    #
    #     # forward prior
    #     inp = self.get_input(h, actions[:, i-1], z_t)
    #     pred_s, (_, h_pred, _) = self.frame_predictor(inp, pred_s)
    #     h_pred = nn.sigmoid(h_pred)
    #     x_pred = self.decoder(jnp.expand_dims(h_pred, 1), skips)[:, 0]
    #     preds.append(x_pred)
    #     prior_means.append(prior_mu)
    #     prior_logvars.append(prior_logvar)
    #
    #     # forward posterior
    #     post_means.append(mu)
    #     post_logvars.append(logvar)
    #
    #     # KL divergence
    #     kld += kl(mu, logvar, prior_mu, prior_logvar)
    #
    #   prior_preds = jnp.stack(preds, axis=1)
    #   post_preds = jnp.zeros_like(prior_preds)  # doesn't matter, not using posterior for eval mode

    post_means = jnp.stack(post_means, axis=1)
    post_logvars = jnp.stack(post_logvars, axis=1)
    prior_mse = utils.l2_loss(prior_preds, video[:, 1:])
    prior_loss = prior_mse + kld * self.beta

    # if self.training:
    post_mse = utils.l2_loss(post_preds, video[:, 1:])
    post_loss = post_mse + kld * self.beta
    # else:
    #   # doesn't matter for eval mode, not using any posterior
    #   post_mse = 0
    #   post_loss = 0


    # Metrics
    metrics = {
      'hist/post_mean': post_means,
      'hist/post_logvars': post_logvars,
      'hist/prior_mean': prior_means,
      'hist/prior_logvars': prior_logvars,
      'loss/prior_mse': prior_mse,
      'loss/kld': kld,
      'loss/prior_all': prior_loss,
    }

    # if self.training:
    metrics['loss/post_mse'] = post_mse
    metrics['loss/post_all'] = post_loss

    loss = post_loss if self.training else prior_loss
    return loss, prior_preds, post_preds, metrics
