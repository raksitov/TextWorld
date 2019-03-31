import os
import shutil
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.contrib import layers
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleGRUCell
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = False
EMBEDDINGS_NAME = 'Embeddings'


def _print_shape(tensor, message):
  if DEBUG:
    return tf.Print(tensor, [tf.shape(tensor)], message)
  return tensor


def _fully_connected_encoder(layer, network_structure, scope_name, keep_prob=1.0):
  with tf.variable_scope(scope_name):
    last_layer = layers.fully_connected(layer, network_structure[0])
    for i in range(1, len(network_structure)):
      last_layer = layers.dropout(last_layer, keep_prob=keep_prob)
      last_layer = layers.fully_connected(last_layer, network_structure[i])
  last_layer = _print_shape(last_layer, 'Fully connected shape ({}): '.format(scope_name))
  return last_layer


class RNNEncoder(object):

  def __init__(self, hidden_size, keep_prob, cell_type='gru', scope='RNNEncoder'):

    self.hidden_size = hidden_size
    self.keep_prob = keep_prob
    self.cell_type = cell_type
    self.rnn_cell = self._dropout()
    self.scope = scope

  def _get_cell(self):
    if self.cell_type == 'gru':
      return CudnnCompatibleGRUCell(self.hidden_size)
    elif self.cell_type == 'lstm':
      return CudnnCompatibleLSTMCell(self.hidden_size)
    else:
      raise Exception('Unknown cell type: {}'.format(self.cell_type))

  def _dropout(self):
    return DropoutWrapper(self._get_cell(), input_keep_prob=self.keep_prob)

  def build_graph(self, inputs, masks=None, initial_states=None):

    with tf.variable_scope(self.scope):
      input_lens = None
      if masks is not None:
        input_lens = tf.reduce_sum(masks, reduction_indices=1)

      outputs, output_state = tf.nn.dynamic_rnn(
          self.rnn_cell,
          inputs,
          initial_state=initial_states,
          sequence_length=input_lens,
          dtype=tf.float32)

      outputs = tf.nn.dropout(outputs, self.keep_prob)

      return outputs, output_state


class Model:

  def __init__(self, config, session, scope, embedding_matrix, summaries_dir=None):
    self.config = config['model']
    self.counter = 0
    self.session = session
    self.last_time = time.time()
    with tf.variable_scope(scope):
      self.states = tf.placeholder(tf.int32, shape=(None, None))
      self.states_mask = tf.placeholder(tf.int32, shape=(None, None))
      self.labels = tf.placeholder(tf.float32, shape=(None,))
      self.actions = tf.placeholder(tf.int32, shape=(None, None))
      self.learning_rate = tf.placeholder_with_default(self.config['learning_rate'], shape=())

      self._add_embedding_layer(embedding_matrix)
      self._build_network()
      if summaries_dir is not None:
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('gradients norm', self.gradients_norm)
        self.merged_summary = tf.summary.merge_all()
        if os.path.exists(summaries_dir):
          shutil.rmtree(summaries_dir)
        self.summary_writer = tf.summary.FileWriter(summaries_dir, self.session.graph)
    return

  def train(self, observations, rewards, actions):
    # print('Train: {}'.format(rewards))
    if DEBUG:
      print('Train: {}, {}, {}'.format(observations.shape, rewards.shape, actions.shape))
    self.counter += 1
    summary, loss, gradients_norm, _ = self.session.run(
        [self.merged_summary, self.loss, self.gradients_norm, self.train_op],
        feed_dict={
            self.states: observations,
            self.states_mask: (observations != 0).astype(np.int32),
            self.labels: rewards,
            self.actions: actions,
        })
    if self.summary_writer is not None:
      if self.counter % 10 == 0:
        self.summary_writer.add_summary(summary, self.counter)
      if self.counter % 100 == 0:
        current_time = time.time()
        print('Batch: {}, loss: {:0.2f}, gradients norm: {:0.2f}, elapsed: {:0.2f}'.format(
            self.counter, loss, gradients_norm, current_time - self.last_time))
        self.last_time = current_time
    return

  def predict(self, observations, actions):
    # print('Predict: {}, {}'.format(observations.shape, actions.shape))
    q_values, probabilities = self.session.run(
        [self.q_values, self.probabilities],
        feed_dict={
            self.states: observations,
            self.states_mask: (observations != 0).astype(np.int32),
            self.actions: actions,
        })
    return q_values, probabilities

  def cleanup(self):
    self.summary_writer.close()
    return

  def _minimize(self, optimizer):
    gradients = optimizer.compute_gradients(self.loss)
    dense_gradients = [
        gradient for gradient in gradients if gradient[1].name.find(EMBEDDINGS_NAME) == -1]
    self.gradients_norm = tf.global_norm(dense_gradients)
    if self.config['clip_by_norm']:
      for i, (gradient, variable) in enumerate(gradients):
        if gradient is not None:
          gradients[i] = (tf.clip_by_norm(gradient, self.config['clip_by_norm']), variable)
    return optimizer.apply_gradients(gradients)

  def _add_embedding_layer(self, emb_matrix):
    # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
    embedding_matrix_init = tf.constant(emb_matrix, tf.float32)
    self.embedding_matrix = tf.get_variable(
        name=EMBEDDINGS_NAME,
        dtype=tf.float32,
        initializer=embedding_matrix_init)

    self.states_embeddings = embedding_ops.embedding_lookup(self.embedding_matrix, self.states)
    self.actions_embeddings = embedding_ops.embedding_lookup(self.embedding_matrix, self.actions)
    self.states_embeddings = _print_shape(self.states_embeddings, 'Embeddings shape (States): ')
    self.actions_embeddings = _print_shape(self.actions_embeddings, 'Embeddings shape (Actions): ')

  def _get_loss(self):
    if self.config['loss'] == 'mean_squared':
      return tf.losses.mean_squared_error(self.labels, self.q_values)
    elif self.config['loss'] == 'huber':
      return tf.losses.huber_loss(self.labels, self.q_values)
    return tf.reduce_sum(tf.square(self.labels - self.q_values))

  def _build_network(self):
    if self.config['state_encoder'] == 'rnn':
      rnn_encoder = RNNEncoder(hidden_size=self.config['actions_network'][-1],
                               keep_prob=self.config['keep_prob'],
                               cell_type=self.config['rnn_cell'])
      # (batch_size, states_len, hidden_size)
      states_output, _ = rnn_encoder.build_graph(
          self.states_embeddings, self.states_mask)
      states_output = states_output[:, -1, :]
    else:
      states_input = tf.reduce_mean(self.states_embeddings, axis=1)
      states_output = _fully_connected_encoder(
          states_input, self.config['states_network'], 'States')

    actions_input = tf.reduce_mean(self.actions_embeddings, axis=1)
    actions_output = _fully_connected_encoder(
        actions_input, self.config['actions_network'], 'Actions')

    self.q_values = tf.reduce_sum(states_output * actions_output, axis=1)
    self.probabilities = self.q_values
    if self.config['softmax_scaling_factor']:
      self.probabilities = tf.multinomial(
          tf.reshape(
              self.q_values,
              (-1,
               1)) * self.config['softmax_scaling_factor'],
          num_samples=1)

    self.labels = _print_shape(self.labels, 'Labels shape: ')
    self.q_values = _print_shape(self.q_values, 'Q-values shape: ')
    self.loss = self._get_loss()
    if self.config['optimizer'] == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    else:
      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.train_op = self._minimize(optimizer)
    return
