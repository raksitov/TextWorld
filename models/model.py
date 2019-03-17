import os
import shutil
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import embedding_ops
from tensorflow.contrib import layers

DEBUG = False


def _print_shape(tensor, message):
  if DEBUG:
    return tf.Print(tensor, [tf.shape(tensor)], message)
  return tensor


def _fully_connected_encoder(layer, network_structure, scope_name):
  with tf.variable_scope(scope_name):
    last_layer = layers.fully_connected(layer, network_structure[0])
    for i in range(1, len(network_structure)):
      last_layer = layers.fully_connected(last_layer, network_structure[i])
  last_layer = _print_shape(last_layer, 'Fully connected shape ({}): '.format(scope_name))
  return last_layer


class BagOfWordsModel:

  def __init__(self, config, session, scope, embedding_matrix, summaries_dir=None):
    self.config = config['model']
    self.counter = 0
    self.session = session
    self.last_time = time.time()
    with tf.variable_scope(scope):
      self.states = tf.placeholder(tf.int32, shape=(None, None))
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
    #print('Train: {}'.format(rewards))
    if DEBUG:
      print('Train: {}, {}, {}'.format(observations.shape, rewards.shape, actions.shape))
    self.counter += 1
    summary, loss, gradients_norm, _ = self.session.run(
        [self.merged_summary, self.loss, self.gradients_norm, self.train_op],
        feed_dict={
            self.states: observations,
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
    #print('Predict: {}, {}'.format(observations.shape, actions.shape))
    q_values, probabilities = self.session.run(
        [self.q_values, self.probabilities],
        feed_dict={
            self.states: observations,
            self.actions: actions,
        })
    return q_values, probabilities

  def cleanup(self):
    self.summary_writer.close()
    return

  def _minimize(self, optimizer):
    gradients = optimizer.compute_gradients(self.loss)
    self.gradients_norm = tf.global_norm(gradients)
    if self.config['clip_by_norm']:
      for i, (gradient, variable) in enumerate(gradients):
        if gradient is not None:
          gradients[i] = (tf.clip_by_norm(gradient, self.config['clip_by_norm']), variable)
    return optimizer.apply_gradients(gradients)

  def _add_embedding_layer(self, emb_matrix):
    # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
    self.embedding_matrix = tf.constant(emb_matrix, tf.float32)
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
    states_input = tf.reduce_mean(self.states_embeddings, axis=1)
    actions_input = tf.reduce_mean(self.actions_embeddings, axis=1)
    states_output = _fully_connected_encoder(states_input, self.config['states_network'], 'States')
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
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.train_op = self._minimize(optimizer)
    return
