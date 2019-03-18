import spacy

import numpy as np
import tensorflow as tf

from model import Model
from util import memoized_string_to_ids, pad_sequences
from replay_memory import State, RecentAndPrioritizedReplayMemory

from numpy.random import RandomState


class Agent():

  def choose_actions(self, observations, infos, dones):
    pass

  def add_state(self, observation, info, action, new_observation, new_info, reward, done):
    pass

  def cleanup(self):
    pass

  def train(self):
    pass

  def end_episode(self):
    pass


class RandomAgent(Agent):

  def __init__(self):
    # TODO: move seed into an argument.
    self.rng = RandomState(self.config['random_seed'])
    return

  def choose_actions(self, observations, infos, dones):
    return [self.rng.choice(info['admissible_commands']) for info in infos]


class TrainableAgent(Agent):

  def __init__(self, config, embedding_matrix, word_ids):
    self.config = config['agent']
    self.word_ids = word_ids
    self.replay_buffer = RecentAndPrioritizedReplayMemory(
        self.config['replay_memory_capacity'],
        self.config['replay_memory_priority_fraction'])
    self.epsilon = self.config['epsilon_start']
    self.episode = 0

    self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
    self.rng = RandomState(self.config['random_seed'])

    tf.reset_default_graph()
    self.session = tf.Session()
    self.model = self.get_model()(
        config,
        self.session,
        'Model',
        embedding_matrix,
        self._get_summary_dir())
    self.session.run(tf.global_variables_initializer())
    return

  def _get_summary_dir(self):
    return self.config['summary_dir']

  def choose_actions(self, observations, infos, dones, evaluation):
    random_plays = np.random.uniform(low=0.0, high=1.0, size=len(observations))
    chosen_actions = []
    for observation, info, done, random_play in zip(observations, infos, dones, random_plays):
      if evaluation or (random_play >= self.epsilon and not done):
        _, probabilities = self.model.predict(
            np.array(self._build_observation_ids(observation, info)).reshape(1, -1),
            self._build_admissible_actions_ids(info, shuffle=False))
        chosen_actions.append(info['admissible_commands'][np.argmax(probabilities)])
      else:
        chosen_actions.append(self.rng.choice(info['admissible_commands']))
    return chosen_actions

  def add_state(self, observation, info, action, new_observation, new_info, reward, done):
    self.replay_buffer.push(State(
        self._build_observation_ids(observation, info),
        self._build_admissible_actions_ids(info, shuffle=self.config['shuffle_actions']),
        self._build_action_ids(action),
        self._build_observation_ids(new_observation, new_info),
        self._build_admissible_actions_ids(
            new_info,
            shuffle=self.config['shuffle_actions']),
        reward,
        done))
    return

  def get_model(self):
    raise NotImplementedError

  def get_observation_padding_size(self):
    raise NotImplementedError

  def get_actions_padding_size(self):
    raise NotImplementedError

  def train(self):
    batch = self.replay_buffer.sample(self.config['training_batch_size'])
    if not batch:
      return
    if self.config['batch_oneshot']:
      observations_ids, rewards, actions_ids = self._preprocess_batch_oneshot(batch)
    else:
      observations_ids, rewards, actions_ids = self._preprocess_batch(batch)
    self.model.train(
        np.stack(pad_sequences(observations_ids, max_len=self.get_observation_padding_size())),
        np.stack(rewards),
        np.stack(pad_sequences(actions_ids, max_len=self.get_actions_padding_size())))
    return

  def end_episode(self):
    self.episode += 1
    if self.episode <= self.config['epsilon_annealing_interval']:
      self.epsilon -= ((self.config['epsilon_start'] - self.config['epsilon_end']) /
                       float(self.config['epsilon_annealing_interval']))
    self.replay_buffer.end_episode()
    return

  def cleanup(self):
    self.model.cleanup()
    self.session.close()
    return

  def _preprocess_batch(self, batch):
    observations_ids = []
    rewards = []
    actions_ids = []
    for sample in batch:
      observations_ids.append(sample.observation_ids)
      reward = sample.reward
      if not sample.done:
        reward += self.config['gamma'] * self._Q(sample)
      rewards.append(reward)
      actions_ids.append(sample.action_ids)
    return observations_ids, rewards, actions_ids

  def _preprocess_batch_oneshot(self, batch):
    observations_ids = []
    rewards = []
    actions_ids = []
    for idx, sample in enumerate(batch):
      observations_ids.append(sample.observation_ids)
      rewards.append(sample.reward)
      actions_ids.append(sample.action_ids)

    q_values, idx_mapping = self._Q_oneshot(batch)
    for idx, sample in enumerate(batch):
      if not sample.done:
        (start, end) = idx_mapping[idx]
        rewards[idx] += self.config['gamma'] * np.max(q_values[start: end])
    return observations_ids, rewards, actions_ids

  def _Q_oneshot(self, batch):
    new_observations_ids = []
    new_admissible_actions_ids = []
    observation_padding_size = 0
    idx_mapping = {}
    current_idx = 0
    for idx, sample in enumerate(batch):
      if sample.done:
        continue
      new_observations_ids.append(sample.new_observation_ids)
      new_admissible_actions_ids.append(sample.new_admissible_actions_ids)
      observation_padding_size = max(observation_padding_size, len(sample.new_observation_ids))
      idx_mapping[idx] = (current_idx, current_idx + len(sample.new_admissible_actions_ids))
      current_idx += len(sample.new_admissible_actions_ids)

    new_observations_ids = pad_sequences(new_observations_ids, max_len=observation_padding_size)
    new_tiled_observations = []
    for idx, observation in enumerate(new_observations_ids):
      num_actions = len(new_admissible_actions_ids[idx])
      new_tiled_observations.append(np.tile(observation, (num_actions, 1)))
    q_values, _ = self.model.predict(np.concatenate(new_tiled_observations),
                                     np.concatenate(new_admissible_actions_ids))
    return q_values, idx_mapping

  def _Q(self, sample):
    q_values, _ = self.model.predict(
        np.array(sample.new_observation_ids).reshape(1, -1),
        sample.new_admissible_actions_ids)
    # print('_Q: {}'.format(q_values))
    return np.max(q_values)

  def _build_observation_ids(self, observation, info):
    observation_ids = memoized_string_to_ids(
        observation, self.word_ids, str_type='feedback', tokenizer=self.nlp)

    description_ids = memoized_string_to_ids(
        info['description'], self.word_ids, tokenizer=self.nlp)

    inventory_ids = memoized_string_to_ids(
        info['inventory'], self.word_ids, tokenizer=self.nlp)

    recipe_ids = memoized_string_to_ids(
        info['extra.recipe'], self.word_ids, tokenizer=self.nlp)
    return observation_ids + description_ids + inventory_ids + recipe_ids

  def _build_admissible_actions_ids(self, info, shuffle):
    admissible_actions_ids = [
        memoized_string_to_ids(
            admissible_action,
            self.word_ids,
            tokenizer=self.nlp) for admissible_action in info['admissible_commands']]

    result = np.array(pad_sequences(
        admissible_actions_ids, max_len=self.get_actions_padding_size()))
    if shuffle:
      np.random.shuffle(result)
    return result

  def _build_action_ids(self, action):
    action_ids = memoized_string_to_ids(action, self.word_ids, tokenizer=self.nlp)
    return action_ids


class CustomizableAgent(TrainableAgent):

  def get_model(self):
    return Model

  def get_observation_padding_size(self):
    return None

  def get_actions_padding_size(self):
    return 10
