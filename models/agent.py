import random
import spacy

import numpy as np
import tensorflow as tf

from model import BagOfWordsModel
from collections import namedtuple
from util import words_to_ids, preproc, pad_sequences, max_len

from numpy.random import RandomState

State = namedtuple(
    'State', [
        'observation_ids',
        'admissible_actions_ids',
        'action_ids',
        'new_observation_ids',
        'new_admissible_actions_ids',
        'reward',
        'done'])


class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0
    return

  def push(self, state):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = state
    self.position = (self.position + 1) % self.capacity
    return

  def sample(self, batch_size):
    if self.__len__() < batch_size:
      return None
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class PrioritizedReplayMemory(object):

  def __init__(self, capacity, priority_fraction):
    self.priority_fraction = priority_fraction
    self.alpha_memory = ReplayMemory(int(capacity * priority_fraction))
    self.beta_memory = ReplayMemory(capacity - self.alpha_memory.capacity)
    return

  def push(self, state):
    if self.priority_fraction > 0. and state.reward != 0:
      self.alpha_memory.push(state)
      return
    self.beta_memory.push(state)
    return

  def sample(self, batch_size):
    from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
    from_beta = min(batch_size - from_alpha, len(self.beta_memory))
    if from_alpha + from_beta < batch_size:
      return None
    result = self.alpha_memory.sample(from_alpha) + self.beta_memory.sample(from_beta)
    random.shuffle(result)
    return result

  def memory(self):
    return self.alpha_memory.memory + self.beta_memory.memory

  def __len__(self):
    return len(self.alpha_memory) + len(self.beta_memory)


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
    self.replay_buffer = PrioritizedReplayMemory(
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

  def choose_actions(self, observations, infos, dones):
    random_plays = np.random.uniform(low=0.0, high=1.0, size=len(observations))
    chosen_actions = []
    for observation, info, done, random_play in zip(observations, infos, dones, random_plays):
      if random_play < self.epsilon or done:
        chosen_actions.append(self.rng.choice(info['admissible_commands']))
      else:
        _, probabilities = self.model.predict(
            np.array(self._build_observation_ids(observation, info)).reshape(1, -1),
            self._build_admissible_actions_ids(info, shuffle=False))
        chosen_actions.append(info['admissible_commands'][np.argmax(probabilities)])
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
    self.print_stats()
    return

  def cleanup(self):
    self.model.cleanup()
    self.session.close()
    return

  def print_stats(self):
    all_states = State(*zip(*self.replay_buffer.memory()))
    print('observation max length: {}, action max length: {}, rewards: {}'.format(
        max_len(all_states.observation_ids),
        max_len(all_states.action_ids),
        sum(all_states.reward)))
    print('alpha buffer: {}, beta buffer: {}'.format(
        len(self.replay_buffer.alpha_memory), len(self.replay_buffer.beta_memory)))

  def _Q(self, sample):
    q_values, _ = self.model.predict(
        np.array(sample.new_observation_ids).reshape(1, -1),
        sample.new_admissible_actions_ids)
    # print('_Q: {}'.format(q_values))
    return np.max(q_values)

  def _build_observation_ids(self, observation, info):
    observation_tokens = preproc(observation, str_type='feedback', tokenizer=self.nlp)
    observation_ids = words_to_ids(observation_tokens, self.word_ids)
    description_tokens = preproc(info['description'], tokenizer=self.nlp)
    description_ids = words_to_ids(description_tokens, self.word_ids)
    return observation_ids + description_ids

  def _build_admissible_actions_ids(self, info, shuffle):
    admissible_actions_tokens = [preproc(admissible_action, tokenizer=self.nlp) for
                                 admissible_action in info['admissible_commands']]
    admissible_actions_ids = [words_to_ids(admissible_action_tokens, self.word_ids) for
                              admissible_action_tokens in admissible_actions_tokens]
    result = np.array(pad_sequences(
        admissible_actions_ids, max_len=self.get_actions_padding_size()))
    if shuffle:
      np.random.shuffle(result)
    return result

  def _build_action_ids(self, action):
    action_tokens = preproc(action, tokenizer=self.nlp)
    action_ids = words_to_ids(action_tokens, self.word_ids)
    return action_ids


class BagOfWordsAgent(TrainableAgent):

  def get_model(self):
    return BagOfWordsModel

  def get_observation_padding_size(self):
    return None

  def get_actions_padding_size(self):
    return None
