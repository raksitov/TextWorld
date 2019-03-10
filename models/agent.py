import spacy
import random

import numpy as np
import tensorflow as tf

from model import BagOfWordsModel#, ConcatenatedWordsModel
from collections import namedtuple
from util import words_to_ids, preproc, pad_sequences, max_len

from numpy.random import RandomState

SUMMARY_DIR = 'Summaries'
GAMMA = 0.5
TRAINING_BATCH_SIZE = 20
REPLAY_MEMORY_SIZE = 10000
RANDOM_SEED = 42
EPSILON_START = 1.
EPSILON_END = 0.01
EPSILON_ANNEALING_INTERVAL = 300
SHUFFLE_ACTIONS = False

State = namedtuple(
    'State', [
        'observation_ids',
        'admissible_actions_ids',
        'action_ids',
        'new_observation_ids',
        'new_admissible_actions_ids',
        'reward',
        'done'])


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
    self.rng = RandomState(RANDOM_SEED)
    return

  def choose_actions(self, observations, infos, dones):
    return [self.rng.choice(info["admissible_commands"]) for info in infos]


class TrainableAgent(Agent):

  def __init__(self, embedding_matrix, word_ids):
    self.word_ids = word_ids
    self.states = []
    self.epsilon = EPSILON_START
    self.episode = 0

    self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
    self.rng = RandomState(RANDOM_SEED)

    tf.reset_default_graph()
    self.session = tf.Session()
    self.model = self.get_model()(
        self.session,
        'Model',
        embedding_matrix,
        self._get_summary_dir())
    self.session.run(tf.global_variables_initializer())
    return

  def _get_summary_dir(self):
    return SUMMARY_DIR

  def choose_actions(self, observations, infos, dones):
    random_plays = np.random.uniform(low=0.0, high=1.0, size=len(observations))
    chosen_actions = []
    for observation, info, done, random_play in zip(observations, infos, dones, random_plays):
      if random_play < self.epsilon or done:
        chosen_actions.append(self.rng.choice(info["admissible_commands"]))
      else:
        _, probabilities = self.model.predict(
            np.tile(
                np.array(self._build_observation_ids(observation, info)),
                (len(info["admissible_commands"]), 1)),
            self._build_admissible_actions_ids(info))
        chosen_actions.append(info["admissible_commands"][np.argmax(probabilities)])
    return chosen_actions

  def add_state(self, observation, info, action, new_observation, new_info, reward, done):
    self.states.append(State(self._build_observation_ids(observation, info),
                             self._build_admissible_actions_ids(info),
                             self._build_action_ids(action),
                             self._build_observation_ids(new_observation, new_info),
                             self._build_admissible_actions_ids(new_info), reward, done))
    return

  def get_model(self):
    raise NotImplementedError

  def get_observation_padding_size(self):
    raise NotImplementedError

  def get_actions_padding_size(self):
    raise NotImplementedError

  def train(self):
    batch = self._get_batch()
    if not batch:
      return
    observations_ids = []
    rewards = []
    actions_ids = []
    for sample in batch:
      observations_ids.append(sample.observation_ids)
      reward = sample.reward
      if not sample.done:
        reward += GAMMA * self._Q(sample)
      rewards.append(reward)
      actions_ids.append(sample.action_ids)
    self.model.train(
        np.stack(pad_sequences(observations_ids, max_len=self.get_observation_padding_size())),
        np.stack(rewards),
        np.stack(pad_sequences(actions_ids, max_len=self.get_actions_padding_size())))
    return

  def end_episode(self):
    self.episode += 1
    if self.episode <= EPSILON_ANNEALING_INTERVAL:
      self.epsilon -= (EPSILON_START - EPSILON_END) / float(EPSILON_ANNEALING_INTERVAL)
    self.print_stats()
    return

  def cleanup(self):
    self.model.cleanup()
    self.session.close()
    return

  def print_stats(self):
    all_states = State(*zip(*self.states))
    print('observation max length: {}, action max length: {}'.format(
      max_len(all_states.observation_ids), max_len(all_states.action_ids)))

  def _recent_memories(self):
    start = 0
    # TODO: replace with circular buffer.
    if len(self.states) >= REPLAY_MEMORY_SIZE:
      start = -REPLAY_MEMORY_SIZE
    return self.states[start:]

  def _get_batch(self):
    if len(self.states) < TRAINING_BATCH_SIZE:
      return None
    return random.sample(self._recent_memories(), TRAINING_BATCH_SIZE)

  def _Q(self, sample):
    q_values, _ = self.model.predict(
        np.tile(
            np.array(sample.new_observation_ids),
            (len(sample.new_admissible_actions_ids), 1)),
        sample.new_admissible_actions_ids)
    #print('_Q: {}'.format(q_values))
    return np.max(q_values)

  def _build_observation_ids(self, observation, info):
    observation_tokens = preproc(observation, str_type='feedback', tokenizer=self.nlp)
    observation_ids = words_to_ids(observation_tokens, self.word_ids)
    description_tokens = preproc(info['description'], tokenizer=self.nlp)
    description_ids = words_to_ids(description_tokens, self.word_ids)
    return observation_ids + description_ids

  def _build_admissible_actions_ids(self, info):
    admissible_actions_tokens = [preproc(admissible_action, tokenizer=self.nlp) for
                                 admissible_action in info['admissible_commands']]
    admissible_actions_ids = [words_to_ids(admissible_action_tokens, self.word_ids) for
                              admissible_action_tokens in admissible_actions_tokens]
    result = np.array(pad_sequences(
      admissible_actions_ids, max_len=self.get_actions_padding_size()))
    if SHUFFLE_ACTIONS:
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
