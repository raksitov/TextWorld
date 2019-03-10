import gym
import time
import tqdm
import textworld.gym
import yaml

import numpy as np

from agent import RandomAgent, BagOfWordsAgent
from glob import glob
from pprint import pprint
from textworld import EnvInfos

from os.path import join

CONFIG = 'config.yaml'


def get_embeddings(config):
  with open(config['glove_path']) as glove:
    glove_vocab = {}
    for vector in glove:
      vector = vector.lstrip().rstrip().split(' ')
      glove_vocab[vector[0]] = list(map(float, vector[1:]))
  with open(config['vocab_path']) as vocab:
    word_vocab = vocab.read().split('\n')

  embedding_matrix = np.zeros((len(word_vocab), config['glove_dim']))
  word_ids = {}
  for idx, word in enumerate(word_vocab):
    word_ids[word] = idx
    if word not in glove_vocab:
      if idx:
        embedding_matrix[idx, :] = np.random.randn(config['glove_dim'])
    else:
      embedding_matrix[idx, :] = glove_vocab[word]

  return embedding_matrix, word_ids


def dict_to_array(d, size):
  return [{k: v[idx] for (k, v) in d.items()} for idx in range(size)]


def train(env, agent, config):
  max_reward = 0
  for episode in tqdm.tqdm(range(config['episodes'])):
    observations, infos = env.reset()
    infos_array = dict_to_array(infos, config['environment_batch_size'])
    rewards = [0] * config['environment_batch_size']
    dones = [False] * config['environment_batch_size']

    step = 0
    # TODO: maybe condition on max_steps as well.
    while not all(dones):
      actions = agent.choose_actions(observations, infos_array, dones)
      new_observations, new_rewards, new_dones, new_infos = env.step(actions)
      new_infos_array = dict_to_array(new_infos, config['environment_batch_size'])
      for idx, done in enumerate(dones):
        if not done:
          agent.add_state(observations[idx],
                          infos_array[idx],
                          actions[idx],
                          new_observations[idx],
                          new_infos_array[idx],
                          new_rewards[idx] - rewards[idx],
                          dones[idx])
      observations = new_observations
      infos_array = new_infos_array
      rewards = new_rewards
      dones = new_dones
      if step % config['update_frequency'] == 0:
        agent.train()
      step += 1
    max_reward = max(max_reward, max(rewards))
    print('Rewards: {}, steps: {}, max: {}'.format(
        np.mean(rewards), step, max_reward))
    agent.end_episode()

  agent.cleanup()
  return


def main():
  with open(CONFIG) as reader:
    config = yaml.safe_load(reader)
  gamefiles = glob(join(config['main']['games_path'], '*.ulx'))
  print('Found {} games.'.format(len(gamefiles)))
  # pprint(gamefiles)
  # Pick a game.
  gamefile = gamefiles[4]

  requested_infos = EnvInfos(
      admissible_commands=True,
      description=True,
      has_lost=True,
      has_won=True,
      max_score=True,
  )
  env_id = textworld.gym.register_games([gamefile], requested_infos)
  env_id = textworld.gym.make_batch(
      env_id,
      batch_size=config['main']['environment_batch_size'],
      parallel=True)
  env = gym.make(env_id)
  agent = BagOfWordsAgent(config, *get_embeddings(config['main']))
  train(env, agent, config['main'])
  return


if __name__ == '__main__':
  main()
