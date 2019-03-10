import gym
import time
import tqdm
import textworld.gym

import numpy as np

from agent import RandomAgent, BagOfWordsAgent
from glob import glob
from pprint import pprint
from textworld import EnvInfos

from os.path import join

GAMES_PATH = "/home/renat/WorkingDir/cs230/TextWorld/sample_games"
VOCAB_PATH = "/home/renat/WorkingDir/cs230/TextWorld/vocab.txt"
GLOVE_PATH = "/home/renat/WorkingDir/cs224n/squad/data/glove.6B.50d.txt"
GLOVE_DIM = 50
EPISODES = 1000
BATCH_SIZE = 20
UPDATE_FREQUENCY = 1


def get_embeddings():
  with open(GLOVE_PATH) as glove:
    glove_vocab = {}
    for vector in glove:
      vector = vector.lstrip().rstrip().split(" ")
      glove_vocab[vector[0]] = list(map(float, vector[1:]))
  with open(VOCAB_PATH) as vocab:
    word_vocab = vocab.read().split("\n")

  embedding_matrix = np.zeros((len(word_vocab), GLOVE_DIM))
  word_ids = {}
  for idx, word in enumerate(word_vocab):
    word_ids[word] = idx
    if word not in glove_vocab:
      if idx:
        embedding_matrix[idx, :] = np.random.randn(GLOVE_DIM)
    else:
      embedding_matrix[idx, :] = glove_vocab[word]

  return embedding_matrix, word_ids


def dict_to_array(d, size):
  return [{k: v[idx] for (k, v) in d.items()} for idx in range(size)]


def train(env, agent):
  max_reward = 0
  for episode in tqdm.tqdm(range(EPISODES)):
    observations, infos = env.reset()
    infos_array = dict_to_array(infos, BATCH_SIZE)
    rewards = [0] * BATCH_SIZE
    dones = [False] * BATCH_SIZE

    step = 0
    # TODO: maybe condition on max_steps as well.
    while not all(dones):
      actions = agent.choose_actions(observations, infos_array, dones)
      new_observations, new_rewards, new_dones, new_infos = env.step(actions)
      new_infos_array = dict_to_array(new_infos, BATCH_SIZE)
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
      if step % UPDATE_FREQUENCY == 0:
        agent.train()
      step += 1
    max_reward = max(max_reward, max(rewards))
    print('Rewards: {}, steps: {}, max: {}'.format(
      np.mean(rewards), step, max_reward))
    agent.end_episode()

  agent.cleanup()
  return


def main():
  gamefiles = glob(join(GAMES_PATH, "*.ulx"))
  print("Found {} games.".format(len(gamefiles)))
  #pprint(gamefiles)
  # Pick a game.
  gamefile = gamefiles[8]

  requested_infos = EnvInfos(
      admissible_commands=True,
      description=True,
      has_lost=True,
      has_won=True,
      max_score=True,
  )
  env_id = textworld.gym.register_games([gamefile], requested_infos)
  env_id = textworld.gym.make_batch(env_id, batch_size=BATCH_SIZE, parallel=True)
  env = gym.make(env_id)
  agent = BagOfWordsAgent(*get_embeddings())
  train(env, agent)
  return


if __name__ == "__main__":
  main()
