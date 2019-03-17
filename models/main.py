import gym
import time
import tqdm
import textworld.gym
import yaml

import numpy as np

from agent import RandomAgent, CustomizableAgent
from glob import glob
from pprint import pprint
from textworld import EnvInfos
from util import get_embeddings, dict_to_array

from os.path import join

CONFIG = 'config.yaml'


def train(env, agent, config):
  max_reward = 0
  max_mean_rewards = 0
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
    max_mean_rewards = max(max_mean_rewards, np.mean(rewards))
    print('Mean rewards: {}, steps: {}, max reward: {}, max mean rewards: {}'.format(
        np.mean(rewards), step, max_reward, max_mean_rewards))
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
      command_templates=True,
      description=True,
      entities=True,
      has_lost=True,
      has_won=True,
      inventory=True,
      max_score=True,
      objective=True,
      verbs=True,
      extras=[
          "recipe",
      ],
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
