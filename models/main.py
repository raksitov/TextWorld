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


def play(env, agent, config, evaluation=False):
  if evaluation:
    print('\nEvaluation:')
  max_reward = 0
  max_mean_rewards = 0
  num_episodes = config['eval_episodes'] if evaluation else config['train_episodes']
  for episode in tqdm.tqdm(range(num_episodes)):
    observations, infos = env.reset()
    infos_array = dict_to_array(infos, config['environment_batch_size'])
    rewards = [0] * config['environment_batch_size']
    dones = [False] * config['environment_batch_size']

    steps = 0
    # TODO: maybe condition on max_steps as well.
    while not all(dones):
      actions = agent.choose_actions(observations, infos_array, dones, evaluation)
      new_observations, new_rewards, new_dones, new_infos = env.step(actions)
      new_infos_array = dict_to_array(new_infos, config['environment_batch_size'])
      for idx, done in enumerate(dones):
        if not done and not evaluation:
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
      if not evaluation and steps % config['update_frequency'] == 0:
        agent.train()
      steps += 1
    mean_rewards = np.mean(rewards)
    max_reward = max(max_reward, max(rewards))
    max_mean_rewards = max(max_mean_rewards, np.mean(rewards))
    max_score = max([info['max_score'] for info in infos_array])
    wins_percentage = sum([info['has_won'] for info in infos_array]) * 100. / len(infos_array)
    print('Mean rewards: {}({}), steps: {}, max reward: {}({}), wins percentage - {}'.format(
        mean_rewards, max_mean_rewards, steps, max_reward, max_score, wins_percentage))
    agent.end_episode()
  return


def main():
  with open(CONFIG) as reader:
    config = yaml.safe_load(reader)
  gamefiles = glob(join(config['main']['games_path'], '*.ulx'))
  print('Found {} games.'.format(len(gamefiles)))
  # pprint(gamefiles)
  # Pick a game.
  gamefile = gamefiles[1]

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

  agent = CustomizableAgent(config, *get_embeddings(config['main']))

  play(env, agent, config['main'])
  play(env, agent, config['main'], evaluation=True)

  agent.cleanup()
  return


if __name__ == '__main__':
  main()
