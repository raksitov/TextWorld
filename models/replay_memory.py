import random

from collections import namedtuple
from util import max_len


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

  def push(self, state):
    raise NotImplementedError

  def sample(self, batch_size):
    raise NotImplementedError

  def end_episode(self):
    raise NotImplementedError

  def __len__(self):
    raise NotImplementedError


class BasicReplayMemory(ReplayMemory):

  def __init__(self, capacity):
    self.capacity = capacity
    self.reset()
    return

  def push(self, state):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = state
    self.position = (self.position + 1) % self.capacity
    return

  def sample(self, batch_size):
    if self.__len__() < batch_size:
      return None
    return random.sample(self.memory, batch_size)

  def end_episode(self):
    return

  def __len__(self):
    return len(self.memory)

  def reset(self):
    self.memory = []
    self.position = 0
    return


class PrioritizedReplayMemory(ReplayMemory):

  def __init__(self, capacity, priority_fraction):
    self.priority_fraction = priority_fraction
    self.alpha_memory = BasicReplayMemory(int(capacity * priority_fraction))
    self.beta_memory = BasicReplayMemory(capacity - self.alpha_memory.capacity)
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

  def __len__(self):
    return len(self.alpha_memory) + len(self.beta_memory)

  def end_episode(self):
    return


class RecentAndPrioritizedReplayMemory(ReplayMemory):

  def __init__(self, capacity, priority_fraction):
    self.priority_fraction = priority_fraction
    self.recent_memory = BasicReplayMemory(capacity)
    self.alpha_memory = BasicReplayMemory(int(capacity * priority_fraction))
    self.beta_memory = BasicReplayMemory(capacity - self.alpha_memory.capacity)
    return

  def push(self, state):
    self.recent_memory.push(state)
    return

  def sample(self, batch_size):
    from_recent = min(int(self.priority_fraction * batch_size), len(self.recent_memory))
    from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
    from_beta = min(batch_size - from_alpha - from_recent, len(self.beta_memory))
    if from_recent + from_alpha + from_beta < batch_size:
      return None
    result = (self.alpha_memory.sample(from_alpha) + self.beta_memory.sample(from_beta) +
              self.recent_memory.sample(from_recent))
    random.shuffle(result)
    return result

  def __len__(self):
    return len(self.alpha_memory) + len(self.beta_memory) + len(self.recent_memory)

  def end_episode(self):
    self._print_stats()
    for state in self.recent_memory.memory:
      self._push_internal(state)
    self.recent_memory.reset()
    return

  def _push_internal(self, state):
    if self.priority_fraction > 0. and state.reward != 0:
      self.alpha_memory.push(state)
      return
    self.beta_memory.push(state)
    return

  def _memory(self):
    return self.recent_memory.memory + self.alpha_memory.memory + self.beta_memory.memory

  def _print_stats(self):
    all_states = State(*zip(*self._memory()))
    print('observation max length: {}, action max length: {}, rewards: {}'.format(
        max_len(all_states.observation_ids),
        max_len(all_states.action_ids),
        sum(all_states.reward)))
    print('alpha buffer: {}, beta buffer: {}, recent memory: {}'.format(
        len(self.alpha_memory), len(self.beta_memory), len(self.recent_memory)))
