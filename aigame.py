"""
AIGame: Run an OpenAIGym
"""

import gym
from gym import wrappers
import numpy as np
from time import time
from collections import deque


class AIGame:
  """ Interface to OpenAI gym game
  """

  def __init__(self, config):  # initialize variables
    # make the environment - env is global so that it only gets created on
    # a single node (important when using MPI with > 1 node)
    env = gym.make(config['env']['name'])
    env.reset()
    self.env = env
    self.actionsPerPlay = config['actionsPerPlay']
    self.observations = deque(maxlen=config['observationsToKeep'])
    self.rewards = deque(maxlen=config['rewardsToKeep'])
    self.count_episodes = 0
    self.count_steps = 0
    self.count_total_steps = 0

  def _clean(self):
    self.observations.clear()
    self.rewards.clear()
    self.count_steps = 0

  def playGame(self, actions):
    current_rewards = []
    done = False

    assert len(actions) == self.actionsPerPlay
    for adx in range(self.actionsPerPlay):
      # for each action generated
      caction = actions[adx]

      observation, reward, done, info = self.env.step(caction)
      self.env.render()

      current_rewards.append(reward)
      self.rewards.append(reward)
      self.observations.append(observation)

      self.count_steps += 1
      self.count_total_steps += 1

      if done:
        self.env.reset()
        self._clean()
        self.count_episodes += 1
        break

    return current_rewards
