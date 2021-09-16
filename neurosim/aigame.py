import gym
from gym import wrappers
import numpy as np
from datetime import datetime
from collections import deque
import random

class AIGame:
  """ Interface to OpenAI gym game
  """

  def __init__(self, config):
    self.conf_env = config['env']
    self.do_render = self.conf_env['render']
    self.actionsPerPlay = config['actionsPerPlay']
    self.observations = deque(maxlen=config['observationsToKeep'])
    self.rewards = deque(maxlen=config['rewardsToKeep'])
    self.count_episodes = 0
    self.count_steps = [0]
    self.tstart = None
    # Setup the environment and set the observation
    self._setup_env()
    observation = self.env.reset()
    self.observations.append(observation)

  def _setup_env(self):
    # make the environment - env is global so that it only gets created on
    # a single node (important when using MPI with > 1 node)
    env = gym.make(self.conf_env['name'])
    if 'seed' in self.conf_env:
      env.seed(self.conf_env['seed'])
      env.action_space.np_random.seed(self.conf_env['seed'])
    if 'rerunEpisode' in self.conf_env:
      # rerunEpisode is 1-based. so 1 is the same as the first episode in evaluation
      for _ in range(self.conf_env['rerunEpisode']-1):
        env.reset()
    self.env = env

  def _clean(self):
    self.observations.clear()
    self.rewards.clear()
    self.count_steps.append(0)
    self.tstart = None
    if 'rerunEpisode' in self.conf_env:
      self._setup_env()
    observation = self.env.reset()
    self.observations.append(observation)

  def randmove(self):
    if 'rerunEpisode' in self.conf_env:
      # For Discrete
      return random.randint(0, self.env.action_space.n-1)
    return self.env.action_space.sample()

  def playGame(self, actions):
    current_rewards = []
    done = False
    if not self.tstart:
      self.tstart = datetime.now()

    assert len(actions) == self.actionsPerPlay
    for adx in range(self.actionsPerPlay):
      # for each action generated
      caction = actions[adx]

      observation, reward, done, info = self.env.step(caction)
      if self.do_render:
        self.env.render()

      current_rewards.append(reward)
      self.rewards.append(reward)
      self.observations.append(observation)

      self.count_steps[-1] += 1

      if done:
        # delta = (datetime.now() - self.tstart) / self.count_steps[-1]
        # print('Python time per step: {} ms'.format(delta.microseconds / 1000))
        self._clean()
        self.count_episodes += 1
        break

    return current_rewards, done
