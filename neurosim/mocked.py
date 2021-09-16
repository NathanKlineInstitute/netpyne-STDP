import random
import numpy as np

class MockAIGame:
  def __init__(self, config):
    obs_space = len(config['env']['observation_map'])
    obs = np.zeros(obs_space)
    self.observations = [obs, obs]
    self.count_steps = [0]
    self._moves = len(config['moves'])
    self._done = False

  def randmove(self):
    return random.randint(0, self._moves-1)

  def playGame(self, actions):
    self.count_steps[-1] += 1
    if self._done:
      self.count_steps.append(0)
      self._done = False
      return 1.0, True
    return 1.0, False

class MockGameInterface:
  def __init__(self, aigame, config):
    self.aigame = aigame
    in_pop = config['net']['inputPop']
    self.pause = 5 # the space in between neurons fire
    self.cnt = config['net']['allpops'][in_pop]
    # self.maxRate = config['net']['InputMaxRate']
    self.maxRate = 87.0
    self._idx_pauses = 0
    self._idx_cnt = 0

  def input_firing_rates(self):
    if self._idx_pauses < (self._idx_cnt + 1) * 5:
      self._idx_pauses += 1
      return np.zeros(self.cnt)
    else:
      res = np.zeros(self.cnt)
      res.put(self._idx_cnt, self.maxRate)
      self._idx_cnt += 1
      if self._idx_cnt == self.cnt:
        self.aigame._done = True
        self._idx_pauses = 0
        self._idx_cnt = 0
      return res
