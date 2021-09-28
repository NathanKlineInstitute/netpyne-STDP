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
    self.pause = 2 # the space in between neurons fire
    self.popsize = config['net']['allpops'][in_pop]
    self.cnt = self.popsize
    # self.maxRate = 87.0
    self.maxRate = config['net']['InputMaxRate']
    self._idx_pauses = 0
    self._idx_cnt = 0
    self.is_all_states = config['env']['mock'] == 2
    if self.is_all_states:
      self.cnt_buckets = [obs['bins'] for obs in config['env']['observation_map']]
      assert sum(self.cnt_buckets) == self.cnt
      self.cnt = np.product(self.cnt_buckets)

      self.curr_step = config['env']['mock_curr_step']
      self.total_steps = config['env']['mock_total_steps']
      assert self.cnt % self.total_steps == 0, 'Total count has to be divisible by total steps'
      self.cnt = int(self.cnt / self.total_steps)
      self._curr_step_idx = self.cnt * self.curr_step
      # print('starting with', self.cnt, self._curr_step_idx)

  def input_firing_rates(self):
    if self._idx_pauses < (self._idx_cnt + 1) * self.pause:
      self._idx_pauses += 1
      return np.zeros(self.popsize)
    else:
      res = np.zeros(self.popsize)
      if self.is_all_states:
        curr = self._idx_cnt + self._curr_step_idx
        curr_sum = 0
        for bucket in self.cnt_buckets:
          res.put(curr % bucket + curr_sum, self.maxRate)
          curr /= bucket
          curr_sum += bucket
      else:
        res.put(self._idx_cnt, self.maxRate)
      self._idx_cnt += 1
      if self._idx_cnt == self.cnt:
        self.aigame._done = True
        self._idx_pauses = 0
        self._idx_cnt = 0
      return res
