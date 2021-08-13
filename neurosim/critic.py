import math
import numpy as np
from collections import deque

EPS = 0.000001

def _modulate_linear(mod_steps, reward, min_steps=1):
  M = len(mod_steps)
  if M < min_steps:
    return reward
  k_pos = len([st for st in mod_steps if st > EPS])
  k_neg = len([st for st in mod_steps if st < -EPS])
  if reward >= 0:
    return reward * (M - k_pos + 1) / (M+1)
  else:
    return reward * (M - k_neg + 1) / (M+1)

class Critic:

  def __init__(self, dconf):
    self.angv_bias = dconf['critic']['angv_bias']
    self.total_gain = dconf['critic']['total_gain']
    self.max_reward = dconf['critic']['max_reward']
    self.posRewardBias = 1.0
    if 'posRewardBias' in dconf['critic']:
      self.posRewardBias = dconf['critic']['posRewardBias']
    if 'posRewardBias' in dconf['net'] and dconf['net']['posRewardBias'] != 1.0:
      self.posRewardBias = dconf['net']['posRewardBias']
    self.modulate = None
    if 'modulation' in dconf['critic']:
      self.mod_steps = dconf['critic']['modulation']['steps']
      self.mod_queue = deque(maxlen=self.mod_steps)
      if dconf['critic']['modulation']['type'] == 'linear':
        self.modulate = _modulate_linear
    self.means = np.array([k['mean'] if 'mean' in k else 0.0 for k in dconf['env']['observation_map']])
    self.stds = np.array([k['std'] if 'std' in k else 1.0 for k in dconf['env']['observation_map']])

  def bad(self):
    return - self.max_reward

  # def _normalize(self, obs):
  #   return (obs - self.means) / self.stds
  # def _loss_inv(self, obs):
  #   _, _, ang, angv = self._normalize(obs)
  #   max_dev = 3.0 # its normalized
  #   dist = np.abs(ang) + np.abs(angv) * self.angv_bias
  #   max_dist = max_dev * (1 + self.angv_bias)
  #   # return np.exp((max_dist - min(dist, max_dist)) / max_dist)
  #   return (max_dist - min(dist, max_dist)) / max_dist

  def _loss(self, obs):
    _, _, ang, angv = obs
    return math.sqrt(ang*ang + self.angv_bias*angv*angv)

  def calc_reward(self, obs1, obs2=None, is_unk_move=False):
    _, _, ang1, angv1 = obs1
    if type(obs2) != np.ndarray or is_unk_move:
      reward = self.bad()
    elif np.abs(ang1) < 0.01 and np.abs(angv1) < 0.01:
        reward = self.max_reward
    else:
      d1 = self._loss(obs1)
      d2 = self._loss(obs2)

      reward = (d2 - d1) * self.total_gain
      # print(d1, d2, reward)

    if reward > 0:
      reward *= self.posRewardBias
    if self.modulate != None:
      new_reward = self.modulate(self.mod_queue, reward, self.mod_steps / 2)
      self.mod_queue.append(reward)
      reward = new_reward

    reward = min(max(-self.max_reward, reward), self.max_reward)

    return reward
