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
    self.total_gain = dconf['critic']['total_gain']
    self.max_reward = dconf['critic']['max_reward']
    self.posRewardBias = 1.0
    self.negRewardBias = 1.0
    if 'posRewardBias' in dconf['critic']:
      self.posRewardBias = dconf['critic']['posRewardBias']
    if 'posRewardBias' in dconf['net'] and dconf['net']['posRewardBias'] != 1.0:
      self.posRewardBias = dconf['net']['posRewardBias']
    if 'negRewardBias' in dconf['critic']:
      self.negRewardBias = dconf['critic']['negRewardBias']
    if 'negRewardBias' in dconf['net'] and dconf['net']['negRewardBias'] != 1.0:
      self.negRewardBias = dconf['net']['negRewardBias']
    self.clear_flips = dconf['critic']['clear_flips'] if 'clear_flips' in dconf['critic'] else 0
    self.modulate = None
    if 'modulation' in dconf['critic']:
      self.mod_steps = dconf['critic']['modulation']['steps']
      self.mod_queue = deque(maxlen=self.mod_steps)
      if dconf['critic']['modulation']['type'] == 'linear':
        self.modulate = _modulate_linear
    self.biases = False
    if 'biases' in dconf['critic']:
      biases = dconf['critic']['biases']
      self.biases = True
      self.biases_vec = np.array([biases[k] if k in biases else 1.0 for k in ['loc', 'vel', 'ang', 'angv']])
    elif 'angv_bias' in dconf['critic']:
      self.angv_bias = dconf['critic']['angv_bias']

    self.means = np.array([k['mean'] if 'mean' in k else 0.0 for k in dconf['env']['observation_map']])
    self.stds = np.array([k['std'] if 'std' in k else 1.0 for k in dconf['env']['observation_map']])

  def bad(self):
    if self.posRewardBias:
      return -self.max_reward / self.posRewardBias
    return -self.max_reward

  def _normalize(self, obs):
    return (obs - self.means) / self.stds

  def _loss(self, obs):
    if self.biases: # this is configuring the whole observation space
      return np.sqrt(np.sum((self._normalize(obs) * self.biases_vec) ** 2))
    else:
      # old method of looking at only ang and angv
      _, _, ang, angv = obs
      return math.sqrt(ang*ang + self.angv_bias*angv*angv)

  def _balanced(self, obs):
    _, _, ang, angv = obs
    return np.abs(ang) < 0.01 and np.abs(angv) < 0.01

  def calc_reward(self, curr_obs, prev_obs=None, is_unk_move=False):
    if type(prev_obs) != np.ndarray:
      if type(curr_obs) != np.ndarray:
        raise Exception('Wrong format for observations!')
      return 0.0
    elif self._balanced(prev_obs):
      return 0.0
    elif is_unk_move:
      reward = self.bad()
    elif self._balanced(curr_obs):
        reward = self.max_reward
    elif self.clear_flips and curr_obs[3] * prev_obs[3] < 0:
      return 0.0
    else:
      curr_loss = self._loss(curr_obs)
      prev_loss = self._loss(prev_obs)

      reward = (prev_loss - curr_loss) * self.total_gain

    if reward > 0:
      reward *= self.posRewardBias
    if reward < 0:
      reward *= self.negRewardBias
    if self.modulate != None:
      new_reward = self.modulate(self.mod_queue, reward, self.mod_steps / 2)
      self.mod_queue.append(reward)
      reward = new_reward

    reward = min(max(-self.max_reward, reward), self.max_reward)
    return reward
