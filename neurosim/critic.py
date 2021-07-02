import math
import numpy as np

class Critic:

  def __init__(self, dconf):
    self.angv_bias = dconf['critic']['angv_bias']
    self.total_gain = dconf['critic']['total_gain']
    self.max_reward = dconf['critic']['max_reward']
    self.posRewardBias = 1.0
    if 'posRewardBias' in dconf['net'] and dconf['net']['posRewardBias'] != 1.0:
      self.posRewardBias = dconf['net']['posRewardBias']

  def bad(self):
    return -0.5 * self.max_reward

  def calc_reward(self, obs1, obs2=None, is_unk_move=False):
    _, _, ang1, angv1 = obs1
    if type(obs2) != np.ndarray or is_unk_move:
      return self.bad()
    _, _, ang2, angv2 = obs2

    # L2 distance to 0 of ang and angv
    d1 = math.sqrt(ang1*ang1 + self.angv_bias*angv1*angv1)
    d2 = math.sqrt(ang2*ang2 + self.angv_bias*angv2*angv2)
    dist_diff = d1 - d2
    reward = -dist_diff * self.total_gain # math.exp()
    reward = min(max(-self.max_reward, reward), self.max_reward)    
    if reward > 0:
      reward *= self.posRewardBias

    return reward
