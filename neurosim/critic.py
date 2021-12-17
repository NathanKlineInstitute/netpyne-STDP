import math
import numpy as np
from collections import deque

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

  def calc_reward(self, curr_action, idealMove=0):
    if type(curr_action) != np.ndarray:
      raise Exception('Wrong format for observations!')
      return 0.0
    elif idealMove==0:
      if curr_action==0:
        reward = 0.5
      elif curr_action==1:
        reward = -0.25
      elif curr_action==2:
        reward = -0.5
    elif idealMove==1:
      if curr_action==0:
        reward = -0.25
      elif curr_action==1:
        reward = 0.5
      elif curr_action==2:
        reward = -0.5
    elif idealMove==2:
      if curr_action==0:
        reward = -0.5
      elif curr_action==1:
        reward = -0.5
      elif curr_action==2:
        reward = 0.5
    else:
      raise Exception('Wrong format for idealMove!')
      return 0.0
      
    reward = (reward) * self.total_gain

    if reward > 0:
      reward *= self.posRewardBias
    if reward < 0:
      reward *= self.negRewardBias

    reward = min(max(-self.max_reward, reward), self.max_reward)
    return reward
