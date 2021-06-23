import math
import numpy as np

def calc_reward(obs1, obs2=None):
  _, _, ang1, angv1 = obs1
  if type(obs2) != np.ndarray:
    return abs(ang1)
  _, _, ang2, angv2 = obs2
  angv_gain = 64.0

  # L2 distance to 0 of ang and angv
  d1 = math.sqrt(ang1*ang1 + angv_gain*angv1*angv1)
  d2 = math.sqrt(ang2*ang2 + angv_gain*angv2*angv2)
  dist_diff = d1 - d2
  reward = -dist_diff # math.exp()
  # print(d1, d2, dist_diff, reward)
  # print('here!!')
  return reward
