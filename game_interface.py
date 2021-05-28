import numpy as np

func_linear = lambda x:x
func_log = lambda x: np.log(x)
func_sigmoid = lambda x: 1/(1 + np.exp(-x))
func_sigmoid_div = lambda y: lambda x: 1/(1 + np.exp(-x / y))

def _map_observation(
    value, minVal, maxVal, minRate, maxRate,
    prevValue=None, func=lambda x:x):
  val = value
  if prevValue:
    val = value - prevValue
  norm_val = (func(val) - func(minVal)) / (func(maxVal) - func(minVal))
  return norm_val * (maxRate - minRate) + minRate

def _parse_obs_map(func_def):
  if func_def['type'] == 'linear':
    return func_linear
  if func_def['type'] == 'log':
    return func_log
  if func_def['type'] == 'sigmoid':
    return func_sigmoid
  if func_def['type'] == 'sigmoid_div':
    return func_sigmoid_div(func_def['scale'])
  raise Exception('Cannot parse func_def {}'.format(func_def))

class GameInterface:

  def __init__(self, aigame, config):
    self.AIGame = aigame
    self.obs_space = self.AIGame.env.observation_space
    self.obs_map = [_parse_obs_map(func_def) for func_def in config['env']['observation_map']]
    self.inputMaxRate = config['net']['InputMaxRate']
    # self.inputPop = config['net']['InputPop']
    # self.inputPopSize = config['net']['allpops'][self.inputPop]

  def input_firing_rates(self):
    return [_map_observation(obsVal,
      minVal=self.obs_space.low[idx],
      maxVal=self.obs_space.high[idx],
      minRate=0, maxRate=self.inputMaxRate,
      func=self.obs_map[idx])
    for idx, obsVal in enumerate(self.AIGame.observations[-1])]

    # rates = np.tile(
    #   np.array(vals),
    #   int(self.inputPopSize / len(vals)) + 1)[:self.inputPopSize]
    # return rates
