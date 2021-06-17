import math
import numpy as np
from scipy.special import expit
from scipy.stats import norm

func_linear = lambda x:x
func_log = lambda x: np.log(x)
func_sigmoid = lambda x: expit(x)
func_sigmoid_div = lambda y: lambda x: expit(x / y)

def _map_observation_to_fr(
    value, minVal, maxVal, minRate, maxRate,
    prevValue=None, func=lambda x:x):
  val = value
  if prevValue:
    val = value - prevValue
  val = max(minVal, min(maxVal, val)) # make sure its within bounds
  norm_val = (func(val) - func(minVal)) / (func(maxVal) - func(minVal))
  return norm_val * (maxRate - minRate) + minRate

def _map_observation_to_rf(
    value, minRate, maxRate, func, prevValue=None):
  val = value
  if prevValue:
    val = value - prevValue
  bin_vals = func(val) # value in [0, 1] for each bin
  return [bin_val * (maxRate - minRate) + minRate for bin_val in bin_vals]

def _parse_obs_map(func_def):
  if func_def['type'].startswith('rf_') or func_def['type'] == 'linear':
    return func_linear
  if func_def['type'] == 'log':
    return func_log
  if func_def['type'] == 'sigmoid':
    return func_sigmoid
  if func_def['type'] == 'sigmoid_div':
    return func_sigmoid_div(func_def['scale'])
  raise Exception('Cannot parse func_def {}'.format(func_def))

def _chain(funcs):
  def _f(x):
    val = x
    for f in funcs:
      val = f(val)
    return val
  return _f

def _onehot(bins):
  def _f(x):
    return [1.0 if x == b else 0.0 for b in range(bins)]
  return _f

def _evensplit_bin_f(imin, step):
  def _f(x):
    return math.floor((x - imin) / step)
  return _f

def _intervals_bin_f(bins, fields_intervals, mean=0.0, std=1.0):
  assert len(fields_intervals) == bins + 1
  def _f(x):
    normalized = (x - mean) / std
    fbin = np.searchsorted(fields_intervals, normalized) - 1
    assert fbin != -1 and fbin != bins, "Value {} (normalized {}) outside of distribution".format(x, normalized)
    return fbin
  return _f

def _parse_rf_map(obs):
  if not obs['type'].startswith('rf_'):
    return None
  bins = obs['bins']
  _func = None
  if obs['type'] == 'rf_evensplit':
    imin, imax = obs['min'], obs['max']
    step = (imax - imin) / bins
    _func = _evensplit_bin_f(imin, step)
  elif obs['type'] == 'rf_normal':
    EPS = 0
    fields_intervals = norm.ppf(np.linspace(EPS, 1.0 - EPS, bins + 1))
    _func = _intervals_bin_f(bins, fields_intervals, obs['mean'], obs['std'])
  elif obs['type'] == 'rf_intervals':
    _func = _intervals_bin_f(bins,
      obs['intervals'],
      obs['mean'] if 'mean' in obs else 0.0,
      obs['std'] if 'std' in obs else 1.0)
  else:
    raise Exception('Unknown Receptive Field function {}'.format(obs['type']))
  return _chain([_func, _onehot(bins)])

class GameInterface:

  def __init__(self, aigame, config):
    self.AIGame = aigame
    self.obs_map = config['env']['observation_map']
    self.obs_func = [_parse_obs_map(func_def) for func_def in config['env']['observation_map']]
    self.obs_rf = [_parse_rf_map(func_def) for func_def in config['env']['observation_map']]
    self.inputMaxRate = config['net']['InputMaxRate']
    self.inputPop = config['net']['inputPop']
    self.inputPopSize = config['net']['allpops'][self.inputPop]

  def _get_limit(self, idx, limit_type='min'):
    if limit_type in self.obs_map[idx]:
      return self.obs_map[idx][limit_type]
    else:
      obs_space = self.AIGame.env.observation_space
      if limit_type == 'min':
        return obs_space.low[idx]
      elif limit_type == 'max':
        return obs_space.high[idx]


  def input_firing_rates(self):
    vals = []
    for idx, obsVal in enumerate(self.AIGame.observations[-1]):
      if self.obs_map[idx]['type'].startswith('rf_'):
        # Create a receptive field (rf)
        vals.extend(_map_observation_to_rf(obsVal,
          minRate=0, maxRate=self.inputMaxRate,
          func=self.obs_rf[idx]))
      else:
        vals.append(_map_observation_to_fr(obsVal,
          minVal=self._get_limit(idx, 'min'),
          maxVal=self._get_limit(idx, 'max'),
          minRate=0, maxRate=self.inputMaxRate,
          func=self.obs_func[idx]))

    assert len(vals) <= self.inputPopSize, 'Population size is smaller than needed inputs'
    rates = np.tile(
      np.array(vals),
      int(self.inputPopSize / len(vals)) + 1)[:self.inputPopSize]
    return rates
