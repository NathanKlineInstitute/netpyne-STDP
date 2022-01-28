import os
import json
import numpy as np

def _get_agg(tr_results, step, func, overlap=True):
  tr_agg = []
  for idx in range(0, len(tr_results) - step, 1 if overlap else step):
    tr_agg.append(func(tr_results[idx:idx+step]))
  return tr_agg

def _get_avg_fast(tr_results, step):
  tr_agg = []
  current_sum = sum(tr_results[:step])
  current_avg_by = step
  tr_agg.append(current_sum / current_avg_by)
  for idx in range(1, len(tr_results) - step):
    current_sum += tr_results[idx+step-1] - tr_results[idx-1]
    tr_agg.append(current_sum / current_avg_by)
  return tr_agg

def _extract_hpsteps(wdir, path_prefix=None):
  configs = []
  wdirs = [wdir]
  while True:
    current_wdir = wdirs[-1]
    if path_prefix:
      current_wdir = os.path.join(path_prefix, current_wdir)
    config_fname = os.path.join(current_wdir, 'backupcfg_sim.json')
    with open(config_fname) as f:
      config = json.load(f)
    configs.append(config)
    if config['simtype']['ResumeSim']:
      wdirs.append(
        os.path.dirname(config['simtype']['ResumeSimFromFile']))
    else:
      break

  configs.reverse()
  wdirs.reverse()
  return wdirs, configs
