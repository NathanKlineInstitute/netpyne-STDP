import fire
import os
import json
import csv
import math
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd

from neurosim.tools.utils import _get_spike_aggs_all, _extract_sorted_min_ids, _group_by_pop
from neurosim.tools.eval_multimodel import _extract_hpsteps, _extract_params

def _extract_stdp_weights(wdir, preduration):
  with open(os.path.join(wdir, 'synWeights.pkl'), 'rb') as f:
    synWeights = pkl.load(f)

  sim_config = os.path.join(wdir, 'sim.pkl')
  with open(sim_config, 'rb') as f:
    sim = pkl.load(f)

  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')
  with open(dconf_path, 'r') as f:
    dconf = json.load(f)

  sorted_min_ids = _extract_sorted_min_ids(sim, dconf, separate_movement=True)
  popWeights = _group_by_pop(synWeights, sorted_min_ids)
  recordWeightStepSize = dconf['sim']['recordWeightStepSize']
  final_weights = {}
  for conn_name, conns in popWeights.items():
    final_weights[conn_name] = {}
    for idx, weights in enumerate(conns):
      final_weights[conn_name][idx * recordWeightStepSize + preduration] = weights
  return final_weights

def stdp_weight_changes(wdirs, outdir, indices=[-1], before_training=True, log_weights=True):
  outputfile = os.path.join(outdir, 'weight_changes.png')
  if type(wdirs) == str:
    wdirs = wdirs.split(',')
  wdirs = [wdir.split(':') for wdir in wdirs]

  wdir_weights = []
  keys = []
  for _,wdir,_ in wdirs:
    all_weights = {}
    wdir_steps, configs = _extract_hpsteps(wdir)
    all_params, _ = _extract_params(wdir_steps, configs)
    key = ":sim:duration"
    # TODO: Take the count of steps
    durations = [params[key] if key in params else None for params in all_params]
    cumm_duration = 0
    for wdir_step, duration in zip(wdir_steps, durations):
      weights = _extract_stdp_weights(wdir_step, cumm_duration)
      for conn_name, conns in weights.items():
        if conn_name not in all_weights:
          all_weights[conn_name] = {}
        for idx, wghts in conns.items():
          all_weights[conn_name][idx] = wghts
      cumm_duration += duration
    wdir_weights.append(all_weights)
    keys.extend(all_weights.keys())

  keys = sorted(list(set(keys)))
  figsize = 21
  ncols = 3
  nrows = math.ceil(len(keys) / ncols)
  if nrows == 1:
    ncols = len(keys)
  _, axs = plt.subplots(
    ncols=ncols, nrows=nrows,
    figsize=(figsize, int(figsize * (nrows / ncols))))
  plt.suptitle('Synaptic Weight distribution before/after training')

  conn_idx = 0
  for axi in axs:
    if nrows == 1:
      axi = [axi]
    for ax in axi:
      nbins = 50
      LOG_PRECISION = -2
      if conn_idx == len(keys):
        continue
      conn = keys[conn_idx]

      curr_legend = []
      wmin, wmax = 1e10, 0
      for widx, (wdir_name, _, _), weights_map in zip(range(len(wdirs)), wdirs, wdir_weights):
        before_index = [0] if widx == 0 and before_training else []
        w_ts = sorted(list(weights_map[conn].keys()))
        for index in before_index + indices:
          ts = w_ts[index]
          weights = weights_map[conn][ts]
          wmin = min(wmin, np.min(weights))
          wmax = max(wmax, np.max(weights))
      if log_weights:
        nbins = 10 ** np.linspace(LOG_PRECISION, np.log10(wmax), nbins)
        nbins = [0.0] + list(nbins)
      for widx, (wdir_name, _, _), weights_map in zip(range(len(wdirs)), wdirs, wdir_weights):
        before_index = [0] if widx == 0 and before_training else []
        w_ts = sorted(list(weights_map[conn].keys()))
        for index in before_index + indices:
          ts = w_ts[index]
          weights = weights_map[conn][ts]
          hist, bins = np.histogram(weights, bins=nbins, range=(wmin, wmax))
          xs = (bins[:-1] + bins[1:])/2
          ax.plot(xs, hist, alpha=0.8)
          if index == 0:
            curr_legend.append('Before Training')
          else:
            curr_legend.append(wdir_name)

      ax.set_title('{} weight distribution'.format(conn))
      ax.set_xlabel('Synaptse Weight values (log scale)')
      ax.set_ylabel('Count of synapses')
      ax.grid(alpha=0.5)
      if log_weights:
        ax.set_xscale('log')
      ax.legend(curr_legend)
      conn_idx += 1

  plt.tight_layout()
  plt.savefig(outputfile)

if __name__ == '__main__':
  fire.Fire({
      'changes': stdp_weight_changes,
  })
