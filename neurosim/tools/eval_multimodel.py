import fire
import os
import json
import csv
import math
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from neurosim.tools.utils import _get_spike_aggs_all, _extract_sorted_min_ids

RANDOM_EVALUATION='results/random_cartpole_ActionsPerEpisode.txt'

def _get_params(config):
  params = {}
  def _extract(j, pname):
    if type(j) == dict:
      for k,v in j.items():
        _extract(v, pname + ':' + k)
    elif type(j) == list:
      for idx,v in enumerate(j):
        _extract(v, pname + ':' + str(idx))
    else:
      params[pname] = j
  _extract(config, '')
  return params


def _extract(wdir, path_prefix=None):
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

def _extract_params(wdirs, configs):
  all_params = []
  keys = []
  for wdir, config in zip(wdirs, configs):
    params = _get_params(config)
    all_params.append(params)
    keys.extend(params.keys())

  keys = sorted(list(set(keys)))
  return all_params, keys

def trace(wdir):
  wdirs, configs = _extract(wdir)

  print('wdirs:')
  for wdir in wdirs:
    print(wdir)

  all_params, keys = _extract_params(wdirs, configs)

  print('params:')
  for key in keys:
    values = [params[key] if key in params else None for params in all_params]
    if len(set(values)) > 1:
      print(key, values)

def _get_evaluation_dir(wdir, idx):
  assert idx in [0, -1], 'Make sure this is getting the correct ts instead of index'
  evaluations = [
    (os.path.join(wdir, fname), int(fname.replace('evaluation_', '')))
    for fname in os.listdir(wdir) if fname.startswith('evaluation_') and 'display' not in fname]
  eval_dir, eval_ts = sorted(evaluations, key=lambda x:x[1])[idx]
  return eval_dir

def _evaluation_actions_per_episode(wdir, idx):
  eval_dir = _get_evaluation_dir(wdir, idx)
  if os.path.isfile(os.path.join(eval_dir, 'ActionsPerEpisode.txt')):
    with open(os.path.join(eval_dir, 'ActionsPerEpisode.txt')) as f:
      return [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]
  return None


def boxplot(wdirs, outdir, include_random=True):
  outputfile = os.path.join(outdir, 'evaluations_boxplot.png')

  wdirs = [wdir.split(':') for wdir in wdirs.split(',')]
  results = []
  if include_random:
    with open(RANDOM_EVALUATION) as f:
      name = 'Random Choice'
      results.append([name, [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]])
  for name,wdir,idx in wdirs:
    results.append([name, _evaluation_actions_per_episode(wdir, int(idx))])


  labels = [k for k,v in results]
  data = [v for k,v in results]

  fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(111)
  # bp = ax.boxplot(data)
  ax = sns.boxplot(data=data)
  ax = sns.swarmplot(data=data, color=".25")
  ax.set_xticklabels(labels, rotation=10)
  ax.set_ylabel('steps/actions per episode')
  ax.set_title('Evaluation of models')

  plt.grid(axis='y')
  plt.tight_layout()
  plt.savefig(outputfile)

def spiking_frequencies(wdirs, outdir):
  outputfile = os.path.join(outdir, 'spiking_frequencies.tsv')

  if type(wdirs) == str:
    wdirs = wdirs.split(',')
  wdirs = [wdir.split(':') for wdir in wdirs]

  header = None
  table = []
  for name,wdir,idx in tqdm(wdirs):
    eval_wdir = _get_evaluation_dir(wdir, int(idx))
    sim_config = os.path.join(eval_wdir, 'sim.pkl')
    with open(sim_config, 'rb') as f:
      sim = pkl.load(f)
    dconf_path = os.path.join(eval_wdir, 'backupcfg_sim.json')
    with open(dconf_path, 'r') as f:
      dconf = json.load(f)
    pop_sizes = dconf['net']['allpops']
    sorted_min_ids = _extract_sorted_min_ids(sim, dconf, separate_movement=True)
    pop_aggs = _get_spike_aggs_all(sim, sorted_min_ids)
    total_time = math.ceil(sim['simData']['spkt'][-1])
    pop_aggs['ALL'] = sum([nr for _,nr in pop_aggs.items()])
    pop_sizes['ALL'] = sum(pop_sizes.values())
    if not header:
      header = [pop for pop in pop_sizes.keys() if pop in pop_aggs]
    spk_freq = [pop_aggs[h] / (total_time / 1000) / pop_sizes[h] for h in header]
    table.append([name] + [round(spk, 2) for spk in spk_freq])

  with open(outputfile, 'w') as out:
    writer = csv.writer(out, delimiter='\t')
    writer.writerow([''] + header)
    for row in table:
      writer.writerow(row)


if __name__ == '__main__':
  fire.Fire({
      'trace': trace,
      'boxplot': boxplot,
      'spk-freq': spiking_frequencies
  })
