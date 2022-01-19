import os
import shutil
import csv
import fire
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
import pickle as pkl

from conf import init_wdir
from sim import NeuroSim
from neurosim.tools.utils import _get_pop_name, _extract_sorted_min_ids
from neurosim.utils.random import pseudo_random

MEDIAN_STEPS = [21,51,101]
FREQ_POPS = ['EM', 'EA']

def _get_runs(keys, params, conditions):
  # Generate all the runs based on the hyperparam search config
  runs = []
  arr = [None for k in keys]
  vals = {}
  newkeys = [k.replace(':', '_').replace('-', '_') for k in keys]
  total_cnt = np.prod([len(v) for k,v in params.items()])
  
  conditions_map = {}
  for cond, index in conditions:
    if index not in conditions_map:
      conditions_map[index] = []
    conditions_map[index].append(cond)

  def _validate(index):
    # make sure it satisfies all conditions at this level
    if index in conditions_map:
      for cond in conditions_map[index]:
        if not eval(cond, vals):
          return False
    return True
  def _create_run(index=0):
    # Recursive call to generate all hpsearch runs
    if _validate(index):
      if index < len(keys):
        for val in params[keys[index]]:
          arr[index] = val
          vals[newkeys[index]] = val
          _create_run(index + 1)
      else:
        runs.append(list(arr))

  _create_run()
  return runs

def _config_replace(config, param_name, param_val):
  # replace a param_name to param_value in config
  toks = param_name.split(':')
  def _repl(curr_obj, tok_id=0):
    if type(curr_obj) == dict:
      curr_obj[toks[tok_id]] = _repl(curr_obj[toks[tok_id]], tok_id+1)
      return curr_obj
    else:
      return param_val

  return _repl(config)

def _config_setup(config, sample, outputdir, random_network=False):
  # Setup the config:
  #   - so it writes to the correct place
  #   - it has the correct hyperparameters
  for param_name, param_val in sample.items():
    if param_name != 'run_id':
      config = _config_replace(config, param_name, param_val)

  if random_network:
    conn_seed = pseudo_random()
    config['sim']["seeds"] = {"conn": conn_seed, "stim": 1, "loc": 1}

  config['sim']['outdir'] = os.path.join(outputdir, 'run_{}'.format(sample['run_id']))
  return config



def _actions_medians(wdir, steps):
  # Extract the max of the medians
  with open(os.path.join(wdir, 'ActionsPerEpisode.txt')) as f:
      training_results = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

  results = []
  for STEP in steps:
      best = 0
      for idx in range(len(training_results) - STEP):
          best = max(np.median(training_results[idx:idx+STEP]), best)
      results.append(best)
  # Get the average over 100 steps
  avg_steps = 100
  best_avg = 0
  for idx in range(len(training_results) - avg_steps):
      best_avg = max(np.average(training_results[idx:idx+avg_steps]), best_avg)
  results.append(best_avg)
  return results

def _frequencies(wdir, freq_pops):
  sim_config = os.path.join(wdir, 'sim.pkl')
  with open(sim_config, 'rb') as f:
    sim = pkl.load(f)

  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')
  with open(dconf_path, 'r') as f:
    dconf = json.load(f)

  sorted_min_ids = _extract_sorted_min_ids(sim, dconf, separate_movement=False)
  
  spkid = sim['simData']['spkid']
  spkt = sim['simData']['spkt']

  spike_aggs = dict([(p, 0) for p in freq_pops])
  for cid, ct in zip(spkid, spkt):
    pop = _get_pop_name(cid, sorted_min_ids)
    if pop in spike_aggs:
      spike_aggs[pop] += 1
  duration = dconf['sim']['duration']
  pop_size = dconf['net']['allpops']
  return [float(spike_aggs[p]) / duration / pop_size[p] for p in freq_pops]


def sample_run(
    outputdir,
    hpconfig_file='hpsearch_config.json',
    config_file='config.json',
    just_init=False,
    random_network=False,
    init_seed_dir=False):

  runs_tsv = os.path.join(outputdir, 'runs.tsv')
  runs_json = os.path.join(outputdir, 'runs.json')
  config_json = os.path.join(outputdir, 'init_config.json')
  results_tsv = os.path.join(outputdir, 'results.tsv')
  hpconfig2_file = os.path.join(outputdir, 'hpsearch_config.json')

  if not os.path.isdir(outputdir):
    os.makedirs(outputdir)

  if os.path.isfile(hpconfig2_file):
    hpconfig_file = hpconfig2_file

  if os.path.isfile(config_json):
    config_file = config_json

  with open(hpconfig_file) as f:
    hpconf = json.load(f)

  if just_init or sum([
      not os.path.isfile(runs_tsv) for fname in [runs_tsv, runs_json, config_json, results_tsv, hpconfig2_file]]) > 0:
    # Generate all runs
    param_keys = list(hpconf['params'].keys())
    allruns = _get_runs(param_keys, hpconf['params'], hpconf['conditions'])
    allruns = [[i] + run for i, run in enumerate(allruns)]

    header = ['run_id'] + param_keys
    with open(runs_tsv, 'w') as out:
      writer = csv.writer(out, delimiter='\t')
      writer.writerow(header)
      for run in allruns:
        writer.writerow(run)

    with open(runs_json, 'w') as out:
      for run in allruns:
        out.write(json.dumps(dict(zip(header, run))) + '\n')

    with open(config_file) as f:
      config = json.load(f)      
    for param_name in hpconf['params'].keys():
      config = _config_replace(config, param_name, None)
    if not os.path.isfile(config_json):
      with open(config_json, 'w') as out:
        out.write(json.dumps(config, indent=4))

    if not os.path.isfile(hpconfig2_file):
      shutil.copy(hpconfig_file, hpconfig2_file)

    with open(results_tsv, 'w') as out:
      writer = csv.writer(out, delimiter='\t')
      writer.writerow(
        ['run_id'] + \
        ['max_median_s{}'.format(step) for step in MEDIAN_STEPS] + \
        ['max_average_s100'] + \
        ['freq_{}'.format(pop) for pop in FREQ_POPS])
    if just_init:
      return
  else:
    # If already created, just extract from the json file
    allruns = []
    header = None
    with open(runs_json) as f:
      for line in f:
        j = json.loads(line)
        if not header:
          header = j.keys()
        allruns.append([j[h] for h in header])

    with open(config_json) as f:
      config = json.load(f)
  
  # sample a new run based on previous runs
  sampled = dict([
    (int(dir_fie.replace('run_', '')), True)
    for dir_fie in os.listdir(outputdir)
    if os.path.isdir(os.path.join(outputdir, dir_fie)) and dir_fie.startswith('run_')])
  run_cnts = [rid for rid, run in enumerate(allruns) if run[0] not in sampled]
  run_sel = run_cnts[pseudo_random() % len(run_cnts)]
  sample = dict(zip(header, allruns[run_sel]))
  run_id = sample['run_id']
  print('Picked {}:'.format(run_id))
  print(sample)

  # setup the config
  config = _config_setup(config, sample, outputdir, random_network)
  if init_seed_dir:
    seed_dirs = [sdir for sdir in os.listdir(init_seed_dir) if sdir.startswith('run_seed')]
    seed_dir_id = pseudo_random() % len(seed_dirs)
    seed_dir = os.path.join(init_seed_dir, seed_dirs[seed_dir_id], 'synWeights.pkl')
    config['simtype']['ResumeSimFromFile'] = seed_dir
  
  # Copied from main.py so that I can trigger SysExit with save
  outdir = config['sim']['outdir']
  if os.path.isdir(outdir):
    evaluations = [fname
      for fname in os.listdir(outdir)
      if fname.startswith('evaluation_') and os.path.isdir(os.path.join(outdir, fname))]
    if len(evaluations) > 0:
      raise Exception(' '.join([
          'You have run evaluations on {}: {}.'.format(outdir, evaluations),
          'This will rewrite!',
          'Please delete to continue!']))
  init_wdir(config)
  runner = NeuroSim(config)
  try:
    runner.run()
  except SystemExit:
    print('Early stopping!')
    runner.save()

  # write results then exit
  medians = _actions_medians(config['sim']['outdir'], MEDIAN_STEPS)
  freqs = _frequencies(config['sim']['outdir'], FREQ_POPS)
  with open(results_tsv, 'a')as out:
    writer = csv.writer(out, delimiter='\t')
    writer.writerow([run_id] + medians + freqs)


if __name__ == '__main__':
  fire.Fire({
      'sample': sample_run
  })
