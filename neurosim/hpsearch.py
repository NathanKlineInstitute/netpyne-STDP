import os
import shutil
import csv
import fire
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

from neurosim.main import main

MEDIAN_STEPS = [21,51,101]

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

def _config_setup(config, sample, outputdir):
  # Setup the config:
  #   - so it writes to the correct place
  #   - it has the correct hyperparameters
  for param_name, param_val in sample.items():
    if param_name != 'run_id':
      config = _config_replace(config, param_name, param_val)

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
  return results

def _pseudo_random(digits=7, iters=2):
  # Used the Middle square method (MSM) for generating pseudorandom numbers
  # use own random function to not mess up with the seed
  current_time = str(datetime.now().timestamp()).replace('.', '')
  seed = int(current_time[-digits:])
  def _iter(nr):
    new_nr = str(nr * nr)
    imin = max(int((len(new_nr) - 7) / 2), 1)
    imax = min(len(new_nr)-1, imin + digits)
    return int(('0' * digits) + new_nr[imin:imax])
  nr = seed
  for _ in range(iters):
    nr = _iter(nr)
  return nr


def sample_run(
    outputdir,
    hpconfig_file='hpsearch_config.json',
    config_file='config.json',
    just_init=False):

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
      writer.writerow(['run_id'] + ['max_median_s{}'.format(step) for step in MEDIAN_STEPS])
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
  run_ids = [rid for rid, run in enumerate(allruns) if run[0] not in sampled]
  run_id = run_ids[_pseudo_random() % len(run_ids)]
  print('Picked {}:'.format(run_id))
  sample = dict(zip(header, allruns[run_id]))
  print(sample)

  # setup the config
  config = _config_setup(config, sample, outputdir)
  main(config)

  # write results then exit
  medians = _actions_medians(config['sim']['outdir'], MEDIAN_STEPS)
  with open(results_tsv, 'a')as out:
    writer = csv.writer(out, delimiter='\t')
    writer.writerow([run_id] + medians)


if __name__ == '__main__':
  fire.Fire({
      'sample': sample_run
  })
