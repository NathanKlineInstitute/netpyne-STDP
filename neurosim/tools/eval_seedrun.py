import os
import fire
import csv
import json
import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from neurosim.tools.eval_multimodel import _get_agg, _extract_hpsteps

def analyze_samples(wdir, modifier=None):
  seedruns = [(wd.replace('run_seed', ''), os.path.join(wdir, wd))
    for wd in os.listdir(wdir) if wd.startswith('run_seed')]

  if modifier:
    seedruns = [(seed, os.path.join(seedrun_f, modifier)) for seed, seedrun_f in seedruns]

  actions = {}
  for seed, seedrun_f in seedruns:
    acts_f = os.path.join(seedrun_f, 'ActionsPerEpisode.txt')
    if os.path.isfile(acts_f):
      actions[seed] = []
      with open(acts_f) as f:
        for row in csv.reader(f, delimiter='\t'):
          if len(row) > 0:
            actions[seed].append(int(float(row[1])))


  akeys = actions.keys()

  aggs_by_step = []
  for step in [21, 51, 101]:
      aggs = [max(_get_agg(actions[k], step, np.median)) for k in akeys]
      aggs_by_step.append(aggs)
      
  aggs = [max(_get_agg(actions[k], 101, np.average)) for k in akeys]
  aggs_by_step.append(aggs)

  labels = ['median over {}steps'.format(s) for s in [21,51,101]] + ['average over 101steps']

  maxes = [200, 300, 500, 501]
  mins = [25, 0]
  curr_min, curr_max = (0, 0)
  for agg_arr in aggs_by_step:
    for agg in list(agg_arr):
      while maxes[curr_max] < agg:
        curr_max += 1
      while mins[curr_min] > agg:
        curr_min += 1

  plt.figure(figsize=(10,6))
  plt.grid(axis='y', alpha=0.3)
  ax = sns.boxplot(data=aggs_by_step)
  ax.set_xticklabels(labels, rotation=5)
  ax.set_ylim(mins[curr_min], maxes[curr_max])
  plt.title('Variance in performance of {} different initial network configurations'.format(len(akeys)))
  # plt.show()
  outputfile = os.path.join(wdir, 'eval_seeds{}.png'.format('' if not modifier else '_' + modifier.replace('/', '_')))
  plt.savefig(outputfile)

# def create_table(wdir, outputfile=None, runs_json_fname='runs.json'):
#   runs_json = os.path.join(wdir, runs_json_fname)
#   results_tsv = os.path.join(wdir, 'results.tsv')

#   outputfile = os.path.join(wdir, 'results_table.tsv')

#   runs = {}
#   with open(runs_json) as f:
#     for line in f:
#       run = json.loads(line)
#       runs[run['run_id']] = run

#   results = []
#   with open(results_tsv) as f:
#     for row in csv.DictReader(f, delimiter='\t'):
#       for k in row.keys():
#         if k.startswith('max_'):
#           row[k] = float(row[k])
#         elif k == 'run_id':
#           row[k] = int(row[k])
#       run_id = row['run_id']
#       for k,v in runs[run_id].items():
#         row[k] = v
#       results.append(row)
#   del runs
#   print('Found {} results'.format(len(results)))

#   results = sorted(results, key=lambda x:x['max_average_s100'], reverse=True)
#   with open(outputfile, 'w') as out:
#     writer = csv.writer(out, delimiter='\t')
#     header = [k for k in results[0].keys()]
#     writer.writerow(header)
#     for row in results:
#       writer.writerow([row[h] for h in header])


def _extract_steps_per_ep(wdir, steps, stype):
  assert stype in ['avg', 'median']
  all_wdir_steps, _ = _extract_hpsteps(wdir)

  training_results = []
  for wdir_steps in all_wdir_steps:
    with open(os.path.join(wdir_steps, 'ActionsPerEpisode.txt')) as f:
        training_results.extend([int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')])

  func = np.average if stype == 'avg' else np.median
  return _get_agg(training_results, steps, func)


def steps_per_eps(wdir, steps=100, stype='avg', modifier=None, parts=1):
  assert stype in ['avg', 'median']

  seedruns = [(wd.replace('run_seed', ''), os.path.join(wdir, wd))
    for wd in os.listdir(wdir) if wd.startswith('run_seed')]

  if modifier:
    seedruns = [(seed, os.path.join(seedrun_f, modifier)) for seed, seedrun_f in seedruns]

  for p in range(parts):
    seedruns_batch = seedruns
    if parts > 1:
      pstep = int(len(seedruns) / parts)
      seedruns_batch = seedruns[p * pstep:min((p+1)*pstep, len(seedruns))]

    plt.figure(figsize=(10,10))
    plt.ylim(0, 200)
    plt.grid(axis='y')
    plt.ylabel('steps/actions per episode')
    plt.title('Performance during training, evaluated using {} over {} episodes'.format(stype, steps))
    for seed, seedrun_wdir in seedruns_batch:
      tr_res = _extract_steps_per_ep(seedrun_wdir, steps, stype)
      plt.plot(tr_res)

    plt.legend([seed for seed,_ in seedruns_batch])
    plt.xlabel('episode')

    outputfile = os.path.join(
      wdir,
      'steps_per_episode{}_{}_{}_{}.png'.format(
        '' if parts == 1 else '_{}of{}'.format(p+1, parts),
        stype, steps, modifier.replace('/', '_')))
    plt.savefig(outputfile)


if __name__ == '__main__':
  fire.Fire({
      'analyze': analyze_samples,
      'train-perf': steps_per_eps,
      # 'combine': create_table
  })