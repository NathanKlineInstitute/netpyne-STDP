import os
import fire
import csv
import json
import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from neurosim.utils.agg import _get_agg, _extract_hpsteps


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
  # print(aggs_by_step)
  ax = sns.boxplot(data=aggs_by_step)
  ax.set_xticklabels(labels, rotation=5)
  ax.set_ylim(mins[curr_min], maxes[curr_max])
  plt.title('Variance in performance of {} different initial network configurations'.format(len(akeys)))
  # plt.show()
  outputfile = os.path.join(wdir, 'eval_seeds{}.png'.format('' if not modifier else '_' + modifier.replace('/', '_')))
  plt.savefig(outputfile)


def show_analyze_once(wdir, modifier=None, find_modifier=False,
                      steps=100, stype='avg', evaltype='stdp', no_swarm=False,
                      cached_eps_dir=None):
  seedruns = [(wd.replace('run_seed', ''), os.path.join(wdir, wd))
    for wd in os.listdir(wdir) if wd.startswith('run_seed')]

  if modifier:
    seedruns = [(seed, os.path.join(seedrun_f, modifier)) for seed, seedrun_f in seedruns]

  assert stype in ['avg', 'median']
  aggs_seed = {}
  for seed, seedrun_f in seedruns:
    if cached_eps_dir:
      tr_res = []
      cached_path = os.path.join(cached_eps_dir, '{}-{}{}.tsv'.format(seed, stype, steps))
      if not os.path.isfile(cached_path):
        continue
      with open(cached_path) as f:
        for t in f:
          tr_res.append(float(t))
    else:
      tr_res = _extract_steps_per_ep(seedrun_f, steps, stype,
          find_modifier=find_modifier, filetype=evaltype)
    print([i for i,k in enumerate(tr_res) if k == max(tr_res)][0])
    aggs_seed[seed] = max(tr_res)

  label = 'MAX {} over {}{}'.format(
    'average' if stype == 'avg' else stype, steps,
    'iterations'if evaltype == 'evol' else 'steps')

  maxes = [200, 300, 500, 501]
  curr_max = 0
  for amax in aggs_seed.values():
    while maxes[curr_max] < amax:
      curr_max += 1

  plt.figure(figsize=(2.5,6))
  plt.grid(axis='y', alpha=0.3)
  aggs_by_step = [list(aggs_seed.values())]
  print('min: ', np.min(aggs_by_step))
  print('mean: ', np.mean(aggs_by_step))
  print('max: ', np.max(aggs_by_step))
  ax = sns.boxplot(data=aggs_by_step)
  if not no_swarm:
    ax = sns.swarmplot(data=aggs_by_step, color=".25", size=10.0)
  ax.set_xticklabels([label]) # , rotation=5
  ax.set_ylim(0, maxes[curr_max])
  # plt.title('Variance in performance of {} different initial network configurations'.format(len(akeys)))
  # plt.show()
  outputfile = os.path.join(wdir, 'eval_{}{}_seeds{}.png'.format(
    stype, steps,'' if not modifier else '_' + modifier.replace('/', '_')))
  plt.tight_layout()
  plt.savefig(outputfile, dpi=300)


def _extract_steps_per_ep(wdir, steps, stype, filetype='stdp', find_modifier=False):
  assert stype in ['avg', 'median']
  if find_modifier:
    while True:
      wdirs = os.listdir(wdir)
      continue_dirs = [wd for wd in wdirs if wd.startswith('continue_')]
      if len(continue_dirs) == 0:
        break
      assert len(continue_dirs) == 1
      wdir = os.path.join(wdir, continue_dirs[0])

  all_wdir_steps, _ = _extract_hpsteps(wdir)
  if filetype == 'stdp':
    training_results = []
    for wdir_steps in all_wdir_steps:
      with open(os.path.join(wdir_steps, 'ActionsPerEpisode.txt')) as f:
          training_results.extend([int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')])

    func = np.average if stype == 'avg' else np.median
    return _get_agg(training_results, steps, func)
  elif filetype == 'evolstdprl':
    training_results = []
    for wdir_steps in all_wdir_steps:
      with open(os.path.join(wdir_steps, 'STDP.txt')) as f:
          training_results.extend([int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')])

    func = np.average if stype == 'avg' else np.median
    return _get_agg(training_results, steps, func)
  elif filetype == 'evol':
    training_results = []
    for wdir_steps in all_wdir_steps:
      with open(os.path.join(wdir_steps, 'es_train.txt')) as f:
          # get the mean
          training_results.extend([int(float(toks[2])) for toks in csv.reader(f, delimiter='\t')])

    func = np.average if stype == 'avg' else np.median
    return _get_agg(training_results, steps, func)


def steps_per_eps(wdir, steps=100, stype='avg', modifier=None, parts=1, seed_nrs=False,
                  cached_eps_dir=None):
  assert stype in ['avg', 'median']

  seedruns = [(wd.replace('run_seed', ''), os.path.join(wdir, wd))
    for wd in os.listdir(wdir) if wd.startswith('run_seed')]

  # valid_seeds = [
  #   '5397326', '7892276', '1932160', '6623146',
  #   '9300610', '5381445', '2544501', '5140568',
  #   '1257804', '1394398']
  # seedruns = [(seed,sr) for seed,sr in seedruns if seed in valid_seeds]

  if modifier:
    seedruns = [(seed, os.path.join(seedrun_f, modifier)) for seed, seedrun_f in seedruns]

  for p in range(parts):
    seedruns_batch = seedruns
    pstep = 0
    if parts > 1:
      pstep = int(len(seedruns) / parts)
      seedruns_batch = seedruns[p * pstep:min((p+1)*pstep, len(seedruns))]

    plt.figure(figsize=(6,6))
    plt.ylim(0, 200)
    plt.grid(axis='y', alpha=0.4)
    plt.ylabel('steps per episode ({} over {} episodes)'.format(
      'average' if stype == 'avg' else stype, steps))
    # plt.title('Performance during training, evaluated using {} over {} episodes'.format(stype, steps))
    for seed, seedrun_wdir in seedruns_batch:
      if cached_eps_dir:
        tr_res = []
        with open(os.path.join(cached_eps_dir, '{}-{}{}.tsv'.format(seed, stype, steps))) as f:
          for t in f:
            tr_res.append(float(t))
      else:
        tr_res = _extract_steps_per_ep(seedrun_wdir, steps, stype)
        # # RM: This code is used to cache results
        # with open(os.path.join('results/seedrun_m1-2022-01-16/steps_per_eps/normal', '{}-{}{}.tsv'.format(seed, stype, steps)), 'w') as out:
        #   for t in tr_res:
        #     out.write(str(t) + '\n')
      plt.plot(tr_res)

    plt.legend(['model seed: {}'.format(sidx+1+pstep*p if seed_nrs else seed)
      for sidx, (seed,_) in enumerate(seedruns_batch)])
    plt.xlabel('episode')

    outputfile = os.path.join(
      wdir,
      'steps_per_episode{}_{}_{}_{}.png'.format(
        '' if parts == 1 else '_{}of{}'.format(p+1, parts),
        stype, steps, modifier.replace('/', '_')))
    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)


def steps_per_eps_evol(wdir, steps=100, stype='avg', parts=1, seed_nrs=False, target_perf=None):
  assert stype in ['avg', 'median']

  seedruns = [(wd.replace('run_seed', ''), os.path.join(wdir, wd))
    for wd in os.listdir(wdir) if wd.startswith('run_seed')]

  for p in range(parts):
    seedruns_batch = seedruns
    pstep = 0
    if parts > 1:
      pstep = int(len(seedruns) / parts)
      seedruns_batch = seedruns[p * pstep:min((p+1)*pstep, len(seedruns))]

    plt.figure(figsize=(5,5))
    plt.ylim(0, 500)
    plt.grid(axis='y', alpha=0.4)
    plt.ylabel('steps per episode ({} over {} iterations)'.format(
      'average' if stype == 'avg' else stype, steps))
    # plt.title('Performance during training, evaluated using {} over {} episodes'.format(stype, steps))
    for seed, seedrun_wdir in seedruns_batch:
      tr_res = _extract_steps_per_ep(seedrun_wdir, steps, stype, 'evol', find_modifier=True)
      plt.plot(tr_res)
      if target_perf:
        print([i for i, k in enumerate(tr_res) if k > target_perf][0])

    plt.legend(['model seed: {}'.format(sidx+1+pstep*p if seed_nrs else seed)
      for sidx, (seed,_) in enumerate(seedruns_batch)])
    plt.xlabel('iteration (10 * 5 episodes)')

    outputfile = os.path.join(
      wdir,
      'steps_per_episode{}_{}_{}.png'.format(
        '' if parts == 1 else '_{}of{}'.format(p+1, parts),
        stype, steps))
    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)


if __name__ == '__main__':
  fire.Fire({
      'analyze': analyze_samples,
      'a1': show_analyze_once,
      'train-perf': steps_per_eps,
      'train-perf-evol': steps_per_eps_evol,
      # 'combine': create_table
  })