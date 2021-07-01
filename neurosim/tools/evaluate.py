import fire
import os
import csv
import json
import math

import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

RANDOM_EVALUATION='results/random_cartpole_ActionsPerEpisode.txt'

def _read_evaluations(wdir, include_random):
  evaluations = [
    (os.path.join(wdir, fname), int(fname.replace('evaluation_', '')))
    for fname in os.listdir(wdir) if fname.startswith('evaluation_')]

  results = {}
  for eval_dir, eval_ts in evaluations:
    if os.path.isfile(os.path.join(eval_dir, 'ActionsPerEpisode.txt')):
      with open(os.path.join(eval_dir, 'ActionsPerEpisode.txt')) as f:
        results[eval_ts] = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

  if include_random:
      with open(RANDOM_EVALUATION) as f:
        results[-1] = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

  return results

def boxplot(wdir, include_random=True, outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'evaluations_boxplot.png')

  results = _read_evaluations(wdir, include_random)

  with open(os.path.join(wdir, 'synWeights.pkl'), 'br') as f:
      synWeights = pkl.load(f)
      
  preid = list(synWeights.keys())[0]
  postid = list(synWeights[preid].keys())[0]

  steps = len(synWeights[preid][postid])
  ts = [s for s,v in synWeights[preid][postid]]

  sorted_results = sorted(list(results.items()), key=lambda x:x[0])

  labels = [k for k,v in sorted_results]
  data = [v for k,v in sorted_results]

  fig = plt.figure(figsize =(10, 10))
  ax = fig.add_subplot(111)
  bp = ax.boxplot(data)
  ticks = (['random_choice'] if include_random else []) + \
      ['step {} (at {} s)'.format(l, round(ts[l] / 1000, 4)) for l in labels if l >= 0]
  ax.set_xticklabels(
      ticks,
      rotation = 80)
  ax.set_ylabel('actions per episode')
  ax.set_title('BoxPlots of ActionsPerEpisode for {} model at different timesteps'.format(wdir))

  plt.tight_layout()
  plt.savefig(outputfile)


def performance(wdir, outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'evaluations_performance.png')
  results = _read_evaluations(wdir)

  fig = plt.figure(figsize =(10, 7))

  data = [(step_ts, acts_per_eps) for step_ts, acts_per_eps in results.items()]
  data = sorted(data, key=lambda x:x[0])
   
  plt.plot([x for x,y in data], [np.mean(y) for x,y in data], '-x')
  plt.plot([x for x,y in data], [np.median(y) for x,y in data], '-x')
  plt.plot([x for x,y in data], [np.std(y) for x,y in data], '-x')

  plt.legend(['mean', 'median', 'std'])

  plt.xlabel('epoch')
  plt.ylabel('actions per episode')
  plt.title('Evolution of performance during training')

  plt.grid()
  plt.tight_layout()
  plt.savefig(outputfile)


def _get_pop_name(cellId, sorted_min_ids):
  # This is O(n), could be O(log(n)) with binary search
  return [pop for pop, minId in sorted_min_ids if cellId >= minId][0]

def frequency(wdir, timestep=1000, outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'frequency.png')

  sim_config = os.path.join(wdir, 'sim.json')
  with open(sim_config) as f:
    sim = json.load(f)

  sorted_min_ids = sorted(list(sim['simData']['dminID'].items()), key=lambda x:x[1], reverse=True)
  spkid = sim['simData']['spkid']
  spkt = sim['simData']['spkt']

  spike_aggs = {}

  for cid, ct in zip(spkid, spkt):
    pop = _get_pop_name(cid, sorted_min_ids)
    bucket = math.floor(ct / timestep)
    if pop not in spike_aggs:
      spike_aggs[pop] = {}
    if bucket not in spike_aggs[pop]:
      spike_aggs[pop][bucket] = 0
    spike_aggs[pop][bucket] += 1

  pop_sizes = dict([(pop, sim['net']['params']['popParams'][pop]['numCells']) for pop in spike_aggs.keys()])

  plt.figure(figsize=(10, 10))
  legend = []
  for pop, pop_spikes in spike_aggs.items():
    spikes = sorted(list(pop_spikes.items()), key=lambda x:x[0])
    plt.plot(
      [x+1 for x,y in spikes],
      [y / pop_sizes[pop] / (timestep / 1000) for x,y in spikes])
    total_spikes = sum([y for x,y in spikes])
    total_time = (spikes[-1][0] + 1) * timestep
    total_freq = total_spikes / pop_sizes[pop] / (total_time / 1000)
    legend.append('{} ({} Hz)'.format(pop, round(total_freq, 1)))

  plt.title('Frequency of populations over time')
  plt.xlabel('t * {}'.format(timestep))
  plt.ylabel('Hz')
  plt.legend(legend)
  plt.savefig(outputfile)

def actions_medians(wdir, steps=[21,51,101], outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'eval_actions.png')

  with open(os.path.join(wdir, 'ActionsPerEpisode.txt')) as f:
      training_results = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

  training_medians = {}
  for STEP in steps:
      training_medians[STEP] = []
      for idx in range(len(training_results) - STEP):
          training_medians[STEP].append(np.median(training_results[idx:idx+STEP]))

  plt.figure(figsize=(10,10))

  plt.plot(list(range(len(training_results))), training_results)
  for STEP, medians in training_medians.items():
      plt.plot([t + STEP for t in range(len(medians))], medians)

  plt.legend(['individual'] + ['median of {}'.format(STEP) for STEP in training_medians.keys()])
  plt.xlabel('episode')
  plt.ylabel('actions per episode')

  plt.savefig(outputfile)

if __name__ == '__main__':
  fire.Fire({
    'frequency': frequency,
    'boxplot': boxplot,
    'perf': performance,
    'medians': actions_medians
  })
