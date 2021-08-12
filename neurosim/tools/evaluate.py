import fire
import os
import csv
import json
import math

import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from neurosim.utils.weights import readWeights

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
  results = _read_evaluations(wdir, False)

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


def _extract_sorted_min_ids(sim, dconf, separate_movement):
  pop_sizes = dconf['net']['allpops']
  sorted_min_ids = sorted(list(sim['simData']['dminID'].items()), key=lambda x:x[1], reverse=True)
  if separate_movement:
    for pop, moves in dconf['pop_to_moves'].items():
      pop_size = pop_sizes[pop]
      move_size = math.floor(pop_size / len(moves))
      smin_dict = dict(sorted_min_ids)
      pop_minId = smin_dict[pop]
      del smin_dict[pop]
      for midx, move in enumerate(moves):
        semi_pop_name = '{}-{}'.format(pop, move)
        smin_dict[semi_pop_name] = pop_minId + midx * move_size
        pop_sizes[semi_pop_name] = move_size
      sorted_min_ids = sorted(list(smin_dict.items()), key=lambda x:x[1], reverse=True)
  return sorted_min_ids

def _get_spike_aggs(sim, sorted_min_ids, timestep):
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
  return spike_aggs

def frequency(wdir, timestep=1000, outputfile=None,
    separate_movement=True, only_movement=False):
  if not outputfile:
    outputfile = os.path.join(wdir, 'frequency{}.png'.format('_OUT' if only_movement else ''))

  sim_config = os.path.join(wdir, 'sim.pkl')
  with open(sim_config, 'rb') as f:
    sim = pkl.load(f)

  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')
  with open(dconf_path, 'r') as f:
    dconf = json.load(f)
  pop_sizes = dconf['net']['allpops']

  sorted_min_ids = _extract_sorted_min_ids(sim, dconf, separate_movement)
  spike_aggs = _get_spike_aggs(sim, sorted_min_ids, timestep)

  plt.figure(figsize=(10, 10))
  legend = []
  for pop, pop_spikes in spike_aggs.items():
    if only_movement and '-' not in pop:
      continue
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

def variance(wdir, timestep=100, outputfile=None, var_of=100,
    separate_movement=True, only_movement=False):
  if not outputfile:
    outputfile = os.path.join(wdir, 'freq_variance{}.png'.format('_OUT' if only_movement else ''))

  sim_config = os.path.join(wdir, 'sim.pkl')
  with open(sim_config, 'rb') as f:
    sim = pkl.load(f)

  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')
  with open(dconf_path, 'r') as f:
    dconf = json.load(f)
  pop_sizes = dconf['net']['allpops']

  sorted_min_ids = _extract_sorted_min_ids(sim, dconf, separate_movement)
  spike_aggs = _get_spike_aggs(sim, sorted_min_ids, timestep)

  # Compute the variances
  variances = {}
  for pop, pop_spikes in spike_aggs.items():
    spikes = sorted(list(pop_spikes.items()), key=lambda x:x[0])
    variances[pop] = []
    curr_idx = 0
    for varidx in range(int(max([x for x,y in spikes]) / var_of)):
      y = []
      for idx in range(varidx*var_of, (varidx+1)*var_of):
        if curr_idx < len(spikes) and idx == spikes[curr_idx][0]:
          y.append(spikes[curr_idx][1])
          curr_idx += 1
        else:
          y.append(0)
      variances[pop].append((varidx, np.var(y)))


  plt.figure(figsize=(10, 10))
  legend = []
  for pop, pop_vars in variances.items():
    if only_movement and '-' not in pop:
      continue
    plt.plot([x+1 for x,y in pop_vars], [y for x,y in pop_vars])
    # pop_avg_vars = np.mean([y for x,y in pop_vars])
    # legend.append('{}'.format(pop))
    legend.append(pop)

  plt.title('Variance of spikes over populations')
  plt.xlabel('t * {}'.format(timestep * var_of))
  plt.ylabel('Variance of ({}) spikes counted in {}ms'.format(var_of, timestep))
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

def rewards_steps(wdir, steps=[25, 50], outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'eval_rewards.png')

  with open(os.path.join(wdir, 'ActionsRewards.txt')) as f:
      training_rewards = [float(r) for _,_,r in csv.reader(f, delimiter='\t')]

  training_medians = {}
  for STEP in steps:
      training_medians[STEP] = []
      for idx in range(len(training_rewards) - STEP):
        rews = training_rewards[idx:idx+STEP]
        pos_rews = [r for r in rews if r > 0]
        training_medians[STEP].append(len(pos_rews) / len(rews))

  plt.figure(figsize=(10,10))

  for STEP, medians in training_medians.items():
      plt.plot([t + STEP for t in range(len(medians))], medians)

  plt.legend(['step of {}'.format(STEP) for STEP in training_medians.keys()])
  plt.xlabel('episode')
  plt.ylabel('len(rewards) / len(total)')

  plt.savefig(outputfile)

def _displayAdj(A):
    A[A == 0] = np.NaN
    vmin = np.amin(A[A > 0])
    vmax = np.amax(A[A > 0])
        
    plt.figure(figsize=(10,10))

    plt.imshow(A, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.clim(vmin, vmax)
    plt.colorbar()

def _get_weights_adj(synWeights_file):  
  with open(synWeights_file, 'rb') as f:
      synWeights = pkl.load(f)
  M = [len(synWeights[n1][n2]) for n1 in list(synWeights.keys())[:1] for n2 in list(synWeights[n1].keys())[:1]][0]
  N = max(list(synWeights.keys()) + [n2 for n1 in synWeights.keys() for n2 in synWeights[n1].keys()]) + 1

  Adj = np.zeros((M, N, N))
  for k in range(M):
    for n1 in synWeights.keys():
        for n2 in synWeights[n1].keys():
            Adj[k][n1][n2] = synWeights[n1][n2][k][1]

  return Adj

def stdp_weights_adj(wdir, index=-1, outputfile=None):
  synWeights_file = os.path.join(wdir, 'synWeights.pkl')
  adj = _get_weights_adj(synWeights_file)
  epochs = adj.shape[0]

  if index < 0:
    index += epochs
  assert 0 <= index and index < epochs
  if not outputfile:
    outputfile = os.path.join(wdir, 'stdp_weights_{}.png'.format(index))

  _displayAdj(adj[index])
  plt.savefig(outputfile)

def stdp_weights_diffs(wdir, index1=0, index2=-1, relative=False, outputfile=None):
  synWeights_file = os.path.join(wdir, 'synWeights.pkl')
  adj = _get_weights_adj(synWeights_file)
  epochs = adj.shape[0]

  indices = [index1, index2]
  for i in range(len(indices)):
    if indices[i] < 0:
      indices[i] += epochs
    assert 0 <= indices[i] and indices[i] < epochs
  if not outputfile:
    outputfile = os.path.join(wdir, 'stdp_weights_diffs{}_{}_to_{}.png'.format(
      '_rel' if relative else '_abs', indices[0], indices[1]))

  matrix = adj[indices[1]] - adj[indices[0]]
  if relative:
    matrix = matrix / adj[indices[0]]
  _displayAdj(matrix)
  plt.savefig(outputfile)


if __name__ == '__main__':
  fire.Fire({
    'frequency': frequency,
    'variance': variance,
    'boxplot': boxplot,
    'perf': performance,
    'medians': actions_medians,
    'rewards': rewards_steps,
    'weights-adj': stdp_weights_adj,
    'weights-diffs': stdp_weights_diffs
  })
