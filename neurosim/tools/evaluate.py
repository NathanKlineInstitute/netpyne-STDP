import fire
import os
import csv
import json
import math

import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from neurosim.utils.weights import readWeights
from neurosim.tools.utils import _get_pop_name, _extract_sorted_min_ids, \
                _get_spike_aggs, _group_by_pop, _read_evaluations


def boxplot(wdir, include_random=True, outputfile=None, inside_dirs=False):
  if not outputfile:
    outputfile = os.path.join(wdir, 'evaluations_boxplot.png')

  results = _read_evaluations(wdir, include_random, inside_dirs)

  ts = None
  if not inside_dirs:
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
      ['step {} (at {} s)'.format(l, round(ts[l] / 1000, 4) if ts else '') for l in labels if l != '-1' and l != -1]
  ax.set_xticklabels(
      ticks,
      rotation = 80)
  ax.set_ylabel('actions per episode')
  ax.set_title('BoxPlots of ActionsPerEpisode for {} model at different timesteps'.format(wdir))

  plt.grid(axis='y')
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

def actions_medians(wdir, steps=[21,51,101], outputfile=None, just_return=False):
  if not outputfile:
    outputfile = os.path.join(wdir, 'eval_actions.png')

  with open(os.path.join(wdir, 'ActionsPerEpisode.txt')) as f:
      training_results = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

  training_medians = {}
  for STEP in steps:
      training_medians[STEP] = []
      for idx in range(len(training_results) - STEP):
          training_medians[STEP].append(np.median(training_results[idx:idx+STEP]))

  if just_return:
    results = []
    for STEP, medians in sorted(list(training_medians.items()), key=lambda x:x[0]):
      results.append(np.amax(medians))
    return results

  plt.figure(figsize=(10,10))

  plt.plot(list(range(len(training_results))), training_results)
  for STEP, medians in training_medians.items():
      plt.plot([t + STEP for t in range(len(medians))], medians)

  plt.legend(['individual'] + ['median of {}'.format(STEP) for STEP in training_medians.keys()])
  plt.xlabel('episode')
  plt.ylabel('actions per episode')
  plt.grid()

  plt.savefig(outputfile)

def rewards_steps(wdir, steps=[25, 50], outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'eval_rewards.png')

  with open(os.path.join(wdir, 'ActionsRewards.txt')) as f:
      training_rewards = [float(toks[2]) for toks in csv.reader(f, delimiter='\t')]

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

def rewards_val_steps(wdir, outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'eval_rewards_val.png')

  with open(os.path.join(wdir, 'ActionsRewards.txt')) as f:
      training_rewards = [float(toks[2]) for toks in csv.reader(f, delimiter='\t')]

  plt.figure(figsize=(10,10))
  plt.plot(list(range(len(training_rewards))), training_rewards, '+')
  plt.title('Reward values')
  plt.xlabel('time steps')
  plt.ylabel('reward value')

  plt.savefig(outputfile)

def _displayAdj(A):
  A[A == 0] = np.NaN
  vmin = np.amin(A[np.logical_not(np.isnan(A))])
  vmax = np.amax(A[np.logical_not(np.isnan(A))])
      
  plt.figure(figsize=(10,10))

  plt.imshow(A, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
  plt.clim(vmin, vmax)
  plt.colorbar()

def eval_moves(wdir, steps=[100, 1000], outputfile=None,
    unk_moves=False, abs_move_diff=False, abs_move_diff_norm=False):
  if not outputfile:
    outputfile = os.path.join(
      wdir,
      'eval_' + ('unk_moves' if unk_moves else 
                'abs_move_diff' if abs_move_diff else
                'abs_move_diff_norm' if abs_move_diff_norm else '') + '.png')


  assert sum([unk_moves, abs_move_diff, abs_move_diff_norm]) == 1

  val_moves = []
  with open(os.path.join(wdir, 'MotorOutputs.txt')) as f:
    for toks in csv.reader(f, delimiter='\t'):
      _, l, r = [int(float(tok)) for tok in toks]
      value = None
      if unk_moves:
        value = (1 if l == r else 0)
      elif abs_move_diff or abs_move_diff_norm:
        value = abs(l - r)
      val_moves.append(value)

  val_moves_avgs = {}
  for STEP in steps:
      val_moves_avgs[STEP] = []
      current_sum = sum(val_moves[:STEP])
      current_avg_by = STEP
      if abs_move_diff_norm:
        current_avg_by = sum([1 for v in val_moves[:STEP] if v > 0])
      val_moves_avgs[STEP].append(current_sum / current_avg_by)
      for idx in range(1, len(val_moves) - STEP):
        current_sum += val_moves[idx+STEP-1] - val_moves[idx-1]
        if abs_move_diff_norm:
          current_avg_by += sum([k / abs(k) for k in [val_moves[idx+STEP-1], -val_moves[idx-1]] if k != 0])
        val_moves_avgs[STEP].append(current_sum / current_avg_by)

  plt.figure(figsize=(10,10))

  for STEP, umoves in val_moves_avgs.items():
      plt.plot([t + STEP for t in range(len(umoves))], umoves)

  plt.legend(['over {} steps'.format(STEP) for STEP in val_moves_avgs.keys()])
  plt.xlabel('steps')
  ylabel = None
  if unk_moves:
    ylabel = 'percentage of unknown moves'
  elif abs_move_diff or abs_move_diff_norm:
    ylabel = 'absolte difference between left and right moves' + (
      ' normalized' if abs_move_diff_norm else '')
  plt.ylabel(ylabel)

  plt.savefig(outputfile)


def eval_motor_balance(wdir, steps=[1000], outputfile=None):
  if not outputfile:
    outputfile = os.path.join(wdir, 'eval_motor_balance.png')

  # TODO: move to model agnostic move choices
  choices = [] # including unknown as -1
  spikes = dict([(k, []) for k in range(2)]) # index by move id (including if same)
  rewards = dict([(k, []) for k in range(2)]) # index by move id (including random)
  with open(os.path.join(wdir, 'MotorOutputs.txt')) as fMO:
    readerFMO = csv.reader(fMO, delimiter='\t')
    with open(os.path.join(wdir, 'ActionsRewards.txt')) as fAR:
      for idx, rAR in enumerate(csv.reader(fAR, delimiter='\t')):
        if idx == 0:
          continue
        rMO = readerFMO.__next__()
        if rAR[0] != rMO[0]:
          raise Exception('Lines not in sync: "{}" vs "{}"'.format(rAR, rMO))
        spk_l, spk_r = int(float(rMO[1])), int(float(rMO[2]))
        choice = 0 if spk_l == spk_r else (-1 if spk_l > spk_r else 1)
        real_choice, reward = [float(k) for k in rAR[1:3]]
        # assert choice == -1 or choice == int(real_choice)
        choices.append(choice)
        spikes[0].append(spk_l)
        spikes[1].append(spk_r)
        other_choice = 1.0 - real_choice
        rewards[int(real_choice)].append(reward)
        rewards[int(other_choice)].append(0)

  def _get_averages(arr, rm_unk=False):
    arr_avgs = {}
    for STEP in steps:
        arr_avgs[STEP] = []
        current_sum = sum(arr[:STEP])
        current_avg_by = STEP
        if rm_unk:
          current_avg_by = sum([abs(a) for a in arr[:STEP]])
        arr_avgs[STEP].append(current_sum / current_avg_by)
        for idx in range(1, len(arr) - STEP):
          current_sum += arr[idx+STEP-1] - arr[idx-1]
          if rm_unk:
            current_avg_by += abs(arr[idx+STEP-1]) - abs(arr[idx-1])
          arr_avgs[STEP].append(current_sum / current_avg_by)
    return arr_avgs

  _, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
  ax1, ax2, ax3 = axs

  spikes_avg_l = _get_averages(spikes[0])
  spikes_avg_r = _get_averages(spikes[1])
  for STEP in steps:
      ax1.plot([t + STEP for t in range(len(spikes_avg_l[STEP]))], spikes_avg_l[STEP])
      ax1.plot([t + STEP for t in range(len(spikes_avg_r[STEP]))], spikes_avg_r[STEP])
  ax1.legend(['{} over {} steps'.format(direction, STEP) for STEP in steps for direction in ['left', 'right']])
  ax1.set_xlabel('steps')
  ax1.set_ylabel('spikes averages')

  rewards_avg_l = _get_averages(rewards[0])
  rewards_avg_r = _get_averages(rewards[1])
  for STEP in steps:
      ax2.plot([t + STEP for t in range(len(rewards_avg_l[STEP]))], rewards_avg_l[STEP])
      ax2.plot([t + STEP for t in range(len(rewards_avg_r[STEP]))], rewards_avg_r[STEP])
  ax2.legend(['{} over {} steps'.format(direction, STEP) for STEP in steps for direction in ['left', 'right']])
  ax2.set_xlabel('steps')
  ax2.set_ylabel('rewards averages')


  choices_avg = _get_averages(choices, rm_unk=True)
  for STEP in steps:
      choices_ratio = [(1.0 - avg) / 2 for avg in choices_avg[STEP]]
      ax3.plot([t + STEP for t in range(len(choices_avg[STEP]))], choices_ratio)
  ax3.legend(['ratio over {} steps'.format(STEP) for STEP in steps])
  ax3.set_xlabel('steps')
  ax3.set_ylabel('ratio of left moves / all non-unk moves')

  plt.savefig(outputfile)


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
    bt = adj[indices[0]]
    bt[bt == 0] = 1e-4
    matrix = matrix / bt
  _displayAdj(matrix)
  plt.savefig(outputfile)


def stdp_weights_changes(wdir, separate_movement=False, outputfile=None, display=False):
  with open(os.path.join(wdir, 'synWeights.pkl'), 'rb') as f:
    synWeights = pkl.load(f)

  sim_config = os.path.join(wdir, 'sim.pkl')
  with open(sim_config, 'rb') as f:
    sim = pkl.load(f)

  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')
  with open(dconf_path, 'r') as f:
    dconf = json.load(f)
  pop_sizes = dconf['net']['allpops']

  sorted_min_ids = _extract_sorted_min_ids(sim, dconf, separate_movement)
  popWeights = _group_by_pop(synWeights, sorted_min_ids)

  ncols = 3
  nbins = 30
  conns = sorted(list(popWeights.keys()))

  figsize = 20 if separate_movement else 15
  nrows = math.ceil(len(conns) / ncols)
  if nrows == 1:
    ncols = len(conns)
  _, axs = plt.subplots(
    ncols=ncols, nrows=nrows,
    subplot_kw=dict(projection="3d"),
    figsize=(figsize, figsize))

  conn_idx = 0
  for axi in axs:
    if nrows == 1:
      axi = [axi]
    for ax in axi:
      if conn_idx == len(conns):
        continue
      conn = conns[conn_idx]
      all_weights = [w for ws in popWeights[conn] for w in ws]
      wmin = np.min(all_weights)
      wmax = np.max(all_weights)

      for z, weights in reversed(list(enumerate(popWeights[conn]))):
          hist, bins = np.histogram(weights, bins=nbins, range=(wmin, wmax))
          xs = (bins[:-1] + bins[1:])/2
          ax.plot(xs, hist, zs=z, zdir='y', alpha=0.8)

      ax.set_title('{} weight changes over time'.format(conn))
      ax.set_xlabel('Weights')
      ax.set_ylabel('Epoch ({} * {}ms)'.format(
        dconf['sim']['recordWeightStepSize'],
        dconf['sim']['tstepPerAction']))
      ax.set_zlabel('Count of neurons')
      conn_idx += 1

  if display:
    plt.show()
  else:
    if not outputfile:
      outputfile = os.path.join(wdir, 'stdp_weight_changes{}.png'.format(
        '_sep_mov' if separate_movement else ''))
    plt.savefig(outputfile)


if __name__ == '__main__':
  fire.Fire({
    'frequency': frequency,
    'variance': variance,
    'eval-moves': eval_moves,
    'eval-motor': eval_motor_balance,
    'boxplot': boxplot,
    'perf': performance,
    'medians': actions_medians,
    'rewards': rewards_steps,
    'rewards-vals': rewards_val_steps,
    'weights-adj': stdp_weights_adj,
    'weights-diffs': stdp_weights_diffs,
    'weights-ch': stdp_weights_changes
  })
