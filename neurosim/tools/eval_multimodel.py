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
  wdirs, configs = _extract_hpsteps(wdir)

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


  labels = [k.replace(' ', '\n') for k,v in results]
  data = [v for k,v in results]

  fig = plt.figure(figsize=(6, 6))
  # ax = fig.add_subplot(111)
  # bp = ax.boxplot(data)
  ax = sns.boxplot(data=data)
  ax = sns.swarmplot(data=data, color=".25", size=2.0)
  ax.set_xticklabels(labels, rotation=5)
  ax.set_ylabel('Actions per episode')
  # ax.set_title('Evaluation of models')

  plt.grid(axis='y', alpha=0.4)
  plt.tight_layout()
  plt.savefig(outputfile, dpi=300)

def spiking_frequencies_table(wdirs, outdir):
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

def steps_per_eps(wdir, wdir_name, outdir, merge_es=False, steps=[100]):
  outputfile = os.path.join(
    outdir,
    'steps_per_episode_during_training_{}{}.png'.format(
      wdir_name.replace(' ', '_'), '_mergedES' if merge_es else ''))

  all_wdir_steps, _ = _extract_hpsteps(wdir)

  training_results = []
  for wdir_steps in all_wdir_steps:
    with open(os.path.join(wdir_steps, 'ActionsPerEpisode.txt')) as f:
        training_results.extend([int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')])

  plt.figure(figsize=(10,10))
  plt.ylim(0, 510)
  plt.grid(axis='y')
  plt.ylabel('steps/actions per episode')
  plt.title('{} performance during training'.format(wdir_name))
  if merge_es:
    STEP = 10
    merged_tr_results = _get_agg(training_results, STEP, np.average, overlap=False)
    min_tr_results = _get_agg(training_results, STEP, np.amin, overlap=False)
    max_tr_results = _get_agg(training_results, STEP, np.amax, overlap=False)

    tr_medians = {STEP: _get_agg(merged_tr_results, STEP, np.median) for STEP in steps}
    tr_averages = {STEP: _get_agg(merged_tr_results, STEP, np.average) for STEP in steps}

    plt.plot(list(range(len(max_tr_results))), max_tr_results, '.')
    plt.plot(list(range(len(merged_tr_results))), merged_tr_results, '.')
    plt.plot(list(range(len(min_tr_results))), min_tr_results, '.')
    for step, medians in tr_medians.items():
        plt.plot([t + step for t in range(len(medians))], medians)
    for step, averages in tr_averages.items():
        plt.plot([t + step for t in range(len(averages))], averages)

    plt.legend(['iteration max', 'iteration average', 'iteration min'] +
      ['median of {} iteration averages'.format(step) for step in tr_medians.keys()] +
      ['averages of {} iteration averages'.format(step) for step in tr_averages.keys()])
    plt.xlabel('iteration ({} episodes)'.format(STEP))

    plt.savefig(outputfile)
    return

  # non merge_es

  tr_medians = {STEP: _get_agg(training_results, STEP, np.median) for STEP in steps}
  tr_averages = {STEP: _get_agg(training_results, STEP, np.average) for STEP in steps}

  plt.plot(list(range(len(training_results))), training_results, '.')
  for STEP, medians in tr_medians.items():
      plt.plot([t + STEP for t in range(len(medians))], medians)
  for STEP, averages in tr_averages.items():
      plt.plot([t + STEP for t in range(len(averages))], averages)

  plt.legend(['individual'] +
    ['median of {}'.format(STEP) for STEP in tr_medians.keys()] +
    ['averages of {}'.format(STEP) for STEP in tr_averages.keys()])
  plt.xlabel('episode')

  plt.savefig(outputfile)


def steps_per_eps_combined(wdirs, outdir, steps=[100]):
  outputfile = os.path.join(
    outdir, 'steps_per_episode_during_training.png')

  if type(wdirs) == str:
    wdirs = wdirs.split(',')
  wdirs = [wdir.split(':') for wdir in wdirs]

  fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
  # plt.suptitle('Performance during training')

  all_train_steps = []
  for _, wdir, _ in wdirs:
    all_wdir_steps, _ = _extract_hpsteps(wdir)

    training_results = []
    for wdir_steps in all_wdir_steps:
      with open(os.path.join(wdir_steps, 'ActionsPerEpisode.txt')) as f:
          training_results.extend([int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')])
    all_train_steps.append(training_results)

  for widx, ax, (wdir_name, wdir, _), train_res in zip(range(len(wdirs)), axs, wdirs, all_train_steps):
    tr_medians = {STEP: _get_agg(train_res, STEP, np.median) for STEP in steps}
    tr_averages = {STEP: _get_agg(train_res, STEP, np.average) for STEP in steps}

    ax.plot(list(range(len(train_res))), train_res, '.')
    for STEP, medians in tr_medians.items():
        ax.plot([t + STEP for t in range(len(medians))], medians)
    for STEP, averages in tr_averages.items():
        ax.plot([t + STEP for t in range(len(averages))], averages)

    curr_legend = ['individual'] + \
      ['median of {}'.format(STEP) for STEP in tr_medians.keys()] + \
      ['averages of {}'.format(STEP) for STEP in tr_averages.keys()]

    for idx in range(len(wdirs)):
      if idx != widx:
        if len(all_train_steps[idx]) < len(train_res):
          ax.axvline(x=len(all_train_steps[idx]), c='r')
          curr_legend.append('{} Episodes'.format(wdirs[idx][0].replace('After ', '')))

    ax.legend(curr_legend)
    ax.set_xlabel('episode')
    ax.set_ylabel('Actions per episode')
    ax.set_ylim(0, 510)
    ax.grid(axis='y', alpha=0.5)
    ax.set_title('{} Performance'.format(wdir_name.replace('After ', '')))

  plt.tight_layout()
  plt.savefig(outputfile, dpi=300)


def undecided_moves(wdirs, outdir, steps=[100, 1000]):
  outputfile = os.path.join(
    outdir, 'undecided_moves_during_training.png')

  if type(wdirs) == str:
    wdirs = wdirs.split(',')
  wdirs = [wdir.split(':') for wdir in wdirs]

  fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20,10))
  plt.suptitle('Undecided moves percentages during training')

  def _len_moves(idx):
    return len(all_unk_moves[idx][steps[0]]) + steps[0]

  all_unk_moves = []
  for _, wdir, _ in wdirs:
    val_moves = []
    for wdir_step, _ in zip(*_extract_hpsteps(wdir)):
        with open(os.path.join(wdir_step, 'MotorOutputs.txt')) as f:
          for toks in csv.reader(f, delimiter='\t'):
            _, l, r = [int(float(tok)) for tok in toks]
            val_moves.append(1 if l == r else 0)
    tr_agg = {STEP: _get_avg_fast(val_moves, STEP) for STEP in steps}
    all_unk_moves.append(tr_agg)

  all_vals = [v for vals in all_unk_moves for vs in vals.values() for v in vs]
  miny, maxy = np.amin(all_vals), np.amax(all_vals)

  for widx, ax, (wdir_name, wdir, _), tr_agg in zip(range(len(wdirs)), axs, wdirs, all_unk_moves):
    for STEP, averages in tr_agg.items():
        ax.plot([t + STEP for t in range(len(averages))], averages)

    curr_legend = ['averages of {}'.format(STEP) for STEP in tr_agg.keys()]

    lw = _len_moves(widx)
    for idx in range(len(wdirs)):
      if idx != widx:
        lidx = _len_moves(idx)
        if lidx < lw:
          ax.axvline(x=lidx, c='r')
          curr_legend.append('{} total moves'.format(
            wdirs[idx][0].replace('Trained ', '')))

    ax.legend(curr_legend)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel('moves')
    ax.set_ylabel('percentage of undecided moves')
    ax.set_title('{} Undecided moves'.format(wdir_name.replace('Trained ', '')))
    ax.grid(axis='y')

  plt.tight_layout()
  plt.savefig(outputfile)

def select_episodes(wdirs, cnt=7):
  if type(wdirs) == str:
    wdirs = wdirs.split(',')

  all_actions = []
  for wdir in wdirs:
    actions = []
    with open(os.path.join(wdir, 'ActionsPerEpisode.txt')) as f:
      for row in csv.reader(f, delimiter='\t'):
        actions.append(int(float(row[1])))
    all_actions.append(actions)

  arr_acts = [(idx+1, list(zip(range(len(wdirs)), toks))) for idx, toks in enumerate(zip(*all_actions))]
  for idx,wdir in enumerate(wdirs):
    print('-' * 40)
    print(wdir)
    arr_acts = sorted(arr_acts, key=lambda x: x[1][idx][1])
    print('min')
    for idx, toks in arr_acts[:cnt]:
      print(idx, [tk[1] for tk in toks])
    print('median')
    st = int(len(arr_acts)/2-cnt/2)
    for idx, toks in arr_acts[st:st+cnt]:
      print(idx, [tk[1] for tk in toks])
    print('max')
    for idx, toks in arr_acts[-cnt:]:
      print(idx, [tk[1] for tk in toks])


def save_episodes_eval(wdirs, outdir, sort_by=None):
  outputfile = os.path.join(outdir, 'evaluations_on_specific_episodes.png')
  if type(wdirs) == str:
    wdirs = wdirs.split(',')
  wdirs = [wdir.split(':') for wdir in wdirs]

  assert len(wdirs) == 2

  all_evals = []
  for _,wdir,_ in wdirs:
    evals = {}
    for fdir in os.listdir(wdir):
      fpath = os.path.join(wdir, fdir)
      if os.path.isdir(fpath) and 'rerunEp' in fdir and 'display' not in fdir:
        eval_id = int(fdir.split('_')[1])
        ep_id = int(fdir.replace('eval_{}_rerunEp'.format(eval_id), ''))
        if eval_id not in evals:
          evals[eval_id] = []
        evals[eval_id].append([ep_id, fpath])
    latest_eval_id = sorted(list(evals.keys()))[-1]
    all_evals.append(evals[latest_eval_id])

  all_ep_ids = sorted(list(set([ep_id for evals in all_evals for ep_id, _ in evals])))
  ep_ids = []
  for ep_id in all_ep_ids:
    count = 0
    for evals in all_evals:
      for ep_id2, fpath in evals:
        if ep_id2 == ep_id and os.path.isfile(os.path.join(fpath, 'ActionsPerEpisode.txt')):
          count += 1
    if count == len(wdirs):
      ep_ids.append(ep_id)
  if sort_by:
    if type(sort_by) == str:
      sort_by = sort_by.split(',')
    sort_by = [int(k) for k in sort_by]
    ep_ids = [k for k in sort_by for ep_id in ep_ids if k == ep_id]
  print('Evaluating on {}'.format(ep_ids))

  table = []
  for idx, ep_id in enumerate(ep_ids):
    all_actions = []
    for wdir_idx, evals in enumerate(all_evals):
      fpath = [fpath for ep_id2, fpath in evals if ep_id2 == ep_id][0]
      with open(os.path.join(fpath, 'ActionsPerEpisode.txt')) as f:
        for row in csv.reader(f, delimiter='\t'):
          actions_per_episode = int(float(row[1]))
          table.append([
            wdirs[wdir_idx][0].replace('After ', '').replace('Training', 'Trained'),
            idx,
            actions_per_episode])

  model_col = 'Model'
  ep_col = 'Unique Initial Game States'
  acts_col = 'Actions per episode'
  df = pd.DataFrame(data=table, columns=[model_col, ep_col, acts_col])

  plt.figure(figsize=(9, 6))
  ax = sns.boxplot(
      x=ep_col, y=acts_col, hue=model_col,
      palette=["g", "r"], data=df)
  # ax.set_xticklabels(ep_ids)
  ax.set_xticklabels(['$S_{r%s}$' % (i+1) for i in range(len(ep_ids))])
  sns.despine(offset=10, trim=True)
  plt.grid(alpha=0.3)
  plt.savefig(outputfile, dpi=300)


if __name__ == '__main__':
  fire.Fire({
      'trace': trace,
      'boxplot': boxplot,
      'spk-freq': spiking_frequencies_table,
      'train-perf': steps_per_eps,
      'train-perf-comb': steps_per_eps_combined,
      'train-unk-moves': undecided_moves,
      'select-eps': select_episodes,
      'eval-selected-eps': save_episodes_eval
  })
