import math
import os
import csv


RANDOM_EVALUATION='results/random_cartpole_ActionsPerEpisode.txt'

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


def _get_spike_aggs_all(sim, sorted_min_ids):
  spkid = sim['simData']['spkid']
  spkt = sim['simData']['spkt']

  spike_aggs = {}
  for cid, ct in zip(spkid, spkt):
    pop = _get_pop_name(cid, sorted_min_ids)
    if pop not in spike_aggs:
      spike_aggs[pop] = 0
    spike_aggs[pop] += 1
  return spike_aggs

def _group_by_pop(synWeights, sorted_min_ids):
  new_map = {}
  for n1, n1conns in synWeights.items():
      n1pop = _get_pop_name(n1, sorted_min_ids)
      for n2, wl in n1conns.items():
          n2pop = _get_pop_name(n2, sorted_min_ids)
          conn_name = '{}->{}'.format(n1pop, n2pop)
          for idx,(t,w) in enumerate(wl):
              if conn_name not in new_map:
                  new_map[conn_name] = []
              if idx == len(new_map[conn_name]):
                  new_map[conn_name].append([])
              new_map[conn_name][idx].append(w)
  return new_map


def _read_evaluations(wdir, include_random, inside_dirs=False):
  if inside_dirs:
    results = {}
    for fname in os.listdir(wdir):
      new_wdir = os.path.join(wdir, fname)
      if os.path.isdir(new_wdir):
        evals = _read_evaluations(new_wdir, include_random=False)
        for eval_ts, res in evals.items():
          results['{}-{}'.format(fname, eval_ts)] = res

    if include_random:
      with open(RANDOM_EVALUATION) as f:
        results['-1'] = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]
    return results

  evaluations = []
  for fname in os.listdir(wdir):
    if fname.startswith('evaluation_') and 'display' not in fname:
      evaluations.append((
        os.path.join(wdir, fname),
        int(fname.replace('evaluation_', ''))))

  results = {}
  for eval_dir, eval_ts in evaluations:
    if os.path.isfile(os.path.join(eval_dir, 'ActionsPerEpisode.txt')):
      with open(os.path.join(eval_dir, 'ActionsPerEpisode.txt')) as f:
        results[eval_ts] = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

  if include_random:
      with open(RANDOM_EVALUATION) as f:
        results[-1] = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

  return results
