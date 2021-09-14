import math

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