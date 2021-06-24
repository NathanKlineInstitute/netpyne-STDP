import fire
import json
import math
import matplotlib.pyplot as plt

def _get_pop_name(cellId, sorted_min_ids):
  # This is O(n), could be O(log(n)) with binary search
  return [pop for pop, minId in sorted_min_ids if cellId >= minId][0]

def frequency(sim_config, timestep=1000, outputfile=None):
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
  if outputfile:
    plt.savefig(outputfile)
  else:
    plt.show()

if __name__ == '__main__':
  fire.Fire({
    'frequency': frequency
  })
