import json
import os
import fire
import numpy as np
import matplotlib.pyplot as plt

from neurosim.game_interface import _parse_rf_map

def _find_boundary(idx, f, imin=None, imax=None, eps=1e-6):
    if imin == None:
        imin = -10000
    if imax == None:
        imax = 10000
        
    midpoint = imin + (imax - imin) / 2
    if f(imin).index(1.0) < idx:
        while f(midpoint).index(1.0) > idx:
            midpoint = imin + (midpoint - imin) / 2
        return _find_boundary(idx, f, midpoint, imax, eps)
    if f(imax).index(1.0) > idx + 1:
        while f(midpoint).index(1.0) < idx + 1:
            midpoint = midpoint + (imax - midpoint) / 2
        return _find_boundary(idx, f, imin, midpoint, eps)
    
    if imax - imin < eps:
        return round(imin, 5), imin, imax
    
    if f(midpoint).index(1.0) == idx:
        return _find_boundary(idx, f, midpoint, imax, eps)
    if f(midpoint).index(1.0) == idx + 1:
        return _find_boundary(idx, f, imin, midpoint, eps)
    


def receptive_fields(outdir):
  outputfile = os.path.join(outdir, 'obs_space_receptive_fields.png')
  with open('config.json') as f:
    config = json.load(f)

  obs_rf = [_parse_rf_map(func_def) for func_def in config['env']['observation_map']]

  space = []
  for obs_idx,obs in enumerate(config['env']['observation_map']):
      obs_space = []
      for idx in range(obs['bins'] - 1):
          boundary, _, _ = _find_boundary(idx, obs_rf[obs_idx])
          obs_space.append(boundary)
      space.append(obs_space)


  obs_names = ['Position', 'Velocity', 'Angle', 'Angular Velocity']
  colors = ['b', 'g', 'r', 'y']

  _,axs = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

  for idx1,axr in enumerate(axs):
      for idx2,ax in enumerate(axr):
          idx = idx1*len(axr) + idx2
          intervals = []
          d = space[idx][1] - space[idx][0]
          intervals.append([space[idx][0] - 1.75 * d, space[idx][0]])
          for si in range(len(space[idx])-1):
              intervals.append([space[idx][si], space[idx][si+1]])
          si = len(space[idx])-1
          d = space[idx][si] - space[idx][si-1]
          intervals.append([space[idx][si], space[idx][si] + 1.75 * d])


          x = np.arange(1, len(intervals)+1) + idx * 20 - 1
          height = [i2-i1 for i1, i2 in intervals]
          bottom = [i1 for i1, i2 in intervals]

          ax.bar(x,height, bottom=bottom, color=colors[idx])
          ax.set_ylim([min(intervals)[0], max(intervals)[1]])
          ax.set_xticks(x)
          ax.set_xlabel('Neuron ID')
          ax.set_ylabel('Receptive Field')
          ax.legend([obs_names[idx]])

          ax.grid(alpha=0.4)
          
  plt.suptitle('Receptive Fields of Neurons ES population')
  plt.tight_layout()
  plt.savefig(outputfile)

if __name__ == '__main__':
  fire.Fire({
      'rf': receptive_fields,
  })
