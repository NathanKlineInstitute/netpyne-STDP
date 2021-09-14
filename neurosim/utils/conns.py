# neuronal network connection functions
import numpy as np


def prob2conv(prob, npre):
  # probability to convergence; prob is connection probability
  # npre is number of presynaptic neurons
  return int(0.5 + prob * npre)


def getconv(cmat, prety, poty, npre):
  # get convergence value from cmat dictionary
  # (uses convergence if specified directly, otherwise uses p to calculate)
  if 'conv' in cmat[prety][poty]:
    return cmat[prety][poty]['conv']
  elif 'p' in cmat[prety][poty]:
    return prob2conv(cmat[prety][poty]['p'], npre)
  return 0


def getInitDelay(dconf, sec):
  dmin, dmax = dconf['net']['delays'][sec]
  if dmin == dmax:
    return dmin
  else:
    return 'uniform(%g,%g)' % (dmin, dmax)


def getSec(prety, poty, sy):
  # Make it more detailed if needed
  return 'soma'


def getDelay(dconf, prety, poty, sy, sec=None):
  if sec == None:
    sec = getSec(prety, poty, sy)
  if sy == 'GA2':
    # longer delay for GA2 only
    return getInitDelay(dconf, sec + '2')
  return getInitDelay(dconf, sec)


def getdminID(sim, lpop):
  # setup min ID for each population in lpop
  # gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
  alltags = sim._gatherAllCellTags()
  dGIDs = {pop: [] for pop in lpop}
  for tinds in range(len(alltags)):
    if alltags[tinds]['pop'] in lpop:
      dGIDs[alltags[tinds]['pop']].append(tinds)
  return {pop: np.amin(dGIDs[pop]) for pop in lpop if len(dGIDs[pop]) > 0}


def setrecspikes(dconf, sim):
  if dconf['sim']['recordStim']:
    sim.cfg.recordCellsSpikes = [-1]  # record from all spikes
  else:
    # make sure to record only from the neurons, not the stimuli - which requires a lot of storage/memory
    sim.cfg.recordCellsSpikes = []
    for pop in sim.net.pops.keys():
      if pop.count('stim') > 0 or pop.count('Noise') > 0:
        continue
      for gid in sim.net.pops[pop].cellGids:
        sim.cfg.recordCellsSpikes.append(gid)
