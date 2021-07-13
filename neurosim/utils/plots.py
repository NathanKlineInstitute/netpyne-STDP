import os
import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plotWeights():
  from pylab import figure, loadtxt, xlabel, ylabel, xlim, ylim, show, pcolor, array, colorbar
  figure()
  weightdata = loadtxt(sim.weightsfilename)
  weightdataT = list(map(list, list(zip(*weightdata))))
  vmax = max([max(row) for row in weightdata])
  vmin = min([min(row) for row in weightdata])
  pcolor(array(weightdataT), cmap='hot_r', vmin=vmin, vmax=vmax)
  xlim((0, len(weightdata)))
  ylim((0, len(weightdata[0])))
  xlabel('Time (weight updates)')
  ylabel('Synaptic connection id')
  colorbar()
  show()


def _getrate(dspkT, dspkID, pop, dnumc, totalDur=None, tlim=None):
  # get average firing rate for the population, over entire simulation
  nspk = len(dspkT[pop])
  ncell = dnumc[pop]
  if tlim is not None:
    spkT = dspkT[pop]
    nspk = len(spkT[(spkT >= tlim[0]) & (spkT <= tlim[1])])
    return 1e3*nspk/((tlim[1]-tlim[0])*ncell)
  else:
    return 1e3*nspk/(totalDur*ncell)

def saveActionsPerEpisode(sim, epCount, output_filename):
  with open(output_filename, 'w') as fid5:
    for i in range(len(epCount)):
      fid5.write('\t%0.1f' % epCount[i])
      fid5.write('\n')


def drawraster(lpop, dspkT, dspkID, dnumc,
               totalDur=None, tlim=None, msz=2, skipstim=True,
               figname=None):
  plt.figure(figsize=(10, 10))
  # draw raster (x-axis: time, y-axis: neuron ID)
  lpop = list(dspkT.keys())
  lpop.reverse()
  lpop = [x for x in lpop if not skipstim or x.count('stim') == 0]
  csm = cm.ScalarMappable(cmap=cm.prism)
  csm.set_clim(0, len(dspkT.keys()))
  lclr = []
  for pdx, pop in enumerate(lpop):
    color = csm.to_rgba(pdx)
    lclr.append(color)
    plot(dspkT[pop], dspkID[pop], 'o', color=color, markersize=msz)
  if tlim is not None:
    xlim(tlim)
  else:
    xlim((0, totalDur))
  xlabel('Time (ms)')
  # lclr.reverse();
  lpatch = [mpatches.Patch(color=c, label=s+' '+str(
      round(_getrate(dspkT, dspkID, s, dnumc, totalDur=totalDur, tlim=tlim), 2))+' Hz')
      for c, s in zip(lclr, lpop)]
  ax = gca()
  ax.legend(handles=lpatch, handlelength=1, loc='best')
  ylim((0, sum([dnumc[x] for x in lpop])))
  if figname:
    plt.savefig(figname)
  else:
    plt.show()


def _prepraster(sim, lpops):
  # lpops = dnumc
  dstartidx, dendidx = {}, {}  # starting,ending indices for each population
  for p in lpops.keys():
    if lpops[p] > 0:
      dstartidx[p] = sim.simData['dminID'][p]
      dendidx[p] = sim.simData['dminID'][p] + lpops[p] - 1
  spkID = np.array(sim.simData['spkid'])
  spkT = np.array(sim.simData['spkt'])
  dspkID, dspkT = {}, {}
  for pop in lpops.keys():
    # if dnumc[pop] > 0:
    dspkID[pop] = spkID[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
    dspkT[pop] = spkT[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
  return dspkID, dspkT


def plotRaster(sim, dconf, dnumc, output_filename):
  lpops = dict([(k, v) for k, v in dconf['net']['allpops'].items() if v > 0])
  for ty in sim.lstimty:
    lpops[ty] = dconf['net']['allpops'][dconf['net']['inputPop']]
  dspkID, dspkT = _prepraster(sim, lpops)
  drawraster(
      [k for k, v in lpops.items() if v > 0],
      dspkT, dspkID, dnumc, totalDur=dconf['sim']['duration'] * 1000,
      figname=output_filename)
