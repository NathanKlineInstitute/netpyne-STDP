import numpy as np
from pylab import *
import pickle
import pandas as pd
import conf
from conf import dconf
import os
import sys
import anim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr


def getrate(dspkT, dspkID, pop, dnumc, totalDur=None, tlim=None):
  # get average firing rate for the population, over entire simulation
  nspk = len(dspkT[pop])
  ncell = dnumc[pop]
  if tlim is not None:
    spkT = dspkT[pop]
    nspk = len(spkT[(spkT >= tlim[0]) & (spkT <= tlim[1])])
    return 1e3*nspk/((tlim[1]-tlim[0])*ncell)
  else:
    return 1e3*nspk/(totalDur*ncell)


def drawraster(lpop, dspkT, dspkID, dnumc,
               totalDur=None, tlim=None, msz=2, skipstim=True,
               figname=None):
  plt.figure(figsize=(10,10))
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
        round(getrate(dspkT, dspkID, s, dnumc, totalDur=totalDur, tlim=tlim), 2))+' Hz')
      for c, s in zip(lclr, lpop)]
  ax = gca()
  ax.legend(handles=lpatch, handlelength=1, loc='best')
  ylim((0, sum([dnumc[x] for x in lpop])))
  if figname:
    plt.savefig(figname)
  else:
    plt.show()

