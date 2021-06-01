from netpyne import specs, sim
from neuron import h

import random
import pickle
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from matplotlib import pyplot as plt

from conf import dconf
from cells import intf7
from connUtils import getconv
from game_interface import GameInterface
from utils import syncdata_alltoall
from simdat import drawraster

# this will not work properly across runs with different number of nodes
random.seed(1234)

sim.davgW = {}  # average adjustable weights on a target population
sim.allTimes = []
sim.allRewards = []  # list to store all rewards
sim.allActions = []  # list to store all actions
sim.allMotorOutputs = []  # list to store firing rate of output motor neurons.
sim.ActionsRewardsfilename = 'data/'+dconf['sim']['name']+'ActionsRewards.txt'
sim.MotorOutputsfilename = 'data/'+dconf['sim']['name']+'MotorOutputs.txt'
sim.WeightsRecordingTimes = []
sim.allRLWeights = []
sim.allNonRLWeights = []

# sim.NonRLweightsfilename = 'data/'+dconf['sim']['name']+'NonRLweights.txt'  # file to store weights
sim.plotWeights = 0  # plot weights
sim.saveWeights = 1  # save weights
if 'saveWeights' in dconf['sim']:
  sim.saveWeights = dconf['sim']['saveWeights']
sim.saveInputImages = 1  # save Input Images (5 game frames)
# whether to save the motion fields
sim.saveMotionFields = dconf['sim']['saveMotionFields']
sim.saveObjPos = 1  # save ball and paddle position to file
sim.saveAssignedFiringRates = dconf['sim']['saveAssignedFiringRates']
recordWeightStepSize = dconf['sim']['recordWeightStepSize']
# recordWeightDT = 1000 # interval for recording synaptic weights (change later)
recordWeightDCells = 1  # to record weights for sub samples of neurons
tstepPerAction = dconf['sim']['tstepPerAction']  # time step per action (in ms)

fid4 = None  # only used by rank 0

allpops = list(dconf['net']['allpops'].keys())
inputPop = dconf['net']['inputPop']
EMotorPops = dconf['net']['EMotorPops']  # excitatory neuron motor populations
# number of neurons of a given type: dnumc
# scales the size of the network (only number of neurons)
scale = dconf['net']['scale']
dnumc = OrderedDict(
    {ty: dconf['net']['allpops'][ty] * scale for ty in allpops})

# connection matrix (for classes, synapses, probabilities [probabilities not used for topological conn])
cmat = dconf['net']['cmat']

# Network parameters
netParams = specs.NetParams()
# spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultThreshold = 0.0

# object of class SimConfig to store simulation configuration
simConfig = specs.SimConfig()
# Simulation options
# 100e3 # 0.1e5                      # Duration of the simulation, in ms
simConfig.duration = dconf['sim']['duration']
# Internal integration timestep to use
simConfig.dt = dconf['sim']['dt']
# make sure temperature is set. otherwise we're at squid temperature
simConfig.hParams['celsius'] = 37
# Show detailed messages
simConfig.verbose = dconf['sim']['verbose']
# Dict with traces to record
simConfig.recordTraces = {'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
# this means record from all neurons - including stim populations, if any
simConfig.recordCellsSpikes = [-1]
# Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.recordStep = dconf['sim']['recordStep']
simConfig.filename = 'data/' + \
    dconf['sim']['name']+'simConfig'  # Set file output name
simConfig.saveJson = True
# Save params, network and sim output to pickle file
simConfig.savePickle = True
simConfig.saveMat = False
simConfig.saveFolder = 'data'
# simConfig.backupCfg = ['sim.json', 'backupcfg/'+dconf['sim']['name']+'sim.json']
simConfig.createNEURONObj = True  # create HOC objects when instantiating network
# create Python structure (simulator-independent) when instantiating network
simConfig.createPyStruct = True
simConfig.analysis['plotTraces'] = {
    'include': [(pop, 0) for pop in dconf['net']['allpops'].keys()]
}
# simConfig.analysis['plotRaster'] = {
#   'popRates': 'overlay',
#   'showFig': dconf['sim']['doplot']
# }
#simConfig.analysis['plot2Dnet'] = True
# simConfig.analysis['plotConn'] = True           # plot connectivity matrix
# simConfig.coreneuron = True
# synaptic weight gain (based on E, I types)
cfg = simConfig
cfg.Gain = dconf['net']['Gain']

# from https://www.neuron.yale.edu/phpBB/viewtopic.php?f=45&t=3770&p=16227&hilit=memory#p16122
# if False removes all data on cell sections prior to gathering from nodes
cfg.saveCellSecs = bool(dconf['sim']['saveCellSecs'])
# if False removes all data on cell connections prior to gathering from nodes
cfg.saveCellConns = bool(dconf['sim']['saveCellConns'])
###

# weight variance -- check if need to vary the initial weights (note, they're over-written if resumeSim==1)
cfg.weightVar = dconf['net']['weightVar']


def isExc(ty): return ty.startswith('E')
def isInh(ty): return ty.startswith('I')
def connType(prety, poty): return prety[0] + poty[0]


def getInitWeight(weight):
  """get initial weight for a connection
     checks if weightVar is non-zero, if so will use a uniform distribution
     with range on interval: (1-var)*weight, (1+var)*weight
  """
  if cfg.weightVar == 0.0:
    return weight
  elif weight <= 0.0:
    return 0.0
  else:
    return 'uniform(%g,%g)' % (max(0, weight*(1.0-cfg.weightVar)), weight*(1.0+cfg.weightVar))


def getInitDelay(sec):
  dmin, dmax = dconf['net']['delays'][sec]
  if dmin == dmax:
    return dmin
  else:
    return 'uniform(%g,%g)' % (dmin, dmax)


def getSec(prety, poty, sy):
  # Make it more detailed if needed
  return 'soma'


def getDelay(prety, poty, sy, sec=None):
  if sec == None:
    sec = getSec(prety, poty, sy)
  if sy == 'GA2':
    # longer delay for GA2 only
    return getInitDelay(sec + '2')
  return getInitDelay(sec)


ECellModel = dconf['net']['ECellModel']
ICellModel = dconf['net']['ICellModel']


def getComp(sy):
  if ECellModel == 'INTF7' or ICellModel == 'INTF7':
    if sy.count('2') > 0:
      return 'Dend'
    return 'Soma'
  else:
    if sy.count('AM') or sy.count('NM'):
      return 'Dend'
    return 'Soma'


# Population parameters
for ty in allpops:
  netParams.popParams[ty] = {
      'cellType': ty,
      'numCells': dnumc[ty],
      'cellModel': ECellModel if isExc(ty) else ICellModel}


def makeECellModel(ECellModel):
  # create rules for excitatory neuron models
  EExcitSec = 'dend'  # section where excitatory synapses placed
  PlastWeightIndex = 0  # NetCon weight index where plasticity occurs
  if ECellModel == 'IntFire4':
    EExcitSec = 'soma'  # section where excitatory synapses placed
    # Dict with traces to record
    simConfig.recordTraces = {'V_soma': {'var': 'm'}}
    netParams.defaultThreshold = 0.0
    for ty in allpops:
      if isExc(ty):
        # netParams.popParams[ty]={'cellType':ty,'numCells':dnumc[ty],'cellModel':ECellModel}#, 'params':{'taue':5.35,'taui1':9.1,'taui2':0.07,'taum':20}}
        netParams.popParams[ty] = {'cellType': ty,
                                   'cellModel': 'IntFire4',
                                   'numCells': dnumc[ty],
                                   'taue': 1.0}  # pop of IntFire4
  elif ECellModel == 'INTF7':
    EExcitSec = 'soma'  # section where excitatory synapses placed
    # Dict with traces to record
    simConfig.recordTraces = {'V_soma': {'var': 'Vm'}}
    netParams.defaultThreshold = -40.0
    for ty in allpops:
      if isExc(ty):
        netParams.popParams[ty] = {'cellType': ty,
                                   'cellModel': 'INTF7',
                                   'numCells': dnumc[ty]}
        for k, v in intf7.INTF7E.dparam.items():
          netParams.popParams[ty][k] = v
    PlastWeightIndex = intf7.dsyn['AM2']
  return EExcitSec, PlastWeightIndex


def makeICellModel(ICellModel):
  # create rules for inhibitory neuron models
  if ICellModel == 'IntFire4':
    # Dict with traces to record
    simConfig.recordTraces = {'V_soma': {'var': 'm'}}
    netParams.defaultThreshold = 0.0
    for ty in allpops:
      if isInh(ty):
        netParams.popParams[ty] = {'cellType': ty, 'cellModel': 'IntFire4',
                                   'numCells': dnumc[ty], 'taue': 1.0}  # pop of IntFire4
  elif ICellModel == 'INTF7':
    EExcitSec = 'soma'  # section where excitatory synapses placed
    # Dict with traces to record
    simConfig.recordTraces = {'V_soma': {'var': 'Vm'}}
    netParams.defaultThreshold = -40.0
    for ty in allpops:
      if isInh(ty):
        netParams.popParams[ty] = {'cellType': ty,
                                   'cellModel': 'INTF7',
                                   'numCells': dnumc[ty]}
        if ty.count('L') > 0:  # LTS
          for k, v in intf7.INTF7IL.dparam.items():
            netParams.popParams[ty][k] = v
        else:  # FS
          for k, v in intf7.INTF7I.dparam.items():
            netParams.popParams[ty][k] = v


EExcitSec, PlastWeightIndex = makeECellModel(ECellModel)
print('EExcitSec,PlastWeightIndex:', EExcitSec, PlastWeightIndex)
makeICellModel(ICellModel)

# Synaptic mechanism parameters
# note that these synaptic mechanisms are not used for the INTF7 neurons
for synMech, synMechParams in dconf['net']['synMechParams'].items():
  netParams.synMechParams[synMech] = synMechParams


def readSTDPParams():
  lsy = ['AMPA', 'NMDA', 'AMPAI']
  gains = [cfg.Gain[gainType] for gainType in ['EE', 'EE', 'EI']]
  dSTDPparams = {}  # STDP-RL/STDPL parameters for AMPA,NMDA synapses;
  # generally uses shorter,longer eligibility traces
  for sy, gain in zip(lsy, gains):
    dSTDPparams[sy] = dconf['STDP'][sy]
    for k in dSTDPparams[sy].keys():
      if k.count('wt') or k.count('wbase') or k.count('wmax'):
        dSTDPparams[sy][k] *= gain

  dSTDPparams['AM2'] = dSTDPparams['AMPA']
  dSTDPparams['NM2'] = dSTDPparams['NMDA']
  return dSTDPparams


dSTDPparams = readSTDPParams()


def getWeightIndex(synmech, cellModel):
  # get weight index for connParams
  if cellModel == 'INTF7':
    return intf7.dsyn[synmech]
  return 0


def setupStimMod():
  # setup variable rate NetStim sources (send spikes based on image contents)
  lstimty = []
  stimModW = dconf['net']['stimModW']
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    lpoty = [inputPop]
    for poty in lpoty:
      if dnumc[poty] <= 0:
        continue
      stimty = 'stimMod'+poty
      lstimty.append(stimty)
      netParams.popParams[stimty] = {
          'cellModel': 'NSLOC',
          'numCells': dnumc[poty],
          'rate': 'variable',
          'noise': 0,
          'start': 0}
      blist = [[i, i] for i in range(dnumc[poty])]
      netParams.connParams[stimty+'->'+poty] = {
          'preConds': {'pop': stimty},
          'postConds': {'pop': poty},
          'weight': stimModW,
          'delay': getInitDelay('STIMMOD'),
          'connList': blist,
          'weightIndex': getWeightIndex('AMPA', ECellModel)}
  return lstimty


# Note: when using IntFire4 cells lstimty has the NetStim populations
# that send spikes to EV1, EV1DE, etc.
sim.lstimty = setupStimMod()
for ty in sim.lstimty:
  allpops.append(ty)

# Stimulation parameters

def setupNoiseStim():
  lnoisety = []
  dnoise = dconf['noise']
  # setup noisy NetStim sources (send random spikes)
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    for poty, dpoty in dnoise.items():
      for sy, dsy in dpoty.items():
        damp = dconf['net']['noiseDamping']['E' if isExc(poty) else 'I']
        Weight, Rate = dsy['w'] * damp, dsy['rate']
        # print(poty, isExc(poty), damp, Weight)
        if Weight > 0.0 and Rate > 0.0:
          # only create the netstims if rate,weight > 0
          stimty = 'stimNoise'+poty+'_'+sy
          netParams.popParams[stimty] = {
              'cellModel': 'NetStim',
              'numCells': dnumc[poty],
              'rate': Rate,
              'noise': 1.0,
              'start': 0}
          blist = [[i, i] for i in range(dnumc[poty])]
          netParams.connParams[stimty+'->'+poty] = {
              'preConds': {'pop': stimty},
              'postConds': {'pop': poty},
              'weight': Weight,
              'delay': getDelay(None, poty, sy),
              'connList': blist,
              'weightIndex': getWeightIndex(sy, ECellModel)}
          lnoisety.append(stimty)
  return lnoisety


sim.lnoisety = setupNoiseStim()
for ty in sim.lnoisety:
  allpops.append(ty)

######################################################################################
#####################################################################################

synToMech = dconf['net']['synToMech']
sytypes = dconf['net']['synToMech'].keys()

# Setup cmat connections
for prety, dprety in cmat.items():
  if dnumc[prety] <= 0:
    continue
  for poty, dconn in dprety.items():
    if dnumc[poty] <= 0:
      continue
    ct = connType(prety, poty)
    for sy in sytypes:
      if sy in cmat[prety][poty] and cmat[prety][poty][sy] > 0:
        k = '{}-{}->{}'.format(prety, sy, poty)
        sec = getSec(prety, poty, sy)
        weight = cmat[prety][poty][sy] * cfg.Gain[ct]
        netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': getconv(cmat, prety, poty, dnumc[prety]),
            'weight': weight,
            'delay': getDelay(prety, poty, sy, sec),
            'synMech': synToMech[sy],
            'sec': sec,
            'loc': 0.5,
            'weightIndex': getWeightIndex(
                sy, ICellModel if isInh(poty) else ECellModel)
        }
        # Setup STDP plasticity rules
        if ct in dconf['STDP'] and dconf['STDP'][ct] and dSTDPparams[synToMech[sy]]['STDPon']:
          netParams.connParams[k]['plast'] = {
              'mech': 'STDP', 'params': dSTDPparams[synToMech[sy]]}
          netParams.connParams[k]['weight'] = getInitWeight(weight)

###################################################################################################################################

sim.AIGame = None  # placeholder

lsynweights = []  # list of syn weights, per node

dsumWInit = {}


def getSumAdjustableWeights(sim):
  dout = {}
  for cell in sim.net.cells:
    W = N = 0.0
    for conn in cell.conns:
      if 'hSTDP' in conn:
        W += float(conn['hObj'].weight[PlastWeightIndex])
        N += 1
    if N > 0:
      dout[cell.gid] = W / N
  # print('getSumAdjustableWeights len=',len(dout))
  return dout


def sumAdjustableWeightsPop(sim, popname):
  # record the plastic weights for specified popname
  # this is the set of MR cells
  lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids]
  W = N = 0
  for cell in lcell:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        W += float(conn['hObj'].weight[PlastWeightIndex])
        N += 1
  return W, N


def recordAdjustableWeights(sim, t, lpop):
  global lsynweights
  """ record the STDP weights during the simulation
  """

  for popname in lpop:
    # record the plastic weights for specified popname
    # this is the set of popname cells
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids]
    for cell in lcell:
      for conn in cell.conns:
        if 'hSTDP' in conn:
          #hstdp = conn.get('hSTDP')
          lsynweights.append(
              [t, conn.preGid, cell.gid, float(conn['hObj'].weight[PlastWeightIndex])])
    # return len(lcell)


def recordWeights(sim, t):
  """ record the STDP weights during the simulation
  """
  #lRcell = [c for c in sim.net.cells if c.gid in sim.net.pops['ER'].cellGids]
  sim.WeightsRecordingTimes.append(t)
  sim.allRLWeights.append([])  # Save this time
  sim.allNonRLWeights.append([])
  for cell in sim.net.cells:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        if conn.plast.params.RLon == 1:
          # save weight only for Rl-STDP conns
          sim.allRLWeights[-1].append(
              float(conn['hObj'].weight[PlastWeightIndex]))
        else:
          # save weight only for nonRL-STDP conns
          sim.allNonRLWeights[-1].append(
              float(conn['hObj'].weight[PlastWeightIndex]))


def saveWeights(sim, downSampleCells):
  ''' Save the weights for each plastic synapse '''
  with open(sim.RLweightsfilename, 'w') as fid1:
    count1 = 0
    for weightdata in sim.allRLWeights:
      # fid.write('%0.0f' % weightdata[0]) # Time
      # print(len(weightdata))
      fid1.write('%0.1f' % sim.WeightsRecordingTimes[count1])
      count1 = count1+1
      for i in range(0, len(weightdata), downSampleCells):
        fid1.write('\t%0.8f' % weightdata[i])
      fid1.write('\n')
  print(('Saved RL weights as %s' % sim.RLweightsfilename))
  with open(sim.NonRLweightsfilename, 'w') as fid2:
    count2 = 0
    for weightdata in sim.allNonRLWeights:
      # fid.write('%0.0f' % weightdata[0]) # Time
      # print(len(weightdata))
      fid2.write('%0.1f' % sim.WeightsRecordingTimes[count2])
      count2 = count2+1
      for i in range(0, len(weightdata), downSampleCells):
        fid2.write('\t%0.8f' % weightdata[i])
      fid2.write('\n')
  print(('Saved Non-RL weights as %s' % sim.NonRLweightsfilename))


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


def getAverageAdjustableWeights(sim, lpop=EMotorPops):
  # get average adjustable weights on a target population
  davg = {pop: 0.0 for pop in lpop}
  for pop in lpop:
    WSum = 0
    NSum = 0
    W, N = sumAdjustableWeightsPop(sim, pop)
    # destlist_on_root = pc.py_gather(srcitem, root)
    lw = sim.pc.py_gather(W, 0)
    ln = sim.pc.py_gather(N, 0)
    if sim.rank == 0:
      WSum = W + np.sum(lw)
      NSum = N + np.sum(ln)
      #print('rank= 0, pop=',pop,'W=',W,'N=',N,'wsum=',WSum,'NSum=',NSum)
      if NSum > 0:
        davg[pop] = WSum / NSum
    else:
      #destitem_from_root = sim.pc.py_scatter(srclist, root)
      pass
      # print('rank=',sim.rank,'pop=',pop,'Wm=',W,'N=',N)
  lsrc = [davg for i in range(sim.nhosts)] if sim.rank == 0 else None
  dest = sim.pc.py_scatter(lsrc, 0)
  return dest


def mulAdjustableWeights(sim, dfctr):
  # multiply adjustable STDP/RL weights by dfctr[pop] value for each population keyed in dfctr
  for pop in dfctr.keys():
    if dfctr[pop] == 1.0:
      continue
    # this is the set of cells
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids]
    for cell in lcell:
      for conn in cell.conns:
        if 'hSTDP' in conn:
          conn['hObj'].weight[PlastWeightIndex] *= dfctr[pop]


def saveGameBehavior(sim):
  with open(sim.ActionsRewardsfilename, 'w') as fid3:
    for i in range(len(sim.allActions)):
      fid3.write('%0.1f' % sim.allTimes[i])
      fid3.write('\t%0.1f' % sim.allActions[i])
      fid3.write('\t%0.5f' % sim.allRewards[i])
      fid3.write('\n')

######################################################################################


def getSpikesWithInterval(trange=None, neuronal_pop=None):
  if len(neuronal_pop) < 1:
    return 0.0
  spkts = sim.simData['spkt']
  spkids = sim.simData['spkid']
  pop_spikes = dict([(v,0) for v in set(neuronal_pop.values())])
  if len(spkts) > 0:
    # if random.random() < 0.005:
    #   print('length', len(spkts), spkts.buffer_size())
    len_skts = len(spkids)
    for idx in range(len_skts):
      i = len_skts - 1 - idx
      if trange[0] <= spkts[i] <= trange[1] and spkids[i] in neuronal_pop:
        pop_spikes[neuronal_pop[spkids[i]]] += 1
      if trange[0] > spkts[i]:
        break
  return pop_spikes


NBsteps = 0  # this is a counter for recording the plastic weights
epCount = []
dSTDPmech = {}  # dictionary of list of STDP mechanisms


def InitializeNoiseRates():
  # initialize the noise firing rates for the primary visual neuron populations (location V1 and direction sensitive)
  # based on image contents
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    # np.random.seed(1234)
    for pop in sim.lnoisety:
      if pop in sim.net.pops:
        for cell in sim.net.cells:
          if cell.gid in sim.net.pops[pop].cellGids:
            cell.hPointp.interval = 2
            cell.hPointp.start = 0  # np.random.uniform(0,1200)


def InitializeInputRates():
  # initialize the source firing rates for the primary visual neuron populations (location V1 and direction sensitive)
  # based on image contents
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    np.random.seed(1234)
    for pop in sim.lstimty:
      if pop in sim.net.pops:
        for cell in sim.net.cells:
          if cell.gid in sim.net.pops[pop].cellGids:
            cell.hPointp.interval = 1e12
            cell.hPointp.start = 0  # np.random.uniform(0,1200)


def updateInputRates():
  input_rates = sim.GameInterface.input_firing_rates()
  # print(input_rates[:4])
  input_rates = syncdata_alltoall(sim, input_rates)

  # if sim.rank == 0: print(dFiringRates['EV1'])
  # update input firing rates for stimuli to ER,EV1 and direction sensitive cells
  # different rules/code when dealing with artificial cells
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    lsz = len('stimMod')  # this is a prefix
    for pop in sim.lstimty:  # go through NetStim populations
      if pop in sim.net.pops:  # make sure the population exists
        # this is the set of NetStim cells
        lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids]
        offset = sim.simData['dminID'][pop]
        for cell in lCell:
          rate = input_rates[int(cell.gid-offset)]
          interval = 1000 / rate if rate != 0 else tstepPerAction
          cell.hPointp.interval = interval


def getActions(t, moves, pop_to_move):
  global fid4, tstepPerAction

  # Get move frequencies
  move_freq = {}
  vec = h.Vector()
  freq = []
  for ts in range(int(dconf['actionsPerPlay'])):
    ts_beg = t-tstepPerAction*(dconf['actionsPerPlay']-ts-1)
    ts_end = t-tstepPerAction*(dconf['actionsPerPlay']-ts)
    cgids_map = {}
    for move in moves:
      pop_name = [p for p, m in pop_to_move.items() if m == move][0]
      for cgid in sim.net.pops[pop_name].cellGids:
        cgids_map[cgid] = move
    freq.append(getSpikesWithInterval([ts_end, ts_beg], cgids_map))
  for move in moves:
    freq_move = [q[move] for q in freq]
    sim.pc.allreduce(vec.from_python(freq_move), 1)  # sum
    move_freq[move] = vec.to_python()

  actions = []

  if sim.rank == 0:
    if fid4 is None:
      fid4 = open(sim.MotorOutputsfilename, 'w')
    if dconf['verbose']:
      print('t={}: {} spikes: {}'.format(
          round(t,2), ','.join(moves), ','.join([str(move_freq[m]) for m in moves])))
    fid4.write('%0.1f' % t)
    for ts in range(int(dconf['actionsPerPlay'])):
      fid4.write(
          '\t' + '\t'.join([str(round(move_freq[m][ts], 1)) for m in moves]))
    fid4.write('\n')

    for ts in range(int(dconf['actionsPerPlay'])):
      no_firing_rates = sum([v[ts] for v in move_freq.values()]) == 0
      if no_firing_rates:
        # Should we initialize with random?
        print('Warning: No firing rates for moves {}!'.format(','.join(moves)))
        actions.append(dconf['moves']['LEFT'])
      else:
        mvsf = [(m, f[ts]) for m, f in move_freq.items()]
        random.shuffle(mvsf)
        best_move, best_move_freq = sorted(
            mvsf, key=lambda x: x[1], reverse=True)[0]
        if dconf['verbose']:
          print('Selected Move', best_move)
        actions.append(dconf['moves'][best_move])

  return actions


def trainAgent(t):
  """ training interface between simulation and game environment
  """
  global NBsteps, epCount, tstepPerAction

  t1 = datetime.now()

  # for the first time interval use randomly selected actions
  if t < (tstepPerAction*dconf['actionsPerPlay']):
    actions = []
    movecodes = [v for k, v in dconf['moves'].items()]
    for _ in range(int(dconf['actionsPerPlay'])):
      action = movecodes[random.randint(0, len(movecodes)-1)]
      actions.append(action)
  # the actions should be based on the activity of motor cortex (EMRIGHT, EMLEFT)
  else:
    actions = getActions(t, dconf['moves'], dconf['pop_to_move'])

  t1 = datetime.now() - t1
  t2 = datetime.now()

  if sim.rank == 0:
    rewards, done = sim.AIGame.playGame(actions)
    if done:
      ep_cnt = dconf['env']['episodes']
      eval_str = ''
      if len(sim.AIGame.count_steps) > ep_cnt:
        # take the steps of the latest `ep_cnt` episodes
        counted = [steps_per_ep for steps_per_ep in sim.AIGame.count_steps if steps_per_ep > 0][-ep_cnt:]
        # get the median
        eval_ep = np.median(counted)
        epCount.append(eval_ep)
        eval_str = '(median: {})'.format(eval_ep)
      
      last_steps = [k for k in sim.AIGame.count_steps if k != 0][-1]
      print('Episode finished in {} steps {}!'.format(last_steps, eval_str))

    # specific for CartPole-v1. TODO: move to a diff file
    if len(sim.AIGame.observations) == 0:
      raise Exception('Failed to get an observation from the Game')
    elif len(sim.AIGame.observations) == 1:
      critic = abs(sim.AIGame.observations[-1][2]) * 100
    else:
      critic = (sim.AIGame.observations[-1][2] -
                sim.AIGame.observations[-2][2]) * 100

    # use py_broadcast to avoid converting to/from Vector
    sim.pc.py_broadcast(critic, 0)  # broadcast critic value to other nodes

  else:  # other workers
    # receive critic value from master node
    critic = sim.pc.py_broadcast(None, 0)

  t2 = datetime.now() - t2
  t3 = datetime.now()

  if critic != 0:  # if critic signal indicates punishment (-1) or reward (+1)
    if dconf['verbose']:
      if sim.rank == 0:
        print('t={} Reward:{}'.format(round(t, 2), critic))
    for STDPmech in dSTDPmech['all']:
      STDPmech.reward_punish(critic)


  t3 = datetime.now() - t3
  t4 = datetime.now()

  if sim.rank == 0:
    sim.allActions.extend(actions)
    sim.allRewards.extend(rewards)
    tvec_actions = []
    for ts in range(len(actions)):
      tvec_actions.append(t-tstepPerAction*(len(actions)-ts-1))
    for ltpnt in tvec_actions:
      sim.allTimes.append(ltpnt)

  updateInputRates()  # update firing rate of inputs to R population (based on game state)


  t4 = datetime.now() - t4
  t5 = datetime.now()


  NBsteps += 1
  if NBsteps % recordWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank == 0:
      print('Weights Recording Time:', t, 'NBsteps:', NBsteps,
            'recordWeightStepSize:', recordWeightStepSize)
    recordAdjustableWeights(sim, t, dconf['pop_to_move'].keys())
    recordWeights(sim, t)

  t5 = datetime.now() - t5
  # if random.random() < 0.005:
  #   print([round(tk.microseconds / 1000, 0) for tk in [t1,t2,t3,t4,t5]])

def getAllSTDPObjects(sim):
  # get all the STDP objects from the simulation's cells
  # dictionary of STDP objects keyed by type (all, for EMRIGHT, EMLEFT populations)
  dSTDPmech = {'all': []}
  for pop in dconf['pop_to_move'].keys():
    dSTDPmech[pop] = []

  for cell in sim.net.cells:
    #if cell.gid in sim.net.pops['EMLEFT'].cellGids and cell.gid==sim.simData['dminID']['EMLEFT']: print(cell.conns)
    for conn in cell.conns:
      # check if the connection has a NEURON STDP mechanism object
      STDPmech = conn.get('hSTDP')
      if STDPmech:
        dSTDPmech['all'].append(STDPmech)
        for pop in dconf['pop_to_move'].keys():
          if cell.gid in sim.net.pops[pop].cellGids:
            dSTDPmech[pop].append(STDPmech)
  return dSTDPmech


# Alternate to create network and run simulation
# create network object and set cfg and net params; pass simulation config and network params as arguments
sim.initialize(simConfig=simConfig, netParams=netParams)

if sim.rank == 0:  # sim rank 0 specific init and backup of config file
  from aigame import AIGame
  sim.AIGame = AIGame(dconf)  # only create AIGame on node 0
  sim.GameInterface = GameInterface(sim.AIGame, dconf)
  # node 0 saves the json config file
  # this is just a precaution since simConfig pkl file has MOST of the info; ideally should adjust simConfig to contain
  # ALL of the required info
  from utils import backupcfg
  backupcfg(dconf['sim']['name'])
  os.makedirs('data', exist_ok=True)


# instantiate network populations
sim.net.createPops()
# instantiate network cells based on defined populations
sim.net.createCells()
# create connections between cells based on params
conns = sim.net.connectCells()
# instantiate netStim
sim.net.addStims()

if sim.rank == 0:
  fconn = 'data/'+dconf['sim']['name']+'_sim'
  sim.saveData(filename=fconn)


def setrecspikes():
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


setrecspikes()
# setup variables to record for each cell (spikes, V traces, etc)
sim.setupRecording()

dSTDPmech = getAllSTDPObjects(sim)  # get all the STDP objects up-front


def resumeSTDPWeights(sim, W):
  # this function assign weights stored in 'ResumeSimFromFile' to all connections by matching pre and post neuron ids
  # get all the simulation's cells (on a given node)
  for cell in sim.net.cells:
    cpostID = cell.gid  # find postID
    # find the record for a connection with post neuron ID
    WPost = W[(W.postid == cpostID)]
    for conn in cell.conns:
      if 'hSTDP' not in conn:
        continue
      cpreID = conn.preGid  # find preID
      if type(cpreID) != int:
        continue
      # find the record for a connection with pre and post neuron ID
      cConnW = WPost[(WPost.preid == cpreID)]
      # find weight for the STDP connection between preID and postID
      for idx in cConnW.index:
        cW = cConnW.at[idx, 'weight']
        conn['hObj'].weight[PlastWeightIndex] = cW
        #hSTDP = conn.get('hSTDP')
        #hSTDP.cumreward = cConnW.at[idx,'cumreward']
        if dconf['verbose'] > 1:
          print('weight updated:', cW)


# if specified 'ResumeSim' = 1, load the connection data from 'ResumeSimFromFile' and assign weights to STDP synapses
if dconf['simtype']['ResumeSim']:
  try:
    from simdat import readweightsfile2pdf
    A = readweightsfile2pdf(dconf['simtype']['ResumeSimFromFile'])
    # take the latest weights saved
    resumeSTDPWeights(sim, A[A.time == max(A.time)])
    sim.pc.barrier()  # wait for other nodes
    if sim.rank == 0:
      print('Updated STDP weights')
    # if 'normalizeWeightsAtStart' in dconf['sim']:
    #   if dconf['sim']['normalizeWeightsAtStart']:
    #     normalizeAdjustableWeights(sim, 0, lrecpop)
    #     print(sim.rank,'normalized adjustable weights at start')
    #     sim.pc.barrier() # wait for other nodes
  except:
    print('Could not restore STDP weights from file.')


def setdminID(sim, lpop):
  # setup min ID for each population in lpop
  # gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
  alltags = sim._gatherAllCellTags()
  dGIDs = {pop: [] for pop in lpop}
  for tinds in range(len(alltags)):
    if alltags[tinds]['pop'] in lpop:
      dGIDs[alltags[tinds]['pop']].append(tinds)
  sim.simData['dminID'] = {pop: np.amin(
      dGIDs[pop]) for pop in lpop if len(dGIDs[pop]) > 0}


setdminID(sim, allpops)
tPerPlay = tstepPerAction*dconf['actionsPerPlay']

# InitializeInputRates()# <-- Do not activate this!
InitializeNoiseRates()

# Plot 2d net
# sim.analysis.plot2Dnet(saveFig='data/net.png', showFig=False)
sim.analysis.plotConn(
          saveFig='data/connsCells.png', showFig=False,
          groupBy='cell', feature='weight')
includePre = list(dconf['net']['allpops'].keys())
sim.analysis.plotConn(saveFig='data/connsPops.png', showFig=False,
  includePre=includePre, includePost=includePre, feature='probability')

# has periodic callback to adjust STDP weights based on RL signal
sim.runSimWithIntervalFunc(tPerPlay, trainAgent)
if sim.rank == 0 and fid4 is not None:
  fid4.close()
if ECellModel == 'INTF7' or ICellModel == 'INTF7':
  intf7.insertSpikes(sim, simConfig.recordStep)
sim.gatherData()  # gather data from different nodes
sim.saveData()  # save data to disk


# print(sim.simData)
# print(dir(sim.simData))
# print(sim.simData['spkt'])
# print(sim.simData['spkid'])
# print(len(sim.simData['spkid']))
# print(sim.simData['spkt'].size())
# print(sim.simData['spkid'].size())
# for c,v in sim.simData['V_soma'].items():
#   print(c, v, v.size(), [v.get(idx) for idx in range(v.size())])

# if sim.simData['spkt'].size() == sim.simData['spkid'].size():
#   print('same size')
#   kt = sim.simData['spkt']
#   kid = sim.simData['spkid']
#   # print([(kt.get(idx), kid.get(idx)) for idx in range(kt.size())])
#   # Check what was stimulated:
#   for idx in range(kt.size()):
#     if kid.get(idx) < 62:
#       print(kt.get(idx), kid.get(idx))


def LSynWeightToD(L):
  # convert list of synaptic weights to dictionary to save disk space
  print('converting synaptic weight list to dictionary...')
  dout = {}
  doutfinal = {}
  for row in L:
    #t,preID,poID,w,cumreward = row
    t, preID, poID, w = row
    if preID not in dout:
      dout[preID] = {}
      doutfinal[preID] = {}
    if poID not in dout[preID]:
      dout[preID][poID] = []
      doutfinal[preID][poID] = []
    # dout[preID][poID].append([t,w,cumreward])
    dout[preID][poID].append([t, w])
  for preID in doutfinal.keys():
    for poID in doutfinal[preID].keys():
      doutfinal[preID][poID].append(dout[preID][poID][-1])
  return dout, doutfinal


def saveSynWeights():
  # save synaptic weights
  fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(sim.rank)+'.pkl'
  # save synaptic weights to disk for this node
  pickle.dump(lsynweights, open(fn, 'wb'))
  sim.pc.barrier()  # wait for other nodes
  time.sleep(1)
  if sim.rank == 0:  # rank 0 reads and assembles the synaptic weights into a single output file
    L = []
    for i in range(sim.nhosts):
      fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(i)+'.pkl'
      while not os.path.isfile(fn):  # wait until the file is written/available
        print('saveSynWeights: waiting for finish write of', fn)
        time.sleep(1)
      lw = pickle.load(open(fn, 'rb'))
      print(fn, 'len(lw)=', len(lw), type(lw))
      os.unlink(fn)  # remove the temporary file
      L = L + lw  # concatenate to the list L
    # pickle.dump(L,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb')) # this would save as a List
    # now convert the list to a dictionary to save space, and save it to disk
    dout, doutfinal = LSynWeightToD(L)
    pickle.dump(dout, open(
        'data/'+dconf['sim']['name']+'synWeights.pkl', 'wb'))
    pickle.dump(doutfinal, open(
        'data/'+dconf['sim']['name']+'synWeights_final.pkl', 'wb'))


if sim.saveWeights:
  saveSynWeights()


def saveAssignedFiringRates(dAllFiringRates):
  pickle.dump(
      dAllFiringRates,
      open('data/'+dconf['sim']['name']+'AssignedFiringRates.pkl', 'wb'))

def prepraster(lpops):
  # lpops = dnumc
  dstartidx,dendidx={},{} # starting,ending indices for each population
  for p in lpops.keys():
    if lpops[p] > 0:
      dstartidx[p] = sim.simData['dminID'][p]
      dendidx[p] = sim.simData['dminID'][p] + lpops[p] - 1
  spkID= np.array(sim.simData['spkid'])
  spkT = np.array(sim.simData['spkt'])
  dspkID,dspkT = {},{}
  for pop in lpops.keys():
    # if dnumc[pop] > 0:
      dspkID[pop] = spkID[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
      dspkT[pop] = spkT[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
  return dspkID, dspkT


if sim.rank == 0:  # only rank 0 should save. otherwise all the other nodes could over-write the output or quit first; rank 0 plots
  if dconf['sim']['doplot']:
    print('plot raster:')
    # sim.analysis.plotData(saveFig='data/data.png')
    # sim.analysis.plotData()
    # sim.analysis.spikes.plotRaster(saveFig='data/raster.png')
  if sim.plotWeights:
    plotWeights()
  saveGameBehavior(sim)
  with open('data/'+dconf['sim']['name']+'ActionsPerEpisode.txt', 'w') as fid5:
    for i in range(len(epCount)):
      fid5.write('\t%0.1f' % epCount[i])
      fid5.write('\n')
  # if sim.saveInputImages: saveInputImages(sim.AIGame.ReducedImages)
  # #anim.savemp4('/tmp/*.png','data/'+dconf['sim']['name']+'randGameBehavior.mp4',10)
  # if sim.saveMotionFields: saveMotionFields(sim.AIGame.ldflow)
  # if sim.saveObjPos: saveObjPos(sim.AIGame.dObjPos)
  # if sim.saveAssignedFiringRates: saveAssignedFiringRates(sim.AIGame.dAllFiringRates)

  lpops = dict([(k,v) for k,v in dconf['net']['allpops'].items() if v > 0])
  for ty in sim.lstimty:
    lpops[ty] = dconf['net']['allpops'][dconf['net']['inputPop']]
  dspkID, dspkT = prepraster(lpops)
  drawraster(
    [k for k,v in lpops.items() if v > 0],
    dspkT, dspkID, dnumc, totalDur=dconf['sim']['duration'],
    figname='data/{}_raster.png'.format(dconf['sim']['name']))

  if dconf['sim']['doquit']:
    quit()
