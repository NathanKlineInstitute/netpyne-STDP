from netpyne import specs, sim
from neuron import h

import random
import pickle
import os
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from matplotlib import pyplot as plt

from cells import intf7
from game_interface import GameInterface
from critic import Critic
from utils.conns import getconv, getSec, getInitDelay, getDelay, setdminID, setrecspikes
from utils.plots import plotRaster, plotWeights, saveActionsPerEpisode
from utils.sync import syncdata_alltoall
from utils.weights import saveSynWeights, readWeights, getWeightIndex, getInitSTDPWeight

# this will not work properly across runs with different number of nodes
random.seed(1234)


def isExc(ty): return ty.startswith('E')
def isInh(ty): return ty.startswith('I')
def connType(prety, poty): return prety[0] + poty[0]


class NeuroSim:
  def __init__(self, dconf):
    self.dconf = dconf

    def outpath(fname): return os.path.join(dconf['sim']['outdir'], fname)
    self.outpath = outpath

    sim.davgW = {}  # average adjustable weights on a target population
    sim.allTimes = []
    sim.allRewards = []  # list to store all rewards
    sim.allActions = []  # list to store all actions
    # list to store firing rate of output motor neurons.
    sim.allMotorOutputs = []
    sim.allSTDPWeights = []

    sim.ActionsRewardsfilename = outpath('ActionsRewards.txt')
    sim.MotorOutputsfilename = outpath('MotorOutputs.txt')
    # sim.NonRLweightsfilename = outpath('NonRLweights.txt')  # file to store weights

    sim.plotWeights = 0  # plot weights
    sim.plotRaster = 1
    if 'plotRaster' in dconf['sim']:
      sim.plotRaster = dconf['sim']['plotRaster']
    sim.doSaveData = 1
    if 'doSaveData' in dconf['sim']:
      sim.doSaveData = dconf['sim']['doSaveData']
    sim.saveWeights = 1  # save weights
    if 'saveWeights' in dconf['sim']:
      sim.saveWeights = dconf['sim']['saveWeights']
    # whether to save the motion fields
    sim.saveMotionFields = dconf['sim']['saveMotionFields']
    sim.saveObjPos = 1  # save ball and paddle position to file
    self.recordWeightStepSize = dconf['sim']['recordWeightStepSize']
    self.normalizeStepSize = dconf['sim']['normalizeStepSize'] if 'normalizeStepSize' in dconf['sim'] else None
    self.normalizeByOutputBalancing = dconf['sim']['normalizeByOutputBalancing'] if 'normalizeByOutputBalancing' in dconf['sim'] else None
    self.normalizeOutBalMinMax = dconf['sim']['normalizeOutBalMinMax'] if 'normalizeOutBalMinMax' in dconf['sim'] else None
    self.normalizeVerbose = dconf['sim']['normalizeVerbose'] if 'normalizeVerbose' in dconf['sim'] else None
    self.normInMeans = None
    self.normOutMeans = None
    # time step per action (in ms)
    self.tstepPerAction = dconf['sim']['tstepPerAction']

    self.allpops = list(dconf['net']['allpops'].keys())
    self.inputPop = dconf['net']['inputPop']
    self.unk_move = dconf['env']['unk_move'] if 'unk_move' in dconf['env'] else min(dconf['moves'].values()) - 1
    # number of neurons of a given type: dnumc
    # scales the size of the network (only number of neurons)
    scale = dconf['net']['scale']
    self.dnumc = OrderedDict(
        {ty: dconf['net']['allpops'][ty] * scale for ty in self.allpops})

    # connection matrix (for classes, synapses, probabilities [probabilities not used for topological conn])
    self.cmat = dconf['net']['cmat']

    # Network parameters
    netParams = specs.NetParams()
    # spike threshold, 10 mV is NetCon default, lower it for all cells
    netParams.defaultThreshold = 0.0
    self.netParams = netParams

    # object of class SimConfig to store simulation configuration
    simConfig = specs.SimConfig()
    # Simulation options
    # Duration of the simulation, in seconds
    simConfig.duration = dconf['sim']['duration'] * 1000
    # Internal integration timestep to use
    simConfig.dt = dconf['sim']['dt']
    # make sure temperature is set. otherwise we're at squid temperature
    simConfig.hParams['celsius'] = 37
    # Show detailed messages
    simConfig.verbose = dconf['sim']['verbose']
    # Dict with traces to record
    simConfig.recordTraces = {'V_soma': {
        'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
    # this means record from all neurons - including stim populations, if any
    simConfig.recordCellsSpikes = [-1]
    # Step size in ms to save data (e.g. V traces, LFP, etc)
    simConfig.recordStep = dconf['sim']['recordStep']
    simConfig.filename = outpath('simConfig')  # Set file output name
    simConfig.saveJson = False
    # Save params, network and sim output to pickle file
    simConfig.savePickle = True
    simConfig.saveMat = False
    simConfig.saveFolder = 'data'
    simConfig.createNEURONObj = True  # create HOC objects when instantiating network
    # create Python structure (simulator-independent) when instantiating network
    simConfig.createPyStruct = True
    simConfig.analysis['plotTraces'] = {
        'include': [(pop, 0) for pop in dconf['net']['allpops'].keys()]
    }
    self.simConfig = simConfig
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
    self.cfg = cfg

    self.ECellModel = dconf['net']['ECellModel']
    self.ICellModel = dconf['net']['ICellModel']

    # Population parameters
    for ty in self.allpops:
      self.netParams.popParams[ty] = {
          'cellType': ty,
          'numCells': self.dnumc[ty],
          'cellModel': self.ECellModel if isExc(ty) else self.ICellModel}

    self.makeECellModel()
    self.makeICellModel()

    # Synaptic mechanism parameters
    # note that these synaptic mechanisms are not used for the INTF7 neurons
    for synMech, synMechParams in dconf['net']['synMechParams'].items():
      self.netParams.synMechParams[synMech] = synMechParams

    self.dSTDPparams = self.readSTDPParams()

    # Note: when using IntFire4 cells lstimty has the NetStim populations
    # that send spikes to EV1, EV1DE, etc.
    sim.lstimty = self.setupStimMod()
    for ty in sim.lstimty:
      self.allpops.append(ty)

    # Setup noise stimulation
    sim.lnoisety = self.setupNoiseStim()
    for ty in sim.lnoisety:
      self.allpops.append(ty)

    self.setupSTDPWeights()

    # Alternate to create network and run simulation
    # create network object and set cfg and net params; pass simulation config and network params as arguments
    sim.initialize(simConfig=self.simConfig, netParams=self.netParams)

    ################################
    ######### End __init__ #########
    ################################

  def makeECellModel(self):
    # create rules for excitatory neuron models
    self.EExcitSec = 'dend'  # section where excitatory synapses placed
    self.PlastWeightIndex = 0  # NetCon weight index where plasticity occurs
    if self.ECellModel == 'IntFire4':
      self.EExcitSec = 'soma'  # section where excitatory synapses placed
      # Dict with traces to record
      self.simConfig.recordTraces = {'V_soma': {'var': 'm'}}
      self.netParams.defaultThreshold = 0.0
      for ty in self.allpops:
        if isExc(ty):
          self.netParams.popParams[ty] = {'cellType': ty,
                                          'cellModel': 'IntFire4',
                                          'numCells': self.dnumc[ty],
                                          'taue': 1.0}  # pop of IntFire4
    elif self.ECellModel == 'INTF7':
      self.EExcitSec = 'soma'  # section where excitatory synapses placed
      # Dict with traces to record
      self.simConfig.recordTraces = {'V_soma': {'var': 'Vm'}}
      self.netParams.defaultThreshold = -40.0
      ecell = intf7.INTF7E(self.dconf)
      for ty in self.allpops:
        if isExc(ty):
          self.netParams.popParams[ty] = {'cellType': ty,
                                          'cellModel': 'INTF7',
                                          'numCells': self.dnumc[ty]}
          for k, v in ecell.dparam.items():
            self.netParams.popParams[ty][k] = v
      self.PlastWeightIndex = intf7.dsyn['AM2']

  def makeICellModel(self):
    # create rules for inhibitory neuron models
    if self.ICellModel == 'IntFire4':
      # Dict with traces to record
      self.simConfig.recordTraces = {'V_soma': {'var': 'm'}}
      self.netParams.defaultThreshold = 0.0
      for ty in self.allpops:
        if isInh(ty):
          self.netParams.popParams[ty] = {'cellType': ty, 'cellModel': 'IntFire4',
                                          'numCells': self.dnumc[ty], 'taue': 1.0}  # pop of IntFire4
    elif self.ICellModel == 'INTF7':
      # Dict with traces to record
      self.simConfig.recordTraces = {'V_soma': {'var': 'Vm'}}
      self.netParams.defaultThreshold = -40.0
      ilcell = intf7.INTF7IL(self.dconf)
      icell = intf7.INTF7I(self.dconf)
      for ty in self.allpops:
        if isInh(ty):
          self.netParams.popParams[ty] = {'cellType': ty,
                                          'cellModel': 'INTF7',
                                          'numCells': self.dnumc[ty]}
          if ty.count('L') > 0:  # LTS
            for k, v in ilcell.dparam.items():
              self.netParams.popParams[ty][k] = v
          else:  # FS
            for k, v in icell.dparam.items():
              self.netParams.popParams[ty][k] = v

  def readSTDPParams(self):
    lsy = ['AMPA', 'NMDA', 'AMPAI']
    gains = [self.cfg.Gain[gainType] for gainType in ['EE', 'EE', 'EI']]
    dSTDPparams = {}  # STDP-RL/STDPL parameters for AMPA,NMDA synapses;
    # generally uses shorter,longer eligibility traces
    for sy, gain in zip(lsy, gains):
      # Parameters defined at:
      # https://github.com/Neurosim-lab/netpyne/blob/development/examples/RL_arm/stdp.mod
      dSTDPparams[sy] = self.dconf['STDP-RL'][sy]
      for k in dSTDPparams[sy].keys():
        if k.count('wt') or k.count('wbase') or k.count('wmax'):
          dSTDPparams[sy][k] *= gain

    dSTDPparams['AM2'] = dSTDPparams['AMPA']
    dSTDPparams['NM2'] = dSTDPparams['NMDA']
    return dSTDPparams

  def setupStimMod(self):
    # setup variable rate NetStim sources (send spikes based on input contents)
    lstimty = []
    stimModW = self.dconf['net']['stimModW']
    if self.ECellModel == 'IntFire4' or self.ECellModel == 'INTF7':
      lpoty = [self.inputPop]
      for poty in lpoty:
        if self.dnumc[poty] <= 0:
          continue
        stimty = 'stimMod'+poty
        lstimty.append(stimty)
        self.netParams.popParams[stimty] = {
            'cellModel': 'NSLOC',
            'numCells': self.dnumc[poty],
            'rate': 'variable',
            'noise': 0,
            'start': 0}
        blist = [[i, i] for i in range(self.dnumc[poty])]
        self.netParams.connParams[stimty+'->'+poty] = {
            'preConds': {'pop': stimty},
            'postConds': {'pop': poty},
            'weight': stimModW,
            'delay': getInitDelay(self.dconf, 'STIMMOD'),
            'connList': blist,
            'weightIndex': getWeightIndex('AMPA', self.ECellModel)}
    return lstimty

  def setupNoiseStim(self):
    lnoisety = []
    dnoise = self.dconf['noise']
    # setup noisy NetStim sources (send random spikes)
    if self.ECellModel == 'IntFire4' or self.ECellModel == 'INTF7':
      for poty, dpoty in dnoise.items():
        for sy, dsy in dpoty.items():
          damp = self.dconf['net']['noiseDamping']['E' if isExc(poty) else 'I']
          Weight, Rate = dsy['w'] * damp, dsy['rate']
          # print(poty, isExc(poty), damp, Weight)
          if Weight > 0.0 and Rate > 0.0:
            # only create the netstims if rate,weight > 0
            stimty = 'stimNoise'+poty+'_'+sy
            self.netParams.popParams[stimty] = {
                'cellModel': 'NetStim',
                'numCells': dnumc[poty],
                'rate': Rate,
                'noise': 1.0,
                'start': 0}
            blist = [[i, i] for i in range(dnumc[poty])]
            self.netParams.connParams[stimty+'->'+poty] = {
                'preConds': {'pop': stimty},
                'postConds': {'pop': poty},
                'weight': Weight,
                'delay': getDelay(self.dconf, None, poty, sy),
                'connList': blist,
                'weightIndex': getWeightIndex(sy, self.ECellModel)}
            lnoisety.append(stimty)
    return lnoisety

  ######################################################################################
  #####################################################################################

  def setupSTDPWeights(self):
    synToMech = self.dconf['net']['synToMech']
    sytypes = self.dconf['net']['synToMech'].keys()

    # Setup cmat connections
    stdpConns = self.dconf['net']['STDPconns']
    for prety, dprety in self.cmat.items():
      if self.dnumc[prety] <= 0:
        continue
      for poty, dconn in dprety.items():
        if self.dnumc[poty] <= 0:
          continue
        ct = connType(prety, poty)
        for sy in sytypes:
          if sy in self.cmat[prety][poty] and self.cmat[prety][poty][sy] > 0:
            k = '{}-{}->{}'.format(prety, sy, poty)
            sec = getSec(prety, poty, sy)
            weight = self.cmat[prety][poty][sy] * self.cfg.Gain[ct]
            self.netParams.connParams[k] = {
                'preConds': {'pop': prety},
                'postConds': {'pop': poty},
                'convergence': getconv(self.cmat, prety, poty, self.dnumc[prety]),
                'weight': weight,
                'delay': getDelay(self.dconf, prety, poty, sy, sec),
                'synMech': synToMech[sy],
                'sec': sec,
                'loc': 0.5,
                'weightIndex': getWeightIndex(
                    sy, self.ICellModel if isInh(poty) else self.ECellModel)
            }
            # Setup STDP plasticity rules
            if ct in stdpConns and stdpConns[ct] and self.dSTDPparams[synToMech[sy]]['RLon']:
              print('Setting RL-STDP on {} ({})'.format(k, weight))
              self.netParams.connParams[k]['plast'] = {
                  'mech': 'STDP', 'params': self.dSTDPparams[synToMech[sy]]}
              self.netParams.connParams[k]['weight'] = getInitSTDPWeight(
                  self.cfg, weight)

  def resumeSTDPWeights(self, sim, W):
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
          conn['hObj'].weight[self.PlastWeightIndex] = cW
          # hSTDP = conn.get('hSTDP')
          #hSTDP.cumreward = cConnW.at[idx,'cumreward']
          if self.dconf['verbose'] != 0:
            print('weight updated:', cW)

  ###################################################################################################################################

  def recordWeights(self, sim, t):
    # record all weights during the simulation
    for cell in sim.net.cells:
      for conn in cell.conns:
        if 'hSTDP' in conn:
          weightItem = [t, conn.preGid, cell.gid, float(
              conn['hObj'].weight[self.PlastWeightIndex])]
          sim.allSTDPWeights.append(weightItem)

  def weightsMean(self, sim, ctype):
    # if ctype == 'in' compute the averages of incoming connections of a neuron
    # if ctype == 'out' compute the averages of outgoing connections of a neuron
    assert ctype in ['in', 'out']
    weights_vec = {}
    for cell in sim.net.cells:
      for conn in cell.conns:
        if 'hSTDP' in conn:
          gid = conn.preGid if ctype == 'out' else cell.gid
          if gid not in weights_vec:
            weights_vec[gid] = []
          weights_vec[gid].append(conn['hObj'].weight[self.PlastWeightIndex])
    return dict([(k, np.mean(v)) for k,v in weights_vec.items()])

  def normalizeInWeights(self, sim):
    curr_means = self.weightsMean(sim, ctype='in')
    norm_means = self.normInMeans
    cell_scales = {}
    for cell in sim.net.cells:
      cell_scale = None
      for conn in cell.conns:
        if 'hSTDP' in conn:
          cell_scale = norm_means[cell.gid] / curr_means[cell.gid]
          conn['hObj'].weight[self.PlastWeightIndex] *= cell_scale
      if cell_scale:
        cell_scales[cell.gid] = cell_scale
    if self.normalizeVerbose:
      llscales = sorted(list(cell_scales.items()), key=lambda x:x[1])
      print('Inminmax scales:', llscales[0], llscales[-1])


  def getSpikesWithInterval(self, trange=None, neuronal_pop=None):
    if len(neuronal_pop) < 1:
      return 0.0
    spkts = sim.simData['spkt']
    spkids = sim.simData['spkid']
    pop_spikes = dict([(v, 0) for v in set(neuronal_pop.values())])
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

  def InitializeNoiseRates(self, sim):
    # initialize the noise firing rates for the primary visual neuron populations (location V1 and direction sensitive)
    if self.ECellModel == 'IntFire4' or self.ECellModel == 'INTF7':
      # np.random.seed(1234)
      for pop in sim.lnoisety:
        if pop in sim.net.pops:
          for cell in sim.net.cells:
            if cell.gid in sim.net.pops[pop].cellGids:
              cell.hPointp.interval = 2
              cell.hPointp.start = 0  # np.random.uniform(0,1200)

  def InitializeInputRates(self, sim):
    # initialize the source firing rates for the primary visual neuron populations (location V1 and direction sensitive)
    if self.ECellModel == 'IntFire4' or self.ECellModel == 'INTF7':
      # np.random.seed(1234)
      for pop in sim.lstimty:
        if pop in sim.net.pops:
          for cell in sim.net.cells:
            if cell.gid in sim.net.pops[pop].cellGids:
              cell.hPointp.interval = 1e12
              cell.hPointp.start = 0  # np.random.uniform(0,1200)

  def updateInputRates(self, sim):
    input_rates = sim.GameInterface.input_firing_rates()
    # print(input_rates[:4])
    input_rates = syncdata_alltoall(sim, input_rates)

    # if sim.rank == 0: print(dFiringRates['EV1'])
    # update input firing rates for stimuli to ER,EV1 and direction sensitive cells
    # different rules/code when dealing with artificial cells
    if self.ECellModel == 'IntFire4' or self.ECellModel == 'INTF7':
      lsz = len('stimMod')  # this is a prefix
      for pop in sim.lstimty:  # go through NetStim populations
        if pop in sim.net.pops:  # make sure the population exists
          # this is the set of NetStim cells
          lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids]
          offset = sim.simData['dminID'][pop]
          for cell in lCell:
            rate = input_rates[int(cell.gid-offset)]
            interval = 1000 / rate if rate != 0 else 1e12
            if cell.hPointp.interval != interval:
              cell.hPointp.interval = interval

  def getAllSTDPObjects(self, sim):
    # get all the STDP objects from the simulation's cells
    # dictionary of STDP objects keyed by type (all, for EMRIGHT, EMLEFT populations)
    dSTDPmech = {'all': [], 'outConns': {}}
    for pop in self.dconf['pop_to_moves'].keys():
      dSTDPmech[pop] = []

    for cell in sim.net.cells:
      #if cell.gid in sim.net.pops['EMLEFT'].cellGids and cell.gid==sim.simData['dminID']['EMLEFT']: print(cell.conns)
      for conn in cell.conns:
        # check if the connection has a NEURON STDP mechanism object
        STDPmech = conn.get('hSTDP')
        if STDPmech:
          dSTDPmech['all'].append(STDPmech)

          # Set all STDP Mechs Output connections indexed by each presynaptic neuron
          if conn.preGid not in dSTDPmech['outConns']:
            dSTDPmech['outConns'][conn.preGid] = []
          dSTDPmech['outConns'][conn.preGid].append(STDPmech)

          # Set all STDP Mechs indexed by each output population
          for pop in self.dconf['pop_to_moves'].keys():
            if cell.gid in sim.net.pops[pop].cellGids:
              dSTDPmech[pop].append(STDPmech)
    return dSTDPmech

  def getActions(self, sim, t, moves, pop_to_moves):
    # Get move frequencies
    move_freq = {}
    vec = h.Vector()
    freq = []
    for ts in range(int(self.dconf['actionsPerPlay'])):
      ts_beg = t-self.tstepPerAction*(self.dconf['actionsPerPlay']-ts-1)
      ts_end = t-self.tstepPerAction*(self.dconf['actionsPerPlay']-ts)
      cgids_map = {}
      for p, pop_moves in pop_to_moves.items():
        if type(pop_moves) == str:
          pop_moves = [pop_moves]
        cells_per_move = math.floor(
            len(sim.net.pops[p].cellGids) / len(pop_moves))
        for idx, cgid in enumerate(sim.net.pops[p].cellGids):
          cgids_map[cgid] = pop_moves[math.floor(idx / cells_per_move)]

      freq.append(self.getSpikesWithInterval([ts_end, ts_beg], cgids_map))
    for move in moves:
      freq_move = [q[move] for q in freq]
      sim.pc.allreduce(vec.from_python(freq_move), 1)  # sum
      move_freq[move] = vec.to_python()

    actions = []

    if sim.rank == 0:
      if self.dconf['verbose']:
        print('t={}: {} spikes: {}'.format(
            round(t, 2), ','.join(moves), ','.join([str(move_freq[m]) for m in moves])))
      with open(sim.MotorOutputsfilename, 'a') as fid4:
        fid4.write('%0.1f' % t)
        for ts in range(int(self.dconf['actionsPerPlay'])):
          fid4.write(
              '\t' + '\t'.join([str(round(move_freq[m][ts], 1)) for m in moves]))
        fid4.write('\n')

      for ts in range(int(self.dconf['actionsPerPlay'])):
        no_firing_rates = sum([v[ts] for v in move_freq.values()]) == 0
        if no_firing_rates:
          # Should we initialize with random?
          if self.dconf['verbose']:
            print('Warning: No firing rates for moves {}!'.format(','.join(moves)))
          else:
            print('.', end='')
          # actions.append(self.dconf['moves']['LEFT'])
          actions.append(self.unk_move)
        else:
          mvsf = [(m, f[ts]) for m, f in move_freq.items()]
          random.shuffle(mvsf)
          mvsf = sorted(mvsf, key=lambda x: x[1], reverse=True)
          best_move, best_move_freq = mvsf[0]
          if best_move_freq == mvsf[-1][1]:
            if self.dconf['verbose']:
              print('Warning: No discrimination between moves, fired: {}!'.format(
                [f for m,f in mvsf]))
            else:
              print(str(round(best_move_freq)) + '-', end='')
            actions.append(self.unk_move)
          else:
            actions.append(self.dconf['moves'][best_move])
          if self.dconf['verbose']:
            print('Selected Move', best_move)

    return actions

  def trainAgent(self, t):
    """ training interface between simulation and game environment
    """
    dconf = self.dconf

    t1 = datetime.now()

    # Measure and cache normalized initial weights
    if self.normalizeStepSize and not self.normInMeans:
      self.normInMeans = self.weightsMean(sim, ctype='in')
    if self.normalizeByOutputBalancing and not self.normOutMeans:
      self.normOutMeans = self.weightsMean(sim, ctype='out')

    # for the first time interval use randomly selected actions
    if t < (self.tstepPerAction*dconf['actionsPerPlay']):
      actions = []
      movecodes = [v for k, v in dconf['moves'].items()]
      for _ in range(int(dconf['actionsPerPlay'])):
        action = movecodes[random.randint(0, len(movecodes)-1)]
        actions.append(action)
    # the actions should be based on the activity of motor cortex (EMRIGHT, EMLEFT)
    else:
      actions = self.getActions(sim, t, dconf['moves'], dconf['pop_to_moves'])

    t1 = datetime.now() - t1
    t2 = datetime.now()

    if sim.rank == 0:
      is_unk_move = len([a for a in actions if a == self.unk_move]) > 0
      actions = [a if a != self.unk_move else sim.AIGame.randmove()
        for a in actions]
      rewards, done = sim.AIGame.playGame(actions)
      if done:
        ep_cnt = dconf['env']['episodes']
        eval_str = ''
        if len(sim.AIGame.count_steps) > ep_cnt:
          # take the steps of the latest `ep_cnt` episodes
          counted = [
              steps_per_ep for steps_per_ep in sim.AIGame.count_steps if steps_per_ep > 0][-ep_cnt:]
          # get the median
          eval_ep = np.median(counted)
          eval_str = '(median: {})'.format(eval_ep)

        last_steps = [k for k in sim.AIGame.count_steps if k != 0][-1]
        self.epCount.append(last_steps)
        print('Episode finished in {} steps {}!'.format(last_steps, eval_str))

      # specific for CartPole-v1. TODO: move to a diff file
      if len(sim.AIGame.observations) == 0:
        raise Exception('Failed to get an observation from the Game')
      else:
        reward = self.critic.calc_reward(
            sim.AIGame.observations[-1],
            sim.AIGame.observations[-2] if len(sim.AIGame.observations) > 1 else None,
            is_unk_move)

      # use py_broadcast to avoid converting to/from Vector
      sim.pc.py_broadcast(reward, 0)  # broadcast reward value to other nodes

    else:  # other workers
      # receive reward value from master node
      reward = sim.pc.py_broadcast(None, 0)

    t2 = datetime.now() - t2
    t3 = datetime.now()

    # if reward signal indicates punishment (-1) or reward (+1)
    if reward != 0:
      if dconf['verbose']:
        if sim.rank == 0:
          print('t={} Reward:{} Actions: {}'.format(round(t, 2), reward, actions))
      if self.normalizeByOutputBalancing:
        # Scale the reward/punishment given to a cell based on its output power:
        # If a neuron already increased all its output weights: 
        #     don't reward that neuron's connections as much, but
        #     punish severely
        # If a neuron has all its output weights decreased:
        #     rewards are having a stronger effect
        #     punishments barely change the weights
        # Effects are limited by a min and a max
        curr_means = self.weightsMean(sim, ctype='out')
        norm_means = self.normOutMeans
        cell_scales = {}
        for cGid, STDPmechs in self.dSTDPmech['outConns'].items():
          cell_scale = norm_means[cGid] / curr_means[cGid]
          if reward > 0: cell_scale = 1.0 / cell_scale
          cell_scale = max(self.normalizeOutBalMinMax[0], min(cell_scale, self.normalizeOutBalMinMax[1]))
          cell_scales[cGid] = cell_scale
          for STDPmech in STDPmechs:
            STDPmech.reward_punish(reward * cell_scale)
        if self.normalizeVerbose:
          llscales = sorted(list(cell_scales.items()), key=lambda x:x[1])
          print('Outminmax scales:', llscales[0], llscales[-1])
      else:
        for STDPmech in self.dSTDPmech['all']:
          STDPmech.reward_punish(reward)

    t3 = datetime.now() - t3
    t4 = datetime.now()

    if sim.rank == 0:
      sim.allActions.extend(actions)
      sim.allRewards.append(reward)
      tvec_actions = []
      for ts in range(len(actions)):
        tvec_actions.append(t-self.tstepPerAction*(len(actions)-ts-1))
      for ltpnt in tvec_actions:
        sim.allTimes.append(ltpnt)

      with open(sim.ActionsRewardsfilename, 'a') as fid3:
        for action, t in zip(actions, tvec_actions):
          fid3.write('%0.1f\t%0.1f\t%0.5f\n' % (t, action, reward))

    # update firing rate of inputs to R population (based on game state)
    self.updateInputRates(sim)

    t4 = datetime.now() - t4
    t5 = datetime.now()

    self.NBsteps += 1
    if self.NBsteps % self.recordWeightStepSize == 0:
      if dconf['verbose'] > 0 and sim.rank == 0:
        print('Weights Recording Time:', t, 'NBsteps:', self.NBsteps,
              'recordWeightStepSize:', self.recordWeightStepSize)
      self.recordWeights(sim, t)
    if self.normalizeStepSize and self.NBsteps % self.normalizeStepSize == 0:
      self.normalizeInWeights(sim)

    t5 = datetime.now() - t5
    if random.random() < 0.001:
      print(t, [round(tk.microseconds / 1000, 0)
                for tk in [t1, t2, t3, t4, t5]])

    if 'sleeptrial' in dconf['sim'] and dconf['sim']['sleeptrial']:
      time.sleep(dconf['sim']['sleeptrial'])

  def run(self):
    sim.AIGame = None  # placeholder

    self.NBsteps = 0  # this is a counter for recording the plastic weights
    self.epCount = []

    if sim.rank == 0:  # sim rank 0 specific init
      from aigame import AIGame
      sim.AIGame = AIGame(self.dconf)  # only create AIGame on node 0
      sim.GameInterface = GameInterface(sim.AIGame, self.dconf)

    self.critic = Critic(self.dconf)

    # instantiate network populations
    sim.net.createPops()
    # instantiate network cells based on defined populations
    sim.net.createCells()
    # create connections between cells based on params
    conns = sim.net.connectCells()
    # instantiate netStim
    sim.net.addStims()

    # get all the STDP objects up-front
    self.dSTDPmech = self.getAllSTDPObjects(sim)

    if self.dconf['simtype']['ResumeSim']:
      # if specified 'ResumeSim' = 1, load the connection data from
      # 'ResumeSimFromFile' and assign weights to STDP synapses
      try:
        A = readWeights(self.dconf['simtype']['ResumeSimFromFile'])
        # take the latest weights saved
        resume_ts = max(A.time)
        if 'ResumeSimFromTs' in self.dconf['simtype']:
          resume_ts = self.dconf['simtype']['ResumeSimFromTs']
        self.resumeSTDPWeights(sim, A[A.time == resume_ts])
        sim.pc.barrier()  # wait for other nodes
        if sim.rank == 0:
          print('Updated STDP weights')
        # if 'normalizeWeightsAtStart' in self.dconf['sim']:
        #   if self.dconf['sim']['normalizeWeightsAtStart']:
        #     normalizeAdjustableWeights(sim, 0, lrecpop)
        #     print(sim.rank,'normalized adjustable weights at start')
        #     sim.pc.barrier() # wait for other nodes
      except Exception as err:
        print('Could not restore STDP weights from file.')
        raise err

    if sim.rank == 0:
      fconn = self.outpath('sim')
      sim.saveData(filename=fconn)

    setrecspikes(self.dconf, sim)
    # setup variables to record for each cell (spikes, V traces, etc)
    sim.setupRecording()

    setdminID(sim, self.allpops)
    tPerPlay = self.tstepPerAction * self.dconf['actionsPerPlay']

    # self.InitializeInputRates(sim)# <-- Do not activate this!
    self.InitializeNoiseRates(sim)

    # Plot 2d net
    # sim.analysis.plot2Dnet(saveFig='data/net.png', showFig=False)
    sim.analysis.plotConn(
        saveFig=self.outpath('connsCells.png'), showFig=False,
        groupBy='cell', feature='weight')
    includePre = list(self.dconf['net']['allpops'].keys())
    sim.analysis.plotConn(saveFig=self.outpath('connsPops.png'), showFig=False,
                          includePre=includePre, includePost=includePre, feature='probability')

    # Record weights at time 0
    self.recordWeights(sim, 0)
    # has periodic callback to adjust STDP weights based on RL signal
    sim.runSimWithIntervalFunc(tPerPlay, self.trainAgent)
    if self.ECellModel == 'INTF7' or self.ICellModel == 'INTF7':
      intf7.insertSpikes(sim, self.simConfig.recordStep)
    sim.gatherData()  # gather data from different nodes
    if sim.doSaveData:
      sim.saveData()  # save data to disk

    if sim.saveWeights:
      saveSynWeights(sim, sim.allSTDPWeights, self.outpath)

    # only rank 0 should save. otherwise all the other nodes could over-write the output or quit first; rank 0 plots
    if sim.rank == 0:
      if sim.plotWeights:
        plotWeights()
      saveActionsPerEpisode(
          sim, self.epCount, self.outpath('ActionsPerEpisode.txt'))
      if sim.plotRaster:
        plotRaster(sim, self.dconf, self.dnumc, self.outpath('raster.png'))
