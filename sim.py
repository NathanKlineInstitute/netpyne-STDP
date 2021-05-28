from netpyne import specs, sim
from neuron import h

import numpy as np
import random
import pandas as pd
import pickle
import os
import time
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import animation

from conf import dconf
from cells import intf7
from connUtils import *
from game_interface import GameInterface
from utils import syncdata_alltoall

random.seed(1234) # this will not work properly across runs with different number of nodes

sim.davgW = {} # average adjustable weights on a target population
sim.allTimes = []
sim.allRewards = [] # list to store all rewards
sim.allActions = [] # list to store all actions
sim.allMotorOutputs = [] # list to store firing rate of output motor neurons.
sim.ActionsRewardsfilename = 'data/'+dconf['sim']['name']+'ActionsRewards.txt'
sim.MotorOutputsfilename = 'data/'+dconf['sim']['name']+'MotorOutputs.txt'
sim.WeightsRecordingTimes = []
sim.allRLWeights = [] # list to store weights --- should remove that
# remove allNonRLWeights
sim.topologicalConns = dict() # dictionary to save topological connections.
sim.lastMove = dconf['moves']['LEFT']
#sim.NonRLweightsfilename = 'data/'+dconf['sim']['name']+'NonRLweights.txt'  # file to store weights
sim.plotWeights = 0  # plot weights
sim.saveWeights = 1  # save weights
if 'saveWeights' in dconf['sim']: sim.saveWeights = dconf['sim']['saveWeights']
sim.saveInputImages = 1 #save Input Images (5 game frames)
sim.saveMotionFields = dconf['sim']['saveMotionFields'] # whether to save the motion fields
sim.saveObjPos = 1 # save ball and paddle position to file
sim.saveAssignedFiringRates = dconf['sim']['saveAssignedFiringRates']
recordWeightStepSize = dconf['sim']['recordWeightStepSize']
normalizeWeightStepSize = dconf['sim']['normalizeWeightStepSize']
#recordWeightDT = 1000 # interval for recording synaptic weights (change later)
recordWeightDCells = 1 # to record weights for sub samples of neurons
tstepPerAction = dconf['sim']['tstepPerAction'] # time step per action (in ms)

fid4=None # only used by rank 0

scale = dconf['net']['scale'] # scales the size of the network (only number of neurons)

ETypes = dconf['net']['ETypes'] # excitatory neuron types
ITypes = dconf['net']['ITypes'] # inhibitory neuron types
allpops = list(dconf['net']['allpops'].keys())
EMotorPops = dconf['net']['EMotorPops'] # excitatory neuron motor populations

cmat = dconf['net']['cmat'] # connection matrix (for classes, synapses, probabilities [probabilities not used for topological conn])

dnumc = OrderedDict({ty:dconf['net']['allpops'][ty]*scale for ty in allpops}) # number of neurons of a given type

lrecpop = ['EMRIGHT', 'EMLEFT'] # which plastic populations to record from

if dnumc['EA']>0 and (dconf['net']['RLconns']['RecurrentANeurons'] or \
                      dconf['net']['STDPconns']['RecurrentANeurons'] or \
                      dconf['net']['RLconns']['FeedbackMtoA'] or \
                      dconf['net']['STDPconns']['FeedbackMtoA']):
  lrecpop.append('EA')

if dnumc['EA2']>0 and (dconf['net']['RLconns']['RecurrentA2Neurons'] or \
                       dconf['net']['STDPconns']['RecurrentA2Neurons'] or \
                       dconf['net']['RLconns']['FeedbackMtoA2'] or \
                       dconf['net']['STDPconns']['FeedbackMtoA2']):
  lrecpop.append('EA2')

if dconf['net']['RLconns']['Visual'] or dconf['net']['STDPconns']['Visual']:
  if lrecpop.count('EV4')==0: lrecpop.append('EV4')
  if lrecpop.count('EMT')==0: lrecpop.append('EMT')


# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells

simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration
#Simulation options
simConfig.duration = dconf['sim']['duration'] # 100e3 # 0.1e5                      # Duration of the simulation, in ms
simConfig.dt = dconf['sim']['dt']                            # Internal integration timestep to use
simConfig.hParams['celsius'] = 37 # make sure temperature is set. otherwise we're at squid temperature
simConfig.verbose = dconf['sim']['verbose']                       # Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordCellsSpikes = [-1] # this means record from all neurons - including stim populations, if any
simConfig.recordStep = dconf['sim']['recordStep'] # Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.filename = 'data/'+dconf['sim']['name']+'simConfig'  # Set file output name
simConfig.saveJson = False
simConfig.savePickle = True            # Save params, network and sim output to pickle file
simConfig.saveMat = False
simConfig.saveFolder = 'data'
# simConfig.backupCfg = ['sim.json', 'backupcfg/'+dconf['sim']['name']+'sim.json']
simConfig.createNEURONObj = True  # create HOC objects when instantiating network
simConfig.createPyStruct = True  # create Python structure (simulator-independent) when instantiating network
simConfig.analysis['plotTraces'] = {'include': [(pop, 0) for pop in ['ER','IR','EV1','EV1DE','ID','IV1','EV4','IV4','EMT','IMT','EMLEFT','EMRIGHT','IM','IML','IMUP','IMDOWN','EA','IA','IAL','EA2','IA2','IA2L']]}
simConfig.analysis['plotRaster'] = {'popRates':'overlay','showFig':dconf['sim']['doplot']}
#simConfig.analysis['plot2Dnet'] = True
#simConfig.analysis['plotConn'] = True           # plot connectivity matrix
# simConfig.coreneuron = True
# synaptic weight gain (based on E, I types)
cfg = simConfig
cfg.EEGain = dconf['net']['EEGain'] # E to E scaling factor
cfg.EIGain = dconf['net']['EIGain'] # E to I scaling factor
cfg.IEGain = dconf['net']['IEGain'] # I to E scaling factor
cfg.IIGain = dconf['net']['IIGain'] # I to I scaling factor

### from https://www.neuron.yale.edu/phpBB/viewtopic.php?f=45&t=3770&p=16227&hilit=memory#p16122
cfg.saveCellSecs = bool(dconf['sim']['saveCellSecs']) # if False removes all data on cell sections prior to gathering from nodes
cfg.saveCellConns = bool(dconf['sim']['saveCellConns']) # if False removes all data on cell connections prior to gathering from nodes
###

# weight variance -- check if need to vary the initial weights (note, they're over-written if resumeSim==1)
cfg.weightVar = dconf['net']['weightVar']
cfg.delayMinDend = dconf['net']['delayMinDend']
cfg.delayMaxDend = dconf['net']['delayMaxDend']
cfg.delayMinSoma = dconf['net']['delayMinSoma']
cfg.delayMaxSoma = dconf['net']['delayMaxSoma']

def getInitWeight (weight):
  """get initial weight for a connection
     checks if weightVar is non-zero, if so will use a uniform distribution
     with range on interval: (1-var)*weight, (1+var)*weight
  """
  if cfg.weightVar == 0.0:
    return weight
  elif weight <= 0.0:
    return 0.0
  else:
    # print('uniform(%g,%g)' % (weight*(1.0-cfg.weightVar),weight*(1.0+cfg.weightVar)))
    return 'uniform(%g,%g)' % (max(0,weight*(1.0-cfg.weightVar)),weight*(1.0+cfg.weightVar))

def getCompFromSy (sy):
  if sy.count('2') > 0: return 'Dend'
  return 'Soma'

def getInitDelay (cmp='Dend'):
  a,b = float(dconf['net']['delayMin'+cmp]), float(dconf['net']['delayMax'+cmp])
  if a==b:
    return a
  else:
    return 'uniform(%g,%g)' % (a,b)

ECellModel = dconf['net']['ECellModel']
ICellModel = dconf['net']['ICellModel']

def getComp (sy):
  if ECellModel == 'INTF7' or ICellModel == 'INTF7':
    if sy.count('2') > 0:
      return 'Dend'
    return 'Soma'
  else:
    if sy.count('AM') or sy.count('NM'): return 'Dend'
    return 'Soma'

#Population parameters
for ty in allpops:
  if ty in ETypes:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ECellModel}
  else:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ICellModel}

def makeECellModel (ECellModel):
  # create rules for excitatory neuron models
  EExcitSec = 'dend' # section where excitatory synapses placed
  PlastWeightIndex = 0 # NetCon weight index where plasticity occurs
  if ECellModel == 'IntFire4':
    EExcitSec = 'soma' # section where excitatory synapses placed
    simConfig.recordTraces = {'V_soma':{'var':'m'}}  # Dict with traces to record
    netParams.defaultThreshold = 0.0
    for ty in ETypes:
      #netParams.popParams[ty]={'cellType':ty,'numCells':dnumc[ty],'cellModel':ECellModel}#, 'params':{'taue':5.35,'taui1':9.1,'taui2':0.07,'taum':20}}
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'IntFire4', 'numCells': dnumc[ty], 'taue': 1.0}  # pop of IntFire4
  elif ECellModel == 'INTF7':
    EExcitSec = 'soma' # section where excitatory synapses placed
    simConfig.recordTraces = {'V_soma':{'var':'Vm'}}  # Dict with traces to record
    netParams.defaultThreshold = -40.0
    for ty in ETypes:
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'INTF7', 'numCells': dnumc[ty]} # pop of IntFire4
      for k,v in intf7.INTF7E.dparam.items(): netParams.popParams[ty][k] = v
    PlastWeightIndex = intf7.dsyn['AM2']
  return EExcitSec, PlastWeightIndex

def makeICellModel (ICellModel):
  # create rules for inhibitory neuron models
  if ICellModel == 'IntFire4':
    simConfig.recordTraces = {'V_soma':{'var':'m'}}  # Dict with traces to record
    netParams.defaultThreshold = 0.0
    for ty in ITypes:
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'IntFire4', 'numCells': dnumc[ty], 'taue': 1.0}  # pop of IntFire4
  elif ICellModel == 'INTF7':
    EExcitSec = 'soma' # section where excitatory synapses placed
    simConfig.recordTraces = {'V_soma':{'var':'Vm'}}  # Dict with traces to record
    netParams.defaultThreshold = -40.0
    for ty in ITypes:
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'INTF7', 'numCells': dnumc[ty]}
      if ty.count('L') > 0: # LTS
        for k,v in intf7.INTF7IL.dparam.items(): netParams.popParams[ty][k] = v
      else: # FS
        for k,v in intf7.INTF7I.dparam.items(): netParams.popParams[ty][k] = v

EExcitSec,PlastWeightIndex = makeECellModel(ECellModel)
print('EExcitSec,PlastWeightIndex:',EExcitSec,PlastWeightIndex)
makeICellModel(ICellModel)

## Synaptic mechanism parameters
# note that these synaptic mechanisms are not used for the INTF7 neurons
# excitatory synaptic mechanism
netParams.synMechParams['AM2'] = netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}
netParams.synMechParams['NM2'] = netParams.synMechParams['NMDA'] = {'mod': 'Exp2Syn', 'tau1': 0.15, 'tau2': 166.0, 'e': 0} # NMDA
# inhibitory synaptic mechanism
netParams.synMechParams['GA'] = netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}

def readSTDPParams():
  ruleParams = []
  for rule in ['RL', 'STDP']:
    lsy = ['AMPA', 'NMDA', 'AMPAI']
    gains = [cfg.EEGain, cfg.EEGain, cfg.EIGain]
    dSTDPparams = {} # STDP-RL/STDPL parameters for AMPA,NMDA synapses; generally uses shorter/longer eligibility traces
    for sy,gain in zip(lsy, gains):
      dSTDPparams[sy] = dconf[rule][sy]
      for k in dSTDPparams[sy].keys():
        if k.count('wt') or k.count('wbase') or k.count('wmax'): dSTDPparams[sy][k] *= gain

    dSTDPparams['AM2']=dSTDPparams['AMPA']
    dSTDPparams['NM2']=dSTDPparams['NMDA']
    ruleParams.append(dSTDPparams)
  return ruleParams

dSTDPparamsRL, dSTDPparams = readSTDPParams()

def getWeightIndex (synmech, cellModel):
  # get weight index for connParams
  if cellModel == 'INTF7': return intf7.dsyn[synmech]
  return 0

def setupStimMod ():
  # setup variable rate NetStim sources (send spikes based on image contents)
  lstimty = []
  inputPop = 'EV1' # which population gets the direct visual inputs (pixels)
  if dnumc['ER']>0: inputPop = 'ER'
  stimModLocW = dconf['net']['stimModVL']
  stimModDirW = dconf['net']['stimModVD']
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    lpoty = [inputPop]
    for poty in ['EV1D'+Dir for Dir in ['E','NE','N', 'NW','W','SW','S','SE']]: lpoty.append(poty)
    wt = stimModLocW
    for poty in lpoty:
      if dnumc[poty] <= 0: continue
      stimty = 'stimMod'+poty
      lstimty.append(stimty)
      netParams.popParams[stimty] = {'cellModel': 'NSLOC', 'numCells': dnumc[poty],'rate': 'variable', 'noise': 0, 'start': 0}
      blist = [[i,i] for i in range(dnumc[poty])]
      netParams.connParams[stimty+'->'+poty] = {
        'preConds': {'pop':stimty},
        'postConds': {'pop':poty},
        'weight':wt,
        'delay': getInitDelay('STIMMOD'),
        'connList':blist,
        'weightIndex': getWeightIndex('AMPA',ECellModel)}
      wt = stimModDirW # rest of inputs use this weight

  return lstimty

sim.lstimty = setupStimMod() # when using IntFire4 cells lstimty has the NetStim populations that send spikes to EV1, EV1DE, etc.
for ty in sim.lstimty: allpops.append(ty)

# Stimulation parameters
def setupNoiseStim ():
  lnoisety = []
  dnoise = dconf['noise']
  # setup noisy NetStim sources (send random spikes)
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    lpoty = dnoise.keys()
    for poty in lpoty:
      lsy = dnoise[poty].keys()
      for sy in lsy:
        Weight,Rate = dnoise[poty][sy]['w'],dnoise[poty][sy]['rate']
        if Weight > 0.0 and Rate > 0.0: # only create the netstims if rate,weight > 0
          stimty = 'stimNoise'+poty+'_'+sy
          netParams.popParams[stimty] = {'cellModel': 'NetStim', 'numCells': dnumc[poty],'rate': Rate, 'noise': 1.0, 'start': 0}
          blist = [[i,i] for i in range(dnumc[poty])]
          netParams.connParams[stimty+'->'+poty] = {
            'preConds': {'pop':stimty},
            'postConds': {'pop':poty},
            'weight':Weight,
            'delay': getInitDelay(getCompFromSy(sy)),
            'connList':blist,
            'weightIndex':getWeightIndex(sy,ECellModel)}
          lnoisety.append(stimty)
  else:
    # setup noise inputs
    lpoty = dnoise.keys()
    for poty in lpoty:
      lsy = dnoise[poty].keys()
      for sy in lsy:
        Weight,Rate = dnoise[poty][sy]['w'],dnoise[poty][sy]['rate']
        if Weight > 0.0 and Rate > 0.0: # only create the netstims if rate,weight > 0
          stimty = poty+'Mbkg'+sy
          netParams.stimSourceParams[stimty] = {'type': 'NetStim', 'rate': Rate, 'noise': 1.0}
          netParams.stimTargetParams[poty+'Mbkg->all'] = {
            'source': stimty, 'conds': {'cellType': EMotorPops}, 'weight': Weight, 'delay': 'max(1, normal(5,2))', 'synMech': sy}
          # lnoisety.append(ty+'Mbkg'+sy)
  return lnoisety

sim.lnoisety = setupNoiseStim()
for ty in sim.lnoisety: allpops.append(ty)

######################################################################################

#####################################################################################

#Local excitation
#E to E recurrent connectivity within visual areas
for epop in EVPops:
  if dnumc[epop] <= 0: continue # skip rule setup for empty population
  prety = poty = epop
  repstr = 'VD' # replacement presynaptic type string (VD -> EV1DE, EV1DNE, etc.; VL -> EV1, EV4, etc.)
  if prety in EVLocPops: repstr = 'VL'
  wAM, wNM = cmat[repstr][repstr]['AM2'], cmat[repstr][repstr]['NM2']
  for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[wAM*cfg.EEGain, wNM*cfg.EEGain]):
    k = strty+prety+'->'+strty+poty
    if weight <= 0.0: continue
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat, repstr, repstr, dnumc[prety]),
      'weight': getInitWeight(weight),
      'delay': getInitDelay('Dend'),
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5,
      'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    useRL = useSTDP = False
    if prety in EVDirPops:
      if dconf['net']['RLconns']['RecurrentDirNeurons']: useRL = True
      if dconf['net']['STDPconns']['RecurrentDirNeurons']: useSTDP = True
    if prety in EVLocPops:
      if dconf['net']['RLconns']['RecurrentLocNeurons']: useRL = True
      if dconf['net']['STDPconns']['RecurrentLocNeurons']: useSTDP = True
    if useRL and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
    elif useSTDP and dSTDPparams[synmech]['STDPon']:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

#E to I within area
if dnumc['ER']>0:
  netParams.connParams['ER->IR'] = {
          'preConds': {'pop': 'ER'},
          'postConds': {'pop': 'IR'},
          'weight': cmat['ER']['IR']['AM2'] * cfg.EIGain,
          'delay': getInitDelay('Dend'),
          'synMech': 'AMPA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('AM2', ICellModel)}
  netParams.connParams['ER->IR']['convergence'] = getconv(cmat, 'ER', 'IR', dnumc['ER'])

netParams.connParams['EV1->IV1'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'IV1'},
        'weight': cmat['EV1']['IV1']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('AM2', ICellModel)}

netParams.connParams['EV1->IV1']['convergence'] = getconv(cmat, 'EV1', 'IV1', dnumc['EV1'])

if dnumc['ID']>0:
  EVDirPops = dconf['net']['EVDirPops']
  IVDirPops = dconf['net']['IVDirPops']
  for prety in EVDirPops:
    for poty in IVDirPops:
      netParams.connParams[prety+'->'+poty] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat, 'VD', 'ID', dnumc[prety]),
        'weight': cmat['VD']['ID']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('AM2', ICellModel)}

netParams.connParams['EV4->IV4'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'IV4'},
        'weight': cmat['EV4']['IV4']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('AM2', ICellModel)}

netParams.connParams['EV4->IV4']['convergence'] = getconv(cmat,'EV4','IV4', dnumc['EV4'])

netParams.connParams['EMT->IMT'] = {
        'preConds': {'pop': 'EMT'},
        'postConds': {'pop': 'IMT'},
        'weight': cmat['EMT']['IMT']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('AM2', ICellModel)}
netParams.connParams['EMT->IMT']['convergence'] = getconv(cmat, 'EMT', 'IMT', dnumc['EMT'])

for prety,poty in zip(['EA','EA','EA2','EA2'],['IA','IAL','IA2','IA2L']):
  if dnumc[prety] <= 0 or dnumc[poty] <= 0: continue
  for sy in ['AM2','NM2']:
    if sy not in cmat[prety][poty]: continue
    k = prety+'->'+poty+sy
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat, prety, poty, dnumc[prety]),
      'weight': cmat[prety][poty][sy] * cfg.EIGain,
      'delay': getInitDelay('Dend'),
      'synMech': sy, 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ICellModel)}
    if sy.count('AM') > 0:
      if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
        netParams.connParams[k]['weight'] = getInitWeight(cmat[prety][poty]['AM2'] * cfg.EIGain)
      elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
        netParams.connParams[k]['weight'] = getInitWeight(cmat[prety][poty]['AM2'] * cfg.EIGain)

for prety in EMotorPops:
  if dnumc[prety] <= 0: continue
  for poty in ['IM', 'IML']:
    if dnumc[poty] <= 0: continue
    for sy in ['AM2','NM2']:
      if sy not in cmat['EM'][poty]: continue
      k = prety+'->'+poty+sy
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat, 'EM', poty, dnumc[prety]),
        'weight': cmat['EM'][poty][sy] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': sy, 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex(sy, ICellModel)}
      if sy.count('AM') > 0:
        if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
          netParams.connParams[k]['weight'] = getInitWeight(cmat['EM'][poty]['AM2'] * cfg.EIGain)
        elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
          netParams.connParams[k]['weight'] = getInitWeight(cmat['EM'][poty]['AM2'] * cfg.EIGain)

# reciprocal inhibition - only active when all relevant populations created - not usually used
for prety in EMotorPops:
  for epoty in EMotorPops:
    if epoty == prety: continue # no self inhib here
    poty = 'IM' + epoty[2:] # change name to interneuron
    k = prety + '->' + poty
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat, 'EM', 'IRecip', dnumc[prety]),
      'weight': cmat['EM']['IRecip']['AM2'] * cfg.EIGain,
      'delay': getInitDelay('Dend'),
      'synMech': 'AMPA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('AM2', ICellModel)}
    if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
      netParams.connParams[k]['weight'] = getInitWeight(cmat['EM']['IRecip']['AM2'] * cfg.EIGain)
    elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
      netParams.connParams[k]['weight'] = getInitWeight(cmat['EM']['IRecip']['AM2'] * cfg.EIGain)

#Local inhibition
#I to E within area
if dnumc['ER']>0:
  netParams.connParams['IR->ER'] = {
          'preConds': {'pop': 'IR'},
          'postConds': {'pop': 'ER'},
          'weight': cmat['IR']['ER']['GA'] * cfg.IEGain,
          'delay': getInitDelay('Soma'),
          'synMech': 'GABA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('GA', ICellModel)}
  netParams.connParams['IR->ER']['convergence'] = getconv(cmat, 'IR', 'ER', dnumc['IR'])

netParams.connParams['IV1->EV1'] = {
  'preConds': {'pop': 'IV1'},
  'postConds': {'pop': 'EV1'},
  'weight': cmat['IV1']['EV1']['GA'] * cfg.IEGain,
  'delay': getInitDelay('Soma'),
  'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
netParams.connParams['IV1->EV1']['convergence'] = getconv(cmat, 'IV1', 'EV1', dnumc['IV1'])

if dnumc['ID']>0:
  IVDirPops = dconf['net']['IVDirPops']
  for prety in IVDirPops:
    for poty in EVDirPops:
      netParams.connParams[prety+'->'+poty] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat, 'ID', 'ED', dnumc['ID']),
        'weight': cmat['ID']['ED']['GA'] * cfg.IEGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}

netParams.connParams['IV4->EV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'EV4'},
        'weight': cmat['IV4']['EV4']['GA'] * cfg.IEGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
netParams.connParams['IV4->EV4']['convergence'] = getconv(cmat,'IV4','EV4', dnumc['IV4'])

netParams.connParams['IMT->EMT'] = {
        'preConds': {'pop': 'IMT'},
        'postConds': {'pop': 'EMT'},
        'weight': cmat['IMT']['EMT']['GA'] * cfg.IEGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
netParams.connParams['IMT->EMT']['convergence'] = getconv(cmat,'IMT','EMT',dnumc['IMT'])

# I -> E for motor populations
for prety,sy in zip(['IM', 'IML'],['GA','GA2']):
  for poty in EMotorPops:
    netParams.connParams[prety+'->'+poty] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat,prety,'EM', dnumc[prety]),
      'weight': cmat[prety]['EM'][sy] * cfg.IEGain,
      'delay': getInitDelay(getCompFromSy(sy)),
      'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ECellModel)}

for prety,poty,sy in zip(['IA','IAL','IA2','IA2L'],['EA','EA','EA2','EA2'],['GA','GA2','GA','GA2']):
  netParams.connParams[prety+'->'+poty] = {
    'preConds': {'pop': prety},
    'postConds': {'pop': poty},
    'convergence': getconv(cmat,prety,poty, dnumc[prety]),
    'weight': cmat[prety][poty][sy] * cfg.IEGain,
    'delay': getInitDelay(getCompFromSy(sy)),
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ECellModel)}

#I to I
for preIType in ITypes:
  sy = 'GA'
  if preIType.count('L') > 0: sy = 'GA2'
  for poIType in ITypes:
    if preIType not in dnumc or poIType not in dnumc: continue
    if dnumc[preIType] <= 0 or dnumc[poIType] <= 0: continue
    if poIType not in cmat[preIType] or \
       getconv(cmat,preIType,poIType,dnumc[preIType])<=0 or \
       cmat[preIType][poIType][sy]<=0: continue
    netParams.connParams[preIType+'->'+poIType] = {
      'preConds': {'pop': preIType},
      'postConds': {'pop': poIType},
      'convergence': getconv(cmat,preIType,poIType,dnumc[preIType]),
      'weight': cmat[preIType][poIType][sy] * cfg.IIGain,
      'delay': getInitDelay(getCompFromSy(sy)),
      'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ICellModel)}

#E to E feedforward connections - AMPA,NMDA
lprety,lpoty,lblist,lconnsCoords = [],[],[],[]
if not dconf['sim']['useReducedNetwork']:
  if dnumc['ER']>0:
    lprety.append('ER')
    lpoty.append('EV1')
    lblist.append(blistERtoEV1)
    lconnsCoords.append(connCoordsERtoEV1)
  lprety.append('EV1'); lpoty.append('EV4'); lblist.append(blistEV1toEV4); lconnsCoords.append(connCoordsEV1toEV4)
  lprety.append('EV4'); lpoty.append('EMT'); lblist.append(blistEV4toEMT); lconnsCoords.append(connCoordsEV4toEMT)
  for prety,poty,blist,connCoords in zip(lprety,lpoty,lblist,lconnsCoords):
    for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat[prety][poty]['AM2']*cfg.EEGain,cmat[prety][poty]['NM2']*cfg.EEGain]):
      k = strty+prety+'->'+strty+poty
      if weight <= 0.0: continue
      netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'weight': weight ,
            'delay': getInitDelay('Dend'),
            'synMech': synmech,'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)}
      netParams.connParams[k]['convergence'] = getconv(cmat,prety,poty,dnumc[prety])
      if dconf['net']['RLconns']['Visual'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['weight'] = getInitWeight(weight) # make sure non-uniform weights
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dconf['net']['STDPconns']['Visual'] and dSTDPparams[synmech]['STDPon']:
        netParams.connParams[k]['weight'] = getInitWeight(weight) # make sure non-uniform weights
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}



  #I to E feedback connections
  netParams.connParams['IV1->ER'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'ER'},
        'connList': blistIV1toER,
        'weight': cmat['IV1']['ER']['GA'] * cfg.IEGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
  sim.topologicalConns['IV1->ER'] = {}
  sim.topologicalConns['IV1->ER']['blist'] = blistIV1toER
  sim.topologicalConns['IV1->ER']['coords'] = connCoordsIV1toER
  netParams.connParams['IV4->EV1'] = {
          'preConds': {'pop': 'IV4'},
          'postConds': {'pop': 'EV1'},
          'connList': blistIV4toEV1,
          'weight': cmat['IV4']['EV1']['GA'] * cfg.IEGain,
          'delay': getInitDelay('Soma'),
          'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
  sim.topologicalConns['IV4->EV1'] = {}
  sim.topologicalConns['IV4->EV1']['blist'] = blistIV4toEV1
  sim.topologicalConns['IV4->EV1']['coords'] = connCoordsIV4toEV1
  netParams.connParams['IMT->EV4'] = {
          'preConds': {'pop': 'IMT'},
          'postConds': {'pop': 'EV4'},
          'connList': blistIMTtoEV4,
          'weight': cmat['IMT']['EV4']['GA'] * cfg.IEGain,
          'delay': getInitDelay('Soma'),
          'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
  sim.topologicalConns['IMT->EV4'] = {'blist':blistIMTtoEV4, 'coords':connCoordsIMTtoEV4}

# add connections from first to second visual association area
# EA -> EA2 (feedforward)
prety = 'EA'; poty = 'EA2'
if dnumc[prety] > 0 and dnumc[poty] > 0:
  lsynw = [cmat[prety][poty]['AM2']*cfg.EEGain, cmat[prety][poty]['NM2']*cfg.EEGain]
  for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],lsynw):
    k = strty+prety+'->'+strty+poty
    if weight <= 0.0: continue
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat,prety,poty,dnumc[prety]),
      'weight': getInitWeight(weight),
      'delay': getInitDelay('Dend'),
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    useRL = useSTDP = False
    ffconnty = 'FeedForwardAtoA2'
    if dconf['net']['RLconns'][ffconnty]: useRL = True
    if dconf['net']['STDPconns'][ffconnty]: useSTDP = True
    if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
    elif dSTDPparams[synmech]['STDPon'] and useSTDP:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

# EA2 -> EA (feedback)
prety = 'EA2'; poty = 'EA'
if dnumc[prety] > 0 and dnumc[poty] > 0:
  lsynw = [cmat[prety][poty]['AM2']*cfg.EEGain, cmat[prety][poty]['NM2']*cfg.EEGain]
  for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],lsynw):
    k = strty+prety+'->'+strty+poty
    if weight <= 0.0: continue
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat,prety,poty,dnumc[prety]),
      'weight': getInitWeight(weight),
      'delay': getInitDelay('Dend'),
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    useRL = useSTDP = False
    fbconnty = 'FeedbackA2toA'
    if dconf['net']['RLconns'][fbconnty]: useRL = True
    if dconf['net']['STDPconns'][fbconnty]: useSTDP = True
    if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
    elif dSTDPparams[synmech]['STDPon'] and useSTDP:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}


# Add connections from visual association areas to motor cortex, and reccurrent conn within visual association areas
for prety,ffconnty,recconnty in zip(['EA', 'EA2'],['FeedForwardAtoM','FeedForwardA2toM'],['RecurrentANeurons','RecurrentA2Neurons']):
  if dnumc[prety] <= 0: continue
  lsynw = [cmat[prety]['EM']['AM2']*cfg.EEGain, cmat[prety]['EM']['NM2']*cfg.EEGain]
  for poty in EMotorPops:
    if dnumc[poty] <= 0: continue
    for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],lsynw):
      k = strty+prety+'->'+strty+poty
      if weight <= 0.0: continue
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat,prety,'EM', dnumc[prety]),
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
      }
      useRL = useSTDP = False
      if dconf['net']['RLconns'][ffconnty]: useRL = True
      if dconf['net']['STDPconns'][ffconnty]: useSTDP = True
      if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dSTDPparams[synmech]['STDPon'] and useSTDP:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}
  # add recurrent plastic connectivity within EA populations
  poty = prety
  if getconv(cmat,prety,poty,dnumc[prety])>0 and dnumc[poty]>0:
    for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat[prety][poty]['AM2']*cfg.EEGain, cmat[prety][poty]['NM2']*cfg.EEGain]):
      k = strty+prety+'->'+strty+poty
      if weight <= 0.0: continue
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat,prety,poty,dnumc[prety]),
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
      }
      if dconf['net']['RLconns'][recconnty] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dconf['net']['STDPconns'][recconnty] and dSTDPparams[synmech]['STDPon']:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

# add recurrent plastic connectivity within EM populations
if getconv(cmat,'EM','EM',dnumc['EMLEFT']) > 0:
  for prety in EMotorPops:
    for poty in EMotorPops:
      if prety==poty or dconf['net']['EEMRecProbCross']: # same types or allowing cross EM population connectivity
        for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat['EM']['EM']['AM2']*cfg.EEGain, cmat['EM']['EM']['NM2']*cfg.EEGain]):
          k = strty+prety+'->'+strty+poty
          if weight <= 0.0: continue
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': getconv(cmat,'EM','EM', dnumc[prety]),
            'weight': getInitWeight(weight),
            'delay': getInitDelay('Dend'),
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
          }
          if dconf['net']['RLconns']['RecurrentMNeurons'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif dconf['net']['STDPconns']['RecurrentMNeurons'] and dSTDPparams[synmech]['STDPon']:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

# add feedback plastic connectivity from EM populations to association populations
if getconv(cmat,'EM','EA',dnumc['EMLEFT'])>0:
  for prety in EMotorPops:
    for poty in ['EA','EA2']:
        for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat['EM'][poty]['AM2']*cfg.EEGain, cmat['EM'][poty]['NM2']*cfg.EEGain]):
          k = strty+prety+'->'+strty+poty
          if weight <= 0.0: continue
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': getconv(cmat,'EM',poty,dnumc[prety]),
            'weight': getInitWeight(weight),
            'delay': getInitDelay('Dend'),
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
          }
          useRL = useSTDP = False
          if poty == 'EA':
            if dconf['net']['RLconns']['FeedbackMtoA']: useRL = True
            if dconf['net']['STDPconns']['FeedbackMtoA']: useSTDP = True
          elif poty == 'EA2':
            if dconf['net']['RLconns']['FeedbackMtoA2']: useRL = True
            if dconf['net']['STDPconns']['FeedbackMtoA2']: useSTDP = True
          if useRL and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif useSTDP and dSTDPparams[synmech]['STDPon']:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

fconn = 'data/'+dconf['sim']['name']+'synConns.pkl'
pickle.dump(sim.topologicalConns, open(fconn, 'wb'))
###################################################################################################################################

sim.AIGame = None # placeholder

lsynweights = [] # list of syn weights, per node

dsumWInit = {}

def getSumAdjustableWeights (sim):
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

def sumAdjustableWeightsPop (sim, popname):
  # record the plastic weights for specified popname
  lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of MR cells
  W = N = 0
  for cell in lcell:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        W += float(conn['hObj'].weight[PlastWeightIndex])
        N += 1
  return W, N

def recordAdjustableWeights (sim, t, lpop):
  global lsynweights
  """ record the STDP weights during the simulation - called in trainAgent
  """

  for popname in lpop:
    # record the plastic weights for specified popname
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of popname cells
    for cell in lcell:
      for conn in cell.conns:
        if 'hSTDP' in conn:
          #hstdp = conn.get('hSTDP')
          lsynweights.append(
            [t, conn.preGid, cell.gid, float(conn['hObj'].weight[PlastWeightIndex])])
    # return len(lcell)

def recordWeights (sim, t):
  """ record the STDP weights during the simulation - called in trainAgent
  """
  #lRcell = [c for c in sim.net.cells if c.gid in sim.net.pops['ER'].cellGids]
  sim.WeightsRecordingTimes.append(t)
  sim.allRLWeights.append([]) # Save this time
  sim.allNonRLWeights.append([])
  for cell in sim.net.cells:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        if conn.plast.params.RLon ==1:
          sim.allRLWeights[-1].append(float(conn['hObj'].weight[PlastWeightIndex])) # save weight only for Rl-STDP conns
        else:
          sim.allNonRLWeights[-1].append(float(conn['hObj'].weight[PlastWeightIndex])) # save weight only for nonRL-STDP conns

def saveWeights(sim, downSampleCells):
  ''' Save the weights for each plastic synapse '''
  with open(sim.RLweightsfilename,'w') as fid1:
    count1 = 0
    for weightdata in sim.allRLWeights:
      #fid.write('%0.0f' % weightdata[0]) # Time
      #print(len(weightdata))
      fid1.write('%0.1f' %sim.WeightsRecordingTimes[count1])
      count1 = count1+1
      for i in range(0,len(weightdata), downSampleCells): fid1.write('\t%0.8f' % weightdata[i])
      fid1.write('\n')
  print(('Saved RL weights as %s' % sim.RLweightsfilename))
  with open(sim.NonRLweightsfilename,'w') as fid2:
    count2 = 0
    for weightdata in sim.allNonRLWeights:
      #fid.write('%0.0f' % weightdata[0]) # Time
      #print(len(weightdata))
      fid2.write('%0.1f' %sim.WeightsRecordingTimes[count2])
      count2 = count2+1
      for i in range(0,len(weightdata), downSampleCells): fid2.write('\t%0.8f' % weightdata[i])
      fid2.write('\n')
  print(('Saved Non-RL weights as %s' % sim.NonRLweightsfilename))

#
def plotWeights():
  from pylab import figure, loadtxt, xlabel, ylabel, xlim, ylim, show, pcolor, array, colorbar
  figure()
  weightdata = loadtxt(sim.weightsfilename)
  weightdataT=list(map(list, list(zip(*weightdata))))
  vmax = max([max(row) for row in weightdata])
  vmin = min([min(row) for row in weightdata])
  pcolor(array(weightdataT), cmap='hot_r', vmin=vmin, vmax=vmax)
  xlim((0,len(weightdata)))
  ylim((0,len(weightdata[0])))
  xlabel('Time (weight updates)')
  ylabel('Synaptic connection id')
  colorbar()
  show()

def getAverageAdjustableWeights (sim, lpop = EMotorPops):
  # get average adjustable weights on a target population
  davg = {pop:0.0 for pop in lpop}
  for pop in lpop:
    WSum = 0; NSum = 0
    W, N = sumAdjustableWeightsPop(sim, pop)
    # destlist_on_root = pc.py_gather(srcitem, root)
    lw = sim.pc.py_gather(W, 0)
    ln = sim.pc.py_gather(N, 0)
    if sim.rank == 0:
      WSum = W + np.sum(lw)
      NSum = N + np.sum(ln)
      #print('rank= 0, pop=',pop,'W=',W,'N=',N,'wsum=',WSum,'NSum=',NSum)
      if NSum > 0: davg[pop] = WSum / NSum
    else:
      #destitem_from_root = sim.pc.py_scatter(srclist, root)
      pass
      #print('rank=',sim.rank,'pop=',pop,'Wm=',W,'N=',N)
  lsrc = [davg for i in range(sim.nhosts)] if sim.rank == 0 else None
  dest = sim.pc.py_scatter(lsrc, 0)
  return dest

def mulAdjustableWeights (sim, dfctr):
  # multiply adjustable STDP/RL weights by dfctr[pop] value for each population keyed in dfctr
  for pop in dfctr.keys():
    if dfctr[pop] == 1.0: continue
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells
    for cell in lcell:
      for conn in cell.conns:
        if 'hSTDP' in conn:
          conn['hObj'].weight[PlastWeightIndex] *= dfctr[pop]


def saveGameBehavior(sim):
  with open(sim.ActionsRewardsfilename,'w') as fid3:
    for i in range(len(sim.allActions)):
      fid3.write('%0.1f' % sim.allTimes[i])
      fid3.write('\t%0.1f' % sim.allActions[i])
      fid3.write('\t%0.5f' % sim.allRewards[i])
      fid3.write('\n')

######################################################################################

def getSpikesWithInterval (trange = None, neuronal_pop = None):
  if len(neuronal_pop) < 1: return 0.0
  spkts = sim.simData['spkt']
  spkids = sim.simData['spkid']
  pop_spikes = 0
  if len(spkts)>0:
    for i in range(len(spkids)):
      if trange[0] <= spkts[i] <= trange[1] and spkids[i] in neuronal_pop:
        pop_spikes += 1
  return pop_spikes

NBsteps = 0 # this is a counter for recording the plastic weights
epCount = []
total_hits = [] #numbertimes ball is hit by racket as ball changes its direction and player doesn't lose a score (assign 1). if player loses
dSTDPmech = {} # dictionary of list of STDP mechanisms

def InitializeNoiseRates ():
  # initialize the noise firing rates for the primary visual neuron populations (location V1 and direction sensitive)
  # based on image contents
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    #np.random.seed(1234)
    for pop in sim.lnoisety:
      if pop in sim.net.pops:
        for cell in sim.net.cells:
          if cell.gid in sim.net.pops[pop].cellGids:
            cell.hPointp.interval = 10
            cell.hPointp.start = 0 # np.random.uniform(0,1200)
  else:
    if dnumc['ER']>0:
      lratepop = ['ER', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']
    else:
      lratepop = ['EV1', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']
    for pop in lratepop:
      lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells
      for cell in lCell:
        for stim in cell.stims:
          if stim['source'] == 'stimMod':
            stim['hObj'].interval = 1e12

def InitializeInputRates ():
  # initialize the source firing rates for the primary visual neuron populations (location V1 and direction sensitive)
  # based on image contents
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    np.random.seed(1234)
    for pop in sim.lstimty:
      if pop in sim.net.pops:
        for cell in sim.net.cells:
          if cell.gid in sim.net.pops[pop].cellGids:
            cell.hPointp.interval = 1e12
            cell.hPointp.start = 0 # np.random.uniform(0,1200)

def updateInputRates():
  input_rates = sim.GameInterface.input_firing_rates()
  input_rates = syncdata_alltoall(sim, input_rates)

  # if sim.rank == 0: print(dFiringRates['EV1'])
  # update input firing rates for stimuli to ER,EV1 and direction sensitive cells
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7': # different rules/code when dealing with artificial cells
    lsz = len('stimMod') # this is a prefix
    for pop in sim.lstimty: # go through NetStim populations
      if pop in sim.net.pops: # make sure the population exists
        lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of NetStim cells
        offset = sim.simData['dminID'][pop]
        #print(pop,pop[lsz:],offset)
        for cell in lCell:
          if dFiringRates[pop[lsz:]][int(cell.gid-offset)]==0:
            cell.hPointp.interval = 1e12
          else:
            cell.hPointp.interval = 1000.0/dFiringRates[pop[lsz:]][int(cell.gid-offset)] #40


def getActions(moves, pop_to_move):
  global fid4, tstepPerAction
  move_freq = {}
  # Iterate over move types
  for move in moves:
    vec = h.Vector()
    freq = []
    for ts in range(int(dconf['actionsPerPlay'])):
      ts_beg = t-tstepPerAction*(dconf['actionsPerPlay']-ts-1)
      ts_end = t-tstepPerAction*(dconf['actionsPerPlay']-ts)
      pop_name = [p for p, m in pop_to_move.items() if m == move][0]
      freq.append(getSpikesWithInterval([ts_end,ts_beg], sim.net.pops['EMRIGHT'].cellGids))

    sim.pc.allreduce(vec.from_python(freq),1) #sum
    move_freq[move] = vec.to_python()

  actions = []

  if sim.rank == 0:
    if fid4 is None: fid4 = open(sim.MotorOutputsfilename,'w')
    print('t={}: {} spikes: {}'.format(round(t,2), ','.join(moves), ','.join([move_freq[m] for m in moves])))
    fid4.write('%0.1f' % t)
    for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t' + '\t'.join([str(round(move_freq[m][ts], 1)) for m in moves]))
    fid4.write('\n')


    for ts in range(int(dconf['actionsPerPlay'])):
      no_firing_rates = sum([v[ts] for v in move_freq.values()]) == 0
      if no_firing_rates:
        # Should we initialize with random?
        print('Warning: No firing rates for moves {}!'.format(','.join(moves)))
      else:
        best_move, best_move_freq = sorted(
          [(m, f[ts]) for m,f in move_freq.items()],
          key=lambda x:x[1],
          reverse=True)[0]
        actions.append(dconf['moves'][best_move])

  return actions


def trainAgent (t):
  """ training interface between simulation and game environment
  """
  global NBsteps, epCount, total_hits, tstepPerAction

  if t<(tstepPerAction*dconf['actionsPerPlay']): # for the first time interval use randomly selected actions
    actions =[]
    for _ in range(int(dconf['actionsPerPlay'])):
      action = dconf['movecodes'][random.randint(0,len(dconf['movecodes'])-1)]
      actions.append(action)
  else: #the actions should be based on the activity of motor cortex (EMRIGHT, EMLEFT)
    actions = getActions(dconf['moves'], dconf['pop_to_move'])


  if sim.rank == 0:
    rewards = sim.AIGame.playGame(actions)

    # specifically for CartPole-v1. TODO: move to a diff file
    
    if len(sim.AIGame.observations) < 2:
      critic = abs(sim.AIGame.observations[-1][2]) * 100
    else:
      critic = (sim.AIGame.observations[-1][2] - sim.AIGame.observations[-2][2]) * 100

    # use py_broadcast to avoid converting to/from Vector
    sim.pc.py_broadcast(critic, 0) # broadcast critic value to other nodes
    
  else: # other workers
    critic = sim.pc.py_broadcast(None, 0) # receive critic value from master node

    if dconf['verbose']>1:
      print('UPactions: ', UPactions,'DOWNactions: ', DOWNactions)

  if critic != 0: # if critic signal indicates punishment (-1) or reward (+1)
    if sim.rank == 0: print('t=',round(t,2),'RLcritic:',critic)
    if dconf['verbose']: print('APPLY RL to both EMRIGHT and EMLEFT')
    for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(critic)

  if sim.rank == 0:
    sim.allActions.extend(actions)
    sim.allRewards.extend(rewards)
    tvec_actions = []
    for ts in range(len(actions)): tvec_actions.append(t-tstepPerAction*(len(actions)-ts-1))
    for ltpnt in tvec_actions: sim.allTimes.append(ltpnt)

  updateInputRates() # update firing rate of inputs to R population (based on game state)

  NBsteps += 1
  if NBsteps % recordWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank==0:
      print('Weights Recording Time:', t, 'NBsteps:',NBsteps,'recordWeightStepSize:',recordWeightStepSize)
    recordAdjustableWeights(sim, t, lrecpop) 
    #recordWeights(sim, t)

def getAllSTDPObjects (sim):
  # get all the STDP objects from the simulation's cells
  dSTDPmech = {'all':[]} # dictionary of STDP objects keyed by type (all, for EMRIGHT, EMLEFT populations)
  for pop in dconf['pop_to_move'].keys(): dSTDPmech[pop] = []

  for cell in sim.net.cells:
    #if cell.gid in sim.net.pops['EMLEFT'].cellGids and cell.gid==sim.simData['dminID']['EMLEFT']: print(cell.conns)
    for conn in cell.conns:
      STDPmech = conn.get('hSTDP')  # check if the connection has a NEURON STDP mechanism object
      if STDPmech:
        dSTDPmech['all'].append(STDPmech)
        for pop in dconf['pop_to_move'].keys():
          if cell.gid in sim.net.pops[pop].cellGids:
            dSTDPmech[pop].append(STDPmech)
  return dSTDPmech

# Alternate to create network and run simulation
# create network object and set cfg and net params; pass simulation config and network params as arguments
sim.initialize(simConfig = simConfig, netParams = netParams)

if sim.rank == 0:  # sim rank 0 specific init and backup of config file
  from aigame import AIGame
  sim.AIGame = AIGame(dconf) # only create AIGame on node 0
  sim.GameInterface = GameInterface(sim.AIGame, dconf)
  # node 0 saves the json config file
  # this is just a precaution since simConfig pkl file has MOST of the info; ideally should adjust simConfig to contain
  # ALL of the required info
  from utils import backupcfg, safemkdir
  backupcfg(dconf['sim']['name'])
  safemkdir('data') # make sure data (output) directory exists

sim.net.createPops()                      # instantiate network populations
sim.net.createCells()                     # instantiate network cells based on defined populations
sim.net.connectCells()                    # create connections between cells based on params
sim.net.addStims()                        #instantiate netStim

def setrecspikes ():
  if dconf['sim']['recordStim']:
    sim.cfg.recordCellsSpikes = [-1] # record from all spikes
  else:
    # make sure to record only from the neurons, not the stimuli - which requires a lot of storage/memory
    sim.cfg.recordCellsSpikes = []
    for pop in sim.net.pops.keys():
      if pop.count('stim') > 0 or pop.count('Noise') > 0: continue
      for gid in sim.net.pops[pop].cellGids: sim.cfg.recordCellsSpikes.append(gid)

setrecspikes()
sim.setupRecording()                  # setup variables to record for each cell (spikes, V traces, etc)

dSTDPmech = getAllSTDPObjects(sim) # get all the STDP objects up-front

def resumeSTDPWeights (sim, W):
  #this function assign weights stored in 'ResumeSimFromFile' to all connections by matching pre and post neuron ids
  # get all the simulation's cells (on a given node)
  for cell in sim.net.cells:
    cpostID = cell.gid#find postID
    WPost = W[(W.postid==cpostID)] #find the record for a connection with post neuron ID
    for conn in cell.conns:
      if 'hSTDP' not in conn: continue
      cpreID = conn.preGid  #find preID
      if type(cpreID) != int: continue
      cConnW = WPost[(WPost.preid==cpreID)] #find the record for a connection with pre and post neuron ID
      #find weight for the STDP connection between preID and postID
      for idx in cConnW.index:
        cW = cConnW.at[idx,'weight']
        conn['hObj'].weight[PlastWeightIndex] = cW
        #hSTDP = conn.get('hSTDP')
        #hSTDP.cumreward = cConnW.at[idx,'cumreward']
        if dconf['verbose'] > 1: print('weight updated:', cW)

#if specified 'ResumeSim' = 1, load the connection data from 'ResumeSimFromFile' and assign weights to STDP synapses
if dconf['simtype']['ResumeSim']:
  try:
    from simdat import readweightsfile2pdf
    A = readweightsfile2pdf(dconf['simtype']['ResumeSimFromFile'])
    resumeSTDPWeights(sim, A[A.time == max(A.time)]) # take the latest weights saved
    sim.pc.barrier() # wait for other nodes
    if sim.rank == 0: print('Updated STDP weights')
    # if 'normalizeWeightsAtStart' in dconf['sim']:
    #   if dconf['sim']['normalizeWeightsAtStart']:
    #     normalizeAdjustableWeights(sim, 0, lrecpop)
    #     print(sim.rank,'normalized adjustable weights at start')
    #     sim.pc.barrier() # wait for other nodes
  except:
    print('Could not restore STDP weights from file.')

def setdminID (sim, lpop):
  # setup min ID for each population in lpop
  alltags = sim._gatherAllCellTags() #gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
  dGIDs = {pop:[] for pop in lpop}
  for tinds in range(len(alltags)):
    if alltags[tinds]['pop'] in lpop:
      dGIDs[alltags[tinds]['pop']].append(tinds)
  sim.simData['dminID'] = {pop:np.amin(dGIDs[pop]) for pop in lpop if len(dGIDs[pop])>0}

setdminID(sim, allpops)
tPerPlay = tstepPerAction*dconf['actionsPerPlay']
InitializeInputRates()
#InitializeNoiseRates()
sim.runSimWithIntervalFunc(tPerPlay,trainAgent) # has periodic callback to adjust STDP weights based on RL signal
if sim.rank == 0 and fid4 is not None: fid4.close()
if ECellModel == 'INTF7' or ICellModel == 'INTF7': intf7.insertSpikes(sim, simConfig.recordStep)
sim.gatherData() # gather data from different nodes
sim.saveData() # save data to disk

def LSynWeightToD (L):
  # convert list of synaptic weights to dictionary to save disk space
  print('converting synaptic weight list to dictionary...')
  dout = {}; doutfinal = {}
  for row in L:
    #t,preID,poID,w,cumreward = row
    t,preID,poID,w = row
    if preID not in dout:
      dout[preID] = {}
      doutfinal[preID] = {}
    if poID not in dout[preID]:
      dout[preID][poID] = []
      doutfinal[preID][poID] = []
    #dout[preID][poID].append([t,w,cumreward])
    dout[preID][poID].append([t,w])
  for preID in doutfinal.keys():
    for poID in doutfinal[preID].keys():
      doutfinal[preID][poID].append(dout[preID][poID][-1])
  return dout, doutfinal

def saveSynWeights ():
  # save synaptic weights
  fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(sim.rank)+'.pkl'
  pickle.dump(lsynweights, open(fn, 'wb')) # save synaptic weights to disk for this node
  sim.pc.barrier() # wait for other nodes
  time.sleep(1)
  if sim.rank == 0: # rank 0 reads and assembles the synaptic weights into a single output file
    L = []
    for i in range(sim.nhosts):
      fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(i)+'.pkl'
      while not os.path.isfile(fn): # wait until the file is written/available
        print('saveSynWeights: waiting for finish write of', fn)
        time.sleep(1)
      lw = pickle.load(open(fn,'rb'))
      print(fn,'len(lw)=',len(lw),type(lw))
      os.unlink(fn) # remove the temporary file
      L = L + lw # concatenate to the list L
    #pickle.dump(L,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb')) # this would save as a List
    # now convert the list to a dictionary to save space, and save it to disk
    dout, doutfinal = LSynWeightToD(L)
    pickle.dump(dout,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb'))
    pickle.dump(doutfinal,open('data/'+dconf['sim']['name']+'synWeights_final.pkl', 'wb'))

if sim.saveWeights: saveSynWeights()


def saveAssignedFiringRates (dAllFiringRates): pickle.dump(dAllFiringRates, open('data/'+dconf['sim']['name']+'AssignedFiringRates.pkl', 'wb'))

def saveInputImages (Images):
  # save input images to txt file (switch to pkl?)
  InputImages = np.array(Images)
  print(InputImages.shape)
  if dconf['net']['useBinaryImage']:
    #InputImages = np.where(InputImages>0,1,0)
    """
    with open('data/'+dconf['sim']['name']+'InputImages.txt', 'w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
      for Input_Image in InputImages:
        np.savetxt(outfile, Input_Image, fmt='%d', delimiter=' ')
        outfile.write('# New slice\n')
    """
    np.save('data/'+dconf['sim']['name']+'InputImages',InputImages)
  else:
    with open('data/'+dconf['sim']['name']+'InputImages.txt', 'w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
      for Input_Image in InputImages:
        np.savetxt(outfile, Input_Image, fmt='%-7.2f', delimiter=' ')
        outfile.write('# New slice\n')

if sim.rank == 0: # only rank 0 should save. otherwise all the other nodes could over-write the output or quit first; rank 0 plots
  if dconf['sim']['doplot']:
    print('plot raster:')
    sim.analysis.plotData()
  if sim.plotWeights: plotWeights()
  saveGameBehavior(sim)
  with open('data/'+dconf['sim']['name']+'ActionsPerEpisode.txt','w') as fid5:
    for i in range(len(epCount)):
      fid5.write('\t%0.1f' % epCount[i])
      fid5.write('\n')
  # if sim.saveInputImages: saveInputImages(sim.AIGame.ReducedImages)
  # #anim.savemp4('/tmp/*.png','data/'+dconf['sim']['name']+'randGameBehavior.mp4',10)
  # if sim.saveMotionFields: saveMotionFields(sim.AIGame.ldflow)
  # if sim.saveObjPos: saveObjPos(sim.AIGame.dObjPos)
  # if sim.saveAssignedFiringRates: saveAssignedFiringRates(sim.AIGame.dAllFiringRates)
  if dconf['sim']['doquit']: quit()
