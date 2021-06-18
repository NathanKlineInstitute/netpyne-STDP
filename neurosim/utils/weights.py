import os
import pickle
import time


def _LSynWeightToD(L):
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
    dout[preID][poID].append([t, w])
  for preID in doutfinal.keys():
    for poID in doutfinal[preID].keys():
      doutfinal[preID][poID].append(dout[preID][poID][-1])
  return dout, doutfinal


def saveSynWeights(sim, lsynweights):
  # save synaptic weights to disk for this node
  with open(sim.outpath(f'synWeights_{str(sim.rank)}.pkl'), 'wb') as f:
    pickle.dump(lsynweights, f)
  sim.pc.barrier()  # wait for other nodes
  if sim.rank == 0:  # rank 0 reads and assembles the synaptic weights into a single output file
    L = []
    for i in range(sim.nhosts):
      fn = sim.outpath(f'synWeights_{str(i)}.pkl')
      while not os.path.isfile(fn):  # wait until the file is written/available
        print('saveSynWeights: waiting for finish write of', fn)
        time.sleep(1)
      with open(fn, 'rb') as f:
        lw = pickle.load(f)
        print(fn, 'len(lw)=', len(lw), type(lw))
      L = L + lw  # concatenate to the list L
    # now convert the list to a dictionary to save space, and save it to disk
    dout, doutfinal = _LSynWeightToD(L)
    with open(sim.outpath('synWeights.pkl'), 'wb') as f:
      pickle.dump(dout, f)
    with open(sim.outpath('synWeights_final.pkl'), 'wb') as f:
      pickle.dump(doutfinal, f)