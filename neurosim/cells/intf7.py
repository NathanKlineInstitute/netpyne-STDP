from neuron import h

# synaptic indices used in intf7.mod NET_RECEIVE
dsyn = {'AM': 0, 'NM': 1, 'GA': 2, 'AM2': 3, 'NM2': 4, 'GA2': 5}
dsyn['AMPA'] = dsyn['AM']
dsyn['NMDA'] = dsyn['NM']
dsyn['GABA'] = dsyn['GA']


class INTF7E():
  # parameters for excitatory neurons
  def __init__(self, dconf):
    self.dparam = dconf['cell']['E']
    cell = self.intf = h.INTF7()


class INTF7I():
  # parameters for fast-spiking interneurons
  def __init__(self, dconf):
    self.dparam = dconf['cell']['I']
    cell = self.intf = h.INTF7()


class INTF7IL():
  # parameters for low-threshold firing interneurons
  def __init__(self, dconf):
    self.dparam = dconf['cell']['IL']
    cell = self.intf = h.INTF7()


def insertSpikes(sim, dt, spkht=50):
  # inserts spikes into voltage traces(paste-on); depends on NetPyNE simulation data format
  import pandas as pd
  import numpy as np
  sampr = 1e3 / dt  # sampling rate
  spkt, spkid = sim.simData['spkt'], sim.simData['spkid']
  spk = pd.DataFrame(np.array([spkid, spkt]).T, columns=['spkid', 'spkt'])
  for kvolt in sim.simData['V_soma'].keys():
    cellID = int(kvolt.split('_')[1])
    spkts = spk[spk.spkid == cellID]
    if len(spkts):
      for idx in spkts.index:
        tdx = int(spk.at[idx, 'spkt'] * sampr / 1e3)
        sim.simData['V_soma'][kvolt][tdx] = spkht
