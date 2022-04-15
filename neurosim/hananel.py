import os
import pickle as pkl
import fire

import netpyne
from sim import NeuroSim
from conf import read_conf, init_wdir
from utils.weights import saveSynWeights

def convert(wdir, conf):
    dconf = read_conf(conf)
    outdir = os.path.join(wdir, 'convert')

    # Initialize the model with dconf config
    dconf['sim']['duration'] = 1e4
    dconf['sim']['recordWeightStepSize'] = 1e4
    dconf['sim']['outdir'] = outdir
    init_wdir(dconf)
    model = NeuroSim(dconf, use_noise=False, save_on_control_c=False)

    weights_path = os.path.join(wdir, 'bestweights.pkl')
    with open(weights_path, 'rb') as f:
        dict_res = pkl.load(f)
        weights = dict_res['best_weights']

    model.setWeightArray(netpyne.sim, weights)
    model.recordWeights(netpyne.sim, 0)
    def _outpath(fname): return os.path.join(dconf['sim']['outdir'], fname)
    saveSynWeights(netpyne.sim, netpyne.sim.allSTDPWeights, _outpath)

if __name__ == '__main__':
  fire.Fire({
    'convert': convert,
  })
