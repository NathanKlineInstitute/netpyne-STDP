import argparse
from copyreg import pickle
import numpy as np
import pickle
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + '/neurosim')

import netpyne
from sim import NeuroSim
from conf import read_conf, init_wdir
from aigame import AIGame

def init(dconf, out_path):
    # Initialize the model with dconf config
    dconf['sim']['duration'] = 1e4
    dconf['sim']['recordWeightStepSize'] = 1e4

    outdir = dconf['sim']['outdir']
    if os.path.isdir(outdir):
        evaluations = [fname 
                        for fname in os.listdir(outdir) 
                        if fname.startswith('evaluation_') and os.path.isdir(os.path.join(outdir, fname))
                        ]
        if len(evaluations) > 0:
            raise Exception(' '.join([
                'You have run evaluations on {}: {}.'.format(outdir, evaluations),
                'This will rewrite!',
                'Please delete to continue!']))

    dconf['sim']['outdir'] = out_path 
    init_wdir(dconf)
    
    return dconf


def run_simulation(child_id, out_path):
    ## loading
    with open(out_path + '/Ready/child_' + str(child_id) +'.pkl', 'rb') as out:
        child_data = pickle.load(out)

    weights = child_data['weights']
    config = child_data['config']
    # - pre (alpha), during (beta), and post (gamma)
    alpha = child_data['alpha']
    beta = child_data['beta']
    gamma = child_data['gamma']
    
    save_flag = child_data['save_flag']

    os.makedirs(out_path + '/WorkingData/', exist_ok=True)

    ### ---Generate model--- ###
    dconf = init(read_conf(config), out_path + '/WorkingData/child_' + str(child_id))
    model = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
    model.setWeightArray(netpyne.sim, weights)
    fres_train = model.outpath(out_path + '/WorkingData/STDP_es_train_' + str(child_id) + '.csv')
    fres_eval = model.outpath(out_path + '/WorkingData/STDP_es_eval_' + str(child_id) + '.csv')
    
    ### --Run-- ###
    
    # alpha: Deactivate STDP and run on just mutations (ES) #
    model.STDP_active = False
    model.end_after_episode = alpha
    model.run()
    alpha_perf = np.mean(model.epCount[-alpha:])
    
    # beta: Activate STDP and run again #
    model.STDP_active = True
    model.end_after_episode = beta
    model.run()
    beta_perf = np.mean(model.epCount[-beta:])
    
    # gamma: Deactivate STDP and run again #
    model.STDP_active = False
    model.end_after_episode = gamma
    model.run()
    gamma_perf = np.mean(model.epCount[-gamma:])

    # # TODO: Uncomment for above, the below is just for testing
    # alpha_perf, beta_perf, gamma_perf = 0, 1, 2

    ## --Write Performance-- ##
    dic_obj = {
        'id': child_id,
        'alpha': alpha_perf,
        'beta': beta_perf,
        'gamma_perf':gamma_perf,
        # what ever you would like to return to parent
    }
    with open(out_path + '/Done/child_' + str(child_id) +'.tmp', 'wb') as out:
        pickle.dump(dic_obj, out)
    
    # Delete data from parent
    os.system('rm ' + out_path + '/Ready/child_' + str(child_id) +'.pkl')
    
    #The closeest to atomic operation
    os.system('mv ' + out_path + '/Done/child_' + str(child_id) +'.tmp ' + out_path + '/child_' + str(child_id) +'.pkl')

    if save_flag:    
        print("\nSaving best weights after iteration")
        # model.setWeightArray(netpyne.sim, best_weights)
        model.recordWeights(netpyne.sim)
        model.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    parser.add_argument("--out_path", type=str)

    args = parser.parse_args()
    run_simulation(**vars(args))