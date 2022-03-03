import argparse
from copyreg import pickle
import numpy as np
import pickle
import os, sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + '/neurosim/')

import netpyne
from sim import NeuroSim
from conf import read_conf, init_wdir
from aigame import AIGame

# Wrapper for netpyne simulation that catched the sys.exit() after one episode (if activated)
def run_episodes(neurosim):
    try:
        # Suppress print statements from the netpyne sim
        # sys.stdout = open(os.devnull, 'w')
        neurosim.run()
    except SystemExit:
        pass
    # Turn printing back on after netpyne sim is done
    sys.stdout = sys.__stdout__
    return

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


def run_simulation(id, out_path):
    ## loading
    with open(os.path.normpath(out_path + '/Ready/child_' + str(id) +'.pkl'), 'rb') as out:
        child_data = pickle.load(out)

    weights = child_data['weights']
    config = child_data['config']
    # - pre (alpha), during (beta), and post (gamma)
    alpha = child_data['alpha']
    beta = child_data['beta']
    gamma = child_data['gamma']
    
    os.makedirs(out_path + '/WorkingData/', exist_ok=True)

    ### ---Generate model--- ###
    dconf = init(read_conf(config), out_path + '/WorkingData/child_' + str(id))
    model = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
    model.setWeightArray(netpyne.sim, weights)
    fres_train = model.outpath(out_path + '/WorkingData/STDP_es_train_' + str(id) + '.csv')
    fres_eval = model.outpath(out_path + '/WorkingData/STDP_es_eval_' + str(id) + '.csv')
    
    ### --Run-- ###
    
    # alpha: Deactivate STDP and run on just mutations (ES) #
    model.STDP_active = False
    model.end_after_episode = alpha
    run_episodes(model)
    alpha_perf = np.nanmean(model.epCount[-alpha:])
    alpha_results = np.copy(model.epCount[-alpha:])
    alpha_post_weights = model.getWeightArray(netpyne.sim)  
    alpha_run_duration_STDP = model.last_times[-1] # TODO: RETURN THIS
    
    # beta: Activate STDP and run again #
    model.STDP_active = True
    model.end_after_episode = beta
    run_episodes(model)
    beta_perf = np.nanmean(model.epCount[-beta:])
    beta_results = np.copy(model.epCount[-beta:])
    beta_post_weights = model.getWeightArray(netpyne.sim)  
    beta_run_duration_STDP = model.last_times[-1] # TODO: RETURN THIS
    
    # gamma: Deactivate STDP and run again #
    model.STDP_active = False
    model.end_after_episode = gamma
    run_episodes(model)
    gamma_perf = np.nanmean(model.epCount[-gamma:])
    gama_results = np.copy(model.epCount[-gamma:])
    gamma_post_weights = model.getWeightArray(netpyne.sim)  
    gamma_run_duration_STDP = model.last_times[-1] # TODO: RETURN THIS

    # # TODO: Uncomment for above, the below is just for testing
    # alpha_perf, beta_perf, gamma_perf = 0, 1, 2

    ## --Write Performance-- ##
    dic_obj = {
        'id': id,
        'alpha': 0 if np.isnan(alpha_perf) else alpha_perf,
        'alpha_results': alpha_results,
        'alpha_post_weights': alpha_post_weights,
        'alpha_run_duration_STDP': alpha_run_duration_STDP,
        'beta': 0 if np.isnan(beta_perf) else beta_perf,
        'beta_results': beta_results,
        'beta_post_weights': beta_post_weights,
        'beta_run_duration_STDP': beta_run_duration_STDP,
        'gamma': 0 if np.isnan(gamma_perf) else gamma_perf,
        'gama_results': gama_results,
        'gamma_post_weights': gamma_post_weights,
        'gamma_run_duration_STDP': gamma_run_duration_STDP,
        # what ever you would like to return to parent
    }
    os.makedirs(out_path + '/Done/', exist_ok=True)
    with open(out_path + '/Done/child_' + str(id) +'.tmp', 'wb') as out:
        pickle.dump(dic_obj, out)
    
    # Delete temp data and data from parent
    os.system('rm "' + out_path + '/Ready/child_' + str(id) +'.pkl"')
    
    #The closeest to atomic operation
    os.system('mv "' + out_path + '/Done/child_' + str(id) +'.tmp"  "' + out_path + '/Done/child_' + str(id) +'.pkl"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    parser.add_argument("--out_path", type=str)

    args = parser.parse_args()
    run_simulation(**vars(args))