import argparse
from copyreg import pickle
from matplotlib.pyplot import flag
import numpy as np
import pickle
import os, sys
import glob
import time

from tqdm import tqdm

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + '/neurosim/')
import netpyne
from sim import NeuroSim
from conf import read_conf, backup_config
from collections import deque

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

def main(id, out_path):
    
    model = None
    
    os.makedirs(out_path + '/WorkingData/', exist_ok=True)
    os.makedirs(out_path + '/Done/', exist_ok=True) 
        
    model = None
        
    while True:
        ## loading
        with open(os.path.normpath(out_path + '/Ready/child_' + str(id) +'.pkl'), 'rb') as out:
            child_data = pickle.load(out)

        weights = child_data['weights']
        config = child_data['config']
        # - pre (alpha), during (beta), and post (gamma)
        alpha = child_data['alpha']
        beta = child_data['beta']
        gamma = child_data['gamma']
        
        if model is None:
            ### ---Generate model--- ###
            dconf = read_conf(config)
            # Initialize the model with dconf config
            dconf['sim']['duration'] = 1e4
            dconf['sim']['recordWeightStepSize'] = 1e4
            dconf['sim']['outdir'] = out_path + '/WorkingData/child_' + str(id) 
            model = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
            fres_train = model.outpath(out_path + '/WorkingData/STDP_es_train_' + str(id) + '.csv')
            fres_eval = model.outpath(out_path + '/WorkingData/STDP_es_eval_' + str(id) + '.csv') 
                 
        # set model weights 
        model.setWeightArray(netpyne.sim, weights)
        
        ### --Run-- ###
    
        # alpha: Deactivate STDP and run on just mutations (ES) #
        model.STDP_active = False
        model.end_after_episode = alpha
        run_episodes(model)
        alpha_perf = 0 if alpha==0 else np.nanmean(model.epCount[-alpha:])
        alpha_results = np.copy(model.epCount[-alpha:]) if alpha>0 else 0
        alpha_post_weights = model.getWeightArray(netpyne.sim) if alpha>0 else 0
        alpha_run_duration_STDP = model.last_times[-1] if alpha>0 else 0 # TODO: RETURN THIS
        
        # beta: Activate STDP and run again #
        model.STDP_active = True
        model.end_after_episode = beta
        run_episodes(model)
        beta_perf = 0 if beta==0 else np.nanmean(model.epCount[-beta:])
        beta_results = 0 if beta==0 else np.copy(model.epCount[-beta:])
        beta_post_weights = 0 if beta==0 else model.getWeightArray(netpyne.sim)  
        beta_run_duration_STDP = 0 if beta==0 else model.last_times[-1] # TODO: RETURN THIS
        
        # gamma: Deactivate STDP and run again #
        model.STDP_active = False
        model.end_after_episode = gamma
        run_episodes(model)
        gamma_perf = 0 if gamma==0 else np.nanmean(model.epCount[-gamma:])
        gama_results = 0 if gamma==0 else np.copy(model.epCount[-gamma:])
        gamma_post_weights = 0 if gamma==0 else model.getWeightArray(netpyne.sim)  
        gamma_run_duration_STDP = 0 if gamma==0 else model.last_times[-1] # TODO: RETURN THIS
        model.epCount.clear()

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
        
        if not os.path.exists(out_path + '/Done/'):
            tqdm.write('Identify parant termination!')
            return
        
        with open(out_path + '/Done/child_' + str(id) +'.tmp', 'wb') as out:
            pickle.dump(dic_obj, out)
        
        # Delete temp data and data from parent
        os.system('rm "' + out_path + '/Ready/child_' + str(id) +'.pkl"')
        
        #The closeest to atomic operation
        os.system('mv "' + out_path + '/Done/child_' + str(id) +'.tmp"  "' + out_path + '/Done/child_' + str(id) +'.pkl"')
        
        ### ---ReGenerate model--- ###
        dconf = read_conf(config)
        # Initialize the model with dconf config
        dconf['sim']['duration'] = 1e4
        dconf['sim']['recordWeightStepSize'] = 1e4
        dconf['sim']['outdir'] = out_path + '/WorkingData/child_' + str(id) 
        model = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
        fres_train = model.outpath(out_path + '/WorkingData/STDP_es_train_' + str(id) + '.csv')
        fres_eval = model.outpath(out_path + '/WorkingData/STDP_es_eval_' + str(id) + '.csv') 

        if child_data['Exit?']:
            # last round!
            break
        else:
            # wait for the next data dump
            while(True):
                if not os.path.exists(out_path + '/Done/'):
                    tqdm.write('Identify parant termination!')
                    return
                
                files = (glob.glob(r'' + out_path + '/Ready/child_' + str(id) +'.pkl'))
                if len(files) > 0:
                    break
                time.sleep(1)
                
    tqdm.write(f'Child - {id} Finish run!')    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    parser.add_argument("--out_path", type=str)

    args = parser.parse_args()
    main(**vars(args))