import sys
sys.path.append('../../neurosim')

import os
import csv
import argparse
import numpy as np
import time
import glob
import pickle
# from neurosim.utils.weights import getInitSTDPWeight


def generate_starting_weights(config) -> np.array:

    # TODO: Re-enable/test this on Hananels system
    # from neurosim.sim import NeuroSim
    # from neurosim.conf import read_conf
    # from neurosim.STDP_ES import init
    # import netpyne
    # dconf = init(read_conf(config))
    # neurosim = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
    # return neurosim.getWeightArray(netpyne.sim)

    return np.ones((100, 200))


def main(
    sim_name,            # Simulation ID (Uniquely identify this run)
    config,              # Network config
    epochs, population,  # Evol. general params
    alpha, beta, gamma,  # STDP+ES params
    sigma, lr            # Evol. learning params
):

    ### ---Assertions--- ###
    assert sim_name is not None, 'Simulation name must be defined'
    assert epochs is not None, 'Number of epochs must be defined'
    assert population is not None, 'Population must be defined'
    assert config is not None, 'Config must be given'
    assert alpha is not None, 'Alpha must be defined'
    assert beta is not None, 'Beta must be defined'
    assert gamma is not None, 'Gamma must be defined'


    ### ---Constants--- ###
    SUB_PROCESS_FILE = 'child_process.py'



    ### ---Initialize--- ###
    parent_weights = generate_starting_weights(config)
    parent_weights[parent_weights < -0.8] = -0.8

    fitness_record = np.zeros((epochs, population, 3))

    # out_path uniquely identified per child
    out_path = os.path.join(os.getcwd(), 'buffers', f'{sim_name}')
    # Establish buffer folders for child outputs
    try:
        os.mkdir(out_path)
        os.mkdir(out_path + '/Ready/')
    except FileExistsError:
        raise Exception("Re-using simulation name, pick a different name")



    ### ---Evolve--- ###
    for epoch in range(epochs):


        # Mutated weights #
        mutations = np.random.normal(0, sigma, (population, *parent_weights.shape))
        mutations[mutations < -0.8] = -0.8


        # Mutate & run population #
        for child_id in range(population):

            child_weights = mutations[child_id, :] + parent_weights

            dic_obj = {
                'child_weights': child_weights, # Mutated weights # TODO: Figure out way to get numpy weights to child
                'config':   config,             # Config file TODO: path?
                'alpha':    alpha,              # Num. Pre-STDP iters
                'beta':     beta,               # Num. STDP iters
                'gamma':    gamma,              # Num. Post-STDP iters
                # what ever you would like to return to parent
            }
            ## --Save data for child process-- ##
            with open(out_path + '/Ready/child_' + str(child_id) +'.pkl', 'wb') as out:
                pickle.dump(dic_obj, out)

            # Prepare command for child process #

            args = {
                'id':       child_id,
                'out_path': "\'"+out_path+"\'"  # Output file path
            }
            shell_command = ' '.join(['python3', SUB_PROCESS_FILE, *(f'--{k} {v}' for k,v in args.items())])

            # Create parallel process #
            os.system(shell_command)


        # Await outputs #
        files = None
        while(True):
            files = (glob.glob(out_path + "/Done/*.pkl"))
            if len(files) >= population:
                break
            time.sleep(1)
        
        child_data = list()
        for file in files:
            with open(file, 'rb') as out:
                child_data.append(pickle.load(out))
            
            #delete the file after we collect the data
            # os.system('rm '+ file)


        # TODO: Add information recording for medians, min, max, etc. & for alpha/gamma
        # Record #
        fitness_record[epoch, child_id, :] = alpha_perf, beta_perf, gamma_perf


        # Evaluate children #
        STDP_perfs = np.expand_dims(fitness_record[epoch, :, 2], axis=(1,2))    # all of ith epochs beta fitness
        normalized_fitness = (STDP_perfs - STDP_perfs.mean()) / (STDP_perfs.std() + 1e-8)
        fitness_weighted_mutations = (normalized_fitness * mutations)

        # Evolve parent #
        parent_weights = parent_weights * (1 + (lr * fitness_weighted_mutations.mean(axis=0)))

        # TODO: Potentially Sigma & LR Decay (not used in original)

    # TODO: Write final results to directory f'./results/{sim_name}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_name", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--beta", type=int)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--lr", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))
