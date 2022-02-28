import os, sys, argparse
import numpy as np
import time, glob, pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + '/neurosim')

def generate_starting_weights(config) -> np.array: 
    import netpyne
    from sim import NeuroSim
    from conf import read_conf, init_wdir
    from aigame import AIGame
    def init(dconf):
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

        init_wdir(dconf)
        return dconf
    
    dconf = init(read_conf(config))
    neurosim = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
    return dconf, neurosim.getWeightArray(netpyne.sim)

    # return np.ones((100, 200))


def main(
    config,              # Network config
):
    # sim_name,            # Simulation ID (Uniquely identify this run)
    # epochs, population,  # Evol. general params
    # alpha, beta, gamma,  # STDP+ES params
    # sigma, lr            # Evol. learning params

    ### ---Assertions--- ###
    assert config is not None, 'Config must be given'
    # assert sim_name is not None, 'Simulation name must be defined'
    # assert epochs is not None, 'Number of epochs must be defined'
    # assert population is not None, 'Population must be defined'
    # assert alpha is not None, 'Alpha must be defined'
    # assert beta is not None, 'Beta must be defined'
    # assert gamma is not None, 'Gamma must be defined'


    ### ---Constants--- ###
    SUB_PROCESS_FILE = 'child_process.py'


    ### ---Initialize--- ###
    dconf, parent_weights = generate_starting_weights(config)
    parent_weights[parent_weights < -0.8] = -0.8
    

    #### --- Set variabels--- ###
    sim_name = dconf['sim']['outdir'].split('/')[-1]
    epochs = dconf['STDP_ES']['iterations']
    population = dconf['STDP_ES']['population_size']
    alpha = dconf['STDP_ES']['alpha_iters']
    beta = dconf['STDP_ES']['beta_iters']
    gamma = dconf['STDP_ES']['gamma_iters']

    SIGMA = dconf['STDP_ES']['sigma'] # 0.1 # standard deviation of perturbations applied to each member of population
    LEARNING_RATE = dconf['STDP_ES']['learning_rate'] # 1 # what percentage of the return normalized perturbations to add to best_weights
    # How much to decay the learning rate and sigma by each episode. In theory
    # this should lead to better
    LR_DECAY = dconf['STDP_ES']['decay_lr'] # 1
    SIGMA_DECAY = dconf['STDP_ES']['decay_sigma'] # 1
    SAVE_WEIGHTS_EVERY_ITER = dconf['STDP_ES']['save_weights_every_iter']


    fitness_record = np.zeros((epochs, population, 3))

    # out_path uniquely identified per child
    out_path = os.path.join(os.getcwd(), 'results', f'{sim_name}')
    # Establish buffer folders for child outputs
    try:
        # os.mkdir(out_path)
        os.mkdir(out_path + '/Ready/')
    except FileExistsError:
        raise Exception("Re-using simulation name, pick a different name")


    save_flag = False

    ### ---Evolve--- ###
    for epoch in tqdm(range(epochs)):

        # Mutated weights #
        mutations = np.random.normal(0, SIGMA, (population, *parent_weights.shape))
        mutations[mutations < -0.8] = -0.8

        # Mutate & run population #
        tqdm.write('Running child process')
        for child_id in range(population):

            child_weights = mutations[child_id, :] + parent_weights

            dic_obj = {
                'weights': child_weights, # Mutated weights # TODO: Figure out way to get numpy weights to child
                'config':   config,             # Config file TODO: path?
                'alpha':    alpha,              # Num. Pre-STDP iters
                'beta':     beta,               # Num. STDP iters
                'gamma':    gamma,              # Num. Post-STDP iters
                'save_flag': save_flag,         # save weights?
                # what ever you would like to return to parent
            }
            ## --Save data for child process-- ##
            with open(out_path + '/Ready/child_' + str(child_id) +'.pkl', 'wb') as out:
                pickle.dump(dic_obj, out)

            save_flag = False

            # Prepare command for child process #

            # args = {
            #     'id':       child_id,
            #     'out_path': "\'"+out_path+"\'",  # Output file path
            # }
            # shell_command = ' '.join(['python3', SUB_PROCESS_FILE, *(f'--{k} {v}' for k,v in args.items())])

            # # Create parallel process #
            # os.system(shell_command)
            from child_process import run_simulation
            run_simulation(child_id=child_id, out_path=out_path)


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


        # # TODO: Add information recording for medians, min, max, etc. & for alpha/gamma
        # # Record #
        # fitness_record[epoch, child_id, :] = alpha_perf, beta_perf, gamma_perf


        # Evaluate children #
        STDP_perfs = np.expand_dims(fitness_record[epoch, :, 2], axis=(1,2))    # all of ith epochs beta fitness
        # normalize the fitness for more stable training
        normalized_fitness = (STDP_perfs - STDP_perfs.mean()) / (STDP_perfs.std() + 1e-8)
        fitness_weighted_mutations = (normalized_fitness * mutations)

        # Evolve parent #
        parent_weights = parent_weights * (1 + (LEARNING_RATE * fitness_weighted_mutations.mean(axis=0)))

        # decay sigma and the learning rate
        SIGMA *= SIGMA_DECAY
        LEARNING_RATE *= LR_DECAY

        # TODO: Potentially Sigma & LR Decay (not used in original)
        
        # This one saves weight data
        if (epoch + 1) % SAVE_WEIGHTS_EVERY_ITER == 0:
            save_flag = True
            

    # TODO: Write final results to directory f'./results/{sim_name}'
    os.makedirs(out_path + '/rerults', exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--sim_name", type=str)
    parser.add_argument("--config", type=str)
    # parser.add_argument("--epochs", type=int)
    # parser.add_argument("--population", type=int)
    # parser.add_argument("--alpha", type=int)
    # parser.add_argument("--beta", type=int)
    # parser.add_argument("--gamma", type=int)
    # parser.add_argument("--sigma", type=int, default=1)
    # parser.add_argument("--lr", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))
