import os, sys, argparse
import numpy as np
import time, glob, pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import signal

import netpyne
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + '/neurosim/')
from sim import NeuroSim
from conf import read_conf, backup_config

def signal_handler(signal, frame):
        done()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
out_path = None

def done():
    global out_path
    os.system('rm -r "' + out_path + '/Ready/"')
    os.system('rm -r "' + out_path + '/WorkingData/"')
    os.system('rm -r "' + out_path + '/Done/"')

def generate_starting_weights(config) -> np.array: 

    def init(dconf):
        # Initialize the model with dconf config
        dconf['sim']['duration'] = 1e4
        dconf['sim']['recordWeightStepSize'] = 1e4

        backup_config(dconf)
        return dconf
    
    dconf = init(config)
    neurosim = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
    return dconf, neurosim.getWeightArray(netpyne.sim)

def plot_performance(open_file, save, title=None):
    data = dict()
    feilds = list()
    n_generations = 0
    with open(open_file,'rt') as f:
        # read lagent
        st = f.readline()
        for field in st.split(','):
            field = field.replace('\n','')
            feilds.append(field)
            data[field] = {
                'data': list(),
                'feild_size': 0
            }
        #read feild size
        st = f.readline().split(',')
        for i, field in enumerate(data.keys()):
            data[field]['feild_size'] = int(st[i])
            
        #space
        st = f.readline().split(',')
        
        #data
        for i_d, d in enumerate(f):
            n_generations += 1
            pop_counter =  i_d % data['pop']['feild_size']
            if pop_counter == 0:
                for k in data.keys():
                    if k == feilds[-1]:
                        continue
                    data[k]['data'].append(np.zeros(data['pop']['feild_size']))
            for i_v, v in enumerate(d.replace('\n','').split(',')):
                data[feilds[i_v]]['data'][-1][pop_counter] = float(v)
                
    # plotting
    plt.figure()
    xAxies = list()
    counter = np.zeros(data['pop']['feild_size'])
    for g in range(n_generations // data['pop']['feild_size']):
        xAxies.extend(counter.tolist())
        counter +=1
    
    if title is None:
        plt.title('')
    else:
        plt.title(title)
    plt.xlabel('Generations') 
    plt.ylabel('Performance') 
    
    for k,v in data.items():
        if k == feilds[-1]:
            continue
        plt.scatter(xAxies, np.array(data[k]['data']).reshape(-1), label = k, s=0.5, alpha=0.3)
    
    plt.legend()
    plt.savefig(save + r".png") 
        
        
def main(
    config,              # Network config
    resume,              # Continue from the last save weights?
):
    ### ---Assertions--- ###
    assert config is not None, 'Config must be given'



    ### ---Constants--- ###
    SUB_PROCESS_FILE = os.path.abspath(os.getcwd()) + '/ChrisHananel/child_process.py'
    Agragate_log_file = 'performance.csv'
    Agragate_Verbos_log_file = 'performance_verbos.csv'


    ### ---Initialize--- ###
    dconf = read_conf(config)

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
    STOP_TRAIN_THRESHOLD = dconf['STDP_ES']['stop_train_threashold']
    STOP_TRAIN_MOVING_AVG = dconf['STDP_ES']['stop_train_moving_avg']
    use_weights_to_mutate = dconf['STDP_ES']['use_weights_to_mutate']

           
    # out_path uniquely identified per child
    global out_path
    out_path = os.path.join(os.getcwd(), 'results', f'{sim_name}')
    
    
    # Establish buffer folders for child outputs
    if resume:
        with open(out_path + '/bestweights.pkl', 'rb') as f:
            parent_weights = pickle.load(f)
        os.system('rm -r "' + out_path + '/Ready/"')
        os.system('rm -r "' + out_path + '/WorkingData/"')
        os.system('rm -r "' + out_path + '/Done/"')
        os.makedirs(out_path + '/Ready/')
    else:            
        try:
            os.makedirs(out_path + '/Ready/')
        except FileExistsError:
            raise Exception("Re-using simulation name, pick a different name")
    
        with open(out_path + '/' + Agragate_log_file,'wt') as f:
            f.write('Alpha,Beta,Gamma,pop\n')
            f.write(f'{alpha},{beta},{gamma},{population}\n')
            f.write(f'\n')
        
        # with open(out_path + '/' + Agragate_Verbos_log_file,'wt') as f:
        #     f.write('Alpha,Beta,Gamma,pop\n')
        #     f.write(f'{population},{alpha},{beta},{gamma}\n')
        #     f.write(f'\n') 

        ### ---Initialize weights--- ###
        dconf, parent_weights = generate_starting_weights(dconf)
    parent_weights[parent_weights < -0.8] = -0.8

    fitness_record = np.zeros((epochs, population, 3))
    moving_performance_log = np.zeros(STOP_TRAIN_MOVING_AVG)
    best_weights = np.copy(parent_weights)
        
    ### ---Evolve--- ###
    for epoch in tqdm(range(epochs), mininterval=1, leave=True,):
        
        if np.mean(moving_performance_log) >= STOP_TRAIN_THRESHOLD:
            break

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
                'Exit?':    (epoch+1)==epochs   # last round?
                # what ever you would like to return to parent
            }
            ## --Save data for child process-- ##
            with open(out_path + '/Ready/child_' + str(child_id) +'.pkl', 'wb') as out:
                pickle.dump(dic_obj, out)

            if epoch == 0:
                # Prepare command for child process #
                args = {
                    'id':       child_id,
                    'out_path': '"' + out_path + '"',  # Output file path
                }
                shell_command = ' '.join(['python3', "'" + SUB_PROCESS_FILE + "'", *(f'--{k} {v}' for k,v in args.items()), '&'])

                # Create parallel process #
                os.system(shell_command)
                
                # from child_process import run_simulation
                # run_simulation(id=child_id, out_path=out_path)


        # Await outputs #
        time.sleep(10)
        files = None
        while(True):
            files = (glob.glob(r'' + out_path + "/Done/*.pkl"))
            if len(files) >= population:
                break
            time.sleep(1)
        
        child_data = list()
        for file in files:
            with open(file, 'rb') as out:
                child_data.append(pickle.load(out))
            
            # delete the file after we collect the data
            os.system('rm "'+ file + '"')

        # TODO: Add information recording for medians, min, max, etc. & for alpha/gamma
        # Record #
        a_list = []; b_list = []; g_list = []
        with open(out_path + '/' + Agragate_log_file,'at') as f:    
            for data in child_data:
                fitness_record[epoch, data['id'], :] = data['alpha'], data['beta'], data['gamma']        
                a_list.append(data['alpha_results'])
                b_list.append(data['beta_results'])
                g_list.append(data['gama_results'])
                f.write(f"{data['alpha']},{data['beta']},{data['gamma']}\n")
        
        # with open(out_path + '/' + Agragate_Verbos_log_file,'at') as f:    
        #     f.write(str(a_list).replace('[','').replace(']',''))
        #     f.write(',' + str(b_list).replace('[','').replace(']',''))
        #     f.write(',' + str(g_list).replace('[','').replace(']',''))
        #     f.write("\n")            
        
        # calculating moving avarage
        moving_performance_log[epoch % STOP_TRAIN_MOVING_AVG] = data['gamma']
        best_fitness = np.mean(moving_performance_log)
        for child in child_data:
            if child['gamma'] > best_fitness:
                best_fitness = child['gamma']
                best_weights = np.copy(child['gamma_post_weights'])

        # This one saves weight data
        if ((epoch + 1) % SAVE_WEIGHTS_EVERY_ITER) == 0 or (epoch+1)==epochs:
            with open(out_path + '/bestweights.pkl', 'wb') as f:
                pickle.dump(best_weights, f)
            plot_performance(open_file=out_path + '/' + Agragate_log_file, 
                             save=out_path + '/performance',
                             title=sim_name
                             )
            # plot_performance_verbos(open_file=out_path + '/' + Agragate_log_file, save=out_path + '/performance')
            
        # Evaluate children #
        STDP_perfs = fitness_record[epoch, :, 2]    # all of ith epochs Gamma fitness
        # normalize the fitness for more stable training
        normalized_fitness = (STDP_perfs - STDP_perfs.mean()) / (STDP_perfs.std() + 1e-8)
        fitness_weighted_mutations = (normalized_fitness.reshape(-1, 1) * mutations)

        # Evolve parent #
        if use_weights_to_mutate:
            parent_weights = np.copy(best_weights)
        parent_weights = parent_weights * (1 + (LEARNING_RATE * fitness_weighted_mutations.mean(axis=0)))

        # TODO: Potentially Sigma & LR Decay (not used in original)
        # decay sigma and the learning rate
        SIGMA *= SIGMA_DECAY
        LEARNING_RATE *= LR_DECAY
    done()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--config", type=str)

    args = parser.parse_args()
    args.resume = True if args.resume == 'True' or args.resume == 'true' else False

    main(**vars(args))
    
    # out_path = '/mnt/d/LocalUserData/Box Sync/git_repo/netpyne-STDP/results/pop-10,alpha-0,beta-30,gama-10.withWights'
    # Agragate_log_file = 'performance.csv'
    # plot_performance(open_file=out_path + '/' + Agragate_log_file, save=out_path + '/performance')

