import os, sys
import fire

import numpy as np
from tqdm import tqdm

sys.path.append('/home/hananel/git_repo/netpyne-STDP/')

from sim import NeuroSim
from conf import read_conf, init_wdir
from aigame import AIGame
from game_interface import GameInterface
import netpyne

import matplotlib
matplotlib.use('Agg') 

from multiprocessing import Process, Queue


# Wrapper for netpyne simulation that catched the sys.exit() after one episode (if activated)
def run_episodes(q, neurosim, EPISODES_PER_ITER_ES, mutated_weights, EPISODES_PER_ITER_STDP):

    # Deactivate STDP & evaluate w/ original mutations
    neurosim.STDP_active = False
    neurosim.end_after_episode = EPISODES_PER_ITER_ES
    neurosim.setWeightArray(netpyne.sim, mutated_weights)
    
    try:
        # Suppress print statements from the netpyne sim
        # sys.stdout = open(os.devnull, 'w')
        neurosim.run()
    except SystemExit:
        pass
    
    fitness_NoSTDP = np.mean(neurosim.epCount[-neurosim.end_after_episode:])
    run_duration = neurosim.last_times[-1]

    # Activate STDP, train, record
    neurosim.STDP_active = True
    neurosim.end_after_episode = EPISODES_PER_ITER_STDP
    try:
        # Suppress print statements from the netpyne sim
        # sys.stdout = open(os.devnull, 'w')
        neurosim.run()
    except SystemExit:
        pass
    fitness_STDP = np.mean(neurosim.epCount[-neurosim.end_after_episode:])
    post_STDP_weights = neurosim.getWeightArray(netpyne.sim)
    run_duration += neurosim.last_times[-1]

    q.put([fitness_STDP, post_STDP_weights, fitness_NoSTDP])

    # Turn printing back on after netpyne sim is done
    sys.stdout = sys.__stdout__
    return


def init(dconf):
  # Initialize the model with dconf config
  if not dconf:
      dconf = read_conf()
  dconf['sim']['duration'] = 1e4
  dconf['sim']['recordWeightStepSize'] = 1e4

  outdir = dconf['sim']['outdir']
  if os.path.isdir(outdir):
    evaluations = [fname
      for fname in os.listdir(outdir)
      if fname.startswith('evaluation_') and os.path.isdir(os.path.join(outdir, fname))]
    if len(evaluations) > 0:
      raise Exception(' '.join([
          'You have run evaluations on {}: {}.'.format(outdir, evaluations),
          'This will rewrite!',
          'Please delete to continue!']))

  init_wdir(dconf)
  return dconf


def train(dconf=None):

    #### CONSTANTS ####
    dconf = init(dconf)
    ITERATIONS = dconf['STDP_ES']['iterations'] # How many iterations to train for
    POPULATION_SIZE = dconf['STDP_ES']['population_size'] # How many perturbations of weights to try per iteration
    SIGMA = dconf['STDP_ES']['sigma'] # 0.1 # standard deviation of perturbations applied to each member of population
    LEARNING_RATE = dconf['STDP_ES']['learning_rate'] # 1 # what percentage of the return normalized perturbations to add to best_weights

    EVAL_FREQUENCY = dconf['STDP_ES']['eval_freq'] # How often to run evaluations on best_weights
    EVAL_SAMPLES = dconf['STDP_ES']['eval_samples'] # Sample size for eval

    # How much to decay the learning rate and sigma by each episode. In theory
    # this should lead to better
    LR_DECAY = dconf['STDP_ES']['decay_lr'] # 1
    SIGMA_DECAY = dconf['STDP_ES']['decay_sigma'] # 1

    EPISODES_PER_ITER_ES = dconf['STDP_ES']['episodes_per_iter_ES'] # 1
    EPISODES_PER_ITER_STDP = dconf['STDP_ES']['episodes_per_iter_STDP'] # 10
    SAVE_WEIGHTS_EVERY_ITER = dconf['STDP_ES']['save_weights_every_iter'] # 10


    #### SETUP NETWORK ####
    neurosim = NeuroSim(dconf, use_noise=False, save_on_control_c=False)

    # randomly initialize best weights to the first weights generated
    best_weights = neurosim.getWeightArray(netpyne.sim)

    fres_train = neurosim.outpath('STDP_es_train.txt')
    fres_eval = neurosim.outpath('STDP_es_eval.txt')


    ### RUN SIM ###
    STDP_performance = []
    EA_performance = []
    total_time = 0
    spkids = []
    spkts = []
    V_somas = {}
    for iteration in range(ITERATIONS):
        print("\n--------------------- STDP_ES iteration", iteration, "---------------------")

        # Generate mutations for ES
        perturbations = np.random.normal(0, SIGMA, (POPULATION_SIZE, best_weights.size))
        perturbations[perturbations < -0.8] = -0.8      # Clip to avoid negative weights

        print("\nSimulating ES episodes ... ")

        # Run networks with mutations of best weights
        fitness_STDP = []
        fitness_NoSTDP = []
        post_STDP_weights = []
        proc = list()
        q = list()
        for i in range(POPULATION_SIZE):

            # Mutate and save pre-STDP weights
            mutated_weights = best_weights * (1 + perturbations[i])
            q.append(Queue())
            proc.append(Process(target=run_episodes, args=(q[-1], neurosim, EPISODES_PER_ITER_ES, mutated_weights, EPISODES_PER_ITER_STDP)))
            proc[-1].start()
        
        for i in range(POPULATION_SIZE):
          returnV = q[i].get()
          fitness_STDP.append(returnV[0])
          post_STDP_weights.append(returnV[1])
          fitness_NoSTDP.append(returnV[2])
          proc[i].join()
        proc.clear()
        q.clear()

            # # Deactivate STDP & evaluate w/ original mutations
            # print("\n### Running Non-STDP ({0}/{1})... ###".format(i, POPULATION_SIZE))
            # neurosim.STDP_active = False
            # neurosim.end_after_episode = EPISODES_PER_ITER_ES
            # neurosim.setWeightArray(netpyne.sim, mutated_weights)
            
            # run_episodes(neurosim)
            # fitness_NoSTDP.append(np.mean(neurosim.epCount[-neurosim.end_after_episode:]))
            # run_duration = neurosim.last_times[-1]

            # # Activate STDP, train, record
            # print("\n### Running STDP ({0}/{1})... ###".format(i, POPULATION_SIZE))
            # neurosim.STDP_active = True
            # neurosim.end_after_episode = EPISODES_PER_ITER_STDP
            # run_episodes(neurosim)
            # fitness_STDP.append(np.mean(neurosim.epCount[-neurosim.end_after_episode:]))
            # post_STDP_weights.append(neurosim.getWeightArray(netpyne.sim))
            # run_duration += neurosim.last_times[-1]
            
            # save the spike and V data
            # spkids.extend(netpyne.sim.simData['spkid'])
            # spkts.extend([(t+total_time) for t in netpyne.sim.simData['spkt']])
            # for kvolt, v_soma in netpyne.sim.simData['V_soma'].items():
            #   if kvolt not in V_somas:
            #     V_somas[kvolt] = []
            #   V_somas[kvolt].extend(v_soma)
            # total_time += run_duration

        # For now, use STDP fitness to assess if it works for this algorithm
        STDP_performance.extend(fitness_STDP)
        EA_performance.extend(fitness_NoSTDP)
        fitness = fitness_STDP.copy()
        fitness = np.expand_dims(np.array(fitness), 1)
        
        fitness_res = [np.median(fitness), fitness.mean(), fitness.min(), fitness.max(),
               best_weights.mean()]
        with open(fres_train, 'a') as out:
          out.write('\t'.join(
            [str(neurosim.end_after_episode)] + [str(r) for r in fitness_res]) + '\n')
        print("\nFitness Median: {}; Mean: {} ([{}, {}]). Mean Weight: {}".format(*fitness_res))

        # normalize the fitness for more stable training
        normalized_fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        # weight the perturbations by their normalized fitness so that perturbations
        # that performed well are added to the best weights and those that performed poorly are subtracted
        fitness_weighted_perturbations = (normalized_fitness * perturbations)

        # apply the fitness_weighted_perturbations to the current best weights proportionally to the LR
        best_weights = best_weights * (1 + (LEARNING_RATE * fitness_weighted_perturbations.mean(axis = 0)))

        # decay sigma and the learning rate
        SIGMA *= SIGMA_DECAY
        LEARNING_RATE *= LR_DECAY

        #### EVALUATIONS ####
        # (non-functional)
        # if EVAL_FREQUENCY > 0 and iteration % EVAL_FREQUENCY == 0:
        #     print("Evaluating best weights ... ")
        #     neurosim.setWeightArray(netpyne.sim, best_weights)
        #     eval_total_fitness = 0
        #     for episode in range(EVAL_SAMPLES):
        #         run_episodes(neurosim)
        #         eval_total_fitness += np.mean(neurosim.epCount[-neurosim.end_after_episode:])
        #     print("Best weights performance: ", eval_total_fitness / EVAL_SAMPLES)

        # if (iteration + 1) % SAVE_WEIGHTS_EVERY_ITER == 0:
        #     print("\nSaving best weights after iteration", iteration)
        #     neurosim.setWeightArray(netpyne.sim, best_weights)
        #     neurosim.recordWeights(netpyne.sim, iteration + 1)


    if ITERATIONS % SAVE_WEIGHTS_EVERY_ITER != 0:
        print("\nSaving best weights after training", iteration)
        neurosim.setWeightArray(netpyne.sim, best_weights)
        neurosim.recordWeights(netpyne.sim, ITERATIONS)

    ### TEMPORARY record data ###
    import csv
    with open("STDP_fitness.csv", 'w') as file:
      writer = csv.writer(file)
      for (STDP_perf, EA_perf) in zip(STDP_performance, EA_performance):
        writer.writerow([STDP_perf, EA_perf])

    # neurosim.dconf['sim']['duration'] = total_time / 1000
    # netpyne.sim.simData['V_soma'] = V_somas
    # netpyne.sim.simData['spkid'] = spkids
    # netpyne.sim.simData['spkt'] = spkts
    # neurosim.save()


def continue_main(wdir, iterations=100, index=None):
  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')

  outdir = os.path.join(wdir, 'continue_{}'.format(1 if index == None else index))
  dconf = read_conf(dconf_path, outdir=outdir)
  synWeights_file = os.path.join(wdir, 'synWeights.pkl')

  dconf['simtype']['ResumeSim'] = 1
  dconf['simtype']['ResumeSimFromFile'] = synWeights_file
  if iterations != None:
    dconf['ES_STDP']['iterations'] = iterations
  dconf['sim']['plotRaster'] = 0

  train(dconf)

if __name__ == "__main__":
  fire.Fire({
      'train': train,
      'continue': continue_main
  })
