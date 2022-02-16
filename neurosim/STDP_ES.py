import os, sys
import fire
import csv

import numpy as np
from tqdm import tqdm

from sim import NeuroSim
from conf import read_conf, init_wdir
from aigame import AIGame
from game_interface import GameInterface
import netpyne

from multiprocessing import Process, Queue


####################################################
# V1.1: 
# - pre (alpha), during (beta), and post (gamma)
# data is now being recorded
####################################################



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


def init(dconf):
  # Initialize the model with dconf config
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


# Wrapper for netpyne simulation that catched the sys.exit() after one episode (if activated)
def run_model(q, neurosim, EPISODES_PER_ITER_ES, mutated_weights, EPISODES_PER_ITER_DURING_STDP, EPISODES_PER_ITER_POST_STDP, i):

  print("Running model", i+1, "from population...")

  if EPISODES_PER_ITER_ES > 0:
    # alpha: Deactivate STDP and run on just mutations (ES)
    neurosim.STDP_active = False
    neurosim.end_after_episode = EPISODES_PER_ITER_ES
    neurosim.setWeightArray(netpyne.sim, mutated_weights)
    run_episodes(neurosim)
    fitness_NoSTDP = np.mean(neurosim.epCount[-neurosim.end_after_episode:])  
    run_duration_NoSTDP = neurosim.last_times[-1] # TODO: RETURN THIS
  else:
    fitness_NoSTDP = 0

  # beta: Activate STDP and run again
  neurosim.STDP_active = True
  neurosim.end_after_episode = EPISODES_PER_ITER_DURING_STDP
  run_episodes(neurosim)
  fitness_STDP = np.mean(neurosim.epCount[-neurosim.end_after_episode:]) 
  post_STDP_weights = neurosim.getWeightArray(netpyne.sim)  
  run_duration_STDP = neurosim.last_times[-1] # TODO: RETURN THIS

  if EPISODES_PER_ITER_POST_STDP > 0:
    # gamma: Deactivate STDP and run again
    neurosim.STDP_active = False
    neurosim.end_after_episode = EPISODES_PER_ITER_POST_STDP
    run_episodes(neurosim)
    fitness_post_STDP = np.mean(neurosim.epCount[-neurosim.end_after_episode:]) 
    run_duration_STDP = neurosim.last_times[-1] # TODO: RETURN THIS
  else:
    fitness_post_STDP = 0

  # Return using queue
  q.put([fitness_STDP, post_STDP_weights, fitness_NoSTDP, fitness_post_STDP])


def train(dconf=None, outdir=None):

    #### CONSTANTS ####
    dconf = init(read_conf(dconf, outdir=outdir))
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

    EPISODES_PER_ITER_ES = dconf['STDP_ES']['alpha_iters']
    EPISODES_PER_ITER_DURING_STDP = dconf['STDP_ES']['beta_iters']
    EPISODES_PER_ITER_POST_STDP = dconf['STDP_ES']['gamma_iters']
    SAVE_WEIGHTS_EVERY_ITER = dconf['STDP_ES']['save_weights_every_iter'] 


    #### SETUP NETWORK ####

    # TODO: Two sims to run STDP and ES on?
    neurosim = NeuroSim(dconf, use_noise=False, save_on_control_c=False)

    # randomly initialize best weights to the first weights generated
    best_weights = neurosim.getWeightArray(netpyne.sim)

    fres_train = neurosim.outpath('STDP_es_train.csv')
    fres_eval = neurosim.outpath('STDP_es_eval.csv')


    ### RUN SIM ###
    total_time_ES = 0
    total_time_STDP = 0
    spkids_ES = []
    spkts_ES = []
    spkids_STDP = []
    spkts_STDP = []
    V_somas = {}
    for iteration in range(ITERATIONS):
        print("\n--------------------- STDP_ES iteration", iteration+1, "---------------------")

        # Generate mutations for ES
        perturbations = np.random.normal(0, SIGMA, (POPULATION_SIZE, best_weights.size))
        perturbations[perturbations < -0.8] = -0.8      # Clip to avoid negative weights

        print("\nSimulating ES episodes ...")

        # Structures for paralellization
        fitness_STDP = []
        fitness_NoSTDP = []
        fitness_post_STDP = []
        # post_STDP_weights = []
        proc = list()
        q = list()

        # Initialize parallel processes
        for i in range(POPULATION_SIZE):
          mutated_weights = best_weights * (1 + perturbations[i])
          q.append(Queue())
          proc.append(
            Process(target=run_model, args=(q[-1], neurosim, EPISODES_PER_ITER_ES, mutated_weights, EPISODES_PER_ITER_DURING_STDP, EPISODES_PER_ITER_POST_STDP, i))
            )
          proc[-1].start()

        # Await returns...
        for i in range(POPULATION_SIZE):
          returnV = q[i].get()
          fitness_STDP.append(returnV[0])
          # post_STDP_weights.append(returnV[1])
          fitness_NoSTDP.append(returnV[2])
          fitness_post_STDP.append(returnV[3])
          proc[i].join()
        proc.clear()
        q.clear()    

        # For now, use STDP as primary fitness
        fitness = fitness_post_STDP
        fitness = np.expand_dims(np.array(fitness), 1)
        
        fitness_NoSTDP = np.expand_dims(np.array(fitness_NoSTDP), 1)
        fitness_STDP = np.expand_dims(np.array(fitness_STDP), 1)

        # Fitness numerics
        fitness_res = [np.median(fitness), fitness.mean(), fitness.min(), fitness.max()]
        ES_fitness_res = [np.median(fitness_NoSTDP), fitness_NoSTDP.mean(), fitness_NoSTDP.min(), fitness_NoSTDP.max(),
               best_weights.mean()]
        fitness_STDP = [np.median(fitness_STDP), fitness_STDP.mean(), fitness_STDP.min(), fitness_STDP.max(),
               fitness_STDP.mean()]

        # Write fitness results for EA and STDP to csv
        with open(fres_train, 'a') as out:
          writer = csv.writer(out)
          row = [str(r) for r in fitness_res] + [str(r) for r in ES_fitness_res] + [str(r) for r in fitness_STDP]
          writer.writerow(row)    
        print("\nFitness Median: {}; Mean: {} ([{}, {}]). Mean Weight: {}".format(*fitness_res, ES_fitness_res[-1]))

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

        ### EVALUATIONS ####
        # (non-functional)
        # if EVAL_FREQUENCY > 0 and iteration % EVAL_FREQUENCY == 0:
        #     print("Evaluating best weights ... ")
        #     neurosim.setWeightArray(netpyne.sim, best_weights)
        #     eval_total_fitness = 0
        #     for episode in range(EVAL_SAMPLES):
        #         run_episodes(neurosim)
        #         eval_total_fitness += np.mean(neurosim.epCount[-neurosim.end_after_episode:])
        #     print("Best weights performance: ", eval_total_fitness / EVAL_SAMPLES)

        # This one saves weight data
        if (iteration + 1) % SAVE_WEIGHTS_EVERY_ITER == 0:
            print("\nSaving best weights after iteration", iteration)
            neurosim.setWeightArray(netpyne.sim, best_weights)
            neurosim.recordWeights(netpyne.sim, iteration + 1)
            neurosim.save()

    # Save final set of best weights
    if ITERATIONS % SAVE_WEIGHTS_EVERY_ITER != 0:
        print("\nSaving best weights after training", iteration)
        neurosim.setWeightArray(netpyne.sim, best_weights)
        neurosim.recordWeights(netpyne.sim, ITERATIONS)

    # # Save ES spike data
    # netpyne.sim.cfg.filename = os.path.join(dconf['sim']['outdir'], "sim_ES")
    # neurosim.dconf['sim']['duration'] = total_time_ES / 1000
    # netpyne.sim.simData['V_soma'] = V_somas
    # netpyne.sim.simData['spkid'] = spkids_ES
    # netpyne.sim.simData['spkt'] = spkts_ES
    # neurosim.save()

    # # Save STDP spike data
    # netpyne.sim.cfg.filename = os.path.join(dconf['sim']['outdir'], "sim_STDP")
    # neurosim.dconf['sim']['duration'] = total_time_STDP / 1000
    # netpyne.sim.simData['V_soma'] = V_somas
    # netpyne.sim.simData['spkid'] = spkids_STDP
    # netpyne.sim.simData['spkt'] = spkts_STDP
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



# # Mutate and save pre-STDP weights
            # mutated_weights = best_weights * (1 + perturbations[i])

            # # Deactivate STDP and run on just mutations
            # print("\n### Running Non-STDP ({0}/{1})... ###".format(i+1, POPULATION_SIZE))
            # neurosim.STDP_active = False
            # neurosim.end_after_episode = EPISODES_PER_ITER_ES
            # neurosim.setWeightArray(netpyne.sim, mutated_weights)
            # run_episodes(neurosim)
            # fitness_NoSTDP.append(np.mean(neurosim.epCount[-neurosim.end_after_episode:]))
            # run_duration = neurosim.last_times[-1]

            # TODO: This can't be done asynchronous
            # # Save ES spike and V data
            # spkids_ES.extend(netpyne.sim.simData['spkid'])
            # spkts_ES.extend([(t+total_time_ES) for t in netpyne.sim.simData['spkt']])
            # for kvolt, v_soma in netpyne.sim.simData['V_soma'].items():
            #   if kvolt not in V_somas:
            #     V_somas[kvolt] = []
            #   V_somas[kvolt].extend(v_soma)
            # total_time_ES += run_duration

            # # Activate STDP and run again
            # print("\n### Running STDP ({0}/{1})... ###".format(i+1, POPULATION_SIZE))
            # neurosim.STDP_active = True
            # neurosim.end_after_episode = EPISODES_PER_ITER_STDP
            # run_episodes(neurosim)
            # fitness_STDP.append(np.mean(neurosim.epCount[-neurosim.end_after_episode:]))
            # post_STDP_weights.append(neurosim.getWeightArray(netpyne.sim))
            # run_duration = neurosim.last_times[-1]
            
            # # Save STDP spike and V data
            # spkids_STDP.extend(netpyne.sim.simData['spkid'])
            # spkts_STDP.extend([(t+total_time_STDP) for t in netpyne.sim.simData['spkt']])
            # for kvolt, v_soma in netpyne.sim.simData['V_soma'].items():
            #   if kvolt not in V_somas:
            #     V_somas[kvolt] = []
            #   V_somas[kvolt].extend(v_soma)
            # total_time_STDP += run_duration
