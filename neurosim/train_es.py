import os, sys
import fire

import numpy as np
from tqdm import tqdm

from sim import NeuroSim
from conf import read_conf, init_wdir
from aigame import AIGame
from game_interface import GameInterface
from utils.weights import readWeights
import netpyne

from multiprocessing import Process, Queue


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


# Wrapper for netpyne simulation that catched the sys.exit() after one episode (if activated)
def run_model(q, neurosim, mutated_weights, i):
  neurosim.setWeightArray(netpyne.sim, mutated_weights)
  run_episodes(neurosim)

  fitness = np.mean(neurosim.epCount[-neurosim.end_after_episode:])
  run_duration = neurosim.last_times[-1]

  # Return using queue
  q.put([fitness, run_duration])


def init(dconf, fnjson=None, outdir=None):
  # Initialize the model with dconf config
  if not dconf:
      dconf = read_conf(fnjson, outdir=outdir)
  dconf['sim']['duration'] = 1e10
  dconf['sim']['recordWeightStepSize'] = 1e10

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

def train(dconf=None, fnjson=None, outdir=None, save_spikes=False):
    dconf = init(dconf, fnjson, outdir=outdir)
    ITERATIONS = dconf['ES']['iterations'] # How many iterations to train for
    POPULATION_SIZE = dconf['ES']['population_size'] # How many perturbations of weights to try per iteration
    SIGMA = dconf['ES']['sigma'] # 0.1 # standard deviation of perturbations applied to each member of population
    LEARNING_RATE = dconf['ES']['learning_rate'] # 1 # what percentage of the return normalized perturbations to add to best_weights

    EVAL_FREQUENCY = dconf['ES']['eval_freq'] # How often to run evaluations on best_weights
    EVAL_SAMPLES = dconf['ES']['eval_samples'] # Sample size for eval

    # How much to decay the learning rate and sigma by each episode. In theory
    # this should lead to better
    LR_DECAY = dconf['ES']['decay_lr'] # 1
    SIGMA_DECAY = dconf['ES']['decay_sigma'] # 1

    EPISODES_PER_ITER = dconf['ES']['episodes_per_iter'] # 5
    SAVE_WEIGHTS_EVERY_ITER = dconf['ES']['save_weights_every_iter'] # 10

    # Setup
    neurosim = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
    neurosim.STDP_active = False # deactivate STDP
    neurosim.end_after_episode = EPISODES_PER_ITER # activate sys.exit() after this many episode

    # randomly initialize best weights to the first weights generated
    best_weights = neurosim.getWeightArray(netpyne.sim)

    fres_train = neurosim.outpath('es_train.txt')
    fres_eval = neurosim.outpath('es_eval.txt')

    total_time = 0
    spkids = []
    spkts = []
    V_somas = {}
    for iteration in range(ITERATIONS):
        print("\n--------------------- ES iteration", iteration, "---------------------")

        # generate unique randomly sampled perturbations to add to each member of
        # this iteration's the training population
        perturbations = np.random.normal(0, SIGMA, (POPULATION_SIZE, best_weights.size))

        # this should rarely be used with reasonable sigma but just in case we
        # clip perturbations that are too low since they can make the weights negative
        perturbations[perturbations < -0.8] = -0.8

        print("\nSimulating episodes ... ")
        proc = list()
        q = list()

        for i in range(POPULATION_SIZE):
            mutated_weights = best_weights * (1 + perturbations[i])
            q.append(Queue())
            proc.append(Process(
              target=run_model,
              args=(q[-1], neurosim, mutated_weights, i)))
            proc[-1].start()


        # get the fitness of each set of perturbations when applied to the current best weights
        # by simulating each network and getting the episode length as the fitness
        fitness = []
        for i in range(POPULATION_SIZE):
            individual_fitness, run_duration = q[i].get()

            if save_spikes:
              # save the spike and V data
              spkids.extend(netpyne.sim.simData['spkid'])
              spkts.extend([(t+total_time) for t in netpyne.sim.simData['spkt']])
              for kvolt, v_soma in netpyne.sim.simData['V_soma'].items():
                if kvolt not in V_somas:
                  V_somas[kvolt] = []
                V_somas[kvolt].extend(v_soma)
            total_time += run_duration

            # Add fitness
            fitness.append(individual_fitness)

        fitness = np.expand_dims(np.array(fitness), 1)

        fitness_res = [np.median(fitness), fitness.mean(), fitness.min(), fitness.max(),
               best_weights.mean()]
        with open(fres_train, 'a') as out:
          out.write('\t'.join(
            [str(EPISODES_PER_ITER)] + [str(r) for r in fitness_res]) + '\n')
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

        if EVAL_FREQUENCY > 0 and iteration % EVAL_FREQUENCY == 0:
            print("Evaluating best weights ... ")
            neurosim.setWeightArray(netpyne.sim, best_weights)
            eval_total_fitness = 0
            for episode in range(EVAL_SAMPLES):
                run_episodes(neurosim)
                eval_total_fitness += np.mean(neurosim.epCount[-neurosim.end_after_episode:])
            print("Best weights performance: ", eval_total_fitness / EVAL_SAMPLES)

        if (iteration + 1) % SAVE_WEIGHTS_EVERY_ITER == 0:
            print("\nSaving best weights after iteration", iteration)
            neurosim.setWeightArray(netpyne.sim, best_weights)
            neurosim.recordWeights(netpyne.sim, iteration + 1)


    if ITERATIONS % SAVE_WEIGHTS_EVERY_ITER != 0:
        print("\nSaving best weights after training", iteration)
        neurosim.setWeightArray(netpyne.sim, best_weights)
        neurosim.recordWeights(netpyne.sim, ITERATIONS)


    neurosim.dconf['sim']['duration'] = total_time / 1000
    if save_spikes:
      netpyne.sim.simData['V_soma'] = V_somas
      netpyne.sim.simData['spkid'] = spkids
      netpyne.sim.simData['spkt'] = spkts
    neurosim.save()

def _saved_timesteps(synWeights_file):
  df = readWeights(synWeights_file)
  return sorted(list(df['time'].unique()))

def continue_main(wdir, iterations=100, index=None):
  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')

  outdir = os.path.join(wdir, 'continue_{}'.format(1 if index == None else index))
  dconf = read_conf(dconf_path, outdir=outdir)
  synWeights_file = os.path.join(wdir, 'synWeights.pkl')

  timesteps = _saved_timesteps(synWeights_file)
  dconf['simtype']['ResumeSim'] = 1
  dconf['simtype']['ResumeSimFromFile'] = synWeights_file
  dconf['simtype']['ResumeSimFromTs'] = float(timesteps[-1])
  if iterations != None:
    dconf['ES']['iterations'] = iterations
  dconf['sim']['plotRaster'] = 0

  train(dconf)


if __name__ == "__main__":
  fire.Fire({
      'train': train,
      'continue': continue_main
  })
