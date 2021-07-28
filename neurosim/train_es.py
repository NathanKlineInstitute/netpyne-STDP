from sim import NeuroSim
from conf import read_conf
from aigame import AIGame
from game_interface import GameInterface
import os, sys
import netpyne

import multiprocessing
from multiprocessing import Pool, Manager

import numpy as np
from tqdm import tqdm

# Wrapper for netpyne simulation that catched the sys.exit() after one episode (if activated)
def run_episodes(neurosim):
    try:
        # Suppress print statements from the netpyne sim
        sys.stdout = open(os.devnull, 'w')
        neurosim.run()
    except SystemExit:
        pass
    # Turn printing back on after netpyne sim is done
    sys.stdout = sys.__stdout__
    return

def train(dconf):
    ITERATIONS = 10 # How many iterations to train for
    POPULATION_SIZE = 3 # How many perturbations of weights to try per iteration
    SIGMA = 0.1 # standard deviation of perturbations applied to each member of population
    LEARNING_RATE = 1 # what percentage of the return normalized perturbations to add to best_weights

    # How much to decay the learning rate and sigma by each episode. In theory
    # this should lead to better
    LR_DECAY = 1
    SIGMA_DECAY = 1

    # # Evaluation frequency and sample size, does not effect training, just gives
    # # a better metric from which to track progress (mean fitness of the current best weights)
    # EVALUATE_EVERY = 10
    # EVALUATION_EPISODES = 10

    # Initialize the model with dconf config
    if not dconf:
        dconf = read_conf()
    dconf['sim']['duration'] = 501

    neurosim = NeuroSim(dconf, use_noise = False)
    neurosim.STDP_active = False # deactivate STDP
    neurosim.end_after_episode = 2 # activate sys.exit() after one episode

    # randomly initialize best weights to the first weights generated
    best_weights = neurosim.getWeightArray(netpyne.sim)

    fres_train = neurosim.outpath('es_train.txt')
    fres_eval = neurosim.outpath('es_eval.txt')

    for iteration in range(ITERATIONS):
        print("\n--------------------- ES iteration", iteration, "---------------------")

        # generate unique randomly sampled perturbations to add to each member of
        # this iteration's the training population
        perturbations = np.random.normal(0, SIGMA, (POPULATION_SIZE, best_weights.size))

        # this should rarely be used with reasonable sigma but just in case we
        # clip perturbations that are too low since they can make the weights negative
        perturbations[perturbations < -0.8] = -0.8

        print("\nSimulating episodes ... ")

        # get the fitness of each set of perturbations when applied to the current best weights
        # by simulating each network and getting the episode length as the fitness
        fitness = []
        for i in range(POPULATION_SIZE):
            neurosim.setWeightArray(netpyne.sim, best_weights * (1 + perturbations[i]))
            run_episodes(neurosim)
            fitness.append(
              np.median(neurosim.epCount[-neurosim.end_after_episode:]))
        fitness = np.expand_dims(np.array(fitness), 1)

        fitness_res = [np.median(fitness), fitness.mean(), fitness.min(), fitness.max(),
               best_weights.mean()]
        with open(fres_train, 'a') as out:
          out.write('\t'.join([str(r) for r in fitness_res]) + '\n')
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

        # # Evaluation - the training progress printouts are for a bunch of noisy versions
        # # of our current best weights. To truly track the progress of our algorithm we need
        # # to actually run our current best weights with a large enough sample size to
        # # decrease variance. We only do it once every several episodes since it is expensive
        # if (iteration + 1) % EVALUATE_EVERY == 0:
        #     print("\nEvaluating best weights after iteration", iteration)
        #     neurosim.setWeightArray(netpyne.sim, best_weights)
        #     print("\nSimulating evaluation iteration ... ")
        #     fitness = []
        #     for i in range(EVALUATION_EPISODES):
        #         run_episodes(neurosim)
        #         fitness.append(
        #           np.median(neurosim.epCount[-neurosim.end_after_episode:]))
        #     fitness = np.array(fitness)
        #     fitness_
        #     print(
        #         "Fitness Median: {}; Mean: {} ([{}, {}])".format(
        #           np.median(fitness),
        #           fitness.mean(),
        #           fitness.min(),
        #           fitness.max())
        #     )

    # print raster
    neurosim.end()

if __name__ == "__main__":
    train(None)
