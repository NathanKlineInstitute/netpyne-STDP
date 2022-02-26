import argparse
import numpy as np
import netpyne
import csv
# from neurosim.sim import NeuroSim


def run_simulation(weights, config, alpha, beta, gamma, out_path):

    # ### ---Generate model--- ###
    # model = NeuroSim(config, use_noise=False, save_on_control_c=False)
    # model.setWeightArray(netpyne.sim, weights)
    #
    # ### --Run-- ###
    #
    # # Alpha #
    # model.STDP_active = False
    # model.end_after_episode = alpha
    # model.run()
    # alpha_perf = np.mean(model.epCount[-alpha:])
    #
    # # Beta #
    # model.STDP_active = True
    # model.end_after_episode = beta
    # model.run()
    # beta_perf = np.mean(model.epCount[-beta:])
    #
    # # Gamma #
    # model.STDP_active = False
    # model.end_after_episode = gamma
    # model.run()
    # gamma_perf = np.mean(model.epCount[-gamma:])

    # TODO: Uncomment for above, the below is just for testing
    alpha_perf, beta_perf, gamma_perf = 0, 1, 2

    ## --Write Performance-- ##
    with open(out_path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow((alpha_perf, beta_perf, gamma_perf))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str)    # TODO: How to do??
    parser.add_argument("--config", type=str)
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--beta", type=int)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--out_path", type=str)

    args = parser.parse_args()
    run_simulation(**vars(args))