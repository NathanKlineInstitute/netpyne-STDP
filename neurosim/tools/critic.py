import os
import json
import csv
import gym
import fire

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from neurosim.game_interface import GameInterface
from neurosim.critic import Critic
from neurosim.conf import now_str

class MockAIGame:
    observations = []
    rewards = []
    env = None
    def __init__(self, env):
        self.env = env
    

def _extract(wdir, critic, verbose=False):
  rew_file = os.path.join(wdir, 'ActionsRewards.txt')
  results_file = os.path.join(wdir, 'ActionsPerEpisode.txt')
  motor_file = os.path.join(wdir, 'MotorOutputs.txt')
  config_file = os.path.join(wdir, 'backupcfg_sim.json')

  with open(config_file) as f:
      config = json.load(f)

  env = gym.make('CartPole-v1')
  game = MockAIGame(env)
  gi = GameInterface(game, config)

  motor = {}
  with open(motor_file) as f:
      for row in csv.reader(f, delimiter='\t'):
          motor[row[0]] = [int(float(i)) for i in row[1:]]

  steps = []

  with open(rew_file) as f:
    prev_obs = None
    for row in csv.reader(f, delimiter='\t'):
        t, move, critic_val, obs_space_j = row
        obs_space = json.loads(obs_space_j)
        game.observations = [obs_space]
        fr = gi.input_firing_rates()
        frind = (fr > 0).nonzero()
        motor_v = motor[t] if len(steps) > 0 else [0,0]
        is_unk_move = motor_v[0] == motor_v[1]
        steps.append({
            'move': move,
            'unk_move': is_unk_move,
            'critic_run': float(critic_val),
            'critic': critic.calc_reward(np.array(obs_space), np.array(prev_obs), is_unk_move),
            'motor': motor_v,
            'fr': list(frind[0])
        })
        prev_obs = obs_space

  if verbose:
    print('total steps: ', len(steps))
          
  results = []
  with open(results_file) as f:
      for row in csv.reader(f, delimiter='\t'):
          results.append(int(float(row[1])))
          
  if verbose:
    print('results:', len(results))
    print('steps from results:', sum(results))

  # separate by iteration
  steps_seq = []

  idx = 0
  for r in results:
      steps_seq.append(steps[idx:idx+r])
      idx += r
      
  if verbose:
    print([len(s) for s in steps_seq])

  return steps_seq


def _evaluate_critic(best_wdir, critic, plot, verbose=False):
  steps_seq = _extract(best_wdir, critic, verbose)

  critic_vals = []
  for steps in steps_seq:
      if len(steps) == 500:
          vals = [j['critic'] for j in steps[1:-1] if not j['unk_move']]
          critic_vals.append(vals)
  if verbose:
    print('steps for each 500eps run that are not unk:', [len(cv) for cv in critic_vals])

  if plot:
    plt.figure(figsize=(10,10))
    plt.hist([v for cv in critic_vals for v in cv], bins=30)
    plt.title('critic on ES best model on {} episodes of 500steps'.format(len(critic_vals)))
    plt.xlabel('critic value')
    plt.ylabel('steps count')
    plt.show()

  avg_res = np.mean([v for cv in critic_vals for v in cv])
  if verbose:
    print('Magnitude of average critic result', round(avg_res, 3))
    print('Magnitude of average critic result per episode', [round(np.mean(cv), 3) for cv in critic_vals])

    print('--')

  ratio_res = np.mean([1 if v > 0 else 0 for cv in critic_vals for v in cv if v != 0])
  if verbose:
    print('Ratio of reward/punishment', round(ratio_res, 3))
    print('Ratio of reward/punishment per episode', [
      round(np.mean([1 if v > 0 else 0 for v in cv if v != 0]), 3)
      for cv in critic_vals])

  return avg_res, ratio_res


def evaluate_critic(best_wdir, critic_config, plot=True, verbose=False):
  with open(critic_config) as f:
    config = json.load(f)
  critic = Critic(config)
  _evaluate_critic(best_wdir, critic, plot, verbose)

def grid_search(
    best_wdir,
    critic_config,
    hp_angv_bias=[0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 3.0, 5.0],
    hp_posRewardBias=[0.5, 1.0, 2.0, 3.0],
    hp_total_gain=[2.0],
    hp_max_reward=[1.0, 2.0, 5.0, 10.0],
    compare='ratio',
    outputfile=None):

  assert compare in ['ratio', 'magnitude']
  cmp_idx = -1 if compare == 'ratio' else -2

  with open(critic_config) as f:
    config = json.load(f)

  results = []
  iterations = len(hp_angv_bias) * len(hp_posRewardBias) * len(hp_total_gain) * len(hp_max_reward)
  best = None
  with tqdm(total=iterations) as t:
    for angv_bias in hp_angv_bias:
      for posRewardBias in hp_posRewardBias:
        for total_gain in hp_total_gain:
          for max_reward in hp_max_reward:
            config['critic'] = {
              'angv_bias': angv_bias,
              'posRewardBias': posRewardBias,
              'total_gain': total_gain,
              'max_reward': max_reward,
            }
            critic = Critic(config)
            magn, ratio = _evaluate_critic(best_wdir, critic, plot=False, verbose=False)
            results.append([
              angv_bias, posRewardBias, total_gain, max_reward, magn, ratio
            ])
            if not best or best[cmp_idx] < results[-1][cmp_idx]:
              best = results[-1]
              print(best)
            t.update()

  if not outputfile:
    outputfile = 'results/critic-hpsearch/{}.tsv'.format(now_str(diplay_time=True))

  with open(outputfile, 'w') as out:
    csvwriter = csv.writer(out, delimiter='\t')
    csvwriter.writerow(['angv_bias', 'posRewardBias', 'total_gain', 'max_reward', 'magniture', 'ratio'])
    for r in results:
      csvwriter.writerow(r)


if __name__ == '__main__':
  fire.Fire({
    'eval': evaluate_critic,
    'hpsearch': grid_search
  })
