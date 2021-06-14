import pytest
import os
import sys

import gym
from neurosim.game_interface import GameInterface, _parse_rf_map

class MockAIGame:
  observations = []
  rewards = []
  env = None

  def __init__(self, env):
    self.env = env

# firing rates config:
mock_config_fr = {
    "env": {
        "name": "CartPole-v1",
        "observation_map": [
            {"type": "linear", "min": -0.7, "max": 0.7},
            {"type": "sigmoid", "scale": 2, "min": -2, "max": 2},
            {"type": "linear", "min": -0.2, "max": 0.2},
            {"type": "sigmoid", "scale": 2, "min": -2.5, "max": 2.5}
        ],
        "episodes": 10
    },
    "net": {
      "InputMaxRate": 150.0,
      "inputPop": 'ES',
      "allpops": {
        "ES": 4
      }
    }
}

# receptive field config:
mock_config_rf = {
    "env": {
        "name": "CartPole-v1",
        "observation_map": [
            {"type": "rf_evensplit", "bins": 5, "min": -1.0, "max": 4.0},
            {"type": "rf_normal", "bins": 5, "mean": 0.0, "std": 1},
            {"type": "rf_normal", "bins": 5, "mean": 1.0, "std": 1},
            {"type": "rf_intervals", "bins": 3, "intervals": [-5, 1, 2, 5]}
        ],
        "episodes": 10
    },
    "net": {
      "InputMaxRate": 10.0,
      "inputPop": 'ES',
      "allpops": {
        "ES": 18
      }
    }
}

obs_count = 4
EPS = 1e-6

def samelist(l1, l2):
  print(l1,l2)
  assert len(l1) == len(l2)
  for i, (a,b) in enumerate(zip(l1, l2)):
    assert abs(a - b) < EPS, 'different at {}: {} != {}'.format(i, a, b)

class TestGameInterface:
    def test_input_firing_rates(self):
      env = gym.make(mock_config_fr['env']['name'])
      game = MockAIGame(env)
      gi = GameInterface(game, mock_config_fr)

      pop_count = mock_config_fr['net']['allpops']['ES']
      test_cases = [
        [[0.0] * obs_count, [75.0] * pop_count],
        [[0.19] * obs_count, [95.35714286, 84.32733355, 146.25, 83.37413618] * int(pop_count / obs_count)],
        [[-0.19] * obs_count, [54.64285714, 65.67266645, 3.75, 66.62586382] * int(pop_count / obs_count)],
      ]

      for obs, expected in test_cases:
        game.observations.append(obs)
        firing_rates = gi.input_firing_rates()
        samelist(firing_rates, expected)

    def test_receptive_field(self):
      env = gym.make(mock_config_rf['env']['name'])
      game = MockAIGame(env)
      gi = GameInterface(game, mock_config_rf)
  
      obs = [0.0] * obs_count
      game.observations.append(obs)
      firing_rates = gi.input_firing_rates()
      expected = [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0]]
      samelist(firing_rates, [l * 10 for ll in expected for l in ll])
      

    def test_rf_fields(self):
      rf1_map = {"type": "rf_evensplit", "bins": 5, "min": -1.0, "max": 1.0}
      f1 = _parse_rf_map(rf1_map)
      # np.linspace(-1, 1, 5+1): array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ])
      samelist(f1(-2), [0.0] * 5)
      samelist(f1(-0.75), [1, 0, 0, 0, 0])
      samelist(f1(-0.25), [0, 1, 0, 0, 0])
      samelist(f1(0.0), [0, 0, 1, 0, 0])
      samelist(f1(0.25), [0, 0, 0, 1, 0])
      samelist(f1(0.75), [0, 0, 0, 0, 1])
      samelist(f1(0.599), [0, 0, 0, 1, 0])
      samelist(f1(0.600), [0, 0, 0, 0, 1])
      samelist(f1(0.601), [0, 0, 0, 0, 1])


      rf2_map = {"type": "rf_normal", "bins": 5, "mean": 0.0, "std": 1.0}
      f2 = _parse_rf_map(rf2_map)
      # norm.ppf(np.linspace(EPS, 1.0-EPS, 5 + 1)): [-6.3613, -0.8416, -0.2533, 0.2533, 0.8416, 6.3613]
      samelist(f2(-1.0), [1, 0, 0, 0, 0])
      samelist(f2(-0.5), [0, 1, 0, 0, 0])
      samelist(f2(0.0), [0, 0, 1, 0, 0])
      samelist(f2(0.5), [0, 0, 0, 1, 0])
      samelist(f2(1.0), [0, 0, 0, 0, 1])
      samelist(f2(0.84), [0, 0, 0, 1, 0])
      samelist(f2(0.85), [0, 0, 0, 0, 1])


      rf3_map = {"type": "rf_intervals", "bins": 3, "intervals": [-5, -3, 1, 5]}
      f3 = _parse_rf_map(rf3_map)
      # norm.ppf(np.linspace(EPS, 1.0-EPS, 5 + 1)): [-6.3613, -0.8416, -0.2533, 0.2533, 0.8416, 6.3613]
      samelist(f3(-4.0), [1, 0, 0])
      samelist(f3(-2.0), [0, 1, 0])
      samelist(f3(0.0), [0, 1, 0])
      samelist(f3(0.99), [0, 1, 0])
      samelist(f3(1.0 + EPS), [0, 0, 1])
      samelist(f3(2.0), [0, 0, 1])


      
      rf4_map = {"type": "rf_normal", "bins": 4, "mean": 10.0, "std": 2.0}
      f4 = _parse_rf_map(rf4_map)
      samelist(f4(8.0), [1, 0, 0, 0])
      samelist(f4(10.0 - EPS), [0, 1, 0, 0])
      samelist(f4(10.0 + EPS), [0, 0, 1, 0])
      samelist(f4(12.0), [0, 0, 0, 1])
