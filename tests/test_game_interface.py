import pytest
import os
import sys

import gym
from game_interface import GameInterface

class MockAIGame:
  observations = []
  rewards = []
  env = None

  def __init__(self, env):
    self.env = env

mock_config = {
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

obs_count = 4
pop_count = mock_config['net']['allpops']['ES']
EPS = 1e-6
test_cases = [
  [[0.0] * obs_count, [75.0] * pop_count],
  [[0.19] * obs_count, [77.96874988, 82.10364272, 109.01936814, 82.10364272] * int(pop_count / obs_count)],
  [[-0.19] * obs_count, [72.03125012, 67.89635728, 40.98063186, 67.89635728] * int(pop_count / obs_count)],
]

class TestGameInterface:
    def test_input_firing_rates(self):
      env = gym.make(mock_config['env']['name'])
      game = MockAIGame(env)
      gi = GameInterface(game, mock_config)

      for obs, expected in test_cases:
        game.observations.append(obs)
        firing_rates = gi.input_firing_rates()
        print(firing_rates)
        assert len(firing_rates) == len(expected)
        for i, (a, b) in enumerate(zip(firing_rates, expected)):
          assert abs(a - b) < EPS, 'different at {}: {} != {}'.format(i, a, b)
