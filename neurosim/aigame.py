from image import ROSImage
import numpy as np
from datetime import datetime
from collections import deque
import random

class AIGame:
  """ Interface to OpenAI gym game
  """

  def __init__(self, config):
    self.conf_env = config['env']
    self.actionsPerPlay = config['actionsPerPlay']
    self.observations = deque(maxlen=config['observationsToKeep'])
    self.count_episodes = 0
    self.count_steps = [0]
    self.tstart = None

  def _clean(self):
    self.observations.clear()
    self.count_steps.append(0)
    self.tstart = None
    #observation = self.env.reset()
    #self.observations.append(observation)

  def randmove(self):
    return random.randint(0, 2)

  def playGame(self, actions):
    done = False
    if not self.tstart:
      self.tstart = datetime.now()

    assert len(actions) == self.actionsPerPlay # in ROS, should
    for adx in range(self.actionsPerPlay):
      # for each action generated
      caction = actions[adx]

      observation= ROSImage.getImage()
      self.observations.append(observation)
      done = ROSImage.done
      self.count_steps[-1] += 1

      if done:
        # delta = (datetime.now() - self.tstart) / self.count_steps[-1]
        # print('Python time per step: {} ms'.format(delta.microseconds / 1000))
        self._clean()
        self.count_episodes += 1
        break

    return done
