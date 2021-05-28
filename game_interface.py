
class GameInterface:

  def __init__(self, aigame, config):
    self.AIGame = aigame
    self.inputMaxRate = config['InputMaxRate']
    self.inputPop = config['net']['InputPop']
    self.inputPopSize = dconf['net']['allpops'][self.inputPop]
    # "FiringRateCutoff": 50.0,

  def input_firing_rates(self):
    # TODO: map from observations to firing rates
    self.AIGame.observations
    return np.reshape(fr_Images, self.inputPopSize)
