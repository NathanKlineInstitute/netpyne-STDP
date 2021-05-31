# neuronal network connection functions
import numpy as np


def prob2conv(prob, npre):
  # probability to convergence; prob is connection probability
  # npre is number of presynaptic neurons
  return int(0.5 + prob * npre)


def getconv(cmat, prety, poty, npre):
  # get convergence value from cmat dictionary
  # (uses convergence if specified directly, otherwise uses p to calculate)
  if 'conv' in cmat[prety][poty]:
    return cmat[prety][poty]['conv']
  elif 'p' in cmat[prety][poty]:
    return prob2conv(cmat[prety][poty]['p'], npre)
  return 0
