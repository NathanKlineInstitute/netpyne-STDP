from datetime import datetime

def pseudo_random(digits=7, iters=2):
  # Used the Middle square method (MSM) for generating pseudorandom numbers
  # use own random function to not mess up with the seed
  current_time = str(datetime.now().timestamp()).replace('.', '')
  seed = int(current_time[-digits:])
  def _iter(nr):
    new_nr = str(nr * nr)
    imin = max(int((len(new_nr) - 7) / 2), 1)
    imax = min(len(new_nr)-1, imin + digits)
    return int(('0' * digits) + new_nr[imin:imax])
  nr = seed
  for _ in range(iters):
    nr = _iter(nr)
  return nr