import os
import datetime


def syncdata_alltoall(sim, data):
  # this py_alltoall seems to work, but is apparently not as fast as py_broadcast (which has problems - wrong-sized array sometimes!)
  root = 0
  nhost = sim.pc.nhost()
  src = [data]*nhost if sim.rank == root else [None]*nhost
  return sim.pc.py_alltoall(src)[0]


def now_str(diplay_time=False):
  now = datetime.datetime.now()
  return now.strftime("%Y%m%d_%H%M%S" if diplay_time else "%Y%m%d")
