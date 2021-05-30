import os


def syncdata_alltoall(sim, data):
  # this py_alltoall seems to work, but is apparently not as fast as py_broadcast (which has problems - wrong-sized array sometimes!)
  root = 0
  nhost = sim.pc.nhost()
  src = [data]*nhost if sim.rank == root else [None]*nhost
  return sim.pc.py_alltoall(src)[0]


def backupcfg(name):
  # backup the config file to backupcfg subdirectory
  os.makedirs('backupcfg', exist_ok=True)
  from conf import fnjson
  fout = 'backupcfg/' + name + 'sim.json'
  if os.path.exists(fout):
    print('removing prior cfg file', fout)
    os.system('rm ' + fout)
  # fcfg created in geom.py via conf.py
  os.system('cp ' + fnjson + '  ' + fout)
