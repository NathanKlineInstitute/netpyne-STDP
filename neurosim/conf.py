import sys
import os
import json

from utils import now_str


def _get_conf_file():
  fnjson = 'config.json'
  for i in range(len(sys.argv)):
    if sys.argv[i].endswith('.json'):
      fnjson = sys.argv[i]
      print('reading', fnjson)
  return fnjson


def _init_conf(fnjson, conf):
  if 'outdir' not in conf['sim'] or not conf['sim']['outdir']:
    conf['sim']['outdir'] = os.path.join('results', now_str())
  os.makedirs(conf['sim']['outdir'], exist_ok=True)
  # Copy the config as a backup
  fout = os.path.join(conf['sim']['outdir'], 'backupcfg_sim.json')
  os.system('cp ' + fnjson + '  ' + fout)


def read_conf(fnjson=None):
  if not fnjson:
    fnjson = _get_conf_file()
  with open(fnjson, 'r') as fp:
    dconf = json.load(fp)
  _init_conf(fnjson, dconf)
  return dconf
