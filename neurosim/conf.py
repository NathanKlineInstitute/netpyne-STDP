import sys
import os
import json
import datetime


def now_str(diplay_time=False):
  now = datetime.datetime.now()
  return now.strftime("%Y%m%d_%H%M%S" if diplay_time else "%Y%m%d")


def _get_conf_file():
  fnjson = 'config.json'
  for i in range(len(sys.argv)):
    if sys.argv[i].endswith('.json'):
      fnjson = sys.argv[i]
      print('reading', fnjson)
  return fnjson

def backup_config(conf):
  # Copy the config as a backup
  fout = os.path.join(conf['sim']['outdir'], 'backupcfg_sim.json')
  with open(fout, 'w') as out:
    out.write(json.dumps(conf, indent=4))

def read_conf(fnjson=None, outdir=None):
  if not fnjson:
    fnjson = _get_conf_file()
  with open(fnjson, 'r') as fp:
    dconf = json.load(fp)
  if outdir: 
    dconf['sim']['outdir'] = outdir
  if 'outdir' not in dconf['sim'] or not dconf['sim']['outdir']:
    dconf['sim']['outdir'] = os.path.join('results', now_str())
  return dconf

def init_wdir(dconf):
  wdir = dconf['sim']['outdir']
  os.makedirs(wdir, exist_ok=True)
  for f in os.listdir(wdir):
      os.remove(os.path.join(wdir, f))
  backup_config(dconf)
