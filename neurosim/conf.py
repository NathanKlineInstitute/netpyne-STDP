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

def backup_config(fnjson, conf):
  # Copy the config as a backup
  fout = os.path.join(conf['sim']['outdir'], 'backupcfg_sim.json')
  with open(fout, 'w') as out:
    out.write(json.dumps(conf, indent=4))

def _init_conf(fnjson, conf, outdir=None):
  if outdir: 
    conf['sim']['outdir'] = outdir
  if 'outdir' not in conf['sim'] or not conf['sim']['outdir']:
    conf['sim']['outdir'] = os.path.join('results', now_str())
  os.makedirs(conf['sim']['outdir'], exist_ok=True)
  backup_config(fnjson, conf)

def read_conf(fnjson=None, outdir=None):
  if not fnjson:
    fnjson = _get_conf_file()
  with open(fnjson, 'r') as fp:
    dconf = json.load(fp)
  _init_conf(fnjson, dconf, outdir)
  return dconf

