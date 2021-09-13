import fire
import os
import json


def _get_params(config):
  params = {}
  def _extract(j, pname):
    if type(j) == dict:
      for k,v in j.items():
        _extract(v, pname + ':' + k)
    elif type(j) == list:
      for idx,v in enumerate(j):
        _extract(v, pname + ':' + str(idx))
    else:
      params[pname] = j
  _extract(config, '')
  return params


def _extract(wdir, path_prefix=None):
  configs = []
  wdirs = [wdir]
  while True:
    current_wdir = wdirs[-1]
    if path_prefix:
      current_wdir = os.path.join(path_prefix, current_wdir)
    config_fname = os.path.join(current_wdir, 'backupcfg_sim.json')
    with open(config_fname) as f:
      config = json.load(f)
    configs.append(config)
    if config['simtype']['ResumeSim']:
      wdirs.append(
        os.path.dirname(config['simtype']['ResumeSimFromFile']))
    else:
      break

  configs.reverse()
  wdirs.reverse()
  return wdirs, configs

def _extract_params(wdirs, configs):
  all_params = []
  keys = []
  for wdir, config in zip(wdirs, configs):
    params = _get_params(config)
    all_params.append(params)
    keys.extend(params.keys())

  keys = sorted(list(set(keys)))
  return all_params, keys

def trace(wdir):
  wdirs, configs = _extract(wdir)

  print('wdirs:')
  for wdir in wdirs:
    print(wdir)

  all_params, keys = _extract_params(wdirs, configs)

  print('params:')
  for key in keys:
    values = [params[key] if key in params else None for params in all_params]
    if len(set(values)) > 1:
      print(key, values)


if __name__ == '__main__':
  fire.Fire({
      'trace': trace,
  })
