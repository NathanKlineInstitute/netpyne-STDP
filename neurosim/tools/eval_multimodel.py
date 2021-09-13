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


def trace(wdir):
  configs = []
  wdirs = [wdir]
  while True:
    config_fname = os.path.join(wdirs[-1], 'backupcfg_sim.json')
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

  all_params = []
  keys = []
  print('wdirs:')
  for wdir, config in zip(wdirs, configs):
    print(wdir)
    params = _get_params(config)
    all_params.append(params)
    keys.extend(params.keys())

  print('params:')
  keys = sorted(list(set(keys)))
  for key in keys:
    values = [params[key] if key in params else None for params in all_params]
    if len(set(values)) > 1:
      print(key, values)




if __name__ == '__main__':
  fire.Fire({
      'trace': trace,
  })
