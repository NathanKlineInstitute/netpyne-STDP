import os
import fire
import csv
import numpy as np

from conf import read_conf, init_wdir, backup_config
from sim import NeuroSim
from utils.weights import readWeights


def main(dconf=None):
  if not dconf:
    dconf = read_conf()

  outdir = dconf['sim']['outdir']
  if os.path.isdir(outdir):
    evaluations = [fname
      for fname in os.listdir(outdir)
      if fname.startswith('evaluation_') and os.path.isdir(os.path.join(outdir, fname))]
    if len(evaluations) > 0:
      raise Exception(' '.join([
          'You have run evaluations on {}: {}.'.format(outdir, evaluations),
          'This will rewrite!',
          'Please delete to continue!']))

  init_wdir(dconf)

  runner = NeuroSim(dconf)
  runner.run()

def continue_main(wdir, duration=None, index=None,
    copy_from_config=None, copy_fields=None, added_params=None):
  dconf_path = os.path.join(wdir, 'backupcfg_sim.json')

  if type(added_params) == str:
    added_params = added_params.split(',')

  synWeights_file = os.path.join(wdir, 'synWeights.pkl')
  timesteps = _saved_timesteps(synWeights_file)
  outdir = os.path.join(wdir, 'continue_{}'.format(1 if index == None else index))
  dconf = read_conf(dconf_path, outdir=outdir)

  if copy_from_config and copy_fields:
    dconf_sep = read_conf(copy_from_config, outdir=outdir)
    if type(copy_fields) == str:
      copy_fields = copy_fields.split(',')
    for field in copy_fields:
      dconf[field] = dconf_sep[field]

  init_wdir(dconf)

  dconf['simtype']['ResumeSim'] = 1
  dconf['simtype']['ResumeSimFromFile'] = synWeights_file
  dconf['simtype']['ResumeSimFromTs'] = float(timesteps[-1])
  if duration != None:
    dconf['sim']['duration'] = duration
  dconf['sim']['plotRaster'] = 0
  if added_params:
    for param in added_params:
      name, val = param.split(':')
      if name == 'recordWeightStepSize':
        dconf['sim']['recordWeightStepSize'] = int(val)
      else:
        raise Exception('Cannot find param {}'.format(name))

  backup_config(dconf)

  runner = NeuroSim(dconf)
  runner.run()


def _saved_timesteps(synWeights_file):
  df = readWeights(synWeights_file)
  return sorted(list(df['time'].unique()))

def print_timesteps(synWeights_file):
  timesteps = _saved_timesteps(synWeights_file)
  print(timesteps)

def _find_best_timestep(wdir, timesteps, step=100):
  actions_per_episode = []
  with open(os.path.join(wdir, 'ActionsPerEpisode.txt')) as f:
    for row in csv.reader(f, delimiter='\t'):
      actions_per_episode.append(int(float(row[1])))

  training_medians = []
  max_median = 0.0
  best_idx = 0
  for idx in range(len(actions_per_episode) - step):
    curr_med = np.median(actions_per_episode[idx:idx+step])
    if max_median < curr_med:
      max_median = curr_med
      best_idx = idx+step
  total_steps = np.sum(actions_per_episode[:best_idx+1])
  return [i for i, t in enumerate(timesteps) if t <= total_steps * 50.0][-1]

def evaluate(eval_dir, duration=None, eps_duration=None, resume_tidx=-1,
    display=False, verbose=False, sleep=False, save_data=False,
    env_seed=None, rerun_episode=None, mock_env=False, resume_best_training=False):
  if (duration == None) and (eps_duration == None):
    duration = 100
  assert (duration == None) ^ (eps_duration == None)

  dconf_path = os.path.join(eval_dir, 'backupcfg_sim.json')

  synWeights_file = os.path.join(eval_dir, 'synWeights.pkl')
  timesteps = _saved_timesteps(synWeights_file)
  if resume_tidx < 0:
    resume_tidx += len(timesteps)
  if resume_best_training:
    resume_tidx = _find_best_timestep(eval_dir, timesteps)
  assert resume_tidx >= 0 and resume_tidx < len(timesteps)

  outdir = os.path.join(eval_dir, 'evaluation{}_{}'.format(
    '_display' if display else '', resume_tidx))
  if rerun_episode != None:
    if not rerun_episode:
      raise Exception('rerun_episode is 1-indexed!')
    outdir = os.path.join(eval_dir, 'eval_{}_rerunEp{}'.format(
      resume_tidx, rerun_episode))
  if mock_env:
    outdir = os.path.join(eval_dir, 'evalmock_{}'.format(resume_tidx))
  dconf = read_conf(dconf_path, outdir=outdir)
  init_wdir(dconf)

  if display:
    dconf['env']['render'] = 1
  if env_seed != None:
    dconf['env']['seed'] = env_seed
  if rerun_episode != None:
    dconf['env']['rerunEpisode'] = rerun_episode
  if mock_env:
    dconf['env']['mock'] = 1
  dconf['simtype']['ResumeSim'] = 1
  dconf['simtype']['ResumeSimFromFile'] = synWeights_file
  dconf['simtype']['ResumeSimFromTs'] = float(timesteps[resume_tidx])
  dconf['verbose'] = 1 if verbose else 0
  dconf['sim']['duration'] = duration if duration != None else eps_duration * 26
  dconf['sim']['saveWeights'] = 0
  dconf['sim']['doSaveData'] = 1 if save_data or mock_env else 0
  dconf['sim']['plotRaster'] = 1 if mock_env else 0
  dconf['sim']['verbose'] = 1 if verbose else 0
  dconf['sim']['sleeptrial'] = sleep if sleep else 0
  dconf['sim']['normalizeByGainControl'] = 0
  dconf['sim']['normalizeByOutputBalancing'] = 0
  dconf['sim']['normalizeSynInputs'] = 0

  for stdp_param in ['STDP', 'STDP-RL']:
    if stdp_param in dconf:
      for k, stdp_map in dconf[stdp_param].items():
        dconf[stdp_param][k]['RLhebbwt'] = 0
        dconf[stdp_param][k]['RLantiwt'] = 0
        dconf[stdp_param][k]['hebbwt'] = 0
        dconf[stdp_param][k]['antiwt'] = 0
        print('Cleaned params for evaluation for {}: {}'.format(stdp_param, k))

  backup_config(dconf)

  runner = NeuroSim(dconf)
  if eps_duration != None:
    runner.end_after_episode = eps_duration
  runner.STDP_active = False

  try:
    runner.run()
  except SystemExit:
    runner.save()


if __name__ == '__main__':
  fire.Fire({
      'run': main,
      'continue': continue_main,
      'eval': evaluate,
      'timesteps': print_timesteps
  })
