import os
import fire
import sys
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

def _saved_timesteps(synWeights_file):
  df = readWeights(synWeights_file)
  return sorted(list(df['time'].unique()))

def evaluate(eval_dir, duration=100, resume_tidx=-1, display=False):
  dconf_path = os.path.join(eval_dir, 'backupcfg_sim.json')

  synWeights_file = os.path.join(eval_dir, 'synWeights.pkl')
  timesteps = _saved_timesteps(synWeights_file)
  if resume_tidx < 0:
    resume_tidx += len(timesteps)
  assert resume_tidx >= 0 and resume_tidx < len(timesteps)

  outdir = os.path.join(eval_dir, 'evaluation_{}'.format(resume_tidx))
  dconf = read_conf(dconf_path, outdir=outdir)
  init_wdir(dconf)

  if display:
    dconf['env']['render'] = 1
  dconf['simtype']['ResumeSim'] = 1
  dconf['simtype']['ResumeSimFromFile'] = synWeights_file
  dconf['simtype']['ResumeSimFromTs'] = float(timesteps[resume_tidx])
  dconf['verbose'] = 0
  dconf['sim']['duration'] = duration
  dconf['sim']['saveWeights'] = 0
  dconf['sim']['doSaveData'] = 0
  dconf['sim']['plotRaster'] = 0
  dconf['sim']['verbose'] = 0

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
  runner.run()


if __name__ == '__main__':
  if sys.argv[-1] == 'run':
    main()
  elif sys.argv[-1] == 'eval':
    evaluate()
  #fire.Fire({
  #    'run': main,
  #    'eval': evaluate
  #})
