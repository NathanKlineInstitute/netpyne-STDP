import os
import fire

from conf import read_conf, backup_config
from sim import NeuroSim
from utils.weights import readWeights


def main(dconf=None):
  if not dconf:
    dconf = read_conf()

  runner = NeuroSim(dconf)
  runner.run()

def _saved_timesteps(synWeights_file):
  df = readWeights(synWeights_file)
  return sorted(list(df['time'].unique()))

def evaluate(eval_dir, duration=200, resume_tidx=-1):
  dconf_path = os.path.join(eval_dir, 'backupcfg_sim.json')

  synWeights_file = os.path.join(eval_dir, 'synWeights.pkl')
  timesteps = _saved_timesteps(synWeights_file)
  if resume_tidx < 0:
    resume_tidx += len(timesteps)
  assert resume_tidx >= 0 and resume_tidx < len(timesteps)

  dconf = read_conf(dconf_path, outdir=os.path.join(eval_dir, 'evaluation_{}'.format(resume_tidx)))

  dconf['simtype']['ResumeSim'] = 1
  dconf['simtype']['ResumeSimFromFile'] = synWeights_file
  dconf['simtype']['ResumeSimFromTs'] = timesteps[resume_tidx]
  dconf['net']['STDPconns'] = {}
  dconf['verbose'] = 0
  dconf['sim']['duration'] = duration
  dconf['sim']['saveWeights'] = 0
  dconf['sim']['doSaveData'] = 0
  dconf['sim']['plotRaster'] = 0
  dconf['sim']['verbose'] = 0
  backup_config(dconf_path, dconf)

  runner = NeuroSim(dconf)
  runner.run()


if __name__ == '__main__':
  fire.Fire({
      'run': main,
      'eval': evaluate
  })
