import os
import fire
import csv
import json
import math
import matplotlib.pyplot as plt

def analyze_samples(wdir, runs_json_fname='runs.json', results_file='results.tsv'):
  runs_json = os.path.join(wdir, runs_json_fname)
  hpconfig_json = os.path.join(wdir, 'hpsearch_config.json')
  results_tsv = os.path.join(wdir, results_file)

  runs = {}
  with open(runs_json) as f:
    for line in f:
      run = json.loads(line)
      runs[run['run_id']] = run

  results = []
  with open(results_tsv) as f:
    for row in csv.DictReader(f, delimiter='\t'):
      for k in row.keys():
        if k.startswith('max_') or k.startswith('freq_'):
          row[k] = float(row[k])
        elif k == 'run_id':
          row[k] = int(row[k])
      row['params'] = runs[row['run_id']]
      results.append(row)
  del runs
  print('Found {} results'.format(len(results)))

  result_keys = [rkey for rkey in results[0].keys() if rkey.startswith('max_') or rkey.startswith('freq_')]

  with open(hpconfig_json) as f:
    hpconfig = json.load(f)

  param_keys = list(hpconfig['params'].keys())
  ncols = 4
  figsize = 25

  for result_key in result_keys:
    nrows = math.ceil(len(param_keys) / ncols)
    if nrows == 1:
      ncols = len(conns)
    fig, axs = plt.subplots(
      ncols=ncols, nrows=nrows,
      figsize=(figsize, figsize))
    fig.suptitle('HyperParam Search Results ({} Episodes) during training'.format(
      result_key.replace('_s', '_over_').replace('_', ' ')))

    param_idx = 0
    for axi in axs:
      if nrows == 1:
        axi = [axi]
      for ax in axi:
        if param_idx == len(param_keys):
          continue

        key = param_keys[param_idx]
        vals = dict([(v, []) for v in hpconfig['params'][key]])
        for res in results:
          param_key = res['params'][key]
          if type(param_key) == str and 'synWeights_final.pkl' in param_key:
            new_param_key = param_key.split('/')[-2]
            if new_param_key not in vals:
              del vals[param_key]
              vals[new_param_key] = []
            param_key = new_param_key
          vals[param_key].append(res[result_key])

        valkeys = sorted(list(vals.keys()))
        ax.boxplot([vals[vk] for vk in valkeys])
        ax.set_xticklabels(
          ['{}({})'.format(vk, len(vals[vk])) for vk in valkeys],
          rotation=8)
        ax.set_ylabel(result_key)
        ax.set_title(key)
        ax.grid(axis='y', alpha=0.5)

        param_idx += 1

    outputfile = os.path.join(wdir, 'eval_params_{}.png'.format(result_key))
    plt.savefig(outputfile)

def create_table(wdir, outputfile=None, runs_json_fname='runs.json'):
  runs_json = os.path.join(wdir, runs_json_fname)
  results_tsv = os.path.join(wdir, 'results.tsv')

  outputfile = os.path.join(wdir, 'results_table.tsv')

  runs = {}
  with open(runs_json) as f:
    for line in f:
      run = json.loads(line)
      runs[run['run_id']] = run

  results = []
  with open(results_tsv) as f:
    for row in csv.DictReader(f, delimiter='\t'):
      for k in row.keys():
        if k.startswith('max_'):
          row[k] = float(row[k])
        elif k == 'run_id':
          row[k] = int(row[k])
      run_id = row['run_id']
      for k,v in runs[run_id].items():
        row[k] = v
      results.append(row)
  del runs
  print('Found {} results'.format(len(results)))

  results = sorted(results, key=lambda x:x['max_average_s100'], reverse=True)
  with open(outputfile, 'w') as out:
    writer = csv.writer(out, delimiter='\t')
    header = [k for k in results[0].keys()]
    writer.writerow(header)
    for row in results:
      writer.writerow([row[h] for h in header])

if __name__ == '__main__':
  fire.Fire({
      'analyze': analyze_samples,
      'combine': create_table
  })
