import os
import csv
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import sys

WDIR = '/u/samn/netpyne-STDP/results/20210701'

"""
evaluations = [(os.path.join(WDIR, fname), int(fname.replace('evaluation_', ''))) for fname in os.listdir(WDIR) if fname.startswith('evaluation_')]

results = {}
for eval_dir, eval_ts in evaluations:
    if os.path.isfile(os.path.join(eval_dir, 'ActionsPerEpisode.txt')):
        with open(os.path.join(eval_dir, 'ActionsPerEpisode.txt')) as f:
            results[eval_ts] = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

with open(os.path.join(WDIR, 'synWeights.pkl'), 'br') as f:
    synWeights = pkl.load(f)
    
preid = list(synWeights.keys())[0]
postid = list(synWeights[preid].keys())[0]

steps = len(synWeights[preid][postid])
ts = [s for s,v in synWeights[preid][postid]]

sorted_results = sorted(list(results.items()), key=lambda x:x[0])

labels = [k for k,v in sorted_results]
data = [v for k,v in sorted_results]

fig = plt.figure(figsize =(10, 7))

# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)

ax.set_xticklabels(
    ['step {} (at {} s)'.format(l, round(ts[l] / 1000, 4)) for l in labels],
    rotation = 80)

ax.set_ylabel('actions per episode')

ax.set_title('BoxPlots of ActionsPerEpisode for 20210630 model at different timesteps')

# show plot
plt.show()

fig = plt.figure(figsize =(10, 7))

data = [(step_ts, acts_per_eps) for step_ts, acts_per_eps in results.items()]
data = sorted(data, key=lambda x:x[0])
 
plt.plot([x for x,y in data], [np.mean(y) for x,y in data], '-x')
plt.plot([x for x,y in data], [np.median(y) for x,y in data], '-x')
plt.plot([x for x,y in data], [np.std(y) for x,y in data], '-x')

plt.legend(['mean', 'median', 'std'])

plt.xlabel('epoch')
plt.ylabel('actions per episode')
plt.title('Evolution of performance during training')

plt.grid()
plt.show()
"""

def plotcartpoleperf ():
    plt.ion()
    with open(os.path.join(WDIR, 'ActionsPerEpisode.txt')) as f:
        training_results = [int(float(eps)) for _,eps in csv.reader(f, delimiter='\t')]

    training_medians = {}
    for STEP in [21, 51, 101]:
        training_medians[STEP] = []
        for idx in range(len(training_results) - STEP):
            training_medians[STEP].append(np.median(training_results[idx:idx+STEP]))



    plt.figure(figsize=(10,10))

    plt.plot(list(range(len(training_results))), training_results)

    for STEP, medians in training_medians.items():
        plt.plot([t + STEP for t in range(len(medians))], medians)

    plt.legend(['individual'] + ['median of {}'.format(STEP) for STEP in training_medians.keys()])

    plt.xlabel('episode')
    plt.ylabel('actions per episode')

    plt.show()

"""
tm = np.array(training_medians) 

first_index = np.where(tm == np.amax(tm))[0][0]

times = [training_results[i] for i in range(first_index)]
cumulative = sum(times)
cumulative += len(times)

print(times)
print(cumulative)

cumulative * 50 / 1000
"""

if __name__ == '__main__':
    if len(sys.argv) > 1:
        WDIR = sys.argv[1]
    print(WDIR)
