#!/bin/bash

WDIR=results/20220123-EVOL_b1-goodseed/continue_1
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "EVOL goodseed Model" --outdir=$WDIR
py neurosim/tools/evaluate.py weights-adj $WDIR

# Validation!
py neurosim/main.py eval $WDIR --resume_tidx=-1 --env-seed 123 \
    --eps-duration 105
py neurosim/main.py eval $WDIR --resume_tidx=30 --env-seed 123 \
    --eps-duration 105
py neurosim/main.py eval $WDIR --resume_tidx=0 --env-seed 123 \
    --eps-duration 105
# Testing!
py neurosim/main.py eval $WDIR --resume_tidx=-1 --env-seed 42 \
    --eps-duration 100 --save-data

py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "EVOL goodseed Model" \
    --outdir=$WDIR --merge-es

WDIR=results/20220123-EVOL_b1-goodseed/continue_1
WDIR=results/20220123-EVOL_b1-goodseed/continue_1/continue_1
WDIR=results/20220123-EVOL_b1-goodseed/continue_1/continue_1/continue_1/continue_1
WDIR=results/20220124-EVOL_b1-badseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
WDIR=results/20220128-EVOL_b5-badseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
WDIR=results/20220129-EVOL_b5-goodseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR

# Figure 4
# A
WDIR=results/20220124-EVOL_b1-badseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR \
    --show_xlabel False --extraticks 1500 --rmticks 1000,2000 --vline 1500
# B
WDIR=results/20220123-EVOL_b1-goodseed/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR \
    --show_xlabel False --extraticks 1500 --rmticks 1000,2000 --vline 1500
# C
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_worst_beta1_pop10.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR \
    --vline 1500 --extraticks 1250,1500 --rmticks 1200,1400 --topticks
# D
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_best_beta1_pop10.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR \
    --vline 1500 --extraticks 1250,1500 --rmticks 1200,1400 --topticks


WDIR=results/20220124-EVOL_b1-badseed/continue_1
WDIR=results/20220124-EVOL_b1-badseed/continue_1/continue_1
WDIR=results/20220124-EVOL_b1-badseed/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "EVOL badseed Model" --outdir=$WDIR
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "EVOL badseed Model" \
    --outdir=$WDIR --merge-es
py neurosim/tools/evaluate.py weights-adj $WDIR

py neurosim/tools/evaluate.py boxplot $WDIR
py neurosim/tools/evaluate.py perf $WDIR

for WDIR in results/20220123-EVOL_b1-goodseed/continue_1 results/20220123-EVOL_b1-goodseed results/20220124-EVOL_b1-badseed/continue_1 results/20220124-EVOL_b1-badseed
do
    for fname in ActionsPerEpisode.txt backupcfg_sim.json es_train.txt "synWeights.pkl"
    do
        git add -f $WDIR/$fname
    done
done


WDIR=results/evol-stdp-rl_2022-01-21/b1
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "EVOL+STDP-RL b1 Model" --outdir=$WDIR

---

# 123 for validation
py neurosim/main.py eval $WDIR --resume_tidx=-1 --env-seed 123 \
    --eps-duration 105
py neurosim/main.py eval $WDIR --resume_tidx=-2 --env-seed 123 \
    --eps-duration 105

# 42 for test
py neurosim/main.py eval $WDIR --resume_tidx=0 --env-seed 42 \
    --eps-duration 100 --save-data
py neurosim/main.py eval $WDIR --resume_tidx=1 --env-seed 42 \
    --eps-duration 100 --save-data
py neurosim/main.py eval $WDIR --resume_tidx=-1 --env-seed 42 \
    --eps-duration 100 --save-data


# Best model:
WDIR=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2/continue_2
# Longest trained:
WDIR=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2/continue_2/continue_2
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "STDP goodseed Model" --outdir=$WDIR
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "STDP goodseed Model" --outdir=$WDIR --delimit_wdirs
py neurosim/tools/evaluate.py weights-adj $WDIR

WDIR=results/seedrun_m1-2022-01-16/run_seed5397326/continue_1/continue_1/continue_2/continue_2/continue_2 
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "STDP badseed Model" --outdir=$WDIR
py neurosim/tools/evaluate.py weights-adj $WDIR

# STDP-RL best models
WDIR=results/seedrun_m1-2022-01-16/run_seed1394398
CONF1="Seed-6(Best) Before Training:$WDIR:0"
WDIR=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2
CONF2="Seed-6(Best) After Training:$WDIR:1"
WDIR=results/seedrun_m1-2022-01-16/run_seed5397326
CONF3="Seed-3(Worst) Before Training:$WDIR:0"
WDIR=results/seedrun_m1-2022-01-16/run_seed5397326/continue_1/continue_1/continue_2/continue_2/continue_2
CONF4="Seed-3(Worst) After Training:$WDIR:-1"
WDIR=results/seedrun_m1-2022-01-16
py neurosim/tools/eval_multimodel.py boxplot "${CONF1},${CONF2},${CONF3},${CONF4}" \
    --outdir=$WDIR --pvals "1,2;3,4" --plog

 # 02/01 EVOL b1
WDIR=results/seedrun_m1-2022-01-16/run_seed1394398
CONF1="Seed-6(Best) Before Training B1:$WDIR:0"
WDIR=results/20220123-EVOL_b1-goodseed/continue_1/continue_1/continue_1/continue_1
CONF2="Seed-6(Best) After Training B1:$WDIR:0" # <- This '0' is correct!
WDIR=results/seedrun_m1-2022-01-16/run_seed5397326
CONF3="Seed-3(Worst) Before Training B1:$WDIR:0"
WDIR=results/20220124-EVOL_b1-badseed/continue_1/continue_1/continue_1/continue_1/continue_1
CONF4="Seed-3(Worst) After Training B1:$WDIR:6"

# EVOL b5
WDIR=results/20220129-EVOL_b5-goodseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
CONF5="Seed-6(Best) After Training B5:$WDIR:1"
WDIR=results/20220128-EVOL_b5-badseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
CONF6="Seed-3(Worst) After Training B5:$WDIR:2"

py neurosim/tools/eval_multimodel.py boxplot "${CONF1},${CONF2},${CONF3},${CONF4}" \
    --outdir=$WDIR --pvals "1,2;3,4" --plog 


# Good Seed B1 / B5
WDIR=results/20220123-EVOL_b1-goodseed/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR
WDIR=results/20220129-EVOL_b5-goodseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR
# Bad Seed B1 / B5
WDIR=results/20220124-EVOL_b1-badseed/continue_1/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR
WDIR=results/20220128-EVOL_b5-badseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR


mkdir $WDIR/evalmockAllStates
STEPS=20
for ((i=0;i<$STEPS;i+=1)); do
    echo "Evaluating Step $i / $STEPS"
    py neurosim/main.py eval $WDIR --resume_tidx=58 --eps-duration 1 \
        --mock-env 2 \
        --duration 1300 \
        --mock_curr_step $i \
        --mock_total_steps $STEPS \
        --outdir $WDIR/evalmockAllStates/step_${i}
done

for stepname in `ls | grep step`
do
    unzip $stepname/sim.pkl.zip
    unzip $stepname/MotorOutputs.txt.zip
    ls $stepname/sim.pkl
    ls $stepname/MotorOutputs.txt
    git rm $stepname/sim.pkl.zip
    git rm $stepname/MotorOutputs.txt.zip
done


WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_best_beta1_pop10.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_best_beta5_pop10.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_worst_beta1_pop10.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_worst_beta5_pop10.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR

WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_worst_beta10_pop5.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_best_beta10_pop5.small
py neurosim/tools/eval_multimodel.py train-perf-evolstdprl $WDIR

WDIR=results/evol-stdp-rl_2022-01-21
py neurosim/tools/eval_multimodel.py train-eval-evolstdprl \
    ${WDIR}/STDP_ES_worst_beta1_pop10.small,${WDIR}/STDP_ES_worst_beta5_pop10.small

eval_for_evolstdprl

for EMODEL in STDP_ES_best_beta1_pop10.small STDP_ES_best_beta5_pop10.small STDP_ES_worst_beta1_pop10.small STDP_ES_worst_beta5_pop10.small
do
    WDIR=results/evol-stdp-rl_2022-01-21/$EMODEL
    git add -f $WDIR/backupcfg_sim.json
    git add -f $WDIR/iters_during_*.png
done



# EVOL-STDP-RL models
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_best_beta1_pop10.small
py neurosim/main.py eval $WDIR --resume_tidx=58 --env-seed 42 \
    --eps-duration 100 --save-data
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_best_beta5_pop10.small
py neurosim/main.py eval $WDIR --resume_tidx=59 --env-seed 42 \
    --eps-duration 100 --save-data
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_worst_beta1_pop10.small
py neurosim/main.py eval $WDIR --resume_tidx=56 --env-seed 42 \
    --eps-duration 100 --save-data
WDIR=results/evol-stdp-rl_2022-01-21/STDP_ES_worst_beta5_pop10.small
py neurosim/main.py eval $WDIR --resume_tidx=52 --env-seed 42 \
    --eps-duration 100 --save-data


# g10 models
WDIR=results/retrain_evol_g10-2022-02-18/run_seed1394398
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR
WDIR=results/retrain_evol_g10-2022-02-18/run_seed5397326
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR


# Figure 4
WDIR=results/20220129-EVOL_b5-goodseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR

WDIR=results/20220128-EVOL_b5-badseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1
py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR


WDIR=results/seedrun_evol-2022-02-20
py neurosim/tools/eval_seedrun.py train-perf-evol $WDIR \
    --stype avg --steps 4 --parts 1

WDIR=results/seedrun_evol-2022-02-20
py neurosim/tools/eval_seedrun.py train-perf-evol $WDIR \
    --stype avg --steps 4 --parts 1 --seed_nrs

python3 neurosim/tools/eval_seedrun.py a1 $WDIR \
    --find_modifier --evaltype evol --stype avg --steps 4 --no_swarm

py neurosim/tools/eval_seedrun.py train-perf-evol $WDIR \
    --stype avg --steps 4 --parts 1 --seed_nrs --target_perf 118


---------



WDIR=results/seedrun_m1-2022-01-16/run_seed1394398
py neurosim/main.py eval $WDIR --resume_tidx=0 --env-seed 42 \
    --eps-duration 100 --save-data

WDIR=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2
py neurosim/main.py eval $WDIR --resume_tidx=1 --env-seed 42 \
    --eps-duration 100 --save-data

WDIR=results/seedrun_m1-2022-01-16/run_seed5397326
py neurosim/main.py eval $WDIR --resume_tidx=0 --env-seed 42 \
    --eps-duration 100 --save-data

WDIR=results/seedrun_m1-2022-01-16/run_seed5397326/continue_1/continue_1/continue_2/continue_2/continue_2
py neurosim/main.py eval $WDIR --resume_tidx=-1 --env-seed 42 \
    --eps-duration 100 --save-data



WDIR=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2


for EPISODE_ID in 74 68 77 55 31 50 32 10 19 59 22
do
    py neurosim/main.py eval $WDIR --resume_tidx=1 \
        --env-seed 42 \
        --eps-duration 25 \
        --save-data \
        --rerun-episode ${EPISODE_ID}
done



WDIR=results/seedrun_m1-2022-01-16/run_seed5397326/continue_1/continue_1/continue_2/continue_2/continue_2

for EPISODE_ID in 74 68 77 55 31 50 32 10 19 59 22
do
    py neurosim/main.py eval $WDIR --resume_tidx=-1 \
        --env-seed 42 \
        --eps-duration 25 \
        --save-data \
        --rerun-episode ${EPISODE_ID}
done


WDIR_S6=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2
WDIR_S3=results/seedrun_m1-2022-01-16/run_seed5397326/continue_1/continue_1/continue_2/continue_2/continue_2
OUTDIR=results/final-results-2021-09

py neurosim/tools/eval_multimodel.py select-eps \
            ${WDIR_S6}/evaluation_1,${WDIR_S3}/evaluation_50

# Figure 6
py neurosim/tools/eval_multimodel.py eval-selected-eps \
            "STDP-RL Seed-6:${WDIR_S6}:1","STDP-RL Seed-3:${WDIR_S3}:50" \
            --outdir=$OUTDIR \
            --sort-by "68,74,10,19,55,32,31,22,77"


WDIR=results/20220129-EVOL_b5-goodseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1

mkdir $WDIR/evalmockAllStates
STEPS=20
for ((i=0;i<$STEPS;i+=1)); do
    echo "Evaluating Step $i / $STEPS"
    py neurosim/main.py eval $WDIR --resume_tidx=1 --eps-duration 1 \
        --mock-env 2 \
        --duration 1300 \
        --mock_curr_step $i \
        --mock_total_steps $STEPS \
        --outdir $WDIR/evalmockAllStates/step_${i}
done


for EPISODE_ID in 68 77
do
    py neurosim/main.py eval $WDIR --resume_tidx=-1 \
        --env-seed 42 \
        --eps-duration 25 \
        --save-data \
        --rerun-episode ${EPISODE_ID}
done

WDIR=results/20220128-EVOL_b5-badseed/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1

for EPISODE_ID in 68 77
do
    py neurosim/main.py eval $WDIR --resume_tidx=-1 \
        --env-seed 42 \
        --eps-duration 25 \
        --save-data \
        --rerun-episode ${EPISODE_ID}
done

# Copy all weights for EVOL into one folder:


mkdir results/seedrun_evol-2022-02-20/weights_EVOL


for seed in run_seed1257804 run_seed1932160 run_seed2544501 run_seed5140568 run_seed5381445 run_seed6623146 run_seed7892276 run_seed9300610
do
    echo $seed
    WDIR=results/seedrun_evol-2022-02-20/$seed/continue_1/
    cp $WDIR/synWeights.pkl results/seedrun_evol-2022-02-20/weights_EVOL/${seed}_synWeights_latest.pkl

    WDIR=results/seedrun_evol-2022-02-20/$seed/
    cp $WDIR/synWeights.pkl results/seedrun_evol-2022-02-20/weights_EVOL/${seed}_synWeights_original.pkl
done


seed=run_seed1394398
cp results/seedrun_evol-2022-02-20/${seed}/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/synWeights.pkl \
    results/seedrun_evol-2022-02-20/weights_EVOL/${seed}_synWeights_latest.pkl
WDIR=results/seedrun_evol-2022-02-20/$seed/
cp $WDIR/synWeights.pkl results/seedrun_evol-2022-02-20/weights_EVOL/${seed}_synWeights_original.pkl

seed=run_seed5397326
cp results/seedrun_evol-2022-02-20/${seed}/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/continue_1/synWeights.pkl \
    results/seedrun_evol-2022-02-20/weights_EVOL/${seed}_synWeights_latest.pkl
WDIR=results/seedrun_evol-2022-02-20/$seed/
cp $WDIR/synWeights.pkl results/seedrun_evol-2022-02-20/weights_EVOL/${seed}_synWeights_original.pkl



# Generate the figure 4 plot by using the notebook: `Plot diffs between plasticity schedules.ipynb`


```
aggs_by_step = [[256.8900, 166.5550, 292.0650, 269.0700, 222.4450, 82.5850, 138.2000, 78.9800, 171.8400, 117.4750],
    [497.75, 496.5, 496.75, 499.25, 490.0, 497.5, 498.0, 500.0, 499.5, 500.0]]
labels = ['EA→EM plastic', 'ES→EA→EM plastic']

```