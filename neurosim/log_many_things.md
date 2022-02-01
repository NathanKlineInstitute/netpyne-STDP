#!/bin/bash

WDIR=results/20220123-EVOL_b1-goodseed/continue_1
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "EVOL goodseed Model" --outdir=$WDIR
py neurosim/tools/evaluate.py weights-adj $WDIR
py neurosim/main.py eval $WDIR --resume_tidx=-1 --env-seed 123 \
    --eps-duration 105
py neurosim/main.py eval $WDIR --resume_tidx=-2 --env-seed 123 \
    --eps-duration 105

py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "EVOL goodseed Model" \
    --outdir=$WDIR --merge-es

py neurosim/tools/eval_multimodel.py train-perf-evol $WDIR


WDIR=results/20220124-EVOL_b1-badseed/continue_1
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

py neurosim/main.py eval $WDIR --resume_tidx=-1 --env-seed 123 \
    --eps-duration 105
py neurosim/main.py eval $WDIR --resume_tidx=-2 --env-seed 123 \
    --eps-duration 105

py neurosim/main.py eval $WDIR --resume_tidx=1 --env-seed 123 \
    --eps-duration 105
py neurosim/main.py eval $WDIR --resume_tidx=0 --env-seed 123 \
    --eps-duration 105

WDIR=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2 
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "STDP goodseed Model" --outdir=$WDIR
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "STDP goodseed Model" --outdir=$WDIR --delimit_wdirs
py neurosim/tools/evaluate.py weights-adj $WDIR

WDIR=results/seedrun_m1-2022-01-16/run_seed5397326/continue_1/continue_1/continue_2/continue_2/continue_2 
py neurosim/tools/eval_multimodel.py train-perf $WDIR --wdir-name "STDP goodseed Model" --outdir=$WDIR
py neurosim/tools/evaluate.py weights-adj $WDIR

WDIR=results/seedrun_m1-2022-01-16/run_seed1394398
CONF1="Seed-6(Best) Before Training:$WDIR:0"
WDIR=results/seedrun_m1-2022-01-16/run_seed1394398/continue_1/continue_1/continue_2/continue_2/continue_2
CONF2="Seed-6(Best) After Training:$WDIR:1"
WDIR=results/seedrun_m1-2022-01-16/run_seed5397326
CONF3="Seed-3(Worst) Before Training:$WDIR:0"
WDIR=results/seedrun_m1-2022-01-16/run_seed5397326/continue_1/continue_1/continue_2/continue_2/continue_2 
CONF4="Seed-3(Worst) After Training:$WDIR:-1"
WDIR=results/seedrun_m1-2022-01-16
py neurosim/tools/eval_multimodel.py boxplot "${CONF1},${CONF2},${CONF3},${CONF4}" --outdir=$WDIR



mkdir $WDIR/evalmockAllStates
STEPS=20
for ((i=0;i<$STEPS;i+=1)); do
    echo "Evaluating Step $i / $STEPS"
    py neurosim/main.py eval $WDIR --resume_tidx=-1 --eps-duration 1 \
        --mock-env 2 \
        --duration 1300 \
        --mock_curr_step $i \
        --mock_total_steps $STEPS \
        --outdir $WDIR/evalmockAllStates/step_${i}
done