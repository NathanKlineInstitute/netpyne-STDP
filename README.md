# NeuroSim CartPole model
Adatapted from [SMARTAgent](https://github.com/NathanKlineInstitute/SMARTAgent) on May 24th 2021

A closed-loop neuronal network model that senses dynamic visual information from the AIgame enviroment and learns to produce actions that maximize game reward through spike-timing dependent reinforcement learning.

This project is related to the paper:

#### Training spiking neuronal networks to perform motor control using reinforcement and evolutionary learning
Hasegan D, Deible M, Earl C, D'Onofrio D, Hazan H, Anwar H, Neymotin SA. 

bioRxiv: [https://doi.org/10.1101/2021.11.20.469405](https://doi.org/10.1101/2021.11.20.469405)

Published in: TODO


---

### Project structure

- `config.json` the current configuration that we will work with. This gets picked by default
- `requirements.txt` the pip packages needed for installation
- `notebooks/` folder with jupyter notebooks that are used for exploration and data analysis before translating into tools
- `results/` default place for storing the resulting trained models
- `neurosim/` the code
    - `main.py` main extry point to the code (run training, evaluation, contiuation)
    - `sim.py` the NetPyNE model setup and run
    - `critic.py` the critic in the actor-critic model for RL (provides reward/punishment)
    - `aigame.py` wrapper for OpenAI gym
    - `game_interface.py` simple interface between the game and firing rates or receptive fields
    - `conf.py` utility for configuration files
    - `utils/` folder for helper functions
    - `tools/` folder of evaluation tools
    - `cells/` folder for defined cells

- `mod/` the NEURON objects that will get compiled by `nvrnivmodl`
- `x86_64/` the compiled NEURON objects

### Install on Mac

install python >3.7 (previous versions might not work)

install ffmpeg

    brew install ffmpeg

create a virtual environment

    virtualenv -p python3 venv

activate the environment

    source ./venv/bin/activate
    export PYTHONPATH="`pwd`"

install all dependencies:

    pip install -r requirements.txt

Compile mod files

    nrnivmodl mod

### Install on Ubuntu 18.04

Running this on a VM on gcloud

    EMAIL="..."
    ssh-keygen -t ed25519 -C $EMAIL
    cat .ssh/id_ed25519.pub 
    # Copy the pub key here: https://github.com/settings/keys
    git clone git@github.com:NathanKlineInstitute/netpyne-STDP.git
    cd netpyne-STDP/

then run the whole script below

    git config --global alias.co checkout
    git config --global alias.br branch
    git config --global alias.ci commit
    git config --global alias.st status
    git config --global user.name "$USER"
    git config --global user.email $EMAIL

    # install pacakges
    sudo apt-get update
    yes | sudo apt-get install python3.7 python3.7-distutils cmake build-essential libz-dev ffmpeg
    
    # install pip and create the environment
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3.7 get-pip.py
    python3.7 -m pip install virtualenv
    python3.7 -m virtualenv venv

    # activate
    source ./venv/bin/activate
    export PYTHONPATH="`pwd`"
    alias py=python3

    # install requirements
    pip install -r requirements.txt

    # compile neuron
    nrnivmodl mod

Then run commands with screen to avoid network disconnecting

    screen -L -Logfile logfile.log ... 

### run

    python3 neurosim/main.py run

Check notebooks:

    jupyter notebook

Run tests:

    pytest tests

### Run ES
    
    python3 neurosim/train_es.py train

Or continue training model

    python3 neurosim/train_es.py continue results/20210907-ES1500it --iterations 500

---

### Tools/Evaluation

Evaluate the model before and after training:
    
    WDIR=results/20210707
    py neurosim/main.py eval $WDIR --resume_tidx=0
    py neurosim/main.py eval $WDIR --resume_tidx=-1 --duration 250

    # To evaluate and use for poster/presentation:
    py neurosim/main.py eval $WDIR --resume_tidx=-1 \
        --env-seed 42 \
        --eps-duration 105 \
        --save-data

    # To evaluate one specific episode over and over again
    # for EPISODE_ID in 74 68 77 55 31 50 32 10 19 59 22
    for EPISODE_ID in 74 68 77 55 31 50 32 10 19 59 22
    do
        py neurosim/main.py eval $WDIR --resume_tidx=-1 \
            --env-seed 42 \
            --eps-duration 25 \
            --save-data \
            --saveEnvObs \
            --rerun-episode ${EPISODE_ID}
    done


    # To display the model run
    py neurosim/main.py eval $WDIR --resume_tidx=-1 --display --env-seed 42 \
        --eps-duration 105

Optional: Maybe evaluate in depth

    py neurosim/main.py eval $WDIR --resume_best_training --env-seed 42 \
        --eps-duration 105

    for ((i=8;i>=0;i-=1)); do
        echo "Evaluating at $i"
        py neurosim/main.py eval $WDIR --resume_tidx=$i --env-seed 42 \
            --eps-duration 105 \
            --save-data
    done

Evaluate how the model is responding to one neuron firing:
    
    py neurosim/main.py eval $WDIR --resume_tidx=-1 --eps-duration 2 \
        --mock-env

    # For a more comprehensive approach to run on all steps

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

Run all evaluation:

    WDIR=results/20210907
    py neurosim/tools/evaluate.py medians $WDIR
    py neurosim/tools/evaluate.py rewards $WDIR
    py neurosim/tools/evaluate.py rewards-vals $WDIR
    py neurosim/tools/evaluate.py eval-motor $WDIR
    py neurosim/tools/evaluate.py eval-moves $WDIR --unk_moves
    py neurosim/tools/evaluate.py eval-moves $WDIR --abs_move_diff

    py neurosim/tools/evaluate.py weights-adj $WDIR
    py neurosim/tools/evaluate.py weights-adj $WDIR --index 0
    py neurosim/tools/evaluate.py weights-diffs $WDIR
    py neurosim/tools/evaluate.py weights-diffs $WDIR --relative
    py neurosim/tools/evaluate.py weights-ch $WDIR
    py neurosim/tools/evaluate.py weights-ch $WDIR --separate_movement True

    py neurosim/tools/evaluate.py frequency $WDIR --timestep 10000
    py neurosim/tools/evaluate.py variance $WDIR

    py neurosim/tools/evaluate.py boxplot $WDIR
    py neurosim/tools/evaluate.py perf $WDIR

Continue training from a already trained model:

    WDIR=results/...
    py neurosim/main.py continue $WDIR --duration 5000
    # note: this script needs more care on how to integrate with different/new params

    # more detailed example:
    py neurosim/main.py continue results/20210801-1000it-1eps/ \
        --copy-from-config config.json \
        --copy-fields critic,sim,STDP-RL \
        --duration 100 \
        --index=3

#### Critic evaluation

    py neurosim/tools/critic.py eval \
        --best-wdir results/20210801-1000it-1eps/500s-evaluation_10 \
        --critic-config results/20210801-1000it-1eps/backupcfg_sim.json \
        --verbose

    py neurosim/tools/critic.py eval \
        --best-wdir results/20210801-1000it-1eps/500s-evaluation_10 \
        --critic-config config.json \
        --verbose

    py neurosim/tools/critic.py hpsearch \
        --best-wdir results/20210801-1000it-1eps/500s-evaluation_10 \
        --critic-config config.json

### Train the same config on many network configs

    WDIR=results/seedrun_m1-2022-01-16
    mkdir $WDIR

    python3 neurosim/main.py seedrun $WDIR --conn-seed 2542033

    for ((i=0;i<20;i+=1)); do
        echo "Running $i th seed"
        python3 neurosim/main.py seedrun $WDIR --fnjson $WDIR/config.json
    done

Evaluate seedrun:

    python3 neurosim/tools/eval_seedrun.py analyze $WDIR

Continue seedrun:
    
    cp results/hpsearch-2022-01-11/best/1_run_2371/backupcfg_sim.json results/seedrun_m1-2022-01-16/config2.json
    py neurosim/main.py cont_seedrun $WDIR/run_seed1139028 $WDIR/config2.json

    

### Run Hyperparameter search

Change `hpsearch_config.json` to the needed params

    WDIR=results/hpsearch-2021-09-13

    # Just for setup:
    py neurosim/hpsearch.py sample $WDIR --just-init

    # run one sample
    py neurosim/hpsearch.py sample $WDIR

    
    # run 100 samples
    for ((i=0;i<100;i+=1)); do
        echo "Sampling $i th run"
        time py neurosim/hpsearch.py sample $WDIR
    done

Alternatively you can run HPSearch with random networks instead:

    # run one sample with a random initialization
    py neurosim/hpsearch.py sample $WDIR --random_network

    # run 100 samples
    for ((i=0;i<100;i+=1)); do
        echo "Sampling $i th run"
        time py neurosim/hpsearch.py sample $WDIR --random_network
    done

Results are posted in `$WDIR/results.tsv`, then you can analyze with:

    py neurosim/tools/eval_hpsearch.py analyze $WDIR
    py neurosim/tools/eval_hpsearch.py combine $WDIR

### Evaluation of best models:

STDP model process:

    WDIR=results/hpsearch-2021-09-01/run_106756
    WDIR=results/hpsearch-2021-09-04/best/1_run_2703
    WDIR=results/hpsearch-2021-09-06/best/1_run_168

ES model:

    WDIR=results/20210907-ES1500it

#### To trace the model

Use this on the latest step of the model

    py neurosim/tools/eval_multimodel.py trace $WDIR

### To generate more results:

Beware: this specific code was used for previous versions. 
New and updated plot-generation scripts are found in the logs of the individual seedruns: `seedrun_m1-2022-01-16` and `seedrun_evol-2022-02-20`.

    BSTDP_WDIR=results/hpsearch-2021-09-06/best/1_run_168
    BEVOL_WDIR=results/20210907-ES1500it
    BCOMB_WDIR=results/evol-stdp-rl
    BEFORE_CONF="Before Training:${BEVOL_WDIR}:0"
    BSTDP_CONF="After STDP-RL Training:${BSTDP_WDIR}:-1"
    BEVOL_CONF="After EVOL Training:${BEVOL_WDIR}:-1"
    BCOMB_CONF="After EVOL+STDP-RL Training:${BCOMB_WDIR}:-1"
    OUTDIR=results/final-results-2021-09

    py neurosim/tools/eval_multimodel.py boxplot "${BEFORE_CONF},${BSTDP_CONF},${BEVOL_CONF},${BCOMB_CONF}" --outdir=$OUTDIR

    py neurosim/tools/evaluate.py frequency ${BSTDP_WDIR}/evaluation_8 --timestep 10000
    py neurosim/tools/evaluate.py frequency ${BEVOL_WDIR}/evaluation_15 --timestep 10000
    py neurosim/tools/evaluate.py frequency ${BEVOL_WDIR}/evaluation_0 --timestep 10000
    py neurosim/tools/evaluate.py variance ${BSTDP_WDIR}/evaluation_8
    py neurosim/tools/evaluate.py variance ${BEVOL_WDIR}/evaluation_15
    py neurosim/tools/evaluate.py variance ${BEVOL_WDIR}/evaluation_0

    py neurosim/tools/eval_multimodel.py spk-freq "${BEFORE_CONF},${BSTDP_CONF},${BEVOL_CONF}" --outdir=$OUTDIR

    py neurosim/tools/eval_multimodel.py train-perf $BSTDP_WDIR --wdir-name "STDP-RL Model" --outdir=$OUTDIR
    py neurosim/tools/eval_multimodel.py train-perf $BEVOL_WDIR --wdir-name "ES Model" --outdir=$OUTDIR
    py neurosim/tools/eval_multimodel.py train-perf $BEVOL_WDIR --wdir-name "ES Model" --outdir=$OUTDIR --merge-es
    py neurosim/tools/eval_multimodel.py train-perf-comb "${BSTDP_CONF},${BEVOL_CONF}" --outdir=$OUTDIR

    py neurosim/tools/eval_multimodel.py train-unk-moves "${BSTDP_CONF},${BEVOL_CONF}" --outdir=$OUTDIR

    py neurosim/tools/eval_multimodel.py select-eps \
            ${BSTDP_WDIR}/evaluation_8,${BEVOL_WDIR}/evaluation_15

    py neurosim/tools/eval_multimodel.py eval-selected-eps \
            ${BSTDP_CONF},${BEVOL_CONF},${BCOMB_CONF} \
            --outdir=$OUTDIR \
            --sort-by "68,74,10,19,55,32,31,22,77"
            

    # Weights:
    py neurosim/tools/eval_multimodel_weights.py changes \
                "${BSTDP_CONF},${BEVOL_CONF}" --outdir=$OUTDIR

    # Observation Space Receptive Fields
    py neurosim/tools/eval_obsspace.py rf $OUTDIR
