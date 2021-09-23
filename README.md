# NeuroSim CartPole model
Adatapted from SMARTAgent on May 24th 2021

A closed-loop neuronal network model that senses dynamic visual information from the AIgame enviroment and learns to produce actions that maximize game reward through spike-timing dependent reinforcement learning.

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

- `mod/` the compiled NEURON objects



### Install on Mac

install python

install ffmpeg

    brew install ffmpeg

create a virtual environment

    virtualenv -p python3 venv

activate the environment

    source ./venv/bin/activate
    export PYTHONPATH="`pwd`"

install all dependencies:

    pip install -r requirements.txt

Add needed dirs

    mkdir data

Compile mod files

    nrnivmodl mod

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
    # for EPISODE_ID in 4 7 12 16 49 60 66 78 83
    for EPISODE_ID in 4 7 12 16 49 66 78 83
    do
        py neurosim/main.py eval $WDIR --resume_tidx=-1 \
            --env-seed 42 \
            --eps-duration 25 \
            --save-data \
            --rerun-episode ${EPISODE_ID}
    done


    # To display the model run
    py neurosim/main.py eval $WDIR --resume_tidx=-1 --display --env-seed 42 \
        --eps-duration 105

Optional: Maybe evaluate in depth

    py neurosim/main.py eval $WDIR --resume_best_training --env-seed 42 \
        --eps-duration 105

    for ((i=30;i<=40;i+=5)); do
        echo "Evaluating at $i"
        py neurosim/main.py eval $WDIR --resume_tidx=$i --env-seed 42 --eps-duration 105
    done

Evaluate how the model is responding to one neuron firing:
    
    py neurosim/main.py eval $WDIR --resume_tidx=-1 --eps-duration 2 --mock-env

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

### Run Hyperparameter search

Change `hpsearch_config.json` to the needed params

    WDIR=results/hpsearch-2021-09-13

    # Just for setup:
    py neurosim/hpsearch.py sample $WDIR --just-init

    # run one sample
    py neurosim/hpsearch.py sample $WDIR
    
    # run 100 samples
    for ((i=0;i<200;i+=1)); do
        echo "Sampling $i th run"
        time py neurosim/hpsearch.py sample $WDIR
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

### To generate all results:

    BSTDP_WDIR=results/hpsearch-2021-09-06/best/1_run_168
    BES_WDIR=results/20210907-ES1500it
    BEFORE_CONF="Before Training:${BES_WDIR}:0"
    BSTDP_CONF="STDP-RL Trained Model:${BSTDP_WDIR}:-1"
    BES_CONF="ES Trained Model:${BES_WDIR}:-1"
    OUTDIR=results/final-results-2021-09

    py neurosim/tools/eval_multimodel.py boxplot "${BEFORE_CONF},${BSTDP_CONF},${BES_CONF}" --outdir=$OUTDIR

    py neurosim/tools/evaluate.py frequency ${BSTDP_WDIR}/evaluation_8 --timestep 10000
    py neurosim/tools/evaluate.py frequency ${BES_WDIR}/evaluation_15 --timestep 10000
    py neurosim/tools/evaluate.py frequency ${BES_WDIR}/evaluation_0 --timestep 10000
    py neurosim/tools/evaluate.py variance ${BSTDP_WDIR}/evaluation_8
    py neurosim/tools/evaluate.py variance ${BES_WDIR}/evaluation_15
    py neurosim/tools/evaluate.py variance ${BES_WDIR}/evaluation_0

    py neurosim/tools/eval_multimodel.py spk-freq "${BEFORE_CONF},${BSTDP_CONF},${BES_CONF}" --outdir=$OUTDIR

    py neurosim/tools/eval_multimodel.py steps-per-eps $BSTDP_WDIR --wdir-name "STDP-RL Model" --outdir=$OUTDIR
    py neurosim/tools/eval_multimodel.py steps-per-eps $BES_WDIR --wdir-name "ES Model" --outdir=$OUTDIR
    py neurosim/tools/eval_multimodel.py steps-per-eps $BES_WDIR --wdir-name "ES Model" --outdir=$OUTDIR --merge-es

    py neurosim/tools/eval_multimodel.py select-eps \
            ${BSTDP_WDIR}/evaluation_8,${BES_WDIR}/evaluation_15

    py neurosim/tools/eval_multimodel.py eval-selected-eps \
            ${BSTDP_CONF},${BES_CONF} \
            --outdir=$OUTDIR \
            --sort-by 16,7,78,66,12,49,83,60,4
            