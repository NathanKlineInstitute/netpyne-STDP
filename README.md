# SMARTAgent
A closed-loop neuronal network model that senses dynamic visual information from the AIgame enviroment and learns to produce actions that maximize game reward through spike-timing dependent reinforcement learning.

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

### Tools/Evaluation

Evaluate the model before and after training:
    
    WDIR=results/20210707
    py neurosim/main.py eval $WDIR --resume_tidx=0
    py neurosim/main.py eval $WDIR --resume_tidx=-1

    # To display the model run
    py neurosim/main.py eval $WDIR --resume_tidx=-1 --display

Optional: Maybe evaluate in depth

    for ((i=0;i<20;i+=2)); do
        echo "Evaluating at $i"
        py neurosim/main.py eval $WDIR --resume_tidx=$i
    done

Run all evaluation:

    WDIR=results/20210809
    py neurosim/tools/evaluate.py frequency $WDIR --timestep 10000
    py neurosim/tools/evaluate.py variance $WDIR
    py neurosim/tools/evaluate.py medians $WDIR
    py neurosim/tools/evaluate.py rewards $WDIR

    py neurosim/tools/evaluate.py weights-adj $WDIR
    py neurosim/tools/evaluate.py weights-adj $WDIR --index 0
    py neurosim/tools/evaluate.py weights-diffs $WDIR
    py neurosim/tools/evaluate.py weights-diffs $WDIR --relative

    py neurosim/tools/evaluate.py boxplot $WDIR
    py neurosim/tools/evaluate.py perf $WDIR

Continue training from a already trained model:

    WDIR=results/...
    py neurosim/main.py continue $WDIR --duration 5000
    # note: this script needs more care on how to integrate with different/new params
