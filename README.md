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

install all dependencies:

    pip install -r requirements.txt

Add needed dirs

    mkdir data

Compile mod files

    nrnivmodl mod

### run

    python3 neurosim/main.py run

Evaluate example:

    py neurosim/main.py eval results/20210629 --resume_tidx=0
    py neurosim/main.py eval results/20210629 --resume_tidx=-1

    for ((i=2;i<=30;i+=2)); do
        echo "Evaluating at $i"
        py neurosim/main.py eval results/20210629-longrun --resume_tidx=$i
    done

Check notebooks:

    jupyter notebook

Run tests:

    pytest tests

Frequency analysis tool:

    py neurosim/tools/evaluate.py frequency \
        results/20210624/sim.json \
        --outputfile results/20210624/frequency.png \
        --timestep 10000

outputs the frequency file
