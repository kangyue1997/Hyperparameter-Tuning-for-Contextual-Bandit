# Hyperparameter-Tuning-for-Contextual-Bandit

## Requirements

To run the code, you will need:
Python3, Numpy, Matplotlib, itertools

## Commands

python3 run.py -algo {algorithm} -gentype {dataset}  

In the above command, replace ``{algorithm}`` with ``linucb``, ``lints``, ``glmucb``, ``laplacets``,``sgdts``, ``glmtsl`` or ``gloc`` to get the results for different algorithms. Replace ``{gentype}`` with ``uniform``, ``normal`` (simulations) or ``movielens`` (real data) to run the experiments for the different scenarios we mention in the paper. When ``{gentype}`` is set as ``uniform``, it means that the feature vectors are changing and re-simulated from the uniform distribution each round. When ``{gentype}`` is set as ``normal``, it means that the feature vectors are changing and re-simulated from the normal distribution each round.

For more commands and their details, please run 

python3 run.py -h
