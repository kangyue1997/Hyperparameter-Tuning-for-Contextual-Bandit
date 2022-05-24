# Hyperparameter-Tuning-for-Contextual-Bandit

## Requirements

To run the code, you will need:
Python3, Numpy, Matplotlib, itertools

## Commands

python3 run.py -algo {algorithm} -gentype {dataset}  

In the above command, replace ``{algorithm}`` with ``linucb``, ``lints`` to get the results for different algorithms. Replace ``{distribution}`` with ``uniform`` or ``uniform_fixed`` to run the experiments for the two scenarios we mention in the paper. When ``dist`` is set as ``uniform``, it means that the feature vectors are changing and re-simulated from the uniform distribution each round. When ``dist`` is set as ``uniform_fixed``, it means that the feature vectors are fixed for all rounds.
