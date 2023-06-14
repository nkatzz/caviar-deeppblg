## Introduction

The whole point of this repository is to apply DeepProblog to CAVIAR an event recognition dataset. The many things 
that have to be done stem from the fact that a naive encoding of an Event Calculus interpeter in Problog results in 
unscalable inference and therefore learning.

```prolog 
holdsAt(Video, ComplexEvent, T) :- previous(T, T1), initiatedAt(Video, ComplexEvent, T1).
holdsAt(Video, ComplexEvent, T) :- holdsAt(Video, ComplexEvent, T1), previous(T, T1), 
    \+terminatedAt(Video, ComplexEvent, T1).
```

This recursive definition of inertia is problematic because the Problog compiler when queried about consecutive time 
steps, i.e. 1, 2, 3, 4, 5 will start building circuits to infer the query probability. These circuits start getting 
unbearably large very soon because each consecutive circuit contains the previous as a subcircuit. This all follow from the 
recursive definition of the inertia law in Event Calculus.

| Time | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| `holdsAt(tensor(train, 0), meeting(p1, p2), T)` | 0 | 1 | 24 | 112 | 245 | 604 | 1304 | 3590 | 7955 | 25774 | 67106 | 199023 | 462658 |

This can also be shown by vizualizing the circuits but this will be done later.

## Changes

The trick is to query the probabilities of complex events in time step T and then use them at T + 1. Whether this can allow the gradients to flow 
back is not certain yet but is maybe possible. This has involved:
- Loading the CAVIAR data and building DeepProblog compatible datasets
- Defining simple LSTM networks 
- Modifying the problog `Model` class in order to dynamically add facts representing the predictions of the previous step to the solver

## Installation
If you want to install everything yourself you should install the torch pipeline seperately and make sure it works. Then install DeepProblog. Consult `pyproject.toml` for necessary packages.
If you use poetry just use the pyproject.toml specification of the environment to install it. 


## Running the code
You can run the code with `PYTHONPATH="." python caviar_deepproblog train.py'
