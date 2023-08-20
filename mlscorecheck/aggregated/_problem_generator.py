"""
This module implements the random problem generator
"""

import copy

import numpy as np

from ._folds import (determine_fold_configurations,
                        random_configurations)

__all__ = ['generate_1_problem_with_folds',
            'generate_problems_with_folds']

def generate_1_problem_with_folds(*,
                                    n_repeats=1,
                                    n_folds=1,
                                    random_repeats=True,
                                    random_folds=True,
                                    max_p=1000,
                                    max_n=1000,
                                    random_state=None,
                                    folding='stratified_sklearn'):
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        random_state = random_state

    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)

    n_folds = min(n_folds, p, n)

    if random_folds and n_folds > 1:
        n_folds = random_state.randint(1, n_folds+1)
    if random_repeats and n_repeats > 1:
        n_repeats = random_state.randint(1, n_repeats+1)

    if folding == 'random':
        folds = random_configurations(p, n, n_folds, n_repeats)
        problem = {'p': p, 'n': n, 'folds': copy.deepcopy(folds)}

        for fold in folds:
            fold['tp'] = random_state.randint(1, fold['p'])
            fold['tn'] = random_state.randint(1, fold['n'])

        problem_with_figures = {'p': p, 'n': n, 'folds': folds}


    elif folding == 'stratified_sklearn':
        folds = determine_fold_configurations(p, n, n_folds, n_repeats, folding)
        problem = {'p': p, 'n': n, 'n_folds': n_folds, 'n_repeats': n_repeats}

        for fold in folds:
            fold['tp'] = random_state.randint(1, fold['p'])
            fold['tn'] = random_state.randint(1, fold['n'])

        problem_with_figures = {'p': p, 'n': n, 'folds': folds}

    return problem_with_figures, problem

def generate_problems_with_folds(n_problems=1,
                                n_repeats=1,
                                n_folds=5,
                                random_repeats=True,
                                random_folds=True,
                                max_p=1000,
                                max_n=1000,
                                random_seed=None,
                                folding='stratified_sklearn'):
    if random_seed is None or isinstance(random_seed, int):
        random_state = np.random.RandomState(random_seed)
    else:
        random_state = random_seed

    with_figures, problems = list(zip(*[generate_1_problem_with_folds(n_repeats=n_repeats,
                                                    n_folds=n_folds,
                                                    random_repeats=random_repeats,
                                                    random_folds=random_folds,
                                                    max_p=max_p,
                                                    max_n=max_n,
                                                    random_state=random_state,
                                                    folding=folding)
                        for _ in range(n_problems)]))
    with_figures, problems = list(with_figures), list(problems)

    return list(with_figures), list(problems)
