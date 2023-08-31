"""
This module implements the random problem generator
"""

import copy

import numpy as np

from ._folds import (determine_fold_configurations,
                        random_configurations)

__all__ = ['generate_1_problem_with_evaluations',
            'generate_problems_with_evaluations',
            #'add_bounds_to_problems'
            ]

def generate_1_problem_with_evaluations(*,
                                    n_repeats=1,
                                    n_folds=1,
                                    random_repeats=True,
                                    random_folds=True,
                                    max_p=1000,
                                    max_n=1000,
                                    random_state=None,
                                    folding='stratified_sklearn'):
    """
    Generate 1 aggregated problem with random tp and tn scores

    Args:
        n_repeats (int): the number of repetitions
        n_folds (int): the number of folds
        random_repeats (bool): whether to consider n_repeats as the upper bound for randomization
        random_folds (bool): whether to consider n_folds as the upper bound for randomization
        max_p (int): the maximum p value
        max_n (int): the maximum n_value
        random_state (None|int/np.random.RandomState): the random state or seed to be used
        folding (str): 'stratified_sklearn'/'random' - the folding strategy

    Returns:
        dict, dict: the problem with evaluations (random tp/tn scores) and the raw problem
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # sampling p and n
    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)

    # sampling the number of folds and the number of repeats
    n_folds = min(n_folds, p, n)

    if random_folds and n_folds > 1:
        n_folds = random_state.randint(1, n_folds+1)
    if random_repeats and n_repeats > 1:
        n_repeats = random_state.randint(1, n_repeats+1)

    # determining the fold configurations
    if folding == 'random':
        folds = random_configurations(p, n, n_folds, n_repeats)
        problem = {'p': p, 'n': n, 'folds': copy.deepcopy(folds)}

    elif folding == 'stratified_sklearn':
        folds = determine_fold_configurations(p, n, n_folds, n_repeats, folding)
        problem = {'p': p, 'n': n,
                    'n_folds': n_folds, 'n_repeats': n_repeats,
                    'folding': folding}

    # populating the folds with the random tp and tn values
    for fold in folds:
        fold['tp'] = random_state.randint(1, fold['p'])
        fold['tn'] = random_state.randint(1, fold['n'])

    # assembling the output
    problem_with_figures = {'p': p, 'n': n, 'folds': folds}

    return problem_with_figures, problem

def generate_problems_with_evaluations(n_problems=1,
                                *,
                                n_repeats=1,
                                n_folds=5,
                                random_repeats=True,
                                random_folds=True,
                                max_p=1000,
                                max_n=1000,
                                random_state=None,
                                folding='stratified_sklearn'):
    """
    Generate multiple aggregated problem with random tp and tn scores

    Args:
        n_problems (int): the number of problems to generate
        n_repeats (int): the number of repetitions
        n_folds (int): the number of folds
        random_repeats (bool): whether to consider n_repeats as the upper bound for randomization
        random_folds (bool): whether to consider n_folds as the upper bound for randomization
        max_p (int): the maximum p value
        max_n (int): the maximum n_value
        random_state (None|int/np.random.RandomState): the random state or seed to be used
        folding (str): 'stratified_sklearn'/'random' - the folding strategy

    Returns:
        list(dict), list(dict): the lists of problem with evaluations (random tp/tn scores) and
                                the raw problems
    """

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    with_figures, problems = list(zip(*[generate_1_problem_with_evaluations(n_repeats=n_repeats,
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
