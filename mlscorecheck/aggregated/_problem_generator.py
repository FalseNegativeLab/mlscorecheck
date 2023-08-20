"""
This module implements the random problem generator
"""

import copy

import numpy as np

from ._folds import (determine_fold_configurations,
                        random_configurations,
                        _expand_datasets)
from ._calculate_scores import (calculate_scores_datasets)

__all__ = ['generate_1_problem_with_folds',
            'generate_problems_with_folds',
            'add_bounds_to_problems']

def add_bounds_to_problems(problems,
                            problems_with_figures,
                            strategy=('mor', 'mor'),
                            bound_strategy=('min', 'min'),
                            bounds=('score', 'tptn')):
    problems = _expand_datasets(problems)

    scores, evaluations = calculate_scores_datasets(problems_with_figures,
                                            strategy=strategy,
                                            return_populated=True)
    min_scores = {'acc': 1, 'sens': 1, 'spec': 1, 'bacc': 1}
    min_tptn = {'tp': np.inf, 'tn': np.inf}

    for dataset in evaluations:
        for fold in dataset['folds']:
            for key in min_scores:
                if fold[key] < min_scores[key]:
                    min_scores[key] = fold[key]
            for key in min_tptn:
                if fold[key] < min_tptn[key]:
                    min_tptn[key] = fold[key]

    score_bounds = {key: (min_scores[key], 1) for key in min_scores}
    tptn_bounds = {key: (min_tptn[key], 1000000) for key in min_tptn}

    for dataset in problems:
        for fold in dataset['folds']:
            fold['score_bounds'] = copy.deepcopy(score_bounds)
            fold['tptn_bounds'] = copy.deepcopy(tptn_bounds)
        dataset['score_bounds'] = copy.deepcopy(score_bounds)
        dataset['tptn_bounds'] = copy.deepcopy(tptn_bounds)

    return problems

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
