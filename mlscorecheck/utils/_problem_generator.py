"""
This module implements a random problem generator
"""

import numpy as np

from ..core import (determine_fold_configurations,
                    score_functions_with_solutions,
                    score_specifications,
                    safe_call)

__all__ = ['generate_problems',
            'calculate_scores_rom',
            'calculate_scores_mor',
            'generate_problems_with_folds',
            'calculate_scores',
            'calculate_all_scores',
            'problem_structure_depth',
            'round_scores']

def round_scores(scores, rounding_decimals=None):
    if rounding_decimals is None:
        return scores

    return {key: np.round(score, rounding_decimals) for key, score in scores.items()}

def calculate_all_scores(problem, *, rounding_decimals=None, additional_symbols=None, scores_only=False):
    if additional_symbols is None:
        additional_symbols = {'sqrt': np.sqrt}

    additional = {}

    if 'fp' not in problem:
        additional['fp'] = problem['n'] - problem['tn']
    if 'fn' not in problem:
        additional['fn'] = problem['p'] - problem['tp']

    results = {score: safe_call(function,
                                {**problem, **additional, **additional_symbols},
                                score_specifications[score].get('nans')) for score, function in score_functions_with_solutions.items()}

    results = round_scores(results, rounding_decimals)

    if scores_only:
        return results

    return {**results, **problem}

def problem_structure_depth(problem):
    if isinstance(problem, dict):
        return 0

    return problem_structure_depth(problem[0]) + 1

def calculate_scores_rom(problem):
    tp = 0
    tn = 0
    p = 0
    n = 0
    for subproblem in problem:
        tp += subproblem['tp']
        tn += subproblem['tn']
        p += subproblem['p']
        n += subproblem['n']

    return calculate_all_scores({'tp': tp,
                                    'tn': tn,
                                    'p': p,
                                    'n': n})

def calculate_scores_mor(problem):
    keys = ['acc', 'sens', 'spec', 'bacc']
    result = {key: 0.0 for key in keys}

    for subproblem in problem:
        for key in keys:
            result[key] += subproblem[key]

    for key in keys:
        result[key] = result[key] / len(problem)

    return result

def calculate_scores(problem,
                        strategy=None,
                        rounding_decimals=None,
                        scores_only=True):
    depth = problem_structure_depth(problem)

    if depth == 0:
        if ('tp' in problem):
            return calculate_all_scores(problem, scores_only=scores_only)
        else:
            return problem

    if isinstance(strategy, str):
        strategy = [strategy] * depth

    if (strategy is None and depth > 0) or (strategy is not None and len(strategy) != depth):
        raise ValueError('problem structure does not match the strategies')

    results = [calculate_scores(subproblem, strategy[1:], scores_only=False) for subproblem in problem]

    if strategy[0] == 'rom':
        scores = calculate_scores_rom(results)
    elif strategy[0] == 'mor':
        scores = calculate_scores_mor(results)
    else:
        raise ValueError(f'Unknown strategy {strategy}')

    if scores_only:
        scores = {key: value for key, value in scores.items()
                    if key not in ['p', 'n', 'tp', 'tn', 'fp', 'fn']}

    return round_scores(scores, rounding_decimals)

def generate_problems(*,
                        n_problems=1,
                        max_p=1000,
                        max_n=1000,
                        zeros=None,
                        add_complements=False,
                        random_seed=None):
    """
    Generates a random problem

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        zeros (None/list): the list of items to set to zero
        random_seed (int, optional): the random seed to use

    Returns:
        dict: the problem
    """
    if random_seed is None or isinstance(random_seed, int):
        random_state = np.random.RandomState(random_seed)
    else:
        random_state = random_seed

    if n_problems > 1:
        return [generate_problems(max_p=max_p,
                                    max_n=max_n,
                                    zeros=zeros,
                                    random_seed=random_state) for _ in range(n_problems)]

    if zeros is None:
        zeros = []

    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)

    if 'tp' in zeros:
        tp = 0
    elif 'fn' in zeros:
        tp = p
    else:
        tp = random_state.randint(1, p)

    if 'tn' in zeros:
        tn = 0
    elif 'fp' in zeros:
        tn = n
    else:
        tn = random_state.randint(1, n)

    result = {'p': p, 'n': n, 'tp': tp, 'tn': tn}

    if add_complements:
        result['fn'] = p - tp
        result['fp'] = n - tn

    return result

def generate_problems_with_folds(n_problems=1,
                                n_repeats=1,
                                n_folds=5,
                                max_p=1000,
                                max_n=1000,
                                random_seed=None,
                                folding='stratified_sklearn'):
    if random_seed is None or isinstance(random_seed, int):
        random_state = np.random.RandomState(random_seed)
    else:
        random_state = random_seed

    if n_problems > 1:
        folds, problems = list(zip(*[generate_problems_with_folds(n_repeats=n_repeats,
                                                        n_folds=n_folds,
                                                        max_p=max_p,
                                                        max_n=max_n,
                                                        random_seed=random_state,
                                                        folding=folding)
                            for _ in range(n_problems)]))
        folds, problems = list(folds), list(problems)

        return list(folds), list(problems)

    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)

    n_folds = min(n_folds, p, n)

    folds = determine_fold_configurations(p, n, n_folds, n_repeats, folding)

    for fold in folds:
        fold['tp'] = random_state.randint(1, fold['p'])
        fold['tn'] = random_state.randint(1, fold['n'])

    if len(folds) == 1:
        folds = folds[0]

    return folds, {'p': p, 'n': n, 'n_folds': n_folds, 'n_repeats': n_repeats}
