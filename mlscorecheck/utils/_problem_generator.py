"""
This module implements a random problem generator
"""

import numpy as np

from ..core import (determine_fold_configurations,
                    score_functions_with_solutions,
                    safe_call,
                    score_functions_standardized_without_complements)

__all__ = ['generate_problem',
            'generate_n_problems',
            'calculate_scores_rom',
            'calculate_scores_mor',
            'generate_folding_problem',
            'calculate_score_mor_grouped',
            'calculate_all_scores']

def calculate_all_scores(problem, *, rounding_decimals=None, additional_symbols=None):
    if additional_symbols is None:
        additional_symbols = {'sqrt': np.sqrt}

    results = {score: safe_call(function, {**problem, **additional_symbols}) for score, function in score_functions_with_solutions.items()}

    if rounding_decimals is not None:
        for score in results:
            results[score] = np.round(results[score], rounding_decimals)

    return results

def generate_problem(*,
                        max_p=1000,
                        max_n=1000,
                        zeros=None,
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
    if zeros is None:
        zeros = []

    if random_seed is None or isinstance(random_seed, int):
        random_state = np.random.RandomState(random_seed)
    else:
        random_state = random_seed

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

    fn = p - tp
    fp = n - tn

    return {'p': p, 'n': n, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def generate_n_problems(n_problems,
                        *,
                        max_p=1000,
                        max_n=1000,
                        random_seed=None):

    random_state = np.random.RandomState(random_seed)

    return [generate_problem(max_p=max_p, max_n=max_n, random_seed=random_state)
            for _ in range(n_problems)]

def generate_folding_problem(n_repeats=1,
                                n_folds=5,
                                max_p=1000,
                                max_n=1000,
                                random_seed=None):
    if random_seed is None or isinstance(random_seed, int):
        random_state = np.random.RandomState(random_seed)
    else:
        random_state = random_seed

    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)

    folds = determine_fold_configurations(p, n, min(n_folds, p, n), n_repeats)

    for fold in folds:
        fold['tp'] = random_state.randint(fold['p'])
        fold['tn'] = random_state.randint(fold['n'])

    return folds, {'p': p, 'n': n, 'n_folds': n_folds, 'n_repeats': n_repeats}

def calculate_scores_mor(problems,
                            rounding_decimals=None):
    acc = [(problem['tp'] + problem['tn'])/(problem['p'] + problem['n'])
            for problem in problems]
    sens = [(problem['tp'])/(problem['p'])
            for problem in problems]
    spec = [(problem['tn'])/(problem['n'])
            for problem in problems]
    bacc = [(problem['tp']/problem['p'] + problem['tn']/problem['n'])/2
            for problem in problems]

    results = {'acc': np.mean(acc),
                'sens': np.mean(sens),
                'spec': np.mean(spec),
                'bacc': np.mean(bacc)}

    if rounding_decimals is not None:
        for key, value in results.items():
            results[key] = np.round(value, rounding_decimals)

    return results

def calculate_scores_rom(problems,
                            rounding_decimals=None,
                            mor=False,
                            additional_symbols=None):
    if additional_symbols is None:
        additional_symbols = {'sqrt': np.sqrt}

    tp = np.sum(problem['tp'] for problem in problems)
    tn = np.sum(problem['tn'] for problem in problems)
    p = np.sum(problem['p'] for problem in problems)
    n = np.sum(problem['n'] for problem in problems)

    tmp = {'tp': tp,
            'tn': tn,
            'fn': p - tp,
            'fp': n - tn,
            'p': p,
            'n': n}

    if mor:
        scores = ['acc', 'sens', 'spec', 'bacc']
    else:
        scores = list(score_functions_with_solutions.keys())

    results = {}

    for score in scores:
        results[score] = safe_call(score_functions_with_solutions[score], {**tmp, **additional_symbols})

    if rounding_decimals is not None:
        for key, value in results.items():
            results[key] = np.round(value, rounding_decimals)

    return results

def calculate_score_mor_grouped(problems,
                                rounding_decimals=None):
    scores = [calculate_scores_mor(problem) for problem in problems]
    final_scores = scores[0]
    for score in scores[1:]:
        for sc in ['acc', 'bacc', 'sens', 'spec']:
            final_scores[sc] += score[sc]

    for sc in ['acc', 'bacc', 'sens', 'spec']:
        final_scores[sc] = final_scores[sc] / len(problems)

    if rounding_decimals is not None:
        for key, value in final_scores.items():
            final_scores[key] = np.round(value, rounding_decimals)

    return final_scores
