"""
This module brings together functionalities related to the systematic
calculation of scores
"""

import math

import numpy as np

from ..core import safe_call, round_scores, logger
from ._score_bundles import score_functions_with_solutions
from ._score_bundles import score_specifications
from ._multiclass_scores import multiclass_score_map

__all__ = ['calculate_scores',
            'calculate_scores_for_lp',
            'calculate_multiclass_scores']

def calculate_scores_for_lp(problem: dict, score_subset: list = None) -> dict:
    """
    Calculate scores for a linear programming problem

    Args:
        problem (dict): the raw figures tp, tn, p and n
        score_subset (list|None): the list of scores to compute

    Returns:
        dict(str,float): the calculated scores
    """
    if score_subset is None:
        score_subset = ['acc', 'sens', 'spec', 'bacc']

    scores = {}

    if 'acc' in score_subset:
        scores['acc'] = (problem['tp'] + problem['tn']) * (1.0 / (problem['p'] + problem['n']))
    if 'sens' in score_subset:
        if problem['p'] > 0:
            scores['sens'] = (problem['tp']) * (1.0 / problem['p'])
        else:
            logger.info('sens cannot be computed since p (%d) is zero', problem["p"])
    if 'spec' in score_subset:
        if problem['n'] > 0:
            scores['spec'] = (problem['tn']) * (1.0 / problem['n'])
        else:
            logger.info('spec cannot be computed since n (%d) is zero', problem["n"])
    if 'bacc' in score_subset:
        if problem['p'] > 0 and problem['n'] > 0:
            scores['bacc'] = ((problem['tp'] * (1.0 / problem['p'])) \
                            + (problem['tn'] * (1.0 / problem['n']))) / 2
        else:
            logger.info('bacc cannot be computed since p (%d) or n (%d) is zero',
                        problem['p'],
                        problem['n'])

    return scores

def calculate_scores(problem: dict,
                    *,
                    rounding_decimals: int = None,
                    additional_symbols: dict = None,
                    subset: list = None) -> dict:
    """
    Calculates all scores with solutions

    Args:
        problem (dict): a problem to calculate the scores for (containing 'p', 'n', 'tp', 'tn')
        rounding_decimals (None|int): the decimal places to round to
        additional_symbols (None|dict): additional symbols for the substitution
        subset (None|list): the subset of scores to calculate

    Returns:
        dict: the calculated scores
    """
    if additional_symbols is None:
        additional_symbols = {'sqrt': math.sqrt}

    additional = {}

    if 'fp' not in problem:
        additional['fp'] = problem['n'] - problem['tn']
    if 'fn' not in problem:
        additional['fn'] = problem['p'] - problem['tp']

    results = {score: safe_call(function,
                                problem | additional | additional_symbols,
                                score_specifications[score].get('nans'))
                for score, function in score_functions_with_solutions.items()
                if subset is None or score in subset}

    results = round_scores(results, rounding_decimals)

    return results

def calculate_multiclass_scores(confusion_matrix: np.array,
                                average=None,
                                *,
                                rounding_decimals: int = None,
                                additional_symbols: dict = None,
                                subset: list = None) -> dict:
    """
    Calculates all scores with solutions

    Args:
        confusion_matrix (np.array): the confusion matrix to calculate scores for
        average (str): the mode of averaging ('macro'/'micro'/'weighted')
        rounding_decimals (None|int): the decimal places to round to
        additional_symbols (None|dict): additional symbols for the substitution
        subset (None|list): the subset of scores to calculate

    Returns:
        dict: the calculated scores
    """
    additional_symbols = {} if additional_symbols is None else additional_symbols
    params = {'confusion_matrix': confusion_matrix,
                'average': average} | additional_symbols
    results = {score: safe_call(function, params)
                for score, function in multiclass_score_map.items()
                if subset is None or score in subset}

    results = round_scores(results, rounding_decimals)

    return results
