"""
This module brings together functionalities related to the systematic
calculation of scores
"""

import math
import numpy as np

from ..core import safe_call
from ..scores import score_functions_with_solutions
from ..scores import score_specifications

__all__ = ['round_scores',
            'calculate_scores',
            'calculate_scores_for_lp']

def round_scores(scores, rounding_decimals=None):
    """
    Rounds the scores in the dictionary

    Args:
        scores (dict): the dictionary of scores to round
        rounding_decimals (None|int): the decimal places to round to

    Returns:
        dict: a dictionary with the rounded scores
    """
    if rounding_decimals is None:
        return scores

    return {key: float(np.round(score, rounding_decimals))
                    for key, score in scores.items()}

def calculate_scores_for_lp(problem, score_subset=None):
    """
    Calculate scores for a linear programming problem

    Args:
        problem (dict): the raw figures tp, tn, p and n
        score_subset (list|None): the list of scores to compute

    Returns:
        dict(str,float): the calculated scores
    """

    scores = {'acc': (problem['tp'] + problem['tn']) * (1.0 / (problem['p'] + problem['n'])),
            'sens': (problem['tp']) * (1.0 / problem['p']),
            'spec': (problem['tn']) * (1.0 / problem['n']),
            'bacc': ((problem['tp'] * (1.0 / problem['p'])) \
                        + (problem['tn'] * (1.0 / problem['n']))) / 2}

    if score_subset is None:
        return scores
    return {key: scores[key] for key in score_subset}

def calculate_scores(problem,
                    *,
                    rounding_decimals=None,
                    additional_symbols=None):
    """
    Calculates all scores with solutions

    Args:
        problem (dict): a problem to calculate the scores for (containing 'p', 'n', 'tp', 'tn')
        rounding_decimals (None|int): the decimal places to round to
        additional_symbols (None|dict): additional symbols for the substitution

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
                                {**problem, **additional, **additional_symbols},
                                score_specifications[score].get('nans'))
                for score, function in score_functions_with_solutions.items()}

    results = round_scores(results, rounding_decimals)

    return results
