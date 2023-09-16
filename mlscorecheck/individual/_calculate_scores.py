"""
This module brings together functionalities related to the systematic
calculation of scores
"""

import math

from ..core import safe_call, round_scores
from ..scores import score_functions_with_solutions
from ..scores import score_specifications

__all__ = ['calculate_scores',
            'calculate_scores_for_lp']

def calculate_scores_for_lp(problem: dict, score_subset: list = None) -> dict:
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

    return scores if score_subset is None else {key: scores[key] for key in score_subset}

def calculate_scores(problem: dict,
                    *,
                    rounding_decimals: int = None,
                    additional_symbols: dict = None) -> dict:
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
