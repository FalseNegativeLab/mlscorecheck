"""
This module implements some utilities used by each data abstraction
"""

import string
import random

import numpy as np

__all__ = ['random_identifier',
            'check_bounds',
            'compare_scores',
            'create_bounds',
            'aggregated_scores']

aggregated_scores = ['acc', 'sens', 'spec', 'bacc']

def random_identifier(length):
    """
    Generating a random identifier

    Args:
        length (int): the length of the string identifier

    Returns:
        str: the identifier
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def check_bounds(scores, bounds, tolerance=1e-5):
    """
    Checks the bounds for the scores

    Args:
        scores (dict(str,float/int)): a dictionary of scores
        bounds (dict(str,tuple(float/int,float/int))): the dictionary of bounds

    Returns:
        None/bool: None if the bounds are not specified, otherwise a flag
                    if the scores are within the bounds
    """

    if bounds is None:
        return None

    flag = True
    for key in bounds:
        if bounds[key][0] is not None and not np.isnan(bounds[key][0]):
            flag = flag and (bounds[key][0]-tolerance <= scores[key])
        if bounds[key][1] is not None and not np.isnan(bounds[key][1]):
            flag = flag and (scores[key] <= bounds[key][1]+tolerance)

    return flag

def compare_scores(scores0, scores1, eps, subset=None, tolerance=1e-5):
    """
    Compares two sets of scores

    Args:
        scores0 (dict(str,float)): the first set of scores
        scores1 (dict(str,float)): the second set of scores
        eps (float/dict(str,float)): the eps value(s)
        subset (list(str)): the subset to compare
        tolerance (float): the additional tolerance for numerical uncertainty
    """
    if subset is not None:
        scores0 = {key: scores0[key] for key in subset}
        scores1 = {key: scores1[key] for key in subset}

    if not isinstance(eps, dict):
        eps = {key: eps for key in scores0}

    return all(abs(scores0[key] - scores1[key]) <= eps[key] + tolerance for key in scores0)

def create_bounds(scores, feasible=True):
    """
    Create bounds for scores depending on the feasibility flag

    Args:
        scores (dict(str,float)): the dictionary of scores to create bounds for
        feasible (bool): if True, the bounds will be feasible, otherwise non-feasible

    Returns:
        dict(str,tuple(float, float)): the bounds
    """
    if feasible:
        score_bounds = {key: (max(scores[key] - 2*1e-1, 0.0),
                                min(1.0, scores[key] + 2*1e-1)) for key in scores}
        return score_bounds

    score_bounds = {}
    for key in scores:
        if scores[key] > 0.5:
            score_bounds[key] = (0.0, max(scores[key] - 1*1e-3, 0))
        else:
            score_bounds[key] = (scores[key] + 1*1e-2, 1.0)

    return score_bounds
