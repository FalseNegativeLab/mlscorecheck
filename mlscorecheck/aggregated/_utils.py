"""
This module implements some utilities used by each data abstraction
"""

import string
import random

import numpy as np

__all__ = ["random_identifier", "check_bounds", "compare_scores", "aggregated_scores"]

aggregated_scores = ["acc", "sens", "spec", "bacc"]


def random_identifier(length: int):
    """
    Generating a random identifier

    Args:
        length (int): the length of the string identifier

    Returns:
        str: the identifier
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def check_bounds(scores: dict, bounds: dict = None, tolerance: float = 1e-5) -> bool:
    """
    Checks the bounds for the scores

    Args:
        scores (dict(str,float|int)): a dictionary of scores
        bounds (dict(str,tuple(float|int,float|int))): the dictionary of bounds
        tolerance (float): the tolerance for the check

    Returns:
        None/bool: None if the bounds are not specified, otherwise a flag
        if the scores are within the bounds
    """

    if bounds is None:
        return None

    flag = True
    for key in bounds:
        if key in scores:
            if bounds[key][0] is not None and not np.isnan(bounds[key][0]):
                flag = flag and (bounds[key][0] - tolerance <= scores[key])
            if bounds[key][1] is not None and not np.isnan(bounds[key][1]):
                flag = flag and (scores[key] <= bounds[key][1] + tolerance)

    return flag


def compare_scores(
    scores0: dict, scores1: dict, eps, subset: list = None, tolerance: float = 1e-5
):
    """
    Compares two sets of scores

    Args:
        scores0 (dict(str,float)): the first set of scores
        scores1 (dict(str,float)): the second set of scores
        eps (float|dict(str,float)): the eps value(s)
        subset (list(str)): the subset to compare
        tolerance (float): the additional tolerance for numerical uncertainty
    """
    if subset is not None:
        scores0 = {key: scores0[key] for key in subset}
        scores1 = {key: scores1[key] for key in subset}

    if not isinstance(eps, dict):
        eps = {key: eps for key in scores0}

    return all(
        abs(scores0[key] - scores1[key]) <= eps[key] + tolerance
        for key in scores1
        if key in scores0
    )
