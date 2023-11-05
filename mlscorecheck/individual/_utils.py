"""
This module implements the main check functionality for individual scores
"""

import itertools

import numpy as np

from ._pair_solutions import solution_specifications
from ..scores import (
    score_functions_without_complements,
    score_functions_standardized_without_complements,
    score_function_aliases,
    score_function_complements,
    score_specifications,
)
from ._interval import Interval, IntervalUnion
from ..core import NUMERICAL_TOLERANCE

__all__ = [
    "create_intervals",
    "create_problems_2",
    "resolve_aliases_and_complements",
    "is_less_than_zero",
    "is_zero",
    "unify_results",
]

solutions = solution_specifications
score_descriptors = score_specifications
supported_scores = {key[0] for key in solutions}.union({key[1] for key in solutions})
aliases = score_function_aliases
complementers = score_function_complements
functions = score_functions_without_complements
functions_standardized = score_functions_standardized_without_complements


def resolve_aliases_and_complements(scores: dict) -> dict:
    """
    Standardizing the scores by resolving aliases and complements

    Args:
        scores (dict(str,float)): the dictionary of scores

    Returns:
        dict(str,float): the resolved scores
    """
    aliased = {}
    for key, val in scores.items():
        if key in score_function_aliases:
            aliased[score_function_aliases[key]] = val
        else:
            aliased[key] = val

    complemented = {}
    for key, val in aliased.items():
        if key in score_function_complements:
            complemented[score_function_complements[key]] = 1.0 - val
        else:
            complemented[key] = val

    return complemented


def create_intervals(
    scores: dict, eps, numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    Turns the scores into intervals using the uncertainty specifications,
    the interval for a score will be (score - eps, score + eps).
    The score set is also standardized by replacing the aliases and replacing
    complements with the corresponding scores from the base set.

    Args:
        scores (dict): the scores to be turned into intervals
        eps (float|dict): the numerical uncertainty
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the score intervals
    """

    # turning the uncertainty into a score specific dictionary if it isnt that
    if not isinstance(eps, dict):
        eps = {key: eps for key in scores}

    # creating the intervals
    intervals = {
        key: Interval(
            val - eps[key] - numerical_tolerance, val + eps[key] + numerical_tolerance
        )
        for key, val in scores.items()
    }

    # trimming the intervals into the domains of the scores
    # for example, to prevent acc - eps < 0 implying a negative subinterval for
    # accuracy
    for key in intervals:
        if key in score_descriptors:
            lower_bound = score_descriptors[key].get("lower_bound", -np.inf)
            upper_bound = score_descriptors[key].get("upper_bound", np.inf)

            intervals[key] = intervals[key].intersection(
                Interval(lower_bound, upper_bound)
            )

    return intervals


def create_problems_2(scores: list) -> list:
    """
    Given a set of scores, this function generates all test case specifications.
    A test case specification consists of two base scores and a third score they
    are checked against.

    Args:
        list(str): the list of scores specified

    Returns:
        list(tuple): all possible triplets of the form (base0, base1, target)
    """
    bases = list(itertools.combinations(scores, 2))
    problems = []
    for base0, base1 in bases:
        problems.extend(
            (base0, base1, score) for score in scores if score not in {base0, base1}
        )

    return problems


def is_less_than_zero(value) -> bool:
    """
    Checks if the parameter is less than zero

    Args:
        value (numeric|Interval|IntervalUnion): the value to check

    Returns:
        bool: True if the parameter is less than zero, False otherwise
    """
    if not isinstance(value, (Interval, IntervalUnion)):
        return value < 0
    if isinstance(value, Interval):
        return value.upper_bound < 0
    return all(interval.upper_bound < 0 for interval in value.intervals)


def is_zero(value, tolerance: float = 1e-8) -> bool:
    """
    Checks if the parameter is zero

    Args:
        value (numeric|Interval|IntervalUnion): the value to check
        tolerance (float): the numerical tolerance of the check

    Returns:
        bool: True if the parameter is zero, False otherwise
    """
    if not isinstance(value, (Interval, IntervalUnion)):
        return abs(value) < tolerance
    return value.contains(0.0)


def unify_results(value_list):
    """
    Unifies the list of solutions

    Args:
        value_list (list): a list of solutions

    Returns:
        obj (list|IntervalUnion): the unified result
    """

    if len(value_list) == 0:
        return None
    if all(value is None for value in value_list):
        return None
    if not any(isinstance(value, (Interval, IntervalUnion)) for value in value_list):
        return [value for value in value_list if value is not None]

    intervals = []
    for interval in value_list:
        if isinstance(interval, Interval):
            intervals.append(interval)
        elif isinstance(interval, IntervalUnion):
            intervals.extend(interval.intervals)
        elif interval is not None:
            intervals.append(Interval(interval, interval))

    intu = IntervalUnion(intervals)

    return intu
