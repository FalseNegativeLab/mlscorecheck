"""
This module tests all solutions
"""

import numpy as np

import pytest

import numbers

from mlscorecheck.utils import (generate_problem,
                                generate_problem_tp0)
from mlscorecheck.core import *

score_functions = score_function_set()

to_test = [globals()[key] for key in globals() if key.startswith('solve_')]

def evaluate_results(results, tp, tn):
    if len(results) == 0:
        return True

    diffs = []

    for result in results:
        diffs.append((abs(result['tp'] - tp) + abs(result['tn'] - tn)) < 1e-4)

    return any(diffs)

@pytest.mark.parametrize("solution", to_test)
def test_solution(solution):
    """
    Testing a solution with scalars
    """

    problem = generate_problem()

    args = solution.__code__.co_varnames[:solution.__code__.co_kwonlyargcount]
    score0 = args[0]
    score1 = args[1]

    tmp_problem = {**problem}

    args = score_functions[score0].__code__.co_varnames[:score_functions[score0].__code__.co_kwonlyargcount]
    tmp_problem[score0] = score_functions[score0](**{key: tmp_problem[key] for key in args})

    args = score_functions[score1].__code__.co_varnames[:score_functions[score1].__code__.co_kwonlyargcount]
    tmp_problem[score1] = score_functions[score1](**{key: tmp_problem[key] for key in args})

    args = solution.__code__.co_varnames[:solution.__code__.co_kwonlyargcount]
    results = solution(**{key: tmp_problem[key] for key in args})

    assert evaluate_results(results, tmp_problem['tp'], tmp_problem['tn'])

@pytest.mark.parametrize("solution", to_test)
def test_solution_tp0(solution):
    """
    Testing a solution with tp = 0
    """

    problem = generate_problem_tp0()

    args = solution.__code__.co_varnames[:solution.__code__.co_kwonlyargcount]
    score0 = args[0]
    score1 = args[1]

    tmp_problem = {**problem}

    try:
        args = score_functions[score0].__code__.co_varnames[:score_functions[score0].__code__.co_kwonlyargcount]
        tmp_problem[score0] = score_functions[score0](**{key: tmp_problem[key] for key in args})

        args = score_functions[score1].__code__.co_varnames[:score_functions[score1].__code__.co_kwonlyargcount]
        tmp_problem[score1] = score_functions[score1](**{key: tmp_problem[key] for key in args})
    except ZeroDivisionError as exc:
        assert True
        return

    args = solution.__code__.co_varnames[:solution.__code__.co_kwonlyargcount]

    results = solution(**{key: tmp_problem[key] for key in args})

    assert evaluate_results(results, tmp_problem['tp'], tmp_problem['tn'])
