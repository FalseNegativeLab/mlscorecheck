"""
This module tests the score calculation capabilities
"""

import numpy as np
import pytest

from mlscorecheck.scores import (
    calculate_multiclass_scores,
    calculate_scores,
    calculate_scores_for_lp,
    score_functions_with_solutions,
)


def test_calculate_scores():
    """
    Testing the score calculation
    """

    scores = calculate_scores(
        {"p": 40, "n": 20, "tp": 34, "tn": 13, "beta_positive": 2, "beta_negative": 2}
    )
    assert len(scores) == len(score_functions_with_solutions)


def test_calculate_scores_for_lp():
    """
    Testing the score calculation for linear programming
    """
    scores = calculate_scores_for_lp({"p": 40, "n": 20, "tp": 34, "tn": 13})
    assert len(scores) == 4

    scores = calculate_scores_for_lp({"p": 0, "n": 20, "tp": 0, "tn": 13})
    assert len(scores) == 2

    scores = calculate_scores_for_lp({"p": 4, "n": 0, "tp": 3, "tn": 0})
    assert len(scores) == 2


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_calculate_multiclass_scores(average):
    """
    Testing the calculation of multiclass scores

    Args:
        average (str): the mode of averaging
    """
    confm = np.ndarray([[5, 8, 3], [3, 10, 2], [2, 4, 11]])
    scores = calculate_multiclass_scores(
        confusion_matrix=confm,
        additional_symbols={"beta_positive": 2, "beta_negative": 2},
        average=average,
    )
    assert len(scores) > 0
