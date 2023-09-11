"""
This module tests the score calculation capabilities
"""

from mlscorecheck.individual import calculate_scores, calculate_scores_for_lp
from mlscorecheck.scores import score_functions_with_solutions

def test_calculate_scores():
    """
    Testing the score calculation
    """

    scores = calculate_scores({'p': 40, 'n': 20, 'tp': 34, 'tn': 13,
                                'beta_positive': 2, 'beta_negative': 2})
    assert len(scores) == len(score_functions_with_solutions)

def test_calculate_scores_for_lp():
    """
    Testing the score calculation for linear programming
    """
    scores = calculate_scores_for_lp({'p': 40, 'n': 20, 'tp': 34, 'tn': 13})
    assert len(scores) == 4
