"""
This module tests the linear programming functionalities
"""

from mlscorecheck.aggregated import (compare_scores,
                                        random_identifier,
                                        check_bounds)

def test_random_identifier():
    """
    Testing the random identifier
    """

    assert len(random_identifier(16)) == 16
    assert random_identifier(10) != random_identifier(10)

def test_check_bounds():
    """
    Testing the check bounds function
    """
    bounds = {'acc': [0.5, 1.0],
                'sens': [0.0, 0.5]}

    assert check_bounds({'acc': 0.7, 'sens': 0.2}, bounds)
    assert not check_bounds({'acc': 0.2, 'sens': 0.8}, bounds)
    assert check_bounds({}, None) is None

def test_compare_scores():
    """
    Testing the score comparison
    """

    scores0 = {'acc': 0.1}
    scores1 = {'acc': 0.1001}

    assert compare_scores(scores0, scores1, eps=0.0, subset=['acc'], tolerance=1e-4)
    assert not compare_scores(scores0, scores1, eps=0.0, subset=['acc'], tolerance=1e-5)
