"""
A function used to test the output of the linear programming
"""

import warnings

from mlscorecheck.aggregated import compare_scores

__all__ = ['evaluate_timeout']

def evaluate_timeout(result, problem, scores, eps, score_subset):
    """
    Evaluate the stopped or succeeded tests

    Args:
        result (pl.LpProblem): the executed problem
        problem (Experiment): the problem to be solved
        scores (dict(str,float)): the scores to match
        eps (float): the tolerance
        score_subset (list): the score subset to use
    """
    if result.status == 1:
        populated = problem.populate(result)

        assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
        assert populated.check_bounds()['bounds_flag'] is True
    else:
        warnings.warn('test timed out')
