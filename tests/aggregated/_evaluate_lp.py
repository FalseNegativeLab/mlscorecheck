"""
A function used to test the output of the linear programming
"""

import warnings

import pulp as pl

from mlscorecheck.aggregated import Experiment, compare_scores

__all__ = ["evaluate_timeout"]


def evaluate_timeout(
    result: pl.LpProblem, problem: Experiment, scores: dict, eps, score_subset: list
):
    """
    Evaluate the stopped or succeeded tests

    Args:
        result (pl.LpProblem): the executed problem
        problem (Experiment): the problem to be solved
        scores (dict(str,float)): the scores to match
        eps (float|dict(str,float)): the tolerance
        score_subset (list): the score subset to use
    """
    if result.status == 1:
        populated = problem.populate(result)

        assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
        assert populated.check_bounds()["bounds_flag"] is True
    elif result.status == 0:
        warnings.warn("test timed out")
