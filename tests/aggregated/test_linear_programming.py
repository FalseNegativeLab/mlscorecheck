"""
Testing some linear programming capabilities
"""

import pulp as pl

from mlscorecheck.aggregated import (Evaluation,
                                    generate_evaluation,
                                    solve,
                                    add_bounds)

def test_solve():
    """
    Testing the fold functionality
    """

    evaluation = Evaluation(**generate_evaluation(random_state=5))
    sample = evaluation.sample_figures()

    scores = sample.calculate_scores()

    eps = {'acc': 0.001, 'sens': 0.001, 'spec': 0.001, 'bacc': 0.001}

    assert solve(evaluation, scores, eps).status == 1

def test_add_bounds():
    """
    Testing the add_bounds function
    """

    problem = pl.LpProblem('dummy')
    problem = add_bounds(problem, {'tn': pl.LpVariable('tn')}, {'tn': (0, 10)}, label='tn')
    assert len(problem.constraints) == 2

    problem = pl.LpProblem('dummy')
    problem = add_bounds(problem, {'tn': pl.LpVariable('tn')}, {'tn': (0, None)}, label='tn')
    assert len(problem.constraints) == 1

    problem = pl.LpProblem('dummy')
    problem = add_bounds(problem, {'tn': pl.LpVariable('tn')}, {'tn': (None, 10)}, label='tn')
    assert len(problem.constraints) == 1
