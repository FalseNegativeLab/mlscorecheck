"""
This module tests the problem generator
"""

import numpy as np

from mlscorecheck.individual import (generate_problems,
                                calculate_scores,
                                round_scores,
                                generate_problem_and_scores)

def test_generate_problem_and_scores():
    """
    Testing the joint problem and score sampling
    """
    _, scores = generate_problem_and_scores(random_state=5,
                                            rounding_decimals=3)
    assert len(scores) > 0

    _, scores = generate_problem_and_scores(random_state=5,
                                            rounding_decimals=3,
                                            score_subset=['acc'])
    assert len(scores) == 1

def test_round_scores():
    """
    Testing the rounding of scores
    """

    assert round_scores({'dummy': 0.12345}) == {'dummy': 0.12345}
    assert round_scores({'dummy': 0.12345}, rounding_decimals=2)['dummy'] == np.round(0.12345, 2)

def test_calculate_all_scores():
    """
    Testing the calculation of all scores
    """

    scores = calculate_scores({'p': 10, 'tp': 5, 'n': 20, 'tn': 15, 'beta_plus': 2, 'beta_minus': 2})
    assert scores['acc'] == 20/30

def test_generate_problems():
    """
    Testing the problem generation
    """

    evaluation, _ = generate_problems(random_state=np.random.RandomState(5))

    evaluation, _ = generate_problems(n_problems=5)
    assert len(evaluation) == 5

    evaluation, _ = generate_problems(add_complements=True)
    assert 'fn' in evaluation

    evaluation, _ = generate_problems(zeros=['tp'])
    assert evaluation['tp'] == 0

    evaluation, _ = generate_problems(zeros=['tn'])
    assert evaluation['tn'] == 0

    evaluation, _ = generate_problems(zeros=['fp'])
    assert evaluation['tn'] == evaluation['n']

    evaluation, _ = generate_problems(zeros=['fn'])
    assert evaluation['tp'] == evaluation['p']
