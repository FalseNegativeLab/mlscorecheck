"""
This module tests the problem generator
"""

import pytest

import numpy as np

from mlscorecheck.utils import (generate_problems,
                                generate_problems_with_folds,
                                problem_structure_depth,
                                calculate_scores,
                                calculate_all_scores,
                                calculate_scores_rom,
                                calculate_scores_mor,
                                round_scores)

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

    scores = calculate_all_scores({'p': 10, 'tp': 5, 'n': 20, 'tn': 15})
    assert scores['acc'] == 20/30

    scores = calculate_all_scores({'p': 10, 'tp': 5, 'n': 20, 'tn': 15}, scores_only=True)
    assert 'tp' not in scores

def test_problem_structure_depth():
    """
    Testing the determination of the problem structure depth
    """

    assert problem_structure_depth({'a': 1}) == 0
    assert problem_structure_depth([{'a': 1}]) == 1
    assert problem_structure_depth([[{'a': 1}]]) == 2

def test_calculate_scores_rom():
    """
    Testing the calculation of scores in RoM manner
    """

    scores = calculate_scores_rom([{'p': 10, 'n': 20, 'tp': 5, 'tn': 8},
                                    {'p': 12, 'n': 26, 'tp': 5, 'tn': 8}])

    assert scores['sens'] == 10/22

def test_calculate_scores_mor():
    """
    Testing the calculation of scores in the MoR manner
    """

    scores = calculate_scores_mor([calculate_all_scores({'p': 10, 'n': 20, 'tp': 5, 'tn': 8}),
                                    calculate_all_scores({'p': 12, 'n': 26, 'tp': 5, 'tn': 8})])

    assert scores['sens'] == (5/10 + 5/12) / 2.0

def test_calculate_scores():
    """
    Testing the score calculation
    """

    assert calculate_scores({'dummy': 1}) == {'dummy': 1}

    with pytest.raises(ValueError):
        assert calculate_scores([[{}]])

    with pytest.raises(ValueError):
        assert calculate_scores([{'p': 10, 'tp': 5, 'n': 20, 'tn': 15}], strategy='dummy')


    scores = calculate_scores([{'p': 10, 'n': 20, 'tp': 5, 'tn': 8},
                                {'p': 12, 'n': 26, 'tp': 5, 'tn': 8}],
                                strategy=('mor'))

    assert len(scores) > 0

def test_generate_problems():
    """
    Testing the problem generation
    """

    problem = generate_problems(random_seed=np.random.RandomState(5))

    problems = generate_problems(n_problems=5)
    assert len(problems) == 5

    problem = generate_problems(add_complements=True)
    assert 'fn' in problem

    problem = generate_problems(zeros=['tp'])
    assert problem['tp'] == 0

    problem = generate_problems(zeros=['tn'])
    assert problem['tn'] == 0

    problem = generate_problems(zeros=['fp'])
    assert problem['tn'] == problem['n']

    problem = generate_problems(zeros=['fn'])
    assert problem['tp'] == problem['p']

def test_generate_problems_with_folds():
    """
    Testing the generation of problems with fold structure
    """

    folds, problems = generate_problems_with_folds(n_problems=2)
    assert len(folds) == len(problems)
