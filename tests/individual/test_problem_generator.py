"""
This module tests the problem generator
"""

import pytest

import numpy as np

from mlscorecheck.individual import (generate_problems,
                                calculate_scores,
                                round_scores)
from mlscorecheck.aggregated import (generate_problems_with_evaluations,
                                        calculate_scores_dataset)

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

    scores = calculate_scores({'p': 10, 'tp': 5, 'n': 20, 'tn': 15})
    assert scores['acc'] == 20/30

    scores = calculate_scores({'p': 10, 'tp': 5, 'n': 20, 'tn': 15}, scores_only=True)
    assert 'tp' not in scores

def test_calculate_scores_rom():
    """
    Testing the calculation of scores in RoM manner
    """

    scores = calculate_scores_dataset({'folds': [{'p': 10, 'n': 20, 'tp': 5, 'tn': 8},
                                        {'p': 12, 'n': 26, 'tp': 5, 'tn': 8}]},
                                            strategy='rom')

    assert scores['sens'] == 10/22

def test_calculate_scores_mor():
    """
    Testing the calculation of scores in the MoR manner
    """

    scores = calculate_scores_dataset({'folds': [{'p': 10, 'n': 20, 'tp': 5, 'tn': 8},
                                        {'p': 12, 'n': 26, 'tp': 5, 'tn': 8}]},
                                            strategy='mor')

    assert scores['sens'] == (5/10 + 5/12) / 2.0

def test_calculate_scores():
    """
    Testing the score calculation
    """

    with pytest.raises(Exception):
        calculate_scores([[{}]])

    with pytest.raises(Exception):
        calculate_scores_dataset([{'p': 10, 'tp': 5, 'n': 20, 'tn': 15}], strategy='dummy')


    scores = calculate_scores_dataset({'folds': [{'p': 10, 'n': 20, 'tp': 5, 'tn': 8},
                                {'p': 12, 'n': 26, 'tp': 5, 'tn': 8}], 'p': 22, 'n': 46},
                                strategy='mor')

    assert len(scores) > 0

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

def test_generate_problems_with_folds():
    """
    Testing the generation of problems with fold structure
    """

    evals, problems = generate_problems_with_evaluations(n_problems=2)
    assert len(evals) == len(problems)
