"""
Testing the consistency tests for regression scores in the 1 testset no kfold case
"""

import pytest

import numpy as np

from mlscorecheck.check.regression import (generate_regression_problem_and_scores,
                                            check_1_testset_no_kfold,
                                            score_formulas,
                                            calculate_regression_scores)

@pytest.mark.parametrize('random_seed', list(range(20)))
@pytest.mark.parametrize('subset', [None, ['mae', 'rmse'],
                                    ['mae', 'mse'], ['mae', 'r2'], ['mae']])
def test_consistency(random_seed, subset):
    """
    Testing a consistent configuration

    Args:
        random_seed (int): the random seed to use
        subset (list): the score subset to use
    """

    random_state = np.random.RandomState(random_seed)

    var, n_samples, scores = generate_regression_problem_and_scores(random_state=random_state,
                                                                    rounding_decimals=4,
                                                                    subset=subset)

    result = check_1_testset_no_kfold(var, n_samples, scores, eps=1e-4)

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(20)))
@pytest.mark.parametrize('subset', [None, ['mae', 'rmse'], ['mae', 'mse'], ['mae', 'r2'], ['mae']])
def test_inconsistency(random_seed, subset):
    """
    Testing an iconsistent configuration

    Args:
        random_seed (int): the random seed to use
        subset (list): the score subset to use
    """

    random_state = np.random.RandomState(random_seed)

    var, n_samples, scores = generate_regression_problem_and_scores(random_state=random_state,
                                                                    rounding_decimals=4,
                                                                    subset=subset)

    scores['mae'] = 0.6
    scores['rmse'] = 0.5

    result = check_1_testset_no_kfold(var, n_samples, scores, eps=1e-4)

    assert result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(20)))
def test_score_formulas(random_seed):
    """
    Testing the score formulas

    Args:
        random_seed (int): the random seed to use
    """

    random_state = np.random.RandomState(random_seed)

    var, n_samples, scores = generate_regression_problem_and_scores(random_state=random_state)

    for key, solutions in score_formulas.items():
        for sol_key, solution in solutions.items():
            score_tmp = eval(solution, scores | {'var': var, 'n_samples': n_samples})
            print(key, sol_key, score_tmp, scores[key])
            assert abs(score_tmp - scores[key]) < 1e-8
