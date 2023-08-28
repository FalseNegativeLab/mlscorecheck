"""
Testing the use case regarding one dataset
"""

import warnings

import pytest

from mlscorecheck.check import (check_1_testset_no_kfold_scores)
from mlscorecheck.individual import calculate_scores, generate_1_problem

k = 4 # pylint: disable=invalid-name
eps = 10**(-k) # pylint: disable=invalid-name

def test_parametrization():
    """
    Testing the parametrization
    """
    with pytest.raises(ValueError):
        check_1_testset_no_kfold_scores({},
                                        1e-4,
                                        testset={'p': 5})

def test_warnings():
    """
    Testing the warning
    """

    problem = {'name': 'common_datasets.ADA',
                'n_repeats': 2}

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        result = check_1_testset_no_kfold_scores(scores,
                                                eps=1e-2,
                                                testset=problem)
        assert len(warns) == 1

    assert not result['inconsistency']

def test_consistent():
    """
    Testing a consistent configuration
    """
    evaluation, problem = generate_1_problem(random_state=5)

    scores = calculate_scores(evaluation,
                                rounding_decimals=k)

    result = check_1_testset_no_kfold_scores(scores,
                                            eps=eps,
                                            testset=problem)

    assert not result['inconsistency']

def test_failure():
    """
    Testing a failure
    """
    evaluation, problem = generate_1_problem(random_state=5)

    scores = calculate_scores(evaluation,
                                rounding_decimals=k)
    scores['bacc'] = 0.9
    scores['acc'] = 0.1

    result = check_1_testset_no_kfold_scores(scores,
                                                eps=eps,
                                                testset=problem)

    assert result['inconsistency']

def test_dataset():
    """
    Testing success with real dataset
    """
    problem = {'name': 'common_datasets.ADA'}

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    result = check_1_testset_no_kfold_scores(scores,
                                            eps=1e-2,
                                            testset=problem)

    assert not result['inconsistency']

def test_dataset_failure():
    """
    Testing failure with real dataset with extreme precision and random scores
    """
    problem = {'name': 'common_datasets.ADA'}

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    result = check_1_testset_no_kfold_scores(scores,
                                            eps=0.00001,
                                            testset=problem)

    assert result['inconsistency']
