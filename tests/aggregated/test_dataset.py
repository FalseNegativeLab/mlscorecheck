"""
This module tests the dataset
"""

import pytest

import numpy as np

from mlscorecheck.aggregated import (Dataset,
                                        Fold,
                                        solve,
                                        generate_dataset_specification)
from ._compare_scores import compare_scores

TOL = 1e-5

two_combs = [['acc', 'sens'], ['acc', 'spec'], ['acc', 'bacc'],
            ['sens', 'spec'], ['sens', 'bacc'], ['spec', 'bacc']]
three_combs = [['acc', 'sens', 'spec'], ['acc', 'sens', 'bacc'],
                ['acc', 'spec', 'bacc'], ['sens', 'spec', 'bacc']]
four_combs = [['acc', 'sens', 'spec', 'bacc']]

random_seeds = list(range(20))

@pytest.mark.parametrize('random_state', random_seeds)
def test_dataset_creation(random_state):
    """
    Testing the dataset creation
    """

    dataset = Dataset(**generate_dataset_specification(random_state=random_state))

    assert dataset is not None

def test_dataset_copy():
    """
    Testing the copying of datasets
    """

    dataset = Dataset(**generate_dataset_specification(random_state=5))

    assert dataset.to_dict() == Dataset(**dataset.to_dict()).to_dict()

def test_dataset_sampling_and_evaluation():
    """
    Testing the dataset sampling and evaluation
    """

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [None, 2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_success(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the successful solving capabilities
    """
    problem = generate_dataset_specification(random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-5

    result = solve(problem, scores, eps)

    assert result.status == 1

    populated = problem.populate(result)

    assert compare_scores(scores, populated.calculate_scores(), score_subset, rounding_decimals)
    assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [None, 2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_success_with_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds, successful case
    """
    problem = generate_dataset_specification(random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_bounds(sample.get_bounds(score_subset, feasible=True))

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-5

    result = solve(problem, scores, eps)

    assert result.status == 1

    populated = problem.populate(result)

    assert compare_scores(scores, populated.calculate_scores(), score_subset, rounding_decimals)
    assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [None, 2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_success_with_fold_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds on folds, successful case
    """
    problem = generate_dataset_specification(random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_fold_bounds(sample.get_fold_bounds(feasible=True))

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-5

    result = solve(problem, scores, eps)

    assert result.status == 1

    populated = problem.populate(result)

    assert compare_scores(scores, populated.calculate_scores(), score_subset, rounding_decimals)
    assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset',three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [None, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_failure(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with failure
    """
    random_state = np.random.RandomState(random_state)

    problem = generate_dataset_specification(random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-5

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.5}
    scores = {key: scores[key] for key in score_subset}

    result = solve(problem, scores, eps)

    assert result.status != 1

@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [None, 2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_failure_with_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds, failure case
    """
    problem = generate_dataset_specification(random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_bounds(sample.get_bounds(score_subset, feasible=False))

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-5

    result = solve(problem, scores, eps)

    assert result.status != 1

@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [None, 2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_failure_with_fold_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds on folds, failure case
    """
    problem = generate_dataset_specification(random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_fold_bounds(sample.get_bounds(score_subset, feasible=False))

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-5

    result = solve(problem, scores, eps)

    assert result.status != 1

def test_others():
    """
    Testing some other functionalities
    """

    with pytest.raises(ValueError):
        Dataset(p=5, n=10, aggregation='dummy')

    assert isinstance(str(Dataset(p=5, n=10, aggregation='mor')), str)

    with pytest.raises(ValueError):
        Dataset(p=5, n=10, name='common_datasets.ADA', aggregation='mor')
