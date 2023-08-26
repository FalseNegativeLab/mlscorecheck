"""
This module tests the dataset

The test cases with complicated bound structures are executed
with timeout to prevent hanging.

When failure with bounds is tested, the rounding precision is
not tested with 2 decimals, since accidentally with whatever
bounds it is likely to become feasible.
"""
import warnings

import pytest

import pulp as pl

import numpy as np

from mlscorecheck.aggregated import (Dataset,
                                        Fold,
                                        solve,
                                        generate_dataset_specification,
                                        compare_scores)

preferred_solver = 'PULP_CBC_CMD'
solvers = pl.listSolvers(onlyAvailable=True)
if preferred_solver not in solvers:
    preferred_solver = solvers[0]
solver = pl.getSolver(preferred_solver)
solver_timeout = pl.getSolver(preferred_solver, timeLimit=2)

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
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_success(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the successful solving capabilities

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation
    """
    problem = generate_dataset_specification(random_state=random_state,
                                                aggregation=aggregation)
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    result = solve(problem, scores, eps, solver=solver)

    assert result.status == 1

    populated = problem.populate(result)

    assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
    assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_success_with_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds, successful case

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation
    """
    problem = generate_dataset_specification(random_state=random_state,
                                                aggregation=aggregation)
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_bounds(sample.get_bounds(score_subset, feasible=True))

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    result = solve(problem, scores, eps, solver=solver_timeout)

    assert result.status >= 0

    if result.status == 1:
        populated = problem.populate(result)

        assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
        assert populated.check_bounds()['bounds_flag'] is True
    else:
        warnings.warn('test timed out')

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_success_with_fold_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds on folds, successful case

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation
    """
    problem_spec = generate_dataset_specification(random_state=random_state,
                                                aggregation=aggregation)
    problem = Dataset(**problem_spec)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_fold_bounds(sample.get_fold_bounds(score_subset, feasible=True))

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    result = solve(problem, scores, eps, solver=solver_timeout)

    assert result.status >= 0

    if result.status == 1:
        populated = problem.populate(result)

        assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
        assert populated.check_bounds()['bounds_flag'] is True
    else:
        warnings.warn('test timed out')

@pytest.mark.parametrize('score_subset',three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_failure(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with failure

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation
    """
    problem = generate_dataset_specification(random_state=random_state,
                                                aggregation=aggregation)
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.5}
    scores = {key: scores[key] for key in score_subset}

    result = solve(problem, scores, eps, solver=solver)

    assert result.status == -1

@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_failure_with_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds, failure case

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation
    """
    problem = generate_dataset_specification(random_state=random_state,
                                                aggregation=aggregation)
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_bounds(sample.get_bounds(score_subset, feasible=False))

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    result = solve(problem, scores, eps, solver=solver)

    assert result.status == -1

@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_failure_with_fold_bounds(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with bounds on folds, failure case

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation
    """
    problem = generate_dataset_specification(random_state=random_state,
                                                aggregation=aggregation)
    problem = Dataset(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_fold_bounds(sample.get_fold_bounds(score_subset, feasible=False))

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    result = solve(problem, scores, eps, solver=solver)

    assert result.status == -1

def test_others():
    """
    Testing some other functionalities
    """

    with pytest.raises(ValueError):
        Dataset(p=5, n=10, aggregation='dummy')

    assert isinstance(str(Dataset(p=5, n=10, aggregation='mor')), str)

    with pytest.raises(ValueError):
        Dataset(p=5, n=10, name='common_datasets.ADA', aggregation='mor')
