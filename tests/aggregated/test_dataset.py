"""
This module tests the dataset

The test cases with complicated bound structures are executed
with timeout to prevent hanging.

It is expected, depending on the solver, that some tests times out.

When failure with bounds is tested, the rounding precision is
not tested with 2 decimals, since accidentally with whatever
bounds it is likely to become feasible.
"""
import warnings

import pytest

import pulp as pl

from mlscorecheck.aggregated import (Dataset,
                                        solve,
                                        generate_dataset_specification,
                                        compare_scores,
                                        create_folds_for_dataset,
                                        generate_dataset_and_scores)

PREFERRED_SOLVER = 'PULP_CBC_CMD'
solvers = pl.listSolvers(onlyAvailable=True)
PREFERRED_SOLVER = PREFERRED_SOLVER if PREFERRED_SOLVER in solvers else solvers[0]
solver = pl.getSolver(PREFERRED_SOLVER)
solver_timeout = pl.getSolver(PREFERRED_SOLVER, timeLimit=2)

two_combs = [['acc', 'sens'], ['acc', 'spec'], ['acc', 'bacc'],
            ['sens', 'spec'], ['sens', 'bacc'], ['spec', 'bacc']]
three_combs = [['acc', 'sens', 'spec'], ['acc', 'sens', 'bacc'],
                ['acc', 'spec', 'bacc'], ['sens', 'spec', 'bacc']]
four_combs = [['acc', 'sens', 'spec', 'bacc']]

random_seeds = list(range(20))

def test_generate_dataset_and_scores():
    """
    Testing the joint generation of dataset and scores
    """
    dataset, scores = generate_dataset_and_scores(random_state=5,
                                                    aggregation='rom')
    assert len(scores) > 0

    dataset, scores = generate_dataset_and_scores(random_state=5,
                                                    aggregation='mor')
    assert len(scores) > 0

    dataset, scores = generate_dataset_and_scores(random_state=5,
                                                    fold_score_bounds=True,
                                                    aggregation='mor')
    assert 'fold_score_bounds' in dataset

@pytest.mark.parametrize('random_state', random_seeds)
def test_dataset_creation(random_state):
    """
    Testing the dataset creation
    """

    dataset = Dataset(**generate_dataset_specification(random_state=random_state)) # pylint: disable=missing-kwoa

    assert dataset is not None

def test_exceptions():
    """
    Testing the exceptions at dataset creation
    """
    with pytest.raises(ValueError):
        create_folds_for_dataset(p=None,
                                    n=None,
                                    n_folds=None,
                                    n_repeats=None,
                                    folds=None,
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='dummy',
                                    name=None,
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=2,
                                    n=None,
                                    n_folds=None,
                                    n_repeats=None,
                                    folds=None,
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='mor',
                                    name=None,
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=2,
                                    n=3,
                                    n_folds=None,
                                    n_repeats=None,
                                    folds=None,
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='mor',
                                    name='name',
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=2,
                                    n=3,
                                    n_folds=None,
                                    n_repeats=None,
                                    folds=[],
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='mor',
                                    name=None,
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=None,
                                    n=None,
                                    n_folds=3,
                                    n_repeats=None,
                                    folds=[],
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='mor',
                                    name=None,
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=None,
                                    n=None,
                                    n_folds=None,
                                    n_repeats=None,
                                    folds=[],
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='mor',
                                    name='name',
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=None,
                                    n=None,
                                    n_folds=None,
                                    n_repeats=None,
                                    folds=None,
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='mor',
                                    name=None,
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=3,
                                    n=5,
                                    n_folds=2,
                                    n_repeats=None,
                                    folds=None,
                                    folding=None,
                                    fold_score_bounds=None,
                                    aggregation='mor',
                                    name=None,
                                    identifier='identifier')

    with pytest.raises(ValueError):
        create_folds_for_dataset(p=3,
                                    n=5,
                                    n_folds=2,
                                    n_repeats=None,
                                    folds=None,
                                    folding=None,
                                    fold_score_bounds={},
                                    aggregation='rom',
                                    name=None,
                                    identifier='identifier')

def test_dataset_copy():
    """
    Testing the copying of datasets
    """

    dataset = Dataset(**generate_dataset_specification(random_state=5)) # pylint: disable=missing-kwoa

    assert dataset.to_dict() == Dataset(**dataset.to_dict()).to_dict()

def test_dataset_creation_without_aggregation():
    """
    Testing the dataset creation without aggregation
    """
    assert Dataset(p=5, n=10) is not None

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
    problem = Dataset(**problem) # pylint: disable=missing-kwoa

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
    problem = Dataset(**problem) # pylint: disable=missing-kwoa

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
def test_solving_success_with_fold_bounds(score_subset,
                                            rounding_decimals,
                                            random_state,
                                            aggregation):
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
    problem = Dataset(**problem_spec) # pylint: disable=missing-kwoa

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
    problem = Dataset(**problem) # pylint: disable=missing-kwoa

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
    problem = Dataset(**problem) # pylint: disable=missing-kwoa

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
def test_solving_failure_with_fold_bounds(score_subset,
                                            rounding_decimals,
                                            random_state,
                                            aggregation):
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
    problem = Dataset(**problem) # pylint: disable=missing-kwoa

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
