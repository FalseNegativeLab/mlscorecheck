"""
Testing the experiment abstraction

The test cases with complicated bound structures are executed
with timeout to prevent hanging.

When failure with bounds is tested, the rounding precision is
not tested with 2 decimals, since accidentally with whatever
bounds it is likely to become feasible.
"""

import pulp as pl

import pytest

import numpy as np

from mlscorecheck.aggregated import (Experiment,
                                        solve,
                                        generate_experiment_specification,
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

def test_basic_functionalities():
    """
    Testing the basic functionalities
    """
    with pytest.raises(ValueError):
        Experiment(datasets=[generate_dataset_specification()],
                    aggregation='dummy')

    experiment = Experiment(**generate_experiment_specification())

    assert isinstance(str(experiment), str)

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('aggregation_ds', ['mor', 'rom'])
#@pytest.mark.skip(reason='asdf')
def test_solving_success(score_subset, rounding_decimals, random_state, aggregation, aggregation_ds):
    """
    Testing the successful solving capabilities

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation (str): 'mor'/'rom' - the mode of aggregation at the experiment level
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation at the dataset level
    """
    problem = generate_experiment_specification(max_n_datasets=5,
                                                max_n_folds=4,
                                                max_n_repeats=3,
                                                random_state=random_state,
                                                aggregation=aggregation,
                                                aggregation_ds=aggregation_ds)
    problem = Experiment(**problem)

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
@pytest.mark.parametrize('aggregation', ['rom', 'mor'])
@pytest.mark.parametrize('aggregation_ds', ['rom', 'mor'])
def test_solving_success_with_bounds(score_subset, rounding_decimals, random_state, aggregation, aggregation_ds):
    """
    Testing the successful solving capabilities with bounds

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation (str): 'mor'/'rom' - the mode of aggregation at the experiment level
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation at the dataset level
    """
    problem = generate_experiment_specification(max_n_datasets=4,
                                                max_n_folds=3,
                                                max_n_repeats=2,
                                                random_state=random_state,
                                                aggregation=aggregation,
                                                aggregation_ds=aggregation_ds)
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    problem = problem.add_dataset_bounds(sample.get_dataset_bounds(score_subset, feasible=True))

    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    result = solve(problem, scores, eps, solver=solver_timeout)

    assert result.status >= 0

    if result.status == 1:
        populated = problem.populate(result)

        assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
        assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['rom', 'mor'])
@pytest.mark.parametrize('aggregation_ds', ['rom', 'mor'])
def test_solving_success_with_minmax_bounds(score_subset, rounding_decimals, random_state, aggregation, aggregation_ds):
    """
    Testing the successful solving capabilities with bounds

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation (str): 'mor'/'rom' - the mode of aggregation at the experiment level
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation at the dataset level
    """
    problem = generate_experiment_specification(max_n_datasets=4,
                                                max_n_folds=3,
                                                max_n_repeats=2,
                                                random_state=random_state,
                                                aggregation=aggregation,
                                                aggregation_ds=aggregation_ds)
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    problem = problem.add_dataset_bounds(sample.get_minmax_bounds(score_subset))

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
@pytest.mark.parametrize('aggregation', ['rom', 'mor'])
@pytest.mark.parametrize('aggregation_ds', ['rom', 'mor'])
def test_solving_success_with_fold_bounds(score_subset, rounding_decimals, random_state, aggregation, aggregation_ds):
    """
    Testing the successful solving capabilities with bounds

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation (str): 'mor'/'rom' - the mode of aggregation at the experiment level
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation at the dataset level
    """
    problem = generate_experiment_specification(max_n_datasets=4,
                                                max_n_folds=3,
                                                max_n_repeats=2,
                                                random_state=random_state,
                                                aggregation=aggregation,
                                                aggregation_ds=aggregation_ds)
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    problem = problem.add_dataset_fold_bounds(sample.get_dataset_fold_bounds(score_subset, feasible=True))

    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    result = solve(problem, scores, eps, solver=solver_timeout)

    assert result.status >= 0

    if result.status == 1:
        populated = problem.populate(result)

        assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
        assert populated.check_bounds()['bounds_flag'] is True


@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('aggregation_ds', ['mor', 'rom'])
def test_solving_failure(score_subset, rounding_decimals, random_state, aggregation, aggregation_ds):
    """
    Testing the solving capabilities with failure

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation (str): 'mor'/'rom' - the mode of aggregation at the experiment level
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation at the dataset level
    """
    random_state = np.random.RandomState(random_state)

    problem = generate_experiment_specification(max_n_datasets=5,
                                                max_n_folds=4,
                                                max_n_repeats=3,
                                                random_state=random_state,
                                                aggregation=aggregation,
                                                aggregation_ds=aggregation_ds)
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.5}
    scores = {key: scores[key] for key in score_subset}

    result = solve(problem, scores, eps, solver=solver)

    assert result.status != 1

@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('aggregation_ds', ['mor', 'rom'])
def test_solving_failure_with_bounds(score_subset, rounding_decimals, random_state, aggregation, aggregation_ds):
    """
    Testing the solving capabilities with failure bounds

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation (str): 'mor'/'rom' - the mode of aggregation at the experiment level
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation at the dataset level
    """
    random_state = np.random.RandomState(random_state)

    problem = generate_experiment_specification(max_n_datasets=5,
                                                max_n_folds=4,
                                                max_n_repeats=3,
                                                random_state=random_state,
                                                aggregation=aggregation,
                                                aggregation_ds=aggregation_ds)
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    problem = problem.add_dataset_bounds(sample.get_dataset_bounds(score_subset, feasible=False))

    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.5}
    scores = {key: scores[key] for key in score_subset}

    result = solve(problem, scores, eps, solver=solver)

    assert result.status != 1

@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('aggregation_ds', ['mor', 'rom'])
def test_solving_failure_with_fold_bounds(score_subset, rounding_decimals, random_state, aggregation, aggregation_ds):
    """
    Testing the solving capabilities with failure fold bounds

    Args:
        score_subset (list): the score subset to test with
        rounding_decimals (int): the number of decimal places to round to
        random_state (int): the random seed to use
        aggregation (str): 'mor'/'rom' - the mode of aggregation at the experiment level
        aggregation_ds (str): 'mor'/'rom' - the mode of aggregation at the dataset level
    """
    random_state = np.random.RandomState(random_state)

    problem = generate_experiment_specification(max_n_datasets=5,
                                                max_n_folds=4,
                                                max_n_repeats=3,
                                                random_state=random_state,
                                                aggregation=aggregation,
                                                aggregation_ds=aggregation_ds)
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    problem = problem.add_dataset_fold_bounds(sample.get_dataset_fold_bounds(score_subset, feasible=False))

    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + 10**(-rounding_decimals-1)

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.5}
    scores = {key: scores[key] for key in score_subset}

    result = solve(problem, scores, eps, solver=solver)

    assert result.status != 1
