"""
Testing the experiment abstraction

The test cases with complicated bound structures are executed
with timeout to prevent hanging.

It is expected, depending on the solver, that some tests times out.

When failure with bounds is tested, the rounding precision is
not tested with 2 decimals, since accidentally with whatever
bounds it is likely to become feasible.
"""

import pulp as pl

import pytest

import numpy as np

from mlscorecheck.aggregated import (Experiment,
                                        solve,
                                        generate_experiment,
                                        compare_scores,
                                        get_dataset_score_bounds)

from ._evaluate_lp import evaluate_timeout

PREFERRED_SOLVER = 'PULP_CBC_CMD'
solvers = pl.listSolvers(onlyAvailable=True)
PREFERRED_SOLVER = PREFERRED_SOLVER if PREFERRED_SOLVER in solvers else solvers[0]
solver = pl.getSolver(PREFERRED_SOLVER)
solver_timeout = pl.getSolver(PREFERRED_SOLVER, timeLimit=3)

two_combs = [['acc', 'sens'], ['acc', 'spec'], ['acc', 'bacc'],
            ['sens', 'spec'], ['sens', 'bacc'], ['spec', 'bacc']]
three_combs = [['acc', 'sens', 'spec'], ['acc', 'sens', 'bacc'],
                ['acc', 'spec', 'bacc'], ['sens', 'spec', 'bacc']]
four_combs = [['acc', 'sens', 'spec', 'bacc']]

random_seeds = list(range(5))

@pytest.mark.parametrize('random_seed', random_seeds)
def test_experiment_instantiation(random_seed):
    """
    Testing the creation of Experiment objects

    Args:
        random_seed (int): the random seed to use
    """

    experiment = generate_experiment(random_state=random_seed)
    experiment = Experiment(**experiment)

    assert experiment is not None

    experiment2 = Experiment(**experiment.to_dict())

    assert experiment.p == experiment2.p and experiment.n == experiment2.n

@pytest.mark.parametrize('random_seed', random_seeds)
def test_sampling_and_scores(random_seed):
    """
    Testing the score calculation in experiments

    Args:
        random_seed (int): the random seed to use
    """

    experiment = generate_experiment(random_state=random_seed)
    experiment = Experiment(**experiment)

    experiment.sample_figures()

    scores = experiment.calculate_scores()

    if experiment.aggregation == 'rom':
        assert abs(scores['sens'] - float(experiment.tp / experiment.p)) < 1e-10
    elif experiment.aggregation == 'mor':
        assert abs(np.mean([evaluation.scores['acc'] for evaluation in experiment.evaluations])\
                - scores['acc']) < 1e-10

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('aggregation_folds', ['mor', 'rom'])
def test_get_dataset_score_bounds(random_seed, aggregation, aggregation_folds):
    """
    Testing the score bounds determination

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation on datasets ('mor'/'rom')
        aggregation_folds (str): the aggregation on folds ('mor'/'rom')
    """

    experiment = generate_experiment(random_state=random_seed,
                                        aggregation=aggregation,
                                        aggregation_folds=aggregation_folds)
    experiment = Experiment(**experiment)
    experiment.sample_figures().calculate_scores()

    score_bounds = get_dataset_score_bounds(experiment, feasible=True)

    for evaluation in experiment.evaluations:
        for key in score_bounds:
            assert score_bounds[key][0] <= evaluation.scores[key] <= score_bounds[key][1]

    score_bounds = get_dataset_score_bounds(experiment, feasible=False)

    for evaluation in experiment.evaluations:
        for key in score_bounds:
            assert not (score_bounds[key][0] <= evaluation.scores[key] <= score_bounds[key][1])

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
def test_linear_programming_success(subset, random_seed, aggregation, rounding_decimals):
    """
    Testing the linear programming functionalities in a successful scenario

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment = generate_experiment(random_state=random_seed,
                                        aggregation=aggregation)
    experiment = Experiment(**experiment)

    experiment.sample_figures(random_state=random_seed)

    scores = experiment.calculate_scores(rounding_decimals, subset)

    print('BBB')

    skeleton = Experiment(**experiment.to_dict())

    lp_program = solve(skeleton, scores, eps=10**(-rounding_decimals))

    assert lp_program.status == 1

    skeleton.populate(lp_program)

    assert compare_scores(scores,
                            skeleton.calculate_scores(),
                            eps=10**(-rounding_decimals),
                            tolerance=1e-6)

    assert skeleton.check_bounds()['bounds_flag']

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
def test_linear_programming_success_with_bounds(subset,
                                                random_seed,
                                                aggregation,
                                                rounding_decimals):
    """
    Testing the linear programming functionalities with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(random_state=random_seed,
                                                aggregation=aggregation,
                                                return_scores=True,
                                                feasible_dataset_score_bounds=True)
    scores = {key: value for key, value in scores.items() if key in subset}

    experiment = Experiment(**experiment)

    lp_program = solve(experiment, scores, eps=10**(-rounding_decimals))

    assert lp_program.status == 1

    experiment.populate(lp_program)

    assert compare_scores(scores,
                            experiment.calculate_scores(),
                            eps=10**(-rounding_decimals),
                            tolerance=1e-6)

    assert experiment.check_bounds()['bounds_flag']

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
def test_linear_programming_failure_with_bounds(subset,
                                                random_seed,
                                                aggregation,
                                                rounding_decimals):
    """
    Testing the linear programming functionalities with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(random_state=random_seed,
                                                aggregation=aggregation,
                                                return_scores=True,
                                                feasible_dataset_score_bounds=False)
    scores = {key: value for key, value in scores.items() if key in subset}

    experiment = Experiment(**experiment)

    lp_program = solve(experiment, scores, eps=10**(-rounding_decimals))

    assert lp_program.status == -1

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('rounding_decimals', [2, 3])
def test_linear_programming_success_both_bounds(subset,
                                                random_seed,
                                                aggregation,
                                                rounding_decimals):
    """
    Testing the linear programming functionalities with both bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(random_state=random_seed,
                                                aggregation=aggregation,
                                                return_scores=True,
                                                rounding_decimals=rounding_decimals,
                                                feasible_dataset_score_bounds=True,
                                                feasible_fold_score_bounds=True)
    scores = {key: value for key, value in scores.items() if key in subset}

    experiment = Experiment(**experiment)

    lp_program = solve(experiment, scores, eps=10**(-rounding_decimals))

    assert lp_program.status == 1

    experiment.populate(lp_program)

    assert compare_scores(scores,
                            experiment.calculate_scores(),
                            eps=10**(-rounding_decimals),
                            tolerance=1e-6)

    assert experiment.check_bounds()['bounds_flag']

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('rounding_decimals', [2, 3])
def test_linear_programming_failure_both_bounds(subset,
                                                random_seed,
                                                aggregation,
                                                rounding_decimals):
    """
    Testing the linear programming functionalities with both bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(random_state=random_seed,
                                                aggregation=aggregation,
                                                return_scores=True,
                                                rounding_decimals=rounding_decimals,
                                                feasible_dataset_score_bounds=False,
                                                feasible_fold_score_bounds=False)
    scores = {key: value for key, value in scores.items() if key in subset}

    experiment = Experiment(**experiment)

    lp_program = solve(experiment, scores, eps=10**(-rounding_decimals))

    assert lp_program.status == -1
