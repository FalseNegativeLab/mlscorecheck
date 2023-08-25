"""
Testing the experiment abstraction
"""

import pytest

import numpy as np

from mlscorecheck.aggregated import (Experiment,
                                        solve,
                                        generate_experiment_specification,
                                        generate_dataset_specification)

from ._compare_scores import compare_scores

TOL = 1e-5

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
@pytest.mark.parametrize('rounding_decimals', [None, 2, 3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_success(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the successful solving capabilities
    """
    problem = generate_experiment_specification(max_n_datasets=5,
                                                max_n_folds=4,
                                                max_n_repeats=3,
                                                random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Experiment(**problem)

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
    Testing the successful solving capabilities with bounds
    """
    problem = generate_experiment_specification(max_n_datasets=5,
                                                max_n_folds=3,
                                                max_n_repeats=2,
                                                random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    problem = problem.add_dataset_bounds(sample.get_dataset_bounds(score_subset, feasible=True))

    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-5

    result = solve(problem, scores, eps)

    assert result.status == 1

    populated = problem.populate(result)

    assert compare_scores(scores, populated.calculate_scores(), score_subset, rounding_decimals)
    assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset',three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [3, 4])
@pytest.mark.parametrize('random_state', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_solving_failure(score_subset, rounding_decimals, random_state, aggregation):
    """
    Testing the solving capabilities with failure
    """
    random_state = np.random.RandomState(random_state)

    problem = generate_experiment_specification(max_n_datasets=5,
                                                max_n_folds=4,
                                                max_n_repeats=3,
                                                random_state=random_state)
    problem['aggregation'] = aggregation
    problem = Experiment(**problem)

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals) if rounding_decimals is not None else 1e-4

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.5}
    scores = {key: scores[key] for key in score_subset}

    result = solve(problem, scores, eps)

    assert result.status != 1
