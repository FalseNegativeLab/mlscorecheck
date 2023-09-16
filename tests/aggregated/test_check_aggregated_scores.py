"""
This module tests the top level interfaces to the aggregated score checks
"""

import pytest

from mlscorecheck.aggregated import (check_aggregated_scores,
                                        generate_experiment,
                                        round_scores)

random_seeds = list(range(20))

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('aggregations', [('rom', 'rom'), ('mor', 'mor'), ('mor', 'rom')])
def test_check_aggregated_scores_feasible(random_seed: int,
                                            rounding_decimals: int,
                                            aggregations: tuple):
    """
    Testing the top level aggregated check function with a feasible problem

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
        aggregations (tuple(str,str)): the aggregations to use
    """
    experiment, scores = generate_experiment(random_state=random_seed,
                                                return_scores=True,
                                                aggregation=aggregations[0],
                                                evaluation_params={'aggregation': aggregations[1]})

    scores = round_scores(scores, rounding_decimals)

    details = check_aggregated_scores(experiment=experiment,
                                            scores=scores,
                                            eps=10**(-rounding_decimals),
                                            timeout=1)

    assert not details['inconsistency']
    assert details['lp_status'] in {'feasible', 'timeout'}
    if details['lp_status'] != 'timeout':
        assert details['lp_configuration_scores_match']
        assert details['lp_configuration_bounds_match']
        assert details['lp_configuration'] is not None

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregations', [('rom', 'rom'), ('mor', 'mor'), ('mor', 'rom')])
def test_check_aggregated_scores_feasible_custom_solver(random_seed: int, aggregations: tuple):
    """
    Testing the top level aggregated check function with a feasible problem
    with custom solver

    Args:
        random_seed (int): the random seed to use
        aggregations (tuple(str,str)): the aggregations to use
    """
    experiment, scores = generate_experiment(random_state=random_seed,
                                                return_scores=True,
                                                aggregation=aggregations[0],
                                                evaluation_params={'aggregation': aggregations[1]})
    scores = round_scores(scores, 4)

    details = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=1e-3,
                                        solver_name='dummy',
                                        timeout=1)

    assert not details['inconsistency']
    assert details['lp_status'] in {'feasible', 'timeout'}
    if details['lp_status'] != 'timeout':
        assert details['lp_configuration_scores_match']
        assert details['lp_configuration_bounds_match']
        assert details['lp_configuration'] is not None

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregations', [('rom', 'rom'), ('mor', 'mor'), ('mor', 'rom')])
def test_check_aggregated_scores_infeasible(random_seed: int, aggregations: tuple):
    """
    Testing the top level aggregated check function with an infeasible problem


    Args:
        random_seed (int): the random seed to use
        aggregations (tuple(str,str)): the aggregations to use
    """
    experiment, scores = generate_experiment(random_state=random_seed,
                                                return_scores=True,
                                                aggregation=aggregations[0],
                                                evaluation_params={'aggregation': aggregations[1]})

    scores = {'acc': 0.1, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.4}

    details = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=1e-4)

    assert details['inconsistency']
    assert details['lp_status'] == 'infeasible'
    assert details['lp_configuration'] is not None

def test_check_aggregated_scores_timeout():
    """
    Testing the top level aggregated check function with an infeasible problem

    Eventually this test can fail, due to the unpredictability of solvers timing out
    """
    experiment, scores = generate_experiment(max_evaluations=20,
                                        evaluation_params={'max_folds': 20,
                                                            'max_repeats': 20,
                                                            'aggregation': 'mor',
                                                            'feasible_fold_score_bounds': True},
                                        random_state=5,
                                        return_scores=True,
                                        aggregation='mor',
                                        feasible_dataset_score_bounds=True)

    scores = round_scores(scores, 7)

    details = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=1e-7,
                                        timeout=0.001,
                                        numerical_tolerance=1e-9)

    assert not details['inconsistency']
    assert details['lp_status'] == 'timeout'
    assert details['lp_configuration'] is not None

def test_no_suitable_score():
    """
    Testing the case when no score is suitable
    """

    details = check_aggregated_scores(experiment=None,
                                        scores={'f1': 0.5},
                                        eps=1e-7,
                                        timeout=0.001,
                                        numerical_tolerance=1e-9)

    assert not details['inconsistency']

def test_others():
    """
    Testing other functionalities
    """

    experiment, scores = generate_experiment(aggregation='rom',
                                                evaluation_params={'aggregation': 'mor'},
                                                return_scores=True)
    with pytest.raises(ValueError):
        check_aggregated_scores(experiment=experiment,
                                scores=scores,
                                eps=1e-4)
