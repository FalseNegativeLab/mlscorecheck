"""
This module tests the top level interfaces to the aggregated score checks
"""

import pytest

from mlscorecheck.aggregated import (check_aggregated_scores,
                                        generate_experiment,
                                        round_scores,
                                        Experiment)

random_seeds = list(range(20))

def check_timeout(details):
    """
    Checks if timeout happened

    Args:
        details (dict): the result of the test
    """
    if details['lp_status'] != 'timeout':
        return (details['lp_configuration_scores_match']
                and details['lp_configuration_bounds_match']
                and details['lp_configuration'] is not None)
    return True

def test_check_timeout():
    """
    Testing the check_timeout function
    """
    assert check_timeout({'lp_status': 'dummy',
                            'lp_configuration_scores_match': True,
                            'lp_configuration_bounds_match': True,
                            'lp_configuration': 'dummy'})

    assert not check_timeout({'lp_status': 'dummy',
                            'lp_configuration_scores_match': True,
                            'lp_configuration_bounds_match': True,
                            'lp_configuration': None})

    assert check_timeout({'lp_status': 'timeout'})

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('aggregations', [('som', 'som'), ('mos', 'mos'), ('mos', 'som')])
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
    assert check_timeout(details)

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregations', [('som', 'som'), ('mos', 'mos'), ('mos', 'som')])
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
    assert check_timeout(details)

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregations', [('som', 'som'), ('mos', 'mos'), ('mos', 'som')])
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
                                                            'aggregation': 'mos',
                                                            'feasible_fold_score_bounds': True},
                                        random_state=5,
                                        return_scores=True,
                                        aggregation='mos',
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

    experiment, scores = generate_experiment(aggregation='som',
                                                evaluation_params={'aggregation': 'mos'},
                                                return_scores=True)
    with pytest.raises(ValueError):
        check_aggregated_scores(experiment=experiment,
                                scores=scores,
                                eps=1e-4)

def test_aggregated_success_with_p_zero():
    """
    Testing a valid configuration with one p in a fold zero
    """
    dataset = {'p': 10, 'n': 20}
    folds = [{'p': 0, 'n': 5}, {'p': 5, 'n': 10}, {'p': 5, 'n': 5}]
    evaluation = {'dataset': dataset,
                    'folding': {'folds': folds},
                    'aggregation': 'mos'}
    experiment = {'evaluations': [evaluation],
                    'aggregation': 'mos'}

    experiment = Experiment(**experiment)

    experiment.sample_figures(score_subset=['acc', 'spec'])

    scores = experiment.calculate_scores(score_subset=['acc', 'spec'],
                                            rounding_decimals=4)

    details = check_aggregated_scores(experiment=experiment.to_dict(),
                                        scores=scores,
                                        eps=1e-4,
                                        solver_name='dummy',
                                        timeout=1)

    assert not details['inconsistency']
    assert details['lp_status'] in {'feasible', 'timeout'}
    assert check_timeout(details)

def test_aggregated_failure_with_p_zero():
    """
    Testing an invalid configuration with one p in a fold zero
    """
    dataset = {'p': 10, 'n': 20}
    folds = [{'p': 0, 'n': 5}, {'p': 5, 'n': 10}, {'p': 5, 'n': 5}]
    evaluation = {'dataset': dataset,
                    'folding': {'folds': folds},
                    'aggregation': 'mos'}
    experiment = {'evaluations': [evaluation],
                    'aggregation': 'mos'}

    scores = {'acc': 0.9584, 'spec': 0.9576}

    details = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=1e-4,
                                        solver_name='dummy',
                                        timeout=1)

    assert details['inconsistency']

def test_aggregated_success_with_n_zero():
    """
    Testing a valid configuration with one n in a fold zero
    """
    dataset = {'p': 10, 'n': 20}
    folds = [{'p': 2, 'n': 0}, {'p': 3, 'n': 15}, {'p': 5, 'n': 5}]
    evaluation = {'dataset': dataset,
                    'folding': {'folds': folds},
                    'aggregation': 'mos'}
    experiment = {'evaluations': [evaluation],
                    'aggregation': 'mos'}

    experiment = Experiment(**experiment)

    experiment.sample_figures(score_subset=['acc', 'sens'])

    scores = experiment.calculate_scores(score_subset=['acc', 'sens'],
                                            rounding_decimals=4)

    details = check_aggregated_scores(experiment=experiment.to_dict(),
                                        scores=scores,
                                        eps=1e-4,
                                        solver_name='dummy',
                                        timeout=1)

    assert not details['inconsistency']
    assert details['lp_status'] in {'feasible', 'timeout'}
    assert check_timeout(details)

def test_aggregated_failure_with_n_zero():
    """
    Testing an invalid configuration with one n in a fold zero
    """
    dataset = {'p': 10, 'n': 20}
    folds = [{'p': 2, 'n': 0}, {'p': 3, 'n': 15}, {'p': 5, 'n': 5}]
    evaluation = {'dataset': dataset,
                    'folding': {'folds': folds},
                    'aggregation': 'mos'}
    experiment = {'evaluations': [evaluation],
                    'aggregation': 'mos'}

    scores = {'acc': 0.9584, 'sens': 0.9576}

    details = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=1e-4,
                                        solver_name='dummy',
                                        timeout=1)

    assert details['inconsistency']

def test_aggregated_success_with_n_or_p_zero():
    """
    Testing a valid configuration with one p and one n in some fold zero
    """
    dataset = {'p': 10, 'n': 20}
    folds = [{'p': 5, 'n': 0}, {'p': 0, 'n': 15}, {'p': 5, 'n': 5}]
    evaluation = {'dataset': dataset,
                    'folding': {'folds': folds},
                    'aggregation': 'mos'}
    experiment = {'evaluations': [evaluation],
                    'aggregation': 'mos'}

    experiment = Experiment(**experiment)

    experiment.sample_figures(score_subset=['acc'])

    scores = experiment.calculate_scores(score_subset=['acc'],
                                            rounding_decimals=4)

    details = check_aggregated_scores(experiment=experiment.to_dict(),
                                        scores=scores,
                                        eps=1e-4,
                                        solver_name='dummy',
                                        timeout=1)

    assert not details['inconsistency']
    assert details['lp_status'] in {'feasible', 'timeout'}
    assert check_timeout(details)

def test_aggregated_failure_with_n_or_p_zero():
    """
    Testing an invalid configuration with one p and one n in some fold zero
    """
    dataset = {'p': 10, 'n': 20}
    folds = [{'p': 5, 'n': 0}, {'p': 0, 'n': 15}, {'p': 5, 'n': 5}]
    evaluation = {'dataset': dataset,
                    'folding': {'folds': folds},
                    'aggregation': 'mos'}
    experiment = {'evaluations': [evaluation],
                    'aggregation': 'mos'}

    scores = {'acc': 0.9584}

    details = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=1e-4,
                                        solver_name='dummy',
                                        timeout=1)

    assert details['inconsistency']
