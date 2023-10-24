"""
Testing the case with multiple datasets and unknown folds in MoS aggregations
"""

import pytest

import numpy as np

from mlscorecheck.check import (check_n_datasets_mos_unknown_folds_mos_scores,
                                estimate_n_experiments)
from mlscorecheck.aggregated import (generate_experiment)

subsets = [['acc', 'sens', 'spec', 'bacc'], ['acc', 'sens'], ['acc', 'spec'], ['acc']]

def test_estimation():
    """
    Testing the evaluation count estimation
    """

    count = estimate_n_experiments(evaluations=[{'dataset': {'p': 5, 'n': 11},
                                                'folding': {'n_folds': 3, 'n_repeats': 2}},
                                                {'dataset': {'p': 6, 'n': 9},
                                                'folding': {'n_folds': 3, 'n_repeats': 2}}],
                                    available_scores=['acc', 'sens', 'spec'])

    assert count == 144

def generate_test_case(random_seed: int,
                        rounding_decimals: int,
                        score_subset: list) -> (dict, dict):
    """
    Generate one random test case

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
        score_subset (list): the list of scores to be used

    Returns:
        dict, dict: the experiment specification and the scores
    """
    evaluation_params = {'max_p': 25,
                        'max_n': 200,
                        'max_folds': 4,
                        'max_repeats': 1,
                        'aggregation': 'mos',
                        'no_folds': True,
                        'no_name': True}

    random_state = np.random.RandomState(random_seed)
    experiment, scores = generate_experiment(random_state=random_state,
                                                rounding_decimals=rounding_decimals,
                                                evaluation_params=evaluation_params,
                                                max_evaluations=2,
                                                aggregation='mos',
                                                return_scores=True)

    n_experiments = estimate_n_experiments(experiment['evaluations'],
                                            list(scores.keys()))

    while n_experiments > 1000\
        or len(experiment['evaluations']) == 1:
        experiment, scores = generate_experiment(random_state=random_state,
                                                rounding_decimals=rounding_decimals,
                                                evaluation_params=evaluation_params,
                                                max_evaluations=2,
                                                aggregation='mos',
                                                return_scores=True)

        n_experiments = estimate_n_experiments(experiment['evaluations'],
                                                list(scores.keys()))
    scores = {key: value for key, value in scores.items() if key in score_subset}
    return experiment, scores

def remove_strategy_fsom_folding(experiment):
    """
    Removes the "strategy" from the folding

    Args:
        experiment (dict): an experiment specification
    """
    for evaluation in experiment['evaluations']:
        del evaluation['folding']['strategy']

def test_remove_strategy_fsom_folding():
    """
    Testing the remove_strategy_fsom_folding function
    """
    experiment = {'evaluations': [{'folding': {'strategy': 'dummy0'}},
                                    {'folding': {'strategy': 'dummy1'}}]}

    remove_strategy_fsom_folding(experiment)

    assert 'strategy' not in experiment['evaluations'][0]['folding']
    assert 'strategy' not in experiment['evaluations'][1]['folding']

@pytest.mark.parametrize('random_seed', list(range(5)))
@pytest.mark.parametrize('subset', subsets)
def test_successful(random_seed: int, subset: list):
    """
    Testing a successful scenario

    Args:
        random_seed (int): the random seed to use
        subset (list): the subset of scores to be used
    """
    experiment, scores = generate_test_case(random_seed, 4, score_subset=subset)

    remove_strategy_fsom_folding(experiment)

    results = check_n_datasets_mos_unknown_folds_mos_scores(evaluations=experiment['evaluations'],
                                                            scores=scores,
                                                            eps=1e-4)

    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(5)))
@pytest.mark.parametrize('subset', subsets)
def test_failure(random_seed: int, subset: list):
    """
    Testing a failure

    Args:
        random_seed (int): the random seed to use
        subset (list): the subset of scores to be used
    """

    experiment, scores = generate_test_case(random_seed, 4, score_subset=subset)

    remove_strategy_fsom_folding(experiment)

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.05}

    results = check_n_datasets_mos_unknown_folds_mos_scores(evaluations=experiment['evaluations'],
                                                            scores=scores,
                                                            eps=1e-4)

    assert results['inconsistency']

def test_exception():
    """
    Testing the exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_mos_unknown_folds_mos_scores(evaluations=[{'aggregation': 'som'}],
                                                        scores={},
                                                        eps=1e-4)

    with pytest.raises(ValueError):
        check_n_datasets_mos_unknown_folds_mos_scores(evaluations=[{'aggregation': 'mos',
                                                                    'fold_score_bounds': {}}],
                                                        scores={},
                                                        eps=1e-4)
