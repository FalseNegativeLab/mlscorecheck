"""
Testing the case with multiple datasets and unknown folds in MoR aggregations
"""

import pytest

import numpy as np

from mlscorecheck.check import (check_n_datasets_mor_unknown_folds_mor_scores,
                                estimate_n_experiments)
from mlscorecheck.aggregated import (generate_experiments_with_all_kfolds,
                                        generate_experiment)

def test_estimation():
    """
    Testing the evaluation count estimation
    """

    count = estimate_n_experiments(experiment={'evaluations':
                                                [{'dataset': {'p': 5, 'n': 11},
                                                'folding': {'n_folds': 3, 'n_repeats': 2}},
                                                {'dataset': {'p': 6, 'n': 9},
                                                'folding': {'n_folds': 3, 'n_repeats': 2}}]})

    assert count == 144

def generate_test_case(random_seed: int, rounding_decimals: int) -> (dict, dict):
    """
    Generate one random test case

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to

    Returns:
        dict, dict: the experiment specification and the scores
    """
    evaluation_params = {'max_p': 25,
                        'max_n': 200,
                        'max_folds': 4,
                        'max_repeats': 1,
                        'aggregation': 'mor',
                        'no_folds': True,
                        'no_name': True}

    random_state = np.random.RandomState(random_seed)
    experiment, scores = generate_experiment(random_state=random_state,
                                                rounding_decimals=rounding_decimals,
                                                evaluation_params=evaluation_params,
                                                max_evaluations=2,
                                                aggregation='mor',
                                                return_scores=True)

    while len(generate_experiments_with_all_kfolds(experiment)) > 1000\
        or len(experiment['evaluations']) == 1:
        experiment, scores = generate_experiment(random_state=random_state,
                                                rounding_decimals=rounding_decimals,
                                                evaluation_params=evaluation_params,
                                                max_evaluations=2,
                                                aggregation='mor',
                                                return_scores=True)
    return experiment, scores

@pytest.mark.parametrize('random_seed', list(range(5)))
def test_successful(random_seed: int):
    """
    Testing a successful scenario

    Args:
        random_seed (int): the random seed to use
    """
    experiment, scores = generate_test_case(random_seed, 4)

    for evaluation in experiment['evaluations']:
        del evaluation['folding']['strategy']

    results = check_n_datasets_mor_unknown_folds_mor_scores(evaluations=experiment['evaluations'],
                                                            scores=scores,
                                                            eps=1e-4)

    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(5)))
def test_failure(random_seed: int):
    """
    Testing a failure

    Args:
        random_seed (int): the random seed to use
    """

    experiment, scores = generate_test_case(random_seed, 4)

    for evaluation in experiment['evaluations']:
        del evaluation['folding']['strategy']

    scores = {'acc': 0.9, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.05}

    results = check_n_datasets_mor_unknown_folds_mor_scores(evaluations=experiment['evaluations'],
                                                            scores=scores,
                                                            eps=1e-4)

    assert results['inconsistency']

def test_exception():
    """
    Testing the exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_mor_unknown_folds_mor_scores(evaluations=[{'aggregation': 'rom'}],
                                                        scores={},
                                                        eps=1e-4)

    with pytest.raises(ValueError):
        check_n_datasets_mor_unknown_folds_mor_scores(evaluations=[{'aggregation': 'mor',
                                                                    'fold_score_bounds': {}}],
                                                        scores={},
                                                        eps=1e-4)
