"""
Testing the test functionality with MoR aggregation and unknown folds
"""

import pytest

import numpy as np

from mlscorecheck.check import (check_1_dataset_unknown_folds_mor_scores,
                                estimate_n_evaluations)
from mlscorecheck.aggregated import Evaluation

def test_estimation():
    """
    Testing the evaluation count estimation
    """

    count = estimate_n_evaluations(evaluation={'dataset': {'p': 5, 'n': 11},
                                                'folding': {'n_folds': 3, 'n_repeats': 2}})

    assert count == 36

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
def test_success(random_seed: int, rounding_decimals: int):
    """
    Testing with successful configuration

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    random_state = np.random.RandomState(random_seed)

    dataset = {'n': random_state.randint(1, 150),
                'p': random_state.randint(1, 20)}
    folding = {'n_folds': random_state.randint(1, 8),
                'n_repeats': random_state.randint(1, 3),
                'strategy': 'stratified_sklearn'}
    folding['n_folds'] = min(folding['n_folds'], dataset['p'], dataset['n'])

    evaluation = Evaluation(dataset=dataset,
                            folding=folding,
                            aggregation='mor')
    evaluation.sample_figures()

    scores = evaluation.calculate_scores(rounding_decimals)

    evaluation = evaluation.to_dict()

    results = check_1_dataset_unknown_folds_mor_scores(
                        scores=scores,
                        eps=(10**(-rounding_decimals)),
                        dataset=evaluation['dataset'],
                        folding=evaluation['folding'],
                        fold_score_bounds=evaluation.get('fold_score_bounds'))

    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
def test_failure(random_seed: int, rounding_decimals: int):
    """
    Testing with successful configuration

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """

    random_state = np.random.RandomState(random_seed)

    dataset = {'n': random_state.randint(1, 150),
                'p': random_state.randint(1, 20)}
    folding = {'n_folds': random_state.randint(1, 8),
                'n_repeats': random_state.randint(1, 3),
                'strategy': 'stratified_sklearn'}
    folding['n_folds'] = min(folding['n_folds'], dataset['p'], dataset['n'])

    evaluation = Evaluation(dataset=dataset,
                            folding=folding,
                            aggregation='mor')
    evaluation.sample_figures()

    evaluation = evaluation.to_dict()
    del evaluation['aggregation']

    results = check_1_dataset_unknown_folds_mor_scores(
                        scores={'sens': 0.7,
                                'spec': 0.7,
                                'bacc': 0.6},
                        eps=(10**(-rounding_decimals)),
                        dataset=evaluation['dataset'],
                        folding=evaluation['folding'],
                        fold_score_bounds=evaluation.get('fold_score_bounds'))

    assert results['inconsistency']
