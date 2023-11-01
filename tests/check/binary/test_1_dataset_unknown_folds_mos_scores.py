"""
Testing the test functionality with MoS aggregation and unknown folds
"""

import pytest

import numpy as np

from mlscorecheck.check.binary import (check_1_dataset_unknown_folds_mos,
                                estimate_n_evaluations)
from mlscorecheck.aggregated import Evaluation

subsets = [['acc', 'sens', 'spec', 'bacc'], ['acc', 'sens'], ['acc', 'spec'], ['acc']]

def test_estimation():
    """
    Testing the evaluation count estimation
    """

    count = estimate_n_evaluations(dataset={'p': 5, 'n': 11},
                                    folding={'n_folds': 3, 'n_repeats': 2},
                                    available_scores=['acc', 'sens', 'spec'])

    assert count == 16

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
@pytest.mark.parametrize('subset', subsets)
def test_success(random_seed: int, rounding_decimals: int, subset: list):
    """
    Testing with successful configuration

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
        subset (list): the subset of scores to work with
    """
    random_state = np.random.RandomState(random_seed)

    dataset = {'n': random_state.randint(1, 150),
                'p': random_state.randint(1, 15)}
    folding = {'n_folds': random_state.randint(1, 5),
                'n_repeats': random_state.randint(1, 3),
                'strategy': 'stratified_sklearn'}
    folding['n_folds'] = min(folding['n_folds'], dataset['p'], dataset['n'])

    evaluation = Evaluation(dataset=dataset,
                            folding=folding,
                            aggregation='mos')
    evaluation.sample_figures(score_subset=subset)

    scores = evaluation.calculate_scores(rounding_decimals, score_subset=subset)

    evaluation = evaluation.to_dict()

    results = check_1_dataset_unknown_folds_mos(
                        scores=scores,
                        eps=(10**(-rounding_decimals)),
                        dataset=evaluation['dataset'],
                        folding=evaluation['folding'],
                        fold_score_bounds=evaluation.get('fold_score_bounds'))

    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
@pytest.mark.parametrize('subset', subsets)
def test_failure(random_seed: int, rounding_decimals: int, subset: list):
    """
    Testing with successful configuration

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
        subset (list): the subset of scores to work with
    """

    random_state = np.random.RandomState(random_seed)

    dataset = {'n': random_state.randint(1, 150),
                'p': random_state.randint(1, 15)}
    folding = {'n_folds': random_state.randint(1, 3),
                'n_repeats': random_state.randint(1, 3),
                'strategy': 'stratified_sklearn'}
    folding['n_folds'] = min(folding['n_folds'], dataset['p'], dataset['n'])

    evaluation = Evaluation(dataset=dataset,
                            folding=folding,
                            aggregation='mos')
    evaluation.sample_figures(score_subset=subset)

    evaluation = evaluation.to_dict()
    del evaluation['aggregation']

    scores = {'sens': 0.7, 'spec': 0.7, 'bacc': 0.6, 'acc': 0.1234}
    scores = {key: value for key, value in scores.items() if key in subset}

    results = check_1_dataset_unknown_folds_mos(
                        scores=scores,
                        eps=(10**(-rounding_decimals)),
                        dataset=evaluation['dataset'],
                        folding=evaluation['folding'],
                        fold_score_bounds=evaluation.get('fold_score_bounds'))

    assert results['inconsistency']
