"""
Testing the test functionality with MoR aggregation and unknown folds
"""

import pytest

import numpy as np

from mlscorecheck.check import check_1_dataset_unknown_folds_mor_scores
from mlscorecheck.aggregated import Evaluation

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
def test_success(random_seed, rounding_decimals):
    """
    Testing with successful configuration
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

    results = check_1_dataset_unknown_folds_mor_scores(scores=scores,
                                                        eps=(10**(-rounding_decimals)),
                                                        evaluation=evaluation)

    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
def test_failure(random_seed, rounding_decimals):
    """
    Testing with successful configuration
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

    results = check_1_dataset_unknown_folds_mor_scores(scores={'sens': 0.7,
                                                                'spec': 0.7,
                                                                'bacc': 0.6},
                                                        eps=(10**(-rounding_decimals)),
                                                        evaluation=evaluation)

    assert results['inconsistency']

def test_exception():
    """
    Testing if the exception is thrown
    """
    with pytest.raises(ValueError):
        check_1_dataset_unknown_folds_mor_scores(evaluation={'aggregation': 'rom'},
                                                    eps=1e-4,
                                                    scores={})
