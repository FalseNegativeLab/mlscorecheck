"""
Testing the checking of scores on 1 dataset using kfold with SoM aggregation
"""

import pytest

from mlscorecheck.check.binary import check_1_dataset_kfold_som
from mlscorecheck.aggregated import (generate_evaluation)

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency(random_seed: int, rounding_decimals: int):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='som',
                                            random_state=random_seed,
                                            return_scores=True,
                                            rounding_decimals=rounding_decimals)

    result = check_1_dataset_kfold_som(dataset=evaluation['dataset'],
                                        folding=evaluation['folding'],
                                        scores=scores,
                                        eps=10**(-rounding_decimals))

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_failure(random_seed: int, rounding_decimals: int):
    """
    Testing with an inconsistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='som',
                                            random_state=random_seed,
                                            rounding_decimals=rounding_decimals,
                                            return_scores=True)
    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_1_dataset_kfold_som(dataset=evaluation['dataset'],
                                        folding=evaluation['folding'],
                                        scores=scores,
                                        eps=10**(-rounding_decimals))

    assert result['inconsistency']

def test_adding_strategy():
    """
    Testing the addition of strategy
    """
    evaluation = {'dataset': {'p': 5, 'n': 6}, 'folding': {'n_folds': 2, 'n_repeats': 1}}
    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_1_dataset_kfold_som(dataset=evaluation['dataset'],
                                        folding=evaluation['folding'],
                                        scores=scores,
                                        eps=10**(-4))

    assert result['inconsistency']
