"""
This module tests the tests developed for the ISIC2016 dataset
"""

import pytest

from mlscorecheck.check.bundles.skinlesion import check_isic2016
from mlscorecheck.aggregated import generate_scores_for_testsets
from mlscorecheck.experiments import get_experiment

@pytest.mark.parametrize('random_seed', [1, 2, 3, 4, 5])
def test_consistency(random_seed):
    """
    Testing a consistent configuration

    Args:
        random_seed (int): the random seed to be used
    """

    testset = get_experiment('skinlesion.isic2016')
    scores = generate_scores_for_testsets([testset],
                                            rounding_decimals=4,
                                            random_state=random_seed)

    results = check_isic2016(scores=scores, eps=1e-4)
    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', [1, 2, 3, 4, 5])
def test_inconsistency(random_seed):
    """
    Testing an inconsistent configuration

    Args:
        random_seed (int): the random seed to be used
    """

    testset = get_experiment('skinlesion.isic2016')
    scores = generate_scores_for_testsets([testset],
                                            rounding_decimals=4,
                                            random_state=random_seed)

    scores['acc'] = (1.0 + scores['spec']) / 2.0

    results = check_isic2016(scores=scores, eps=1e-4)
    assert results['inconsistency']
