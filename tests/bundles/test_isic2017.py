"""
This module tests the tests developed for the ISIC2017 dataset
"""

import pytest

from mlscorecheck.bundles.skinlesion import check_isic2017, _prepare_testset_isic2017
from mlscorecheck.aggregated import generate_scores_for_testsets

subsets = [(['M'], ['SK', 'N']),
            (['SK'], ['M', 'N']),
            (['N'], ['SK', 'M']),
            (['M', 'SK'], ['N']),
            (['M', 'N'], ['SK']),
            (['SK', 'N'], ['M'])]

@pytest.mark.parametrize('random_seed', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('target_against', subsets)
def test_consistency(random_seed, target_against):
    """
    Testing a consistent configuration

    Args:
        random_seed (int): the random seed to be used
        target_against (tuple(list,list)): the target and against classes
    """

    testset = _prepare_testset_isic2017(target=target_against[0],
                                        against=target_against[1])
    scores = generate_scores_for_testsets([testset],
                                            rounding_decimals=4,
                                            random_state=random_seed,
                                            subset=['acc', 'sens', 'spec', 'f1p'],
                                            aggregation='som')

    results = check_isic2017(target=target_against[0],
                                against=target_against[1],
                                scores=scores,
                                eps=1e-4)
    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('target_against', subsets)
def test_inconsistency(random_seed, target_against):
    """
    Testing an inconsistent configuration

    Args:
        random_seed (int): the random seed to be used
        target_against (tuple(list,list)): the target and against classes
    """

    testset = _prepare_testset_isic2017(target=target_against[0],
                                        against=target_against[1])
    scores = generate_scores_for_testsets([testset],
                                            rounding_decimals=4,
                                            random_state=random_seed,
                                            subset=['acc', 'sens', 'spec', 'f1p'],
                                            aggregation='som')

    scores['acc'] = (1.0 + scores['spec']) / 2.0

    results = check_isic2017(target=target_against[0],
                                against=target_against[1],
                                scores=scores,
                                eps=1e-4)
    assert results['inconsistency']
