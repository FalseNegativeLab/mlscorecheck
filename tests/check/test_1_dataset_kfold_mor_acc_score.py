"""
This module tests the accuracy score testing functionality in the kfold MoR case
with unknown folds (only k and the number of repetitions known)
"""

import pytest

from mlscorecheck.check import check_kfold_accuracy_score
from mlscorecheck.aggregated import (Dataset,
                                        generate_dataset_specification)

@pytest.mark.parametrize('random_seed', range(20))
def test_consistency(random_seed):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
    """
    dataset_spec = generate_dataset_specification(aggregation='mor',
                                                    random_state=random_seed)
    dataset = Dataset(**dataset_spec) # pylint: disable=missing-kwoa
    sample = dataset.sample()
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_kfold_accuracy_score(dataset_spec,
                                        scores['acc'],
                                        1e-4)
    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', range(20))
def test_failure(random_seed):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
    """
    dataset_spec = generate_dataset_specification(aggregation='mor',
                                                    random_state=random_seed)
    dataset = Dataset(**dataset_spec) # pylint: disable=missing-kwoa
    sample = dataset.sample()
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_kfold_accuracy_score(dataset_spec,
                                        0.1234,
                                        1e-5)
    assert not result['inconsistency']
