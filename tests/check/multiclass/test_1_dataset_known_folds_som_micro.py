"""
This module tests the 1 dataset, kfold SoM, micro averaging consistency test
"""

import pytest

from mlscorecheck.check.multiclass import check_1_dataset_known_folds_som_micro
from mlscorecheck.aggregated import generate_dataset_folding_multiclass

@pytest.mark.parametrize("random_seed", range(20))
def test_consistent(random_seed: int):
    """
    Testing a consistent configuration

    Args:
        random_seed (int): the random seed to be used
    """

    dataset, folding, scores = generate_dataset_folding_multiclass(
        random_state=random_seed,
        average="micro",
        aggregation="som",
        rounding_decimals=4,
    )

    result = check_1_dataset_known_folds_som_micro(
        dataset=dataset, folding=folding, scores=scores, eps=1e-4
    )

    assert not result["inconsistency"]


@pytest.mark.parametrize("random_seed", range(20))
def test_inconsistent(random_seed: int):
    """
    Testing a consistent configuration

    Args:
        random_seed (int): the random seed to be used
    """

    dataset, folding, scores = generate_dataset_folding_multiclass(
        random_state=random_seed,
        average="micro",
        aggregation="som",
        rounding_decimals=4,
    )

    scores["acc"] = (1.0 + scores["spec"]) / 2.0

    result = check_1_dataset_known_folds_som_micro(
        dataset=dataset, folding=folding, scores=scores, eps=1e-4
    )

    assert result["inconsistency"]

