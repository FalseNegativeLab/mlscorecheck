"""
Testing the checking of scores on multiple datasets with
mean-of-scores aggregation over the datasets
"""

import pytest

from mlscorecheck.aggregated import generate_experiment
from mlscorecheck.check.binary import check_n_testsets_mos_no_kfold


@pytest.mark.parametrize("random_seed", list(range(10)))
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_consistency(random_seed: int, rounding_decimals: int):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(
        aggregation="mos",
        evaluation_params={
            "aggregation": "mos",
            "max_folds": 1,
            "max_repeats": 1,
            "no_name": True,
        },
        random_state=random_seed,
        rounding_decimals=rounding_decimals,
        return_scores=True,
    )
    testsets = [evaluation["dataset"] for evaluation in experiment["evaluations"]]

    result = check_n_testsets_mos_no_kfold(
        testsets=testsets, scores=scores, eps=10 ** (-rounding_decimals)
    )

    assert not result["inconsistency"]


@pytest.mark.parametrize("random_seed", list(range(10)))
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_failure(random_seed: int, rounding_decimals: int):
    """
    Testing with an inconsistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    experiment, scores = generate_experiment(
        aggregation="mos",
        evaluation_params={
            "aggregation": "mos",
            "max_folds": 1,
            "max_repeats": 1,
            "no_name": True,
        },
        random_state=random_seed,
        rounding_decimals=rounding_decimals,
        return_scores=True,
    )
    testsets = [evaluation["dataset"] for evaluation in experiment["evaluations"]]

    scores = {"acc": 0.9, "sens": 0.3, "spec": 0.5, "f1": 0.1}

    result = check_n_testsets_mos_no_kfold(
        testsets=testsets, scores=scores, eps=10 ** (-rounding_decimals)
    )

    assert result["inconsistency"]
