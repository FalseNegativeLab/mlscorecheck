"""
Testing the checking of scores on multiple datasets using kfold with
mean-of-scores aggregation over the folds and mean-of-scores aggregation
over the datasets
"""

import pytest

from mlscorecheck.check.binary import check_n_datasets_mos_known_folds_mos
from mlscorecheck.aggregated import generate_experiment


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
        evaluation_params={"aggregation": "mos"},
        rounding_decimals=rounding_decimals,
        random_state=random_seed,
        return_scores=True,
    )

    result = check_n_datasets_mos_known_folds_mos(
        evaluations=experiment["evaluations"],
        scores=scores,
        eps=10 ** (-rounding_decimals),
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
        evaluation_params={"aggregation": "mos"},
        rounding_decimals=rounding_decimals,
        random_state=random_seed,
        return_scores=True,
    )

    scores = {"acc": 0.9, "sens": 0.3, "spec": 0.5, "f1": 0.1}

    result = check_n_datasets_mos_known_folds_mos(
        evaluations=experiment["evaluations"],
        scores=scores,
        eps=10 ** (-rounding_decimals),
    )

    assert result["inconsistency"]


@pytest.mark.parametrize("random_seed", list(range(10)))
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_consistency_bounds(random_seed: int, rounding_decimals: int):
    """
    Testing with a consistent setup and bounds

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    experiment, scores = generate_experiment(
        aggregation="mos",
        evaluation_params={"aggregation": "mos"},
        rounding_decimals=rounding_decimals,
        random_state=random_seed,
        feasible_dataset_score_bounds=True,
        return_scores=True,
    )

    result = check_n_datasets_mos_known_folds_mos(
        evaluations=experiment["evaluations"],
        dataset_score_bounds=experiment["dataset_score_bounds"],
        scores=scores,
        eps=10 ** (-rounding_decimals),
        timeout=2,
    )

    assert not result["inconsistency"] or result["lp_status"] == "timeout"


@pytest.mark.parametrize("random_seed", list(range(10)))
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_failure_bounds(random_seed: int, rounding_decimals: int):
    """
    Testing with a inconsistent setup and bounds

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    experiment, scores = generate_experiment(
        aggregation="mos",
        evaluation_params={"aggregation": "mos"},
        rounding_decimals=rounding_decimals,
        random_state=random_seed,
        feasible_dataset_score_bounds=True,
        return_scores=True,
    )

    scores = {"acc": 0.5, "sens": 0.1, "spec": 0.2, "npv": 0.1, "ppv": 0.1, "f1": 0.9}

    result = check_n_datasets_mos_known_folds_mos(
        evaluations=experiment["evaluations"],
        dataset_score_bounds=experiment["dataset_score_bounds"],
        scores=scores,
        eps=10 ** (-rounding_decimals),
        timeout=2,
    )

    assert result["inconsistency"] or result["lp_status"] == "timeout"


def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_mos_known_folds_mos(
            evaluations=[{"aggregation": "som"}], scores={}, eps=1e-4
        )

    with pytest.raises(ValueError):
        check_n_datasets_mos_known_folds_mos(
            evaluations=[{"aggregation": "mos", "fold_score_bounds": {}}],
            scores={},
            eps=1e-4,
        )
