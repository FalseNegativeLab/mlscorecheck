"""
This module tests the test bundle for the DIARETDB0 dataset
"""

import pytest

from mlscorecheck.check.bundles.retina import check_diaretdb0_class
from mlscorecheck.check.bundles.retina import _prepare_configuration_diaretdb0
from mlscorecheck.aggregated import generate_scores_for_testsets


@pytest.mark.parametrize(
    "class_name",
    [
        "neovascularisation",
        "hardexudates",
        "hemorrhages",
        "softexudates",
        "redsmalldots",
    ],
)
@pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("aggregation", ["mos", "som"])
def test_success(class_name, random_seed, aggregation):
    """
    Testing consistent configurations

    Args:
        class_name (str): the name of the class to test
        random_seed (int): the random seed to be used
        aggregation (str): the aggregation to be used
    """

    testsets = _prepare_configuration_diaretdb0("test", "all", class_name)
    scores = generate_scores_for_testsets(
        testsets=testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec", "f1p"],
        aggregation=aggregation,
        random_state=random_seed,
    )

    results = check_diaretdb0_class(
        subset="test", batch="all", class_name=class_name, scores=scores, eps=1e-4
    )

    assert not results["inconsistency"][f"inconsistency_{aggregation}"]

    testsets = _prepare_configuration_diaretdb0("test", ["1", "2"], class_name)
    scores = generate_scores_for_testsets(
        testsets=testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec", "f1p"],
        aggregation=aggregation,
        random_state=random_seed,
    )

    results = check_diaretdb0_class(
        subset="test", batch=["1", "2"], class_name=class_name, scores=scores, eps=1e-4
    )

    assert not results["inconsistency"][f"inconsistency_{aggregation}"]


@pytest.mark.parametrize(
    "class_name",
    [
        "neovascularisation",
        "hardexudates",
        "hemorrhages",
        "softexudates",
        "redsmalldots",
    ],
)
@pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("aggregation", ["mos", "som"])
def test_failure(class_name, random_seed, aggregation):
    """
    Testing inconsistent configurations

    Args:
        class_name (str): the name of the class to test
        random_seed (int): the random seed to be used
        aggregation (str): the aggregation to be used
    """

    testsets = _prepare_configuration_diaretdb0("test", "all", class_name)
    scores = generate_scores_for_testsets(
        testsets=testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec", "f1p"],
        aggregation=aggregation,
        random_state=random_seed,
    )

    scores["acc"] = (1.0 + scores["spec"]) / 2.0

    results = check_diaretdb0_class(
        subset="test", batch="all", class_name=class_name, scores=scores, eps=1e-4
    )

    assert results["inconsistency"][f"inconsistency_{aggregation}"]

    results = check_diaretdb0_class(
        subset="test", batch=["1", "2"], class_name=class_name, scores=scores, eps=1e-4
    )

    assert results["inconsistency"][f"inconsistency_{aggregation}"]
