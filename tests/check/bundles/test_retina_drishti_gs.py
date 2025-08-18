"""
This module tests the tests developed for the DRISHTI_GS dataset
"""

import pytest

from mlscorecheck.aggregated import generate_scores_for_testsets
from mlscorecheck.check.bundles.retina import (
    _prepare_testsets_drishti_gs,
    check_drishti_gs_segmentation_aggregated,
    check_drishti_gs_segmentation_image,
)
from mlscorecheck.experiments import get_experiment

data = get_experiment("retina.drishti_gs")
test_identifiers = list(data["test"].keys())


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("image_identifier", test_identifiers)
@pytest.mark.parametrize("target", ["OD", "OC"])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
def test_image_consistent(
    random_seed: int, image_identifier: str, target: str, confidence: float
):
    """
    Testing the image level tests with a consistent configuration

    Args:
        random_seed (int): the random seed to use
        image_identifier (str): the image identifier
        target (str): the target
        confidence (float): the confidence level
    """
    testset = _prepare_testsets_drishti_gs(
        subset=[image_identifier], target=target, confidence=confidence
    )[0]

    scores = generate_scores_for_testsets(
        [testset],
        rounding_decimals=4,
        subset=["acc", "sens", "spec", "f1p"],
        aggregation="mos",
        random_state=random_seed,
    )

    results = check_drishti_gs_segmentation_image(
        image_identifier=image_identifier,
        confidence=confidence,
        target=target,
        scores=scores,
        eps=1e-4,
    )

    assert not results["inconsistency"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("image_identifier", test_identifiers)
@pytest.mark.parametrize("target", ["OD", "OC"])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
def test_image_inconsistent(
    random_seed: int, image_identifier: str, target: str, confidence: float
):
    """
    Testing the image level tests with an inconsistent configuration

    Args:
        random_seed (int): the random seed to use
        image_identifier (str): the image identifier
        target (str): the target
        confidence (float): the confidence level
    """
    testset = _prepare_testsets_drishti_gs(
        subset=[image_identifier], target=target, confidence=confidence
    )[0]

    scores = generate_scores_for_testsets(
        [testset],
        rounding_decimals=4,
        subset=["acc", "sens", "spec", "f1p"],
        aggregation="mos",
        random_state=random_seed,
    )

    scores["acc"] = (1.0 + scores["spec"]) / 2.0

    results = check_drishti_gs_segmentation_image(
        image_identifier=image_identifier,
        confidence=confidence,
        target=target,
        scores=scores,
        eps=1e-4,
    )

    assert results["inconsistency"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("subset", ["train", "test", test_identifiers])
@pytest.mark.parametrize("target", ["OD", "OC"])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("aggregation", ["som", "mos"])
def test_aggregated_consistent(
    random_seed: int, subset, target: str, confidence: float, aggregation: str
):
    """
    Testing the consistency test for aggregated scores in a consistent case

    Args:
        random_seed (int): the random seed to use
        subset (str|list): the image subset to be used
        target (str): the target
        confidence (float): the confidence level
        aggregation (str): the aggregation to be used
    """
    testsets = _prepare_testsets_drishti_gs(
        subset=subset, target=target, confidence=confidence
    )

    scores = generate_scores_for_testsets(
        testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec", "f1p"],
        aggregation=aggregation,
        random_state=random_seed,
    )

    results = check_drishti_gs_segmentation_aggregated(
        subset=subset, confidence=confidence, target=target, scores=scores, eps=1e-4
    )

    assert not results["inconsistency"][f"inconsistency_{aggregation}"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("subset", ["train", "test", test_identifiers])
@pytest.mark.parametrize("target", ["OD", "OC"])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("aggregation", ["som", "mos"])
def test_aggregated_inconsistent(
    random_seed: int, subset, target: str, confidence: float, aggregation: str
):
    """
    Testing the consistency test for aggregated scores in an inconsistent case

    Args:
        random_seed (int): the random seed to use
        subset (str|list): the image subset to be used
        target (str): the target
        confidence (float): the confidence level
        aggregation (str): the aggregation to be used
    """
    testsets = _prepare_testsets_drishti_gs(
        subset=subset, target=target, confidence=confidence
    )

    scores = generate_scores_for_testsets(
        testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec", "f1p"],
        aggregation=aggregation,
        random_state=random_seed,
    )

    scores["acc"] = (1.0 + scores["sens"]) / 2.0

    results = check_drishti_gs_segmentation_aggregated(
        subset=subset, confidence=confidence, target=target, scores=scores, eps=1e-4
    )

    assert results["inconsistency"][f"inconsistency_{aggregation}"]
