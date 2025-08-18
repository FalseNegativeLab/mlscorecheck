"""
This module tests the test functionalities implemented for the DIARETDB1 dataset
"""

import pytest

from mlscorecheck.aggregated import generate_scores_for_testsets
from mlscorecheck.check.bundles.retina import (
    _prepare_configuration_diaretdb1,
    check_diaretdb1_class,
    check_diaretdb1_segmentation_aggregated,
    check_diaretdb1_segmentation_aggregated_assumption,
    check_diaretdb1_segmentation_image,
)
from mlscorecheck.experiments import get_experiment

class_names = [
    ["hardexudates"],
    ["softexudates"],
    ["hemorrhages"],
    ["redsmalldots"],
    ["hardexudates", "softexudates"],
    ["hemorrhages", "redsmalldots"],
]

data = get_experiment("retina.diaretdb1")
test_identifiers = data["test"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("subset", ["train", "test"])
@pytest.mark.parametrize("class_name", class_names)
def test_check_success_class(
    random_seed: str, confidence: float, subset: str, class_name
):
    """
    Testing the image labeling in a consistent setup

    Args:
        random_seed (int): the random seed to use
        confidence (float): the confidence for thresholding
        subset (str): the subset to use ('train'/'test')
        class_name (str|list): the names of the lesions constituting the positive class
    """

    testset = _prepare_configuration_diaretdb1(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        pixel_level=False,
        assumption="fov",
    )

    scores = generate_scores_for_testsets(
        [testset],
        rounding_decimals=4,
        subset=["acc", "sens", "spec"],
        aggregation="som",
        random_state=random_seed,
    )

    results = check_diaretdb1_class(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        scores=scores,
        eps=1e-4,
    )

    assert not results["inconsistency"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("subset", ["train", "test"])
@pytest.mark.parametrize("class_name", class_names)
def test_check_failure_class(
    random_seed: int, confidence: float, subset: str, class_name
):
    """
    Testing the image labeling in an inconsistent setup

    Args:
        random_seed (int): the random seed to use
        confidence (float): the confidence for thresholding
        subset (str): the subset to use ('train'/'test')
        class_name (str|list): the names of the lesions constituting the positive class
    """

    testset = _prepare_configuration_diaretdb1(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        pixel_level=False,
        assumption="fov",
    )

    scores = generate_scores_for_testsets(
        [testset],
        rounding_decimals=4,
        subset=["acc", "sens", "spec"],
        aggregation="som",
        random_state=random_seed,
    )
    scores["acc"] = (1.0 + scores["spec"]) / 2.0

    results = check_diaretdb1_class(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        scores=scores,
        eps=1e-4,
    )

    assert results["inconsistency"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("class_name", class_names)
@pytest.mark.parametrize("assumption", ["fov", "all"])
def test_check_success_segmentation_image(
    random_seed: int, confidence: float, class_name, assumption: str
):
    """
    Testing the image labeling in a consistent setup

    Args:
        random_seed (int): the random seed to use
        confidence (float): the confidence for thresholding
        class_name (str|list): the names of the lesions constituting the positive class
        assumption (str): the assumption on the region of evaluation ('fov'/'all')
    """

    testsets = _prepare_configuration_diaretdb1(
        subset="train",
        class_name=class_name,
        confidence=confidence,
        pixel_level=True,
        assumption=assumption,
    ) + _prepare_configuration_diaretdb1(
        subset="test",
        class_name=class_name,
        confidence=confidence,
        pixel_level=True,
        assumption=assumption,
    )

    test_idx = 0
    for test_idx, test_item in enumerate(testsets):
        if test_item["p"] > 0:
            break

    scores = generate_scores_for_testsets(
        [testsets[test_idx]],
        rounding_decimals=4,
        subset=["acc", "sens", "spec"],
        aggregation="som",
        random_state=random_seed,
    )

    results = check_diaretdb1_segmentation_image(
        image_identifier=testsets[test_idx]["identifier"],
        class_name=class_name,
        confidence=confidence,
        scores=scores,
        eps=1e-4,
    )

    assert not results["inconsistency"][f"inconsistency_{assumption}"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("class_name", class_names)
@pytest.mark.parametrize("assumption", ["fov", "all"])
def test_check_failure_segmentation_image(
    random_seed: int, confidence: float, class_name, assumption: str
):
    """
    Testing the image labeling in an inconsistent setup

    Args:
        random_seed (int): the random seed to use
        confidence (float): the confidence for thresholding
        class_name (str|list): the names of the lesions constituting the positive class
        assumption (str): the assumption on the region of evaluation ('fov'/'all')
    """

    testsets = _prepare_configuration_diaretdb1(
        subset="train",
        class_name=class_name,
        confidence=confidence,
        pixel_level=True,
        assumption=assumption,
    ) + _prepare_configuration_diaretdb1(
        subset="test",
        class_name=class_name,
        confidence=confidence,
        pixel_level=True,
        assumption=assumption,
    )

    test_idx = 0
    for test_idx, test_item in enumerate(testsets):
        if test_item["p"] > 0:
            break

    scores = generate_scores_for_testsets(
        [testsets[test_idx]],
        rounding_decimals=4,
        subset=["acc", "sens", "spec"],
        aggregation="som",
        random_state=random_seed,
    )

    scores["acc"] = (1.0 + scores["spec"]) / 2.0

    results = check_diaretdb1_segmentation_image(
        image_identifier=testsets[test_idx]["identifier"],
        class_name=class_name,
        confidence=confidence,
        scores=scores,
        eps=1e-4,
    )

    assert results["inconsistency"][f"inconsistency_{assumption}"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("subset", ["train", "test", test_identifiers])
@pytest.mark.parametrize("class_name", class_names)
@pytest.mark.parametrize("assumption", ["fov", "all"])
@pytest.mark.parametrize("aggregation", ["mos", "som"])
def test_check_success_segmentation_aggregated(
    *,
    random_seed: int,
    confidence: float,
    subset,
    class_name,
    assumption: str,
    aggregation: str,
):
    """
    Testing the image labeling in a consistent setup with all images

    Args:
        random_seed (int): the random seed to use
        confidence (float): the confidence for thresholding
        subset (str|list): the subset to use ('train'/'test')
        class_name (str|list): the names of the lesions constituting the positive class
        assumption (str): the assumption on the region of evaluation ('fov'/'all')
        aggregation (str): the mode of aggregation ('som'/'mos')
    """

    testsets = _prepare_configuration_diaretdb1(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        pixel_level=True,
        assumption=assumption,
        only_valid=True,
    )

    scores = generate_scores_for_testsets(
        testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec"],
        aggregation=aggregation,
        random_state=random_seed,
    )

    results = check_diaretdb1_segmentation_aggregated(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        only_valid=True,
        scores=scores,
        eps=1e-4,
    )

    assert not results["inconsistency"][f"inconsistency_{assumption}_{aggregation}"]


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4])
@pytest.mark.parametrize("confidence", [0.5, 0.75])
@pytest.mark.parametrize("subset", ["train", "test", test_identifiers])
@pytest.mark.parametrize("class_name", class_names)
@pytest.mark.parametrize("assumption", ["fov", "all"])
@pytest.mark.parametrize("aggregation", ["som", "mos"])
def test_check_failure_segmentation_aggregated(
    *,
    random_seed: int,
    confidence: float,
    subset,
    class_name,
    assumption: str,
    aggregation: str,
):
    """
    Testing the image labeling in an inconsistent setup with all images

    Args:
        random_seed (int): the random seed to use
        confidence (float): the confidence for thresholding
        subset (str|list): the subset to use ('train'/'test')
        class_name (str|list): the names of the lesions constituting the positive class
        assumption (str): the assumption on the region of evaluation ('fov'/'all')
        aggregation (str): the mode of aggregation ('som'/'mos')
    """

    testsets = _prepare_configuration_diaretdb1(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        pixel_level=True,
        assumption=assumption,
        only_valid=True,
    )

    scores = generate_scores_for_testsets(
        testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec"],
        aggregation=aggregation,
        random_state=random_seed,
    )

    scores["acc"] = (1.0 + scores["spec"]) / 2.0

    results = check_diaretdb1_segmentation_aggregated(
        subset=subset,
        class_name=class_name,
        confidence=confidence,
        only_valid=True,
        scores=scores,
        eps=1e-4,
    )

    assert results["inconsistency"][f"inconsistency_{assumption}_{aggregation}"]


def test_no_mos():
    """
    Testing the case when MoS cannot be tested
    """

    testsets = _prepare_configuration_diaretdb1(
        subset=test_identifiers[:10],
        class_name="hardexudates",
        confidence=0.75,
        pixel_level=True,
        assumption="fov",
    )

    scores = generate_scores_for_testsets(
        testsets,
        rounding_decimals=4,
        subset=["acc", "sens", "spec"],
        aggregation="som",
        random_state=1,
    )

    results = check_diaretdb1_segmentation_aggregated_assumption(
        subset=test_identifiers[:10],
        class_name="hardexudates",
        confidence=0.75,
        assumption="fov",
        only_valid=False,
        scores=scores,
        eps=1e-4,
    )

    assert "details_mos" not in results
