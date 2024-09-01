"""
This module tests the AUC functionalities related to single test sets
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    acc_min,
    acc_rmin,
    acc_max,
    acc_rmax,
    macc_min,
    acc_from,
    max_acc_from,
    acc_upper_from,
    max_acc_upper_from,
)

auc_scenarios = [{"auc": 0.8, "p": 20, "n": 60}, {"auc": 0.4, "p": 50, "n": 51}]


def test_acc_min():
    """
    Testing the minimum accuracy estimation
    """
    np.testing.assert_almost_equal(acc_min(0.8, 10, 20), 10 / 30 * 0.8)


def test_acc_rmin():
    """
    Testing the minimum accuracy assuming the curve does not go below
    the random classification line
    """
    np.testing.assert_almost_equal(acc_rmin(0.6, 10, 20), 10 / 30)

    with pytest.raises(ValueError):
        acc_rmin(0.2, 10, 20)


@pytest.mark.parametrize("scenario", auc_scenarios)
def test_acc_max(scenario):
    """
    Testing the maximum accuracy estimation
    """
    auc = scenario["auc"]
    p = scenario["p"]
    n = scenario["n"]

    tpr_at_fpr_0 = auc
    fpr_at_tpr_1 = 1 - auc

    acc_a = (tpr_at_fpr_0 * p + n) / (p + n)
    acc_b = ((1 - fpr_at_tpr_1) * n + p) / (p + n)

    np.testing.assert_almost_equal(acc_max(auc, p, n), max(acc_a, acc_b))


@pytest.mark.parametrize("scenario", auc_scenarios)
def test_acc_rmax(scenario):
    """
    Testing the maximum accuracy estimation assuming the curve does
    not go below the random classification line
    """
    auc = scenario["auc"]
    p = scenario["p"]
    n = scenario["n"]

    if auc < 0.5:
        with pytest.raises(ValueError):
            acc_rmax(auc, p, n)
        return

    tpr_at_fpr_0 = np.sqrt((auc - 0.5) * 2)
    fpr_at_tpr_1 = 1 - np.sqrt((auc - 0.5) * 2)

    acc_a = (tpr_at_fpr_0 * p + n) / (p + n)
    acc_b = ((1 - fpr_at_tpr_1) * n + p) / (p + n)

    np.testing.assert_almost_equal(acc_rmax(auc, p, n), max(acc_a, acc_b))


def test_macc_min():
    """
    Testing the estimation of the minimum of the maximum accuracy
    """
    auc = 0.9
    p = 50
    n = 60

    np.testing.assert_almost_equal(
        macc_min(auc, p, n), 1 - (np.sqrt(2 * p * n - 2 * auc * p * n)) / (p + n)
    )

    np.testing.assert_almost_equal(macc_min(0.1, p, n), max(p, n) / (p + n))

    np.testing.assert_almost_equal(
        macc_min(1 - min(p, n) / (2 * max(p, n)), p, n), max(p, n) / (p + n)
    )


def test_acc_from():
    """
    Testing the acc_from functionality
    """

    with pytest.raises(ValueError):
        acc_from(scores={}, eps=1e-4, p=50, n=100)

    with pytest.raises(ValueError):
        acc_upper_from(scores={}, eps=1e-4, p=50, n=100)

    for lower in ["min", "rmin"]:
        for upper in ["max", "rmax"]:
            tmp = acc_from(
                scores={"auc": 0.9}, eps=1e-4, p=20, n=50, lower=lower, upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        acc_from(scores={"auc": 0.9}, eps=1e-4, p=20, n=50, lower="dummy")

    with pytest.raises(ValueError):
        acc_from(scores={"auc": 0.9}, eps=1e-4, p=20, n=50, upper="dummy")


def test_max_acc_from():
    """
    Testing the max_acc_from functionality
    """

    with pytest.raises(ValueError):
        max_acc_from(scores={}, eps=1e-4, p=50, n=100)

    with pytest.raises(ValueError):
        max_acc_upper_from(scores={}, eps=1e-4, p=50, n=100)

    for lower in ["min"]:
        for upper in ["max", "rmax"]:
            tmp = max_acc_from(
                scores={"auc": 0.9}, eps=1e-4, p=20, n=50, lower=lower, upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        max_acc_from(scores={"auc": 0.9}, eps=1e-4, p=20, n=50, lower="dummy")

    with pytest.raises(ValueError):
        max_acc_from(scores={"auc": 0.9}, eps=1e-4, p=20, n=50, upper="dummy")
