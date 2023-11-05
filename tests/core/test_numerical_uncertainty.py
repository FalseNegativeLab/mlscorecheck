"""
Testing the handling of uncertainty and tolerance
"""

import pytest

from mlscorecheck.core import check_uncertainty_and_tolerance, update_uncertainty


def test_uncertainty_and_tolerance():
    """
    Testing the checking of the relation of uncertainty and tolerance
    """

    with pytest.raises(ValueError):
        check_uncertainty_and_tolerance({"acc": 1e-7, "sens": 1e-4}, 2 * 1e-8)

    with pytest.raises(ValueError):
        check_uncertainty_and_tolerance(1e-7, 2 * 1e-8)


def test_update_uncertainty():
    """
    Testing the updating of the uncertainty
    """

    assert abs(update_uncertainty(1e-2, 1e-3) - (1e-2 + 1e-3)) <= 1e-5
    assert abs(update_uncertainty({"acc": 1e-2}, 1e-3)["acc"] - (1e-2 + 1e-3)) <= 1e-5
