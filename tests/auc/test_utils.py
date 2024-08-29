"""
This module tests the AUC related utilities
"""

import numpy as np
import pytest

from mlscorecheck.auc import (
    R,
    F,
    multi_perturb_solutions,
    check_cvxopt,
    translate_folding,
    translate_scores,
    prepare_intervals,
)


def test_r_f():
    """
    Testing the R and F functions
    """

    lower = np.array([0.1, 0.2, 0.3])
    upper = np.array([0.9, 0.8, 0.8])
    x = 0.78

    r = 1 - R(x, lower=lower, upper=upper, k=3)
    f = F(R(1 - x, lower=1 - F(upper), upper=1 - F(lower), k=3))

    np.testing.assert_array_almost_equal(r, f)

    with pytest.raises(ValueError):
        R(0.1, 3, np.array([0.5, 0.5, 0.5]), np.array([0.8, 0.8, 0.8]))


def test_multi_perturb():
    """
    Testing the iterated perturbation
    """

    x = np.array([0.2, 0.8, 0.5])
    lower = np.array([0.1, 0.2, 0.3])
    upper = np.array([0.8, 0.9, 0.7])

    x_new = multi_perturb_solutions(5, x, lower, upper, 5)

    np.testing.assert_almost_equal(np.mean(x), np.mean(x_new))


def test_check_cvxopt():
    """
    Testing the check_cvxopt function
    """

    with pytest.raises(ValueError):
        check_cvxopt({"status": "dummy"}, "")


def test_translate_scores():
    """
    Testing the translation of scores
    """

    with pytest.raises(ValueError):
        translate_scores({"sens": 0.6, "tpr": 0.7})

    with pytest.raises(ValueError):
        translate_scores({"sens": 0.6, "fnr": 0.3})

    with pytest.raises(ValueError):
        translate_scores({"spec": 0.6, "tnr": 0.7})

    with pytest.raises(ValueError):
        translate_scores({"spec": 0.6, "fpr": 0.3})

    tmp = translate_scores({"spec": 0.7})
    np.testing.assert_almost_equal(tmp["fpr"], 0.3)

    tmp = translate_scores({"tnr": 0.7})
    np.testing.assert_almost_equal(tmp["fpr"], 0.3)

    tmp = translate_scores({"sens": 0.7})
    np.testing.assert_almost_equal(tmp["tpr"], 0.7)

    tmp = translate_scores({"fnr": 0.7})
    np.testing.assert_almost_equal(tmp["tpr"], 0.3)


def test_prepare_intervals():
    """
    Testing the preparation of intervals
    """

    intervals = prepare_intervals({"tpr": 0.8, "f1": 0.7}, eps=0.1)

    np.testing.assert_almost_equal(intervals["tpr"][0], 0.7)
    np.testing.assert_almost_equal(intervals["tpr"][1], 0.9)

    assert len(intervals) == 1


def test_translate_folding():
    """
    Testing the folding extraction
    """

    folding = {
        "p": 10,
        "n": 20,
        "n_repeats": 1,
        "n_folds": 5,
        "folding": "stratified_sklearn",
    }

    ps, ns = translate_folding(folding)

    assert len(ps) == 5 and len(ns) == 5
