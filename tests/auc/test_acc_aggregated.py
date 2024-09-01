"""
This module tests the aggregated AUC related functionalities
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    acc_min,
    acc_max,
    acc_rmin,
    acc_rmax,
    acc_min_aggregated,
    acc_rmin_aggregated,
    acc_max_aggregated,
    acc_rmax_aggregated,
    perturb_solutions,
    acc_from_aggregated,
    acc_upper_from_aggregated,
    FAccRMax,
)

random_seeds = list(range(100))

acc_auc_confs = [
    {"auc": 0.8, "ps": [10, 10, 10], "ns": [19, 19, 20]},
    {"auc": 0.6, "ps": [10, 10, 10], "ns": [19, 19, 20]},
    {"auc": 0.4, "ps": [10, 10, 10], "ns": [19, 19, 20]},
    {"auc": 0.8, "ps": [10, 11, 12, 13, 14], "ns": [19, 18, 17, 16, 15]},
    {"auc": 0.6, "ps": [10, 11, 12, 13, 14], "ns": [19, 18, 17, 16, 15]},
]


@pytest.mark.parametrize("conf", acc_auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_acc_min_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf["auc"]
    ps = conf["ps"]
    ns = conf["ns"]

    acc, (aucs, ps, ns, lower, upper) = acc_min_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_min(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc <= acc_tmp


@pytest.mark.parametrize("conf", acc_auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_acc_max_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf["auc"]
    ps = conf["ps"]
    ns = conf["ns"]

    acc, (aucs, ps, ns, lower, upper) = acc_max_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_max(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc >= acc_tmp


@pytest.mark.parametrize("conf", acc_auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_acc_rmin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf["auc"]
    ps = conf["ps"]
    ns = conf["ns"]

    acc, (aucs, lower, upper) = acc_rmin_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_rmin(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc <= acc_tmp


@pytest.mark.parametrize("conf", acc_auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_acc_rmax_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf["auc"]
    ps = conf["ps"]
    ns = conf["ns"]

    if auc >= 0.5:
        acc, (aucs, ps, ns, lower, upper) = acc_rmax_aggregated(auc, ps, ns, True)
    else:
        with pytest.raises(ValueError):
            acc, (aucs, ps, ns, lower, upper) = acc_rmax_aggregated(auc, ps, ns, True)
        return
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_rmax(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc >= acc_tmp


def test_acc_rmax_exception():
    """
    Testing the exception throwing in acc_rmax_aggregated
    """

    with pytest.raises(ValueError):
        acc_rmax_aggregated(0.3, [], [])


def test_acc_rmax_edge():
    """
    Testing the edge cases for acc_rmax
    """
    np.testing.assert_almost_equal(
        acc_rmax_aggregated(
            auc=0.5, ps=np.array([10, 10]), ns=np.array([20, 20]), return_solutions=True
        )[0],
        2 / 3,
    )

    assert (
        acc_rmax_aggregated(
            auc=0.5001,
            ps=np.array([10, 10]),
            ns=np.array([20, 20]),
            return_solutions=True,
        )
        is not None
    )

    assert (
        acc_rmax_aggregated(
            auc=0.501,
            ps=np.array([10, 10]),
            ns=np.array([20, 20]),
            return_solutions=True,
        )
        is not None
    )

    assert (
        acc_rmax_aggregated(
            auc=0.51,
            ps=np.array([10, 10]),
            ns=np.array([20, 20]),
            return_solutions=True,
        )
        is not None
    )

    assert (
        acc_rmax_aggregated(auc=0.5001, ps=np.array([10, 10]), ns=np.array([20, 20]))
        is not None
    )

    assert (
        acc_rmax_aggregated(auc=0.501, ps=np.array([10, 10]), ns=np.array([20, 20]))
        is not None
    )

    assert (
        acc_rmax_aggregated(auc=0.51, ps=np.array([10, 10]), ns=np.array([20, 20]))
        is not None
    )


def test_acc_from_aggregated():
    """
    Testing the acc_from_aggregated functionality
    """

    ps = [60, 70, 80]
    ns = [80, 90, 80]

    with pytest.raises(ValueError):
        acc_from_aggregated(scores={}, eps=1e-4, ps=ps, ns=ns)

    with pytest.raises(ValueError):
        acc_upper_from_aggregated(scores={}, eps=1e-4, ps=ps, ns=ns)

    for lower in ["min", "rmin"]:
        for upper in ["max", "rmax"]:
            tmp = acc_from_aggregated(
                scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, lower=lower, upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        acc_from_aggregated(scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, lower="dummy")

    with pytest.raises(ValueError):
        acc_from_aggregated(scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, upper="dummy")

    with pytest.raises(ValueError):
        acc_from_aggregated(
            scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, upper="dummy", folding={}
        )

    with pytest.raises(ValueError):
        acc_upper_from_aggregated(
            scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, upper="dummy", folding={}
        )


def test_acc_from_aggregated_folding():
    """
    Testing the acc_from_aggregated with folding
    """

    result = acc_from_aggregated(
        scores={"auc": 0.8},
        eps=0.01,
        folding={
            "p": 20,
            "n": 80,
            "n_repeats": 1,
            "n_folds": 5,
            "folding": "stratified_sklearn",
        },
    )

    assert result is not None


def test_acc_from_aggregated_error():
    """
    Testing the acc_from_aggregated throwing exception
    """

    with pytest.raises(ValueError):
        acc_from_aggregated(
            scores={"acc": 0.9},
            eps=0.01,
            ps=[10, 20],
            ns=[20, 30],
            folding={
                "p": 20,
                "n": 80,
                "n_repeats": 1,
                "n_folds": 5,
                "folding": "stratified_sklearn",
            },
        )


def test_faccrmax():
    """
    Testing some functionalities from FAccRMax
    """

    obj = FAccRMax(np.array([10, 10]), np.array([20, 20]))

    assert obj(np.array([[-1], [-1]])) == (None, None)
