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
    macc_min,
    acc_min_aggregated,
    acc_rmin_aggregated,
    acc_max_aggregated,
    acc_rmax_aggregated,
    macc_min_aggregated,
    perturb_solutions,
    acc_from_aggregated,
    max_acc_from_aggregated,
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


@pytest.mark.parametrize("conf", acc_auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_macc_min_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf["auc"]
    ps = conf["ps"]
    ns = conf["ns"]

    lower_bounds = 1.0 - np.array([min(p, n) / (2 * max(p, n)) for p, n in zip(ps, ns)])
    if auc < np.mean(lower_bounds):
        with pytest.raises(ValueError):
            acc, (aucs, lower, upper) = macc_min_aggregated(auc, ps, ns, True)
        return

    acc, (aucs, lower, upper) = macc_min_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)

    acc_tmp = np.mean([macc_min(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc <= acc_tmp


def test_macc_min_aggregated_extreme():
    """
    Testing the edge case of macc_min_aggregated
    """

    acc, (_, _, _) = macc_min_aggregated(1.0, [10, 10], [20, 20], True)

    assert acc == 1.0


def test_acc_from_aggregated():
    """
    Testing the acc_from_aggregated functionality
    """

    ps = [60, 70, 80]
    ns = [80, 90, 80]

    with pytest.raises(ValueError):
        acc_from_aggregated(scores={}, eps=1e-4, ps=ps, ns=ns)

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


def test_max_acc_from_aggregated():
    """
    Testing the max_acc_from functionality
    """

    ps = [60, 70, 80]
    ns = [80, 90, 80]

    with pytest.raises(ValueError):
        max_acc_from_aggregated(scores={}, eps=1e-4, ps=ps, ns=ns)

    for lower in ["min"]:
        for upper in ["max", "rmax"]:
            tmp = max_acc_from_aggregated(
                scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, lower=lower, upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        max_acc_from_aggregated(
            scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, lower="dummy"
        )

    with pytest.raises(ValueError):
        max_acc_from_aggregated(
            scores={"auc": 0.9}, eps=1e-4, ps=ps, ns=ns, upper="dummy"
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


def test_max_acc_from_aggregated_folding():
    """
    Testing the max_acc_from_aggregated with folding
    """

    result = max_acc_from_aggregated(
        scores={"auc": 0.9},
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


def test_max_acc_from_aggregated_error():
    """
    Testing the max_acc_from_aggregated throwing exception
    """

    with pytest.raises(ValueError):
        max_acc_from_aggregated(
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
