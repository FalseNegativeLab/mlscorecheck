"""
This module tests the aggregated AUC related functionalities
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    auc_min,
    auc_max,
    auc_rmin,
    auc_maxa,
    auc_amin,
    auc_amax,
    auc_armin,
    auc_min_aggregated,
    auc_max_aggregated,
    auc_rmin_aggregated,
    auc_maxa_aggregated,
    auc_amin_aggregated,
    auc_amax_aggregated,
    auc_armin_aggregated,
    perturb_solutions,
    auc_from_aggregated,
    estimate_acc_interval,
    estimate_tpr_interval,
    estimate_fpr_interval,
    augment_intervals_aggregated,
    check_applicability_upper_aggregated,
    auc_upper_from_aggregated,
)

random_seeds = list(range(100))

auc_confs = [
    {"tpr": 0.8, "fpr": 0.1, "k": 5},
    {"tpr": 0.4, "fpr": 0.3, "k": 5},
    {"tpr": 0.3, "fpr": 0.4, "k": 5},
    {"tpr": 0.8, "fpr": 0.1, "k": 10},
    {"tpr": 0.4, "fpr": 0.3, "k": 10},
]

auc_acc_confs = [
    {"acc": 0.8, "ps": [10, 10, 10], "ns": [19, 19, 20]},
    {"acc": 0.4, "ps": [10, 10, 10], "ns": [19, 19, 20]},
    {"acc": 0.8, "ps": [10, 11, 12, 13, 14], "ns": [19, 18, 17, 16, 15]},
    {"acc": 0.4, "ps": [10, 11, 12, 13, 14], "ns": [19, 18, 17, 16, 15]},
]


def test_interval_estimation():
    """
    Testing the interval estimation
    """

    ps = [10, 11, 12]
    ns = [20, 19, 21]

    tps = [8, 7, 8]
    tns = [15, 16, 17]

    tpr = np.mean([tp / p for tp, p in zip(tps, ps)])
    fpr = np.mean([1 - tn / n for tn, n in zip(tns, ns)])
    acc = np.mean([(tp + tn) / (p + n) for tp, tn, p, n in zip(tps, tns, ps, ns)])

    tpr_interval = (tpr - 1e-4, tpr + 1e-4)
    fpr_interval = (fpr - 1e-4, fpr + 1e-4)
    acc_interval = (acc - 1e-4, acc + 1e-4)

    tpr_est = estimate_tpr_interval(fpr_interval, acc_interval, ps, ns)
    assert tpr_est[0] <= tpr <= tpr_est[1]

    fpr_est = estimate_fpr_interval(tpr_interval, acc_interval, ps, ns)
    assert fpr_est[0] <= fpr <= fpr_est[1]

    acc_est = estimate_acc_interval(fpr_interval, tpr_interval, ps, ns)
    assert acc_est[0] <= acc <= acc_est[1]


def test_augment_intervals_aggregated():
    """
    Testing the aggregated interval augmentation
    """

    ps = [10, 11, 12]
    ns = [20, 19, 21]

    tps = [8, 7, 8]
    tns = [15, 16, 17]

    tprs = [tp / p for tp, p in zip(tps, ps)]
    fprs = [1 - tn / n for tn, n in zip(tns, ns)]
    accs = [(tp + tn) / (p + n) for tp, tn, p, n in zip(tps, tns, ps, ns)]

    tpr = np.mean(tprs)
    fpr = np.mean(fprs)
    acc = np.mean(accs)

    tpr_interval = (tpr - 1e-4, tpr + 1e-4)
    fpr_interval = (fpr - 1e-4, fpr + 1e-4)
    acc_interval = (acc - 1e-4, acc + 1e-4)

    intervals = {"tpr": tpr_interval, "fpr": fpr_interval}
    intervals = augment_intervals_aggregated(intervals, ps, ns)
    assert intervals["acc"][0] <= acc <= intervals["acc"][1]

    intervals = {"tpr": tpr_interval, "acc": acc_interval}
    intervals = augment_intervals_aggregated(intervals, ps, ns)
    assert intervals["fpr"][0] <= fpr <= intervals["fpr"][1]

    intervals = {"fpr": fpr_interval, "acc": acc_interval}
    intervals = augment_intervals_aggregated(intervals, ps, ns)
    assert intervals["tpr"][0] <= tpr <= intervals["tpr"][1]


@pytest.mark.parametrize("conf", auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_auc_min_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    tpr = conf["tpr"]
    fpr = conf["fpr"]
    k = conf["k"]

    # try:
    auc, (fprs, tprs, lower, upper) = auc_min_aggregated(fpr, tpr, k, True)
    # except ValueError as exc:
    #    return
    auc = np.round(auc, 8)

    tmp_fprs = perturb_solutions(fprs, lower, upper, random_seed)
    tmp_tprs = perturb_solutions(tprs, lower, upper, random_seed + 1)
    auc_tmp = np.mean([auc_min(fpr, tpr) for fpr, tpr in zip(tmp_fprs, tmp_tprs)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp


@pytest.mark.parametrize("conf", auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_auc_max_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    tpr = conf["tpr"]
    fpr = conf["fpr"]
    k = conf["k"]

    # try:
    auc, (fprs, tprs, lower, upper) = auc_max_aggregated(fpr, tpr, k, True)
    # except ValueError as exc:
    #    return
    auc = np.round(auc, 8)

    tmp_fprs = perturb_solutions(fprs, lower, upper, random_seed)
    tmp_tprs = perturb_solutions(tprs, lower, upper, random_seed + 1)
    auc_tmp = np.mean([auc_max(fpr, tpr) for fpr, tpr in zip(tmp_fprs, tmp_tprs)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc >= auc_tmp


@pytest.mark.parametrize("conf", auc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_auc_rmin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    tpr = conf["tpr"]
    fpr = conf["fpr"]
    k = conf["k"]

    if tpr >= fpr:
        auc, (fprs, tprs, lower, upper) = auc_rmin_aggregated(fpr, tpr, k, True)
    else:
        with pytest.raises(ValueError):
            auc, (fprs, tprs, lower, upper) = auc_rmin_aggregated(fpr, tpr, k, True)
        return

    auc = np.round(auc, 8)

    tmp_fprs = perturb_solutions(fprs, lower, tprs, random_seed)
    tmp_tprs = perturb_solutions(tprs, tmp_fprs, upper, random_seed + 1)
    auc_tmp = np.mean([auc_rmin(fpr, tpr) for fpr, tpr in zip(tmp_fprs, tmp_tprs)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp


def test_auc_rmin_aggregated_exception():
    """
    Testing the exception for auc_rmin_aggregated
    """

    with pytest.raises(ValueError):
        auc_rmin_aggregated(0.8, 0.1, 5)


@pytest.mark.parametrize("conf", auc_acc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_auc_maxa_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf["acc"]
    ps = conf["ps"]
    ns = conf["ns"]

    if acc < np.mean([max(p, n) / (p + n) for p, n in zip(ps, ns)]):
        with pytest.raises(ValueError):
            auc, (accs, lower, upper) = auc_maxa_aggregated(acc, ps, ns, True)
        return

    auc, (accs, lower, upper) = auc_maxa_aggregated(acc, ps, ns, True)

    auc = np.round(auc, 8)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_maxa(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc >= auc_tmp


@pytest.mark.parametrize("conf", auc_acc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_auc_amin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf["acc"]
    ps = conf["ps"]
    ns = conf["ns"]

    auc, (accs, _, _, lower, upper) = auc_amin_aggregated(acc, ps, ns, True)
    auc = np.round(auc, 8)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_amin(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp


@pytest.mark.parametrize("conf", auc_acc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_auc_amax_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf["acc"]
    ps = conf["ps"]
    ns = conf["ns"]

    auc, (accs, _, _, lower, upper) = auc_amax_aggregated(acc, ps, ns, True)
    auc = np.round(auc, 8)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_amax(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc >= auc_tmp


@pytest.mark.parametrize("conf", auc_acc_confs)
@pytest.mark.parametrize("random_seed", random_seeds)
def test_auc_armin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf["acc"]
    ps = conf["ps"]
    ns = conf["ns"]

    lower = np.array([min(p, n) / (p + n) for p, n in zip(ps, ns)])

    if acc < np.mean(lower):
        with pytest.raises(ValueError):
            auc, (accs, _, _, lower, upper) = auc_armin_aggregated(acc, ps, ns, True)
        return

    auc, (accs, _, _, lower, upper) = auc_armin_aggregated(acc, ps, ns, True)
    auc = np.round(auc, 8)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_armin(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp


def test_auc_from_aggregated():
    """
    Testing the auc_from_aggregated function
    """

    ps = [60, 70, 80]
    ns = [80, 90, 80]

    with pytest.raises(ValueError):
        auc_from_aggregated(scores={"tpr": 0.9}, eps=0.01, k=5)

    with pytest.raises(ValueError):
        check_applicability_upper_aggregated(
            intervals={"tpr": 0.9}, upper="max", ps=None, ns=None
        )

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={"tpr": 0.9, "fpr": 0.1}, eps=0.01, k=5, lower="amin"
        )

    with pytest.raises(ValueError):
        check_applicability_upper_aggregated(
            intervals={"tpr": 0.9, "fpr": 0.1}, upper="amax", ps=None, ns=None
        )

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={"tpr": 0.9},
            eps=0.01,
            k=len(ps),
            lower="amin",
            upper="amax",
            ps=ps,
            ns=ns,
        )

    with pytest.raises(ValueError):
        check_applicability_upper_aggregated(
            intervals={"tpr": 0.9, "fpr": 0.1}, upper="amax", ps=[], ns=[]
        )

    for lower in ["min", "rmin", "amin", "armin"]:
        for upper in ["max", "amax", "maxa"]:
            tmp = auc_from_aggregated(
                scores={"tpr": 0.9, "fpr": 0.1},
                eps=1e-4,
                k=len(ps),
                ps=ps,
                ns=ns,
                lower=lower,
                upper=upper,
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={"tpr": 0.9, "fpr": 0.1},
            eps=1e-4,
            k=len(ps),
            ps=ps,
            ns=ns,
            lower="dummy",
        )

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={"tpr": 0.9, "fpr": 0.1},
            eps=1e-4,
            k=len(ps),
            ps=ps,
            ns=ns,
            upper="dummy",
        )


def test_auc_from_aggregated_folding():
    """
    Testing the auc_from_aggregated with folding
    """

    result = auc_from_aggregated(
        scores={"tpr": 0.9, "fpr": 0.1},
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


def test_auc_from_aggregated_error():
    """
    Testing the auc_from_aggregated with folding
    """

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={"tpr": 0.9, "fpr": 0.1},
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

    with pytest.raises(ValueError):
        auc_upper_from_aggregated(
            scores={"tpr": 0.9, "fpr": 0.1},
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
