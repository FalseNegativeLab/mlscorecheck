"""
This module tests the AUC estimation related functionalities
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    prepare_intervals_for_auc_estimation,
    auc_from_sens_spec,
    acc_from_auc,
    generate_kfold_sens_spec_fix_problem,
    auc_from_sens_spec_kfold,
    generate_average,
    translate,
)

EPS = 0.04
ITERATIONS = 5000
ITERATIONS_ACC = 10000
RANDOM_SEED = 5


def test_translate():
    """
    Testing the score translation
    """

    assert "sens" in translate({"tpr": 0.9})
    assert "spec" in translate({"tnr": 0.9})
    assert "spec" in translate({"fpr": 0.9})

    with pytest.raises(ValueError):
        translate({"tpr": 0.9, "sens": 0.8})
    with pytest.raises(ValueError):
        translate({"tnr": 0.9, "spec": 0.8})
    with pytest.raises(ValueError):
        translate({"fpr": 0.9, "spec": 0.8})


def test_generate_average():
    """
    Testing the average population generation
    """

    with pytest.raises(ValueError):
        generate_average(0.8, 5, 0.9)


@pytest.mark.parametrize("sens_spec", [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize("k_p_n", [(5, 100, 100), (10, 100, 100)])
def test_kfold_min(sens_spec, k_p_n):
    """
    Tests the average AUC lower bound with "min" estimation

    Args:
        sens_spec (tuple(float, float)): the sensitivity and specificity values
        k_p_n (tuple(int, int, int)): the number of folds, positives and negatives
    """
    # min
    aucs = []
    senss = []
    specs = []

    k, p, n = k_p_n

    random_state = np.random.RandomState(RANDOM_SEED)

    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(
            sens=sens_spec[0], spec=sens_spec[1], k=k, random_state=random_state
        )
        aucs.append(np.mean(problem["spec"] * problem["sens"]))
        senss.append(problem["sens"])
        specs.append(problem["spec"])

    interval = auc_from_sens_spec_kfold(
        scores={"sens": sens_spec[0], "spec": sens_spec[1]},
        eps=1e-4,
        p=p,
        n=n,
        k=k,
        lower="min",
    )
    assert np.abs(np.min(aucs) - interval[0]) < EPS


@pytest.mark.parametrize("sens_spec", [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize("k_p_n", [(5, 100, 100), (10, 100, 100)])
def test_kfold_cmin(sens_spec, k_p_n):
    """
    Tests the average AUC lower bound with "cmin" estimation

    Args:
        sens_spec (tuple(float, float)): the sensitivity and specificity values
        k_p_n (tuple(int, int, int)): the number of folds, positives and negatives
    """
    aucs = []
    senss = []
    specs = []

    k, p, n = k_p_n

    random_state = np.random.RandomState(RANDOM_SEED)

    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(
            sens=sens_spec[0], spec=sens_spec[1], k=k, random_state=random_state
        )
        if np.any(problem["sens"] < 1 - problem["spec"]):
            continue
        aucs.append(np.mean(0.5 + (1 - problem["spec"] - problem["sens"]) ** 2 / 2.0))
        senss.append(problem["sens"])
        specs.append(problem["spec"])

    interval = auc_from_sens_spec_kfold(
        scores={"sens": sens_spec[0], "spec": sens_spec[1]},
        eps=1e-4,
        p=p,
        n=n,
        k=k,
        lower="cmin",
    )
    assert np.abs(np.min(aucs) - interval[0]) < EPS


@pytest.mark.parametrize("sens_spec", [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize("k_p_n", [(5, 100, 100), (10, 100, 100)])
def test_kfold_max(sens_spec, k_p_n):
    """
    Tests the average AUC upper bound with "max" estimation

    Args:
        sens_spec (tuple(float, float)): the sensitivity and specificity values
        k_p_n (tuple(int, int, int)): the number of folds, positives and negatives
    """
    aucs = []
    senss = []
    specs = []

    k, p, n = k_p_n

    random_state = np.random.RandomState(RANDOM_SEED)

    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(
            sens=sens_spec[0], spec=sens_spec[1], k=k, random_state=random_state
        )
        aucs.append(1 - np.mean((1 - problem["spec"]) * (1 - problem["sens"])))
        senss.append(problem["sens"])
        specs.append(problem["spec"])
    np.max(aucs)

    interval = auc_from_sens_spec_kfold(
        scores={"sens": sens_spec[0], "spec": sens_spec[1]},
        eps=1e-4,
        p=p,
        n=n,
        k=k,
        upper="max",
    )
    assert np.abs(np.max(aucs) - interval[1]) < EPS


@pytest.mark.parametrize("sens_spec", [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize("k_p_n", [(5, 100, 100), (10, 100, 100)])
def test_kfold_amax(sens_spec, k_p_n):
    """
    Tests the average AUC upper bound with "amax" estimation

    Args:
        sens_spec (tuple(float, float)): the sensitivity and specificity values
        k_p_n (tuple(int, int, int)): the number of folds, positives and negatives
    """
    aucs = []
    senss = []
    specs = []

    k, p, n = k_p_n

    random_state = np.random.RandomState(RANDOM_SEED)

    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(
            sens=sens_spec[0], spec=sens_spec[1], k=k, random_state=random_state
        )
        accs = (problem["sens"] + problem["spec"]) / 2.0
        aucs.append(1 - np.mean(((1 - accs) * (p + n)) ** 2 / (2 * p * n)))
        senss.append(problem["sens"])
        specs.append(problem["spec"])

    interval = auc_from_sens_spec_kfold(
        scores={"sens": sens_spec[0], "spec": sens_spec[1]},
        eps=1e-4,
        p=p,
        n=n,
        k=k,
        upper="amax",
    )
    assert np.abs(np.max(aucs) - interval[1]) < EPS


@pytest.mark.parametrize("auc", [0.7, 0.8])
@pytest.mark.parametrize("k_p_n", [(4, 100, 100), (3, 30, 45)])
def test_kfold_acc_min_max(auc, k_p_n):
    accs = []

    k, p, n = k_p_n

    random_state = np.random.RandomState(RANDOM_SEED)

    for _ in range(ITERATIONS_ACC):
        aucs = generate_average(
            auc,
            k,
            lower_bound=1 - min(p, n) / (2 * max(p, n)) + 1e-3,
            random_state=random_state,
        )
        tmp = []
        for a in aucs:
            if a < 1 - min(p, n) / (2 * max(p, n)) + 1e-3:
                break
            acc_int = acc_from_auc(scores={"auc": a}, eps=1e-4, p=p, n=n, upper="max")
            tmp.append(random_state.random() * (acc_int[1] - acc_int[0]) + acc_int[0])
        if len(tmp) < k:
            continue

        accs.append(np.mean(tmp))

    interval = acc_from_auc(scores={"auc": auc}, eps=1e-4, p=p, n=n, upper="max")
    assert np.abs(np.max(accs) - interval[1]) < EPS
    assert np.abs(np.min(accs) - interval[0]) < EPS


@pytest.mark.parametrize("auc", [0.7, 0.8])
@pytest.mark.parametrize("k_p_n", [(4, 100, 100), (3, 30, 45)])
def test_kfold_acc_min_cmax(auc, k_p_n):
    accs = []

    k, p, n = k_p_n

    random_state = np.random.RandomState(RANDOM_SEED)

    for _ in range(ITERATIONS_ACC):
        aucs = generate_average(
            auc,
            k,
            lower_bound=1 - min(p, n) / (2 * max(p, n)) + 1e-3,
            random_state=random_state,
        )
        tmp = []
        for a in aucs:
            if a < 1 - min(p, n) / (2 * max(p, n)) + 1e-3:
                break
            acc_int = acc_from_auc(scores={"auc": a}, eps=1e-4, p=p, n=n, upper="cmax")
            tmp.append(random_state.random() * (acc_int[1] - acc_int[0]) + acc_int[0])

        if len(tmp) < k:
            continue

        accs.append(np.mean(tmp))

    interval = acc_from_auc(scores={"auc": auc}, eps=1e-4, p=p, n=n, upper="cmax")
    assert np.abs(np.max(accs) - interval[1]) < EPS
    assert np.abs(np.min(accs) - interval[0]) < EPS


def test_prepare_intervals_for_auc_estimation():
    """
    Testing the perparation of intervals
    """

    p = 20
    n = 80
    eps = 1e-4

    scores = {"acc": 0.6, "sens": 0.55, "spec": 0.7, "asdf": "dummy"}

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3

    scores = {"sens": 0.55, "spec": 0.7}

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3
    acc = (p * scores["sens"] + n * scores["spec"]) / (p + n)
    assert abs(np.mean(intervals["acc"]) - acc) < 1e-5

    scores = {"acc": 0.6, "spec": 0.7}

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3
    sens = ((p + n) * scores["acc"] - n * scores["spec"]) / p
    print(intervals, sens)
    assert abs(np.mean(intervals["sens"]) - sens) < 1e-5

    scores = {"acc": 0.6, "sens": 0.7}

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3
    spec = ((p + n) * scores["acc"] - p * scores["sens"]) / n
    assert abs(np.mean(intervals["spec"]) - spec) < 1e-5


def test_auc_from():
    """
    Testing the auc estimation functionalities
    """

    p = 20
    n = 80
    eps = 1e-4

    scores = {"acc": 0.6}

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores, 
            eps=eps, 
            p=p, 
            n=n, 
            lower="min", 
            upper="max", 
            raise_errors=True
        )

    scores = {"acc": 0.6, "sens": 0.55, "spec": 0.7, "asdf": "dummy"}

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores, 
            eps=eps, 
            p=p, 
            n=n, 
            lower="cmin", 
            upper="amax", 
            raise_errors=True
        )

    scores = {"acc": 0.9, "sens": 0.92, "spec": 0.95, "asdf": "dummy"}

    auc0 = auc_from_sens_spec(
        scores=scores, eps=eps, p=p, n=n, lower="min", upper="max"
    )

    auc1 = auc_from_sens_spec(
        scores=scores, eps=eps, p=p, n=n, lower="cmin", upper="amax"
    )

    assert auc0[0] < auc1[0]
    assert auc0[1] > auc1[1]

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores | {"sens": 0.1, "spec": 0.8},
            eps=eps,
            p=p,
            n=n,
            lower="cmin",
            upper="amax",
            raise_errors=True
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores, 
            eps=eps, 
            p=p, 
            n=n, 
            lower="dummy", 
            upper="max",
            raise_errors=True
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores, 
            eps=eps, 
            p=p, 
            n=n, 
            lower="min", 
            upper="dummy",
            raise_errors=True
        )


def test_auc_from_kfold():
    """
    Testing the auc estimation functionalities with kfold
    """

    p = 20
    n = 80
    eps = 1e-4

    scores = {"acc": 0.6}

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores, eps=eps, p=p, n=n, k=5, lower="min", upper="max"
        )

    scores = {"acc": 0.6, "sens": 0.55, "spec": 0.7, "asdf": "dummy"}

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores, eps=eps, p=p, n=n, k=5, lower="cmin", upper="amax"
        )

    scores = {
        "acc": (0.3 * p + 0.5 * n) / (p + n),
        "sens": 0.3,
        "spec": 0.5,
        "asdf": "dummy",
    }

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores, eps=eps, p=p, n=n, k=5, lower="cmin", upper="amax"
        )

    scores = {"acc": 0.9, "sens": 0.92, "spec": 0.95, "asdf": "dummy"}

    auc0 = auc_from_sens_spec_kfold(
        scores=scores, eps=eps, p=p, n=n, lower="min", upper="max", k=5
    )

    auc1 = auc_from_sens_spec_kfold(
        scores=scores, eps=eps, p=p, n=n, lower="cmin", upper="amax", k=5
    )

    assert auc0[0] < auc1[0]
    assert auc0[1] > auc1[1]

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores | {"sens": 0.1, "spec": 0.8},
            eps=eps,
            p=p,
            n=n,
            lower="cmin",
            upper="amax",
            k=5,
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores | {"sens": 0.1, "spec": 0.8},
            eps=eps,
            lower="cmin",
            upper="amax",
            k=5,
            p=None,
            n=None,
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores | {"sens": 0.1, "spec": 0.8},
            eps=eps,
            lower="cmin",
            upper="amax",
            p=31,
            n=42,
            k=5,
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores, eps=eps, p=p, n=n, lower="dummy", upper="max", k=5
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec_kfold(
            scores=scores, eps=eps, p=p, n=n, lower="min", upper="dummy", k=5
        )


def test_acc_from_auc():
    """
    Testing the accuracy estimation
    """

    scores = {"auc": 0.9}
    p = 20
    n = 80
    eps = 1e-4

    acc = acc_from_auc(scores=scores, eps=eps, p=p, n=n)

    assert acc[1] > acc[0]

    scores = {"auc": 1 - min(p, n) / (2 * max(p, n)) - 0.1}

    with pytest.raises(ValueError):
        acc_from_auc(scores=scores, eps=eps, p=p, n=n, raise_errors=True)
