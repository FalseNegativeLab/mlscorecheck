"""
This module tests the operations related to fold structures
"""

import itertools

import pytest

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mlscorecheck.individual import generate_multiclass_dataset
from mlscorecheck.aggregated import (
    stratified_configurations_sklearn,
    determine_fold_configurations,
    _create_folds,
    repeated_kfolds_generator,
    fold_partitioning_generator,
    _check_specification_and_determine_p_n,
    determine_min_max_p,
    multiclass_stratified_folds,
    transform_multiclass_fold_to_binary,
    create_folds_multiclass,
    multiclass_fold_partitioning_generator_22,
    multiclass_fold_partitioning_generator_2n,
    multiclass_fold_partitioning_generator_kn
)


def test_generate_datasets_with_all_kfolds():
    """
    Testing the generation of datasets with all kfold configurations
    """
    evaluation = {"dataset": {"p": 5, "n": 7}, "folding": {"n_folds": 3}}

    datasets = list(
        repeated_kfolds_generator(
            evaluation, available_scores=["acc", "bacc", "sens", "spec"]
        )
    )
    assert len(datasets) == 2

    evaluation = {
        "dataset": {"p": 5, "n": 7},
        "folding": {"n_folds": 3, "n_repeats": 2},
    }

    datasets = list(
        repeated_kfolds_generator(
            evaluation, available_scores=["acc", "bacc", "sens", "spec"]
        )
    )
    assert len(datasets) == 4

    evaluation = {
        "dataset": {"p": 5, "n": 7},
        "folding": {"n_folds": 3, "n_repeats": 2},
        "fold_score_bounds": {"acc": (0.0, 1.0)},
    }

    datasets = list(
        repeated_kfolds_generator(
            evaluation, available_scores=["acc", "bacc", "sens", "spec"]
        )
    )
    assert "fold_score_bounds" in datasets[0]

    evaluation = {
        "dataset": {"dataset_name": "common_datasets.appendicitis"},
        "folding": {"n_folds": 3},
    }

    datasets = list(
        repeated_kfolds_generator(
            evaluation, available_scores=["acc", "bacc", "sens", "spec"]
        )
    )


def test_exceptions():
    """
    Testing if exceptions are thrown
    """

    with pytest.raises(ValueError):
        _check_specification_and_determine_p_n(None, {"folds": []})

    with pytest.raises(ValueError):
        _check_specification_and_determine_p_n({"p": 2}, {})

    with pytest.raises(ValueError):
        _check_specification_and_determine_p_n(
            {"p": 2, "n": 5, "dataset_name": "dummy"}, {}
        )


def test_create_folds():
    """
    Testing the creation of folds
    """
    folds = _create_folds(
        5, 10, score_bounds={"acc": (0.0, 1.0)}, n_repeats=1, n_folds=1
    )
    assert len(folds) == 1

    folds = _create_folds(
        p=5, n=10, n_folds=2, n_repeats=2, score_bounds={"acc": (0.0, 1.0)}
    )
    assert len(folds) == 1

    folds = _create_folds(
        p=5,
        n=10,
        n_folds=2,
        n_repeats=2,
        folding="stratified_sklearn",
        score_bounds={"acc": (0.0, 1.0)},
    )
    assert len(folds) == 4


def sklearn_configurations(y_labels, n_splits):
    """
    Generating the sklearn fold configurations

    Args:
        y_labels (np.array): an array of y labels
        n_splits (int): the number of splits

    Returns:
        list(tuple): the fold configurations
    """
    validator = StratifiedKFold(n_splits=n_splits)

    return [
        tuple(np.bincount(y_labels[test]).tolist())
        for _, test in validator.split(y_labels, y_labels)
    ]


@pytest.mark.parametrize("random_state", list(range(500)))
def test_stratified_configurations_sklearn(random_state):
    """
    Testing the determination of the stratified sklearn fold configurations

    Args:
        random_state (int): the random seed to use
    """

    random_state = np.random.RandomState(random_state)

    n_splits = random_state.randint(2, 40)
    n_items = random_state.randint(n_splits * 2, n_splits * 100)
    n_1 = random_state.randint(n_splits, n_items - n_splits + 1)
    n_0 = n_items - n_1

    y_labels = np.hstack([np.repeat(0, n_0), np.repeat(1, n_1)])

    assert stratified_configurations_sklearn(
        n_1, n_0, n_splits
    ) == sklearn_configurations(y_labels, n_splits)


def test_determine_fold_configurations():
    """
    Testing the determination of fold configurations
    """

    conf = determine_fold_configurations(10, 20, 4, 1)

    conf = [(tmp["n"], tmp["p"]) for tmp in conf]

    assert conf == stratified_configurations_sklearn(10, 20, 4)

    with pytest.raises(ValueError):
        determine_fold_configurations(10, 20, 4, 1, "dummy")


def any_zero(values):
    """
    Tests is any of the values is zero

    Args:
        values (list): a list of values

    Returns:
        bool: True if any of the values is zero, False otherwise
    """
    return any(val == 0 for val in values)


def test_any_zero():
    """
    Testing the any_zero function
    """
    assert any_zero([0, 1])
    assert not any_zero([1, 1])


def test_fold_partitioning_generator():
    """
    Testing the fold partitioning generator
    """

    folds = fold_partitioning_generator(p=6, n=6, k=3, p_non_zero=True, n_non_zero=True)

    assert all((not any_zero(fold[0])) and (not any_zero(fold[1])) for fold in folds)

    folds = list(
        fold_partitioning_generator(p=6, n=6, k=3, p_non_zero=False, n_non_zero=True)
    )

    assert all(not any_zero(fold[1]) for fold in folds)
    assert any(any_zero(fold[0]) for fold in folds)

    folds = list(
        fold_partitioning_generator(p=6, n=6, k=3, p_non_zero=True, n_non_zero=False)
    )

    assert all(not any_zero(fold[0]) for fold in folds)
    assert any(any_zero(fold[1]) for fold in folds)

    folds = list(
        fold_partitioning_generator(p=6, n=6, k=3, p_non_zero=False, n_non_zero=False)
    )

    assert any(any_zero(fold[0]) for fold in folds)
    assert any(any_zero(fold[1]) for fold in folds)


def test_fold_partitioning_generator_p_min():
    """
    Testing the fold partitioning generator with p_min
    """

    folds = fold_partitioning_generator(
        p=6, n=7, k=3, p_non_zero=True, n_non_zero=True, p_min=2
    )

    assert len(list(folds)) == 1

    folds = list(
        fold_partitioning_generator(
            p=6, n=7, k=3, p_non_zero=False, n_non_zero=True, p_min=2
        )
    )

    assert len(list(folds)) == 1

    folds = list(
        fold_partitioning_generator(
            p=6, n=7, k=3, p_non_zero=True, n_non_zero=False, p_min=2
        )
    )

    assert len(list(folds)) == 1

    folds = list(
        fold_partitioning_generator(
            p=6, n=7, k=3, p_non_zero=False, n_non_zero=False, p_min=2
        )
    )

    assert len(list(folds)) == 1


def exhaustive_min_max_p(
    *, p, k_a, k_b, c_a, c_b, p_non_zero, n_non_zero
):  # pylint: disable=too-many-locals
    """
    Exhaustive search for the minimum and maximum p in folds of type A

    Args:
        p (int): the overall number of positives
        k_a (int): the number of folds of type A
        k_b (int): the number of folds of type B
        c_a (int): the count of elements in folds of type A
        c_b (int): the count of elements in folds of type B
        p_non_zero (bool): wether p can be zero in any fold (False if not)
        n_non_zero (bool): wether n can be zero in any fold (False if not)

    Returns:
        int, int: the minimum and maximum number of positives total in folds of
        type A
    """
    a_folds_p = [list(range(p + 1))] * k_a
    b_folds_p = [list(range(p + 1))] * k_b

    min_p_a = p
    max_p_a = 0

    for p_a in itertools.product(*a_folds_p):
        if any(p_tmp > c_a or (p_non_zero and p_tmp == 0) for p_tmp in p_a):
            continue
        p_a_sum = sum(p_a)
        if p_a_sum > k_a * c_a or p_a_sum > p:
            continue

        n_a = [c_a - p_tmp for p_tmp in p_a]

        for p_b in itertools.product(*b_folds_p):
            if any(p_tmp > c_b or (p_non_zero and p_tmp == 0) for p_tmp in p_b):
                continue
            p_b_sum = sum(p_b)

            if p_b_sum > k_b * c_b or p_b_sum != p - p_a_sum:
                continue

            n_b = [c_b - p_tmp for p_tmp in p_b]

            n_all = n_a + n_b

            if n_non_zero and any(n_tmp == 0 for n_tmp in n_all):
                continue

            if sum(p_a) < min_p_a:
                min_p_a = sum(p_a)
            if sum(p_a) > max_p_a:
                max_p_a = sum(p_a)

    return min_p_a, max_p_a


@pytest.mark.parametrize("p", list(range(2, 20, 3)))
@pytest.mark.parametrize("n", list(range(5, 30, 3)))
@pytest.mark.parametrize("k", list(range(2, 6)))
@pytest.mark.parametrize("p_non_zero", [True, False])
@pytest.mark.parametrize("n_non_zero", [True, False])
def test_determine_min_max_p(
    p, n, k, p_non_zero, n_non_zero
):  # pylint: disable=invalid-name
    """
    Testing the determination of minimum and maximum p with exhaustive search

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        k (int): the number of folds
        p_non_zero (bool): wether p can be zero in any fold (False if not)
        n_non_zero (bool): wether n can be zero in any fold (False if not)
    """
    if (p_non_zero and p < k) or (n_non_zero and n < k):
        return

    k_div = (p + n) // k
    k_mod = (p + n) % k

    k_b = k - k_mod
    k_a = k_mod
    c_a = k_div + 1
    c_b = k_div
    min_p, max_p = determine_min_max_p(
        p=p,
        n=n,
        k_a=k_a,
        k_b=k_b,
        c_a=c_a,
        p_non_zero=p_non_zero,
        n_non_zero=n_non_zero,
    )

    min_p_full, max_p_full = exhaustive_min_max_p(
        p=p,
        k_a=k_a,
        k_b=k_b,
        c_a=c_a,
        c_b=c_b,
        p_non_zero=p_non_zero,
        n_non_zero=n_non_zero,
    )

    assert min_p == min_p_full
    assert max_p == max_p_full


def test_multiclass_stratified_folds():
    """
    Testing the generation of multiclass stratified folds
    """

    dataset = generate_multiclass_dataset(random_state=5)

    folds = multiclass_stratified_folds(dataset, n_folds=3)

    assert len(folds) == 3

    counts = [0] * len(dataset)

    for fold in folds:
        for class_, count in fold.items():
            counts[class_] += count

    assert counts == list(dataset.values())


def test_transform_multiclass_fold_to_binary():
    """
    Testing the transformation of a multiclass fold to binary folds
    """

    dataset = generate_multiclass_dataset(random_state=5)

    bfolds = transform_multiclass_fold_to_binary(dataset)

    assert len(bfolds) == len(dataset)


def test_multiclass_create_folds_exception():
    """
    Testing the exception throwing of the multiclass fold creation
    """

    with pytest.raises(ValueError):
        create_folds_multiclass(
            dataset={"p": 5, "n": 7}, folding={"folds": "dummy", "n_repeats": 5}
        )

def test_multiclass_fold_partitioning_generator_22():
    """
    Smoke-test for the multiclass fold partitioning generator (22)
    """

    count = 0
    for _  in multiclass_fold_partitioning_generator_22(10, 10, 10):
        count += 1
    assert count > 0

def test_multiclass_fold_partitioning_generator_2n():
    """
    Smoke-test for the multiclass fold partitioning generator (2n)
    """

    count = 0
    for _ in multiclass_fold_partitioning_generator_2n(10, 10, [10, 6, 4]):
        count += 1
    assert count > 0

def test_multiclass_fold_partitioning_generator_kn():
    """
    Smoke-test for the multiclass fold partitioning generator (kn)
    """

    count = 0
    for _ in multiclass_fold_partitioning_generator_kn([10, 7, 3], [10, 6, 4]):
        count += 1
    assert count > 0

    count = 0
    for _ in multiclass_fold_partitioning_generator_kn([10, 10, 10], [10, 10, 10]):
        count += 1
    assert count > 0
