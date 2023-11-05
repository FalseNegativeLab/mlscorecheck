"""
This module implements some functionalities related to folding
"""

import copy

import numpy as np

from sklearn.model_selection import StratifiedKFold

from ._utils import random_identifier

__all__ = [
    "stratified_configurations_sklearn",
    "determine_fold_configurations",
    "_create_folds",
    "multiclass_stratified_folds",
    "transform_multiclass_fold_to_binary",
    "create_folds_multiclass",
]


def stratified_configurations_sklearn(p: int, n: int, n_splits: int) -> list:
    """
    The sklearn stratification strategy

    Args:
        p (int): number of positives
        n (int): number of negatives
        n_splits (int): the number of splits

    Returns:
        list(tuple): the list of the structure of the folds
    """
    p_base, p_remainder = divmod(p, n_splits)
    n_base, n_remainder = divmod(n, n_splits)

    results = [(n_base, p_base)] * n_splits

    idx = 0
    while n_remainder > 0:
        results[idx] = (results[idx][0] + 1, results[idx][1])
        n_remainder -= 1
        idx += 1
        idx %= n_splits
    while p_remainder > 0:
        results[idx] = (results[idx][0], results[idx][1] + 1)
        p_remainder -= 1
        idx += 1
        idx %= n_splits

    return results


def determine_fold_configurations(
    p: int, n: int, n_folds: int, n_repeats: int, folding: str = "stratified_sklearn"
) -> list:
    """
    Determine fold configurations according to a folding

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int): the number of folds
        n_repeats (int): the number of repeats
        folding (str): 'stratified_sklearn' - the folding strategy

    Returns:
        list(dict): the list of folds

    Raises:
        ValueError: if the folding is not supported
    """
    if folding != "stratified_sklearn":
        raise ValueError(f"folding strategy {folding} is not supported yet")

    configurations = stratified_configurations_sklearn(p=p, n=n, n_splits=n_folds)
    configurations = [{"n": conf[0], "p": conf[1]} for conf in configurations]
    return [{**item} for item in configurations for _ in range(n_repeats)]


def _create_folds(
    p: int,
    n: int,
    *,
    n_folds: int = None,
    n_repeats: int = None,
    folding: str = None,
    score_bounds: dict = None,
    identifier: str = None,
) -> list:
    """
    Given a dataset, adds folds to it

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int/None): the number of folds (defaults to 1)
        n_repeats (int|None): the number of repeats (defaults to 1)
        folding (str): the folding strategy ('stratified_sklearn')
        score_bounds (dict(str,tuple(float,float))): the score bounds
        identifier (str|None): the identifier

    Returns:
        list(dict): the list of fold specifications

    Raises:
        ValueError: if the folding is not supported
    """

    if n_folds == 1:
        folds = [
            {"p": p, "n": n, "identifier": f"{identifier}_0_r{idx}"}
            for idx in range(n_repeats)
        ]

    elif folding is None:
        folds = [
            {"p": p * n_repeats, "n": n * n_repeats, "identifier": f"{identifier}_0"}
        ]
    else:
        folds = determine_fold_configurations(p, n, n_folds, n_repeats, folding)
        n_fold = 0
        n_repeat = 0
        for fold in folds:
            fold["identifier"] = f"{identifier}_{n_repeat}_{n_fold}"
            n_fold += 1
            if n_fold % n_folds == 0:
                n_fold = 0
                n_repeat += 1

    for fold in folds:
        if score_bounds is not None:
            fold["score_bounds"] = {**score_bounds}

    return folds


def multiclass_stratified_folds(dataset: dict, n_folds: int) -> list:
    """
    Generating the folds for an sklearn stratified multiclass setup

    Args:
        dataset (dict): the specification of the dataset
        n_folds (int): the number of folds

    Returns:
        list(dict): the list of fold specifications
    """
    folds = []
    labels = np.hstack([np.repeat(key, value) for key, value in dataset.items()])
    for _, test in StratifiedKFold(n_splits=n_folds).split(
        labels.reshape(-1, 1), labels, labels
    ):
        folds.append(dict(enumerate(np.bincount(labels[test]))))

    return folds


def transform_multiclass_fold_to_binary(fold: dict) -> list:
    """
    Transforms a multiclass fold specification to a list of binary folds

    Args:
        fold (dict): a multiclass fold specification

    Returns:
        list(dict): the list of binary folds
    """
    n_total = sum(fold.values())
    folds = [{"p": value, "n": n_total - value} for value in fold.values()]
    identifier = fold.get("identifier", random_identifier(4))

    for idx, fold_ in enumerate(folds):
        fold_["identifier"] = f"{identifier}_{idx}"

    return folds


def create_folds_multiclass(dataset: dict, folding: dict) -> list:
    """
    Create the folds for the multiclass setup

    Args:
        dataset (dict): the dataset specification
        folding (dict): the folding specification

    Returns:
        list(dict): the list of fold specifications
    """
    if folding.get("folds") is not None and (
        folding.get("n_repeats") is not None
        or folding.get("strategy") is not None
        or folding.get("n_folds") is not None
    ):
        raise ValueError("either specify the folds or the folding strategy")

    if "folds" in folding:
        return folding["folds"]
    if folding.get("strategy") == "stratified_sklearn":
        folds = multiclass_stratified_folds(dataset, folding.get("n_folds", 1))
    else:
        folds = [dataset]

    n_repeats = folding.get("n_repeats", 1)

    folds = folds * n_repeats
    folds = [copy.deepcopy(fold) for fold in folds]

    return folds
