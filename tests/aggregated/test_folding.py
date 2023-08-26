"""
This module tests the operations related to fold structures
"""

import pytest

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mlscorecheck.aggregated import (stratified_configurations_sklearn,
                                        determine_fold_configurations,
                                        _create_folds)

def test_create_folds():
    """
    Testing the creation of folds
    """
    folds = _create_folds(5, 10, score_bounds={'acc': (0.0, 1.0)})
    assert len(folds) == 1

    folds = _create_folds(5, 10,
                            n_folds=2,
                            n_repeats=2,
                            score_bounds={'acc': (0.0, 1.0)})
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

    results = []

    for _, test in validator.split(y_labels, y_labels, y_labels):
        results.append(tuple(np.bincount(y_labels[test]).tolist()))

    return results

def test_stratified_configurations_sklearn():
    """
    Testing the determination of the stratified sklearn fold configurations
    """

    random_state = np.random.RandomState(5)

    for _ in range(1000):
        n_splits = random_state.randint(2, 40)
        n_items = random_state.randint(n_splits * 2, n_splits*100)
        n_1 = random_state.randint(n_splits, n_items - n_splits + 1)
        n_0 = n_items - n_1

        y_labels = np.hstack([np.repeat(0, n_0), np.repeat(1, n_1)])

        assert stratified_configurations_sklearn(n_1, n_0, n_splits) \
                    == sklearn_configurations(y_labels, n_splits)

def test_determine_fold_configurations():
    """
    Testing the determination of fold configurations
    """

    conf = determine_fold_configurations(10, 20, 4, 1)

    conf = [(tmp['n'], tmp['p']) for tmp in conf]

    assert conf == stratified_configurations_sklearn(10, 20, 4)

    with pytest.raises(ValueError):
        determine_fold_configurations(10, 20, 4, 1, 'dummy')