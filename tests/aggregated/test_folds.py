"""
This module tests the operations related to fold structures
"""

import pytest

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mlscorecheck.aggregated import (stratified_configurations_sklearn,
                                        determine_fold_configurations,
                                        _create_folds,
                                        _expand_datasets,
                                        random_folds,
                                        random_configurations)

def test_create_folds():
    """
    Testing the creation of folds
    """

    dataset = {'folds': [{'p': 5, 'n': 6}], 'p': 5, 'n': 6}
    with_folds = _create_folds(dataset)
    assert dataset == with_folds

    dataset = {'p': 5, 'n': 6, 'folding': 'stratified_sklearn'}
    with_folds = _create_folds(dataset)
    assert 'folds' in with_folds

    dataset = {'dataset': 'common_datasets.ADA', 'folding': 'stratified_sklearn'}
    with_folds = _create_folds(dataset)
    assert 'folds' in with_folds

    dataset = {'p': 5, 'n': 6, 'folding': 'stratified_sklearn',
                'score_bounds': {'acc': (0.5, 1.0)},
                'tptn_bounds': {'tp': (0, 5)}}
    with_folds = _create_folds(dataset)
    assert 'folds' in with_folds and 'score_bounds' in with_folds and 'tptn_bounds' in with_folds

    dataset = {'p': 5, 'n': 6, 'folding': 'stratified_sklearn',
                'fold_score_bounds': {'acc': (0.5, 1.0)},
                'fold_tptn_bounds': {'tp': (0, 5)}}
    with_folds = _create_folds(dataset)
    assert 'folds' in with_folds
    assert 'score_bounds' in with_folds['folds'][0]
    assert 'tptn_bounds' in with_folds['folds'][0]

def test_expand_datasets():
    """
    Testing the expanding of datasets
    """
    dataset = {'p': 5, 'n': 6, 'folding': 'stratified_sklearn'}
    expanded = _expand_datasets(dataset)
    assert 'folds' in expanded

    datasets = [{'p': 5, 'n': 6, 'folding': 'stratified_sklearn'},
                {'dataset': 'common_datasets.ADA', 'p': 5, 'n': 6,
                    'folding': 'stratified_sklearn'}]
    expanded = _expand_datasets(datasets)
    assert 'folds' in expanded[0] and 'folds' in expanded[1]

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

def test_random_folds():
    """
    Testing the random folds function
    """

    for idx in range(1, 100):
        assert np.sum(random_folds(100, idx, 5)) == 100

def test_random_configurations():
    """
    Testing the random configurations
    """

    conf = random_configurations(10, 20, 4, 2, random_state=5)
    assert len(conf) == 8

def test_determine_fold_configurations():
    """
    Testing the determination of fold configurations
    """

    conf = determine_fold_configurations(10, 20, 4, 1)

    conf = [(tmp['n'], tmp['p']) for tmp in conf]

    assert conf == stratified_configurations_sklearn(10, 20, 4)

    with pytest.raises(ValueError):
        determine_fold_configurations(10, 20, 4, 1, 'dummy')
