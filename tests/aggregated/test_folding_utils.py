"""
This module tests the operations related to fold structures
"""

import pytest

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mlscorecheck.aggregated import (stratified_configurations_sklearn,
                                        determine_fold_configurations,
                                        _create_folds,
                                        create_all_kfolds,
                                        generate_evaluations_with_all_kfolds,
                                        fold_partitioning_generator,
                                        _check_specification_and_determine_p_n)

def test_create_all_kfolds():
    """
    Testing the generation of all k-fold configurations
    """
    assert len(create_all_kfolds(5, 7, 3)) == 2

def test_create_all_kfolds_many_p():
    """
    Testing the generation of all k-fold configurations with many p
    """
    assert len(create_all_kfolds(15, 7, 3)) == 2

@pytest.mark.parametrize('random_seed', list(range(30)))
def test_create_all_kfolds_many_p_many_folds(random_seed):
    """
    Testing the generation of all k-fold configurations with many p and many folds

    Args:
        random_seed (int): the random seed of the test
    """

    random_state = np.random.RandomState(random_seed)
    p = random_state.randint(5, 30)
    n = random_state.randint(5, 30)
    n_folds = random_state.randint(2, 6)
    assert len(create_all_kfolds(p, n, n_folds)) == 2

def test_generate_datasets_with_all_kfolds():
    """
    Testing the generation of datasets with all kfold configurations
    """
    evaluation = {'dataset': {'p': 5, 'n': 7}, 'folding': {'n_folds': 3}}

    datasets = generate_evaluations_with_all_kfolds(evaluation,
                                                    available_scores=['acc', 'bacc',
                                                                        'sens', 'spec'])
    assert len(datasets) == 2

    evaluation = {'dataset': {'p': 5, 'n': 7},
                    'folding': {'n_folds': 3, 'n_repeats': 2}}

    datasets = generate_evaluations_with_all_kfolds(evaluation,
                                                    available_scores=['acc', 'bacc',
                                                                        'sens', 'spec'])
    assert len(datasets) == 4

    evaluation = {'dataset': {'p': 5, 'n': 7},
                    'folding': {'n_folds': 3, 'n_repeats': 2},
                    'fold_score_bounds': {'acc': (0.0, 1.0)}}

    datasets = generate_evaluations_with_all_kfolds(evaluation,
                                                    available_scores=['acc', 'bacc',
                                                                        'sens', 'spec'])
    assert 'fold_score_bounds' in datasets[0]

    evaluation = {'dataset': {'dataset_name': 'common_datasets.appendicitis'},
                    'folding': {'n_folds': 3}}

    datasets = generate_evaluations_with_all_kfolds(evaluation,
                                                    available_scores=['acc', 'bacc',
                                                                        'sens', 'spec'])

def test_exceptions():
    """
    Testing if exceptions are thrown
    """

    with pytest.raises(ValueError):
        _check_specification_and_determine_p_n(None, {'folds': []})

    with pytest.raises(ValueError):
        _check_specification_and_determine_p_n({'p': 2}, {})

    with pytest.raises(ValueError):
        _check_specification_and_determine_p_n({'p': 2, 'n': 5, 'dataset_name': 'dummy'}, {})

def test_create_folds():
    """
    Testing the creation of folds
    """
    folds = _create_folds(5, 10, score_bounds={'acc': (0.0, 1.0)},
                            n_repeats=1, n_folds=1)
    assert len(folds) == 1

    folds = _create_folds(p=5, n=10,
                            n_folds=2,
                            n_repeats=2,
                            score_bounds={'acc': (0.0, 1.0)})
    assert len(folds) == 1

    folds = _create_folds(p=5, n=10,
                            n_folds=2,
                            n_repeats=2,
                            folding='stratified_sklearn',
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

    return [tuple(np.bincount(y_labels[test]).tolist())
            for _, test in validator.split(y_labels, y_labels, y_labels)]

@pytest.mark.parametrize('random_state', list(range(500)))
def test_stratified_configurations_sklearn(random_state):
    """
    Testing the determination of the stratified sklearn fold configurations

    Args:
        random_state (int): the random seed to use
    """

    random_state = np.random.RandomState(random_state)

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

    folds = fold_partitioning_generator(6, 6, 3, False, False)

    assert all((not any_zero(fold[0])) and (not any_zero(fold[1])) for fold in folds)

    folds = list(fold_partitioning_generator(6, 6, 3, True, False))

    assert all(not any_zero(fold[1]) for fold in folds)
    assert any(any_zero(fold[0]) for fold in folds)

    folds = list(fold_partitioning_generator(6, 6, 3, False, True))

    assert all(not any_zero(fold[0]) for fold in folds)
    assert any(any_zero(fold[1]) for fold in folds)

    folds = list(fold_partitioning_generator(6, 6, 3, True, True))

    assert any(any_zero(fold[0]) for fold in folds)
    assert any(any_zero(fold[1]) for fold in folds)
