"""
This module implements some functionalities related to fold structures
"""

import numpy as np

from ..experiments import resolve_pn

__all__ = ['stratified_configurations_sklearn',
            'determine_fold_configurations',
            '_create_folds',
            '_expand_datasets',
            'random_folds',
            'random_configurations',
            '_create_folds_pure']

def stratified_configurations_sklearn(p, n, n_splits):
    """
    The sklearn stratification strategy

    Args:
        p (int): number of positives
        n (int): number of negatives
        n_splits (int): the number of splits

    Returns:
        list(tuple): the list of the structure of the folds
    """
    p_base = p // n_splits
    n_base = n // n_splits
    p_remainder = p % n_splits
    n_remainder = n % n_splits

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

def random_folds(n, n_folds, random_state=None):
    """
    Create random folds for n items

    Args:
        n (int): the number of items
        n_folds (int): the number of folds
        random_state (None/int/np.random.RandomState): the random seed or random state to use

    Returns:
        np.array: the list of the number of items in the folds
    """
    if random_state is None or not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    numbers = np.repeat(int(n/n_folds)*2, n_folds).astype(int)

    while np.sum(numbers) != n:
        ridx = random_state.randint(numbers.shape[0])
        if numbers[ridx] > 1:
            numbers[ridx] -= 1

    return numbers

def random_configurations(p, n, n_folds, n_repeats, random_state=None):
    """
    Create random fold configurations

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int): the number of folds
        n_repeats (int): the number of repeats
        random_state (None/int/np.random.RandomState): the random seed or random state to use

    Returns:
        list(dict): the list of the fold configurations
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    configurations = []

    for _ in range(n_repeats):
        folds_p = random_folds(p, n_folds, random_state)
        folds_n = random_folds(n, n_folds, random_state)
        configurations.extend({'p': fn0, 'n': fn1}
                                for fn0, fn1 in zip(folds_p.tolist(), folds_n.tolist()))

    # asserting if the statistics are correct
    total_p, total_n = 0, 0
    for conf in configurations:
        total_p += conf['p']
        total_n += conf['n']
        assert conf['p'] > 0
        assert conf['n'] > 0

    assert total_p == p * n_repeats
    assert total_n == n * n_repeats

    return configurations

def determine_fold_configurations(p, n, n_folds, n_repeats, folding='stratified_sklearn'):
    """
    Determine fold configurations according to a folding

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int): the number of folds
        n_repeats (int): the number of repeats
        folding (str): 'stratified_sklearn' - the folding strategy
    """
    if folding == 'stratified_sklearn':
        confs = stratified_configurations_sklearn(p=p, n=n, n_splits=n_folds)
        confs = [{'n': conf[0], 'p': conf[1]} for conf in confs]
        results = []
        for _ in range(n_repeats):
            for item in confs:
                results.append({**item})
    else:
        raise ValueError(f'folding strategy {folding} is not supported yet')

    return results

def _create_folds_pure(p,
                        n,
                        n_folds,
                        n_repeats,
                        folding,
                        score_bounds=None,
                        tptn_bounds=None,
                        id=None):
    """
    Given a dataset, adds folds to it

    Args:
        dataset (dict): a dataset specification

    Returns:
        dict: the dataset specification with folds
    """

    if n_folds is None:
        n_folds = 1
    if n_repeats is None:
        n_repeats = 1

    folds = determine_fold_configurations(p,
                                    n,
                                    n_folds,
                                    n_repeats,
                                    folding)
    n_fold = 0
    n_repeat = 0
    for idx, fold in enumerate(folds):
        if score_bounds is not None:
            fold['score_bounds'] = {**score_bounds}
        if tptn_bounds is not None:
            fold['tptn_bounds'] = {**tptn_bounds}
        fold['id'] = f'{id}_{n_repeat}_{n_fold}'
        n_fold += 1
        if n_fold % n_folds == 0:
            n_fold = 0
            n_repeat += 1

    return folds

def _create_folds(dataset):
    """
    Given a dataset, adds folds to it

    Args:
        dataset (dict): a dataset specification

    Returns:
        dict: the dataset specification with folds
    """
    results = {}
    if 'folds' not in dataset:
        if 'p' not in dataset:
            dataset = resolve_pn(dataset)
        results['folds'] = determine_fold_configurations(dataset['p'],
                                                            dataset['n'],
                                                            dataset.get('n_folds', 1),
                                                            dataset.get('n_repeats', 1),
                                                            dataset.get('folding'))
        results['p'] = dataset['p'] * dataset.get('n_repeats', 1)
        results['n'] = dataset['n'] * dataset.get('n_repeats', 1)
    else:
        results['folds'] = dataset['folds']
        results['p'] = sum(fold['p'] for fold in dataset['folds'])
        results['n'] = sum(fold['n'] for fold in dataset['folds'])

    if 'score_bounds' in dataset:
        results['score_bounds'] = {**dataset['score_bounds']}
    if 'tptn_bounds' in dataset:
        results['tptn_bounds'] = {**dataset['tptn_bounds']}

    if 'fold_score_bounds' in dataset:
        for fold in results['folds']:
            fold['score_bounds'] = {**dataset['fold_score_bounds']}
    if 'fold_tptn_bounds' in dataset:
        for fold in results['folds']:
            fold['tptn_bounds'] = {**dataset['fold_tptn_bounds']}

    return results

def _expand_datasets(datasets):
    """
    Expand the datasets by adding fold structures

    Args:
        dataset (dict/list(dict)): a dataset specification or a list of specifications

    Returns:
        dict/list(dict): an updated dataset specification or list of specifications
    """
    if isinstance(datasets, dict):
        return _create_folds(datasets)
    return [_create_folds(dataset) for dataset in datasets]
