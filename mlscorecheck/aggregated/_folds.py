"""
This module implements some functionalities related to fold structures
"""

import numpy as np

from ..datasets import _resolve_pn

__all__ = ['stratified_configurations_sklearn',
            'determine_fold_configurations',
            '_create_folds',
            '_expand_datasets',
            'random_folds',
            'random_configurations']

def stratified_configurations_sklearn(n0, n1, n_splits):
    n0_base = n0 // n_splits
    n1_base = n1 // n_splits
    n0_remainder = n0 % n_splits
    n1_remainder = n1 % n_splits

    results = [(n0_base, n1_base)] * n_splits

    idx = 0
    while n0_remainder > 0:
        results[idx] = (results[idx][0] + 1, results[idx][1])
        n0_remainder -= 1
        idx += 1
        idx %= n_splits
    while n1_remainder > 0:
        results[idx] = (results[idx][0], results[idx][1] + 1)
        n1_remainder -= 1
        idx += 1
        idx %= n_splits

    return results

def random_folds(n, n_folds, random_state):
    cum = np.cumsum(random_state.random_sample(n_folds))
    cum = cum / cum[-1]
    numbers = np.diff(np.hstack([[0], (n*cum).astype(int)]))
    numbers += 1

    while np.sum(numbers) != n:
        if np.sum(numbers) < n:
            numbers[random_state.randint(numbers.shape[0])] += 1
        else:
            ridx = random_state.randint(numbers.shape[0])
            if numbers[ridx] > 1:
                numbers[ridx] -= 1

    return numbers

def random_configurations(n0, n1, n_folds, n_repeats, random_state=None):
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        random_state = random_state

    configurations = []

    for _ in range(n_repeats):
        folds_n0 = random_folds(n0, n_folds, random_state)
        folds_n1 = random_folds(n1, n_folds, random_state)
        for fn0, fn1 in zip(folds_n0.tolist(), folds_n1.tolist()):
            configurations.append({'p': fn0, 'n': fn1})

    total_p, total_n = 0, 0
    for conf in configurations:
        total_p += conf['p']
        total_n += conf['n']
        assert conf['p'] > 0
        assert conf['n'] > 0
    assert total_p == n0 * n_repeats
    assert total_n == n1 * n_repeats

    return configurations

def determine_fold_configurations(p, n, n_folds, n_repeats, folding='stratified_sklearn'):
    if folding is None:
        folding = 'stratified_sklearn'

    if folding == 'stratified_sklearn':
        confs = stratified_configurations_sklearn(p, n, n_folds)
        confs = [{'p': conf[0], 'n': conf[1]} for conf in confs]
        results = []
        for _ in range(n_repeats):
            for item in confs:
                results.append({**item})

    return results

def _create_folds(dataset):
    results = {}
    if 'folds' not in dataset:
        if 'p' not in dataset:
            dataset = _resolve_pn(dataset)
        results['folds'] = determine_fold_configurations(dataset['p'],
                                                            dataset['n'],
                                                            dataset.get('n_folds', 1),
                                                            dataset.get('n_repeats', 1))
        results['p'] = dataset['p'] * dataset.get('n_repeats', 1)
        results['n'] = dataset['n'] * dataset.get('n_repeats', 1)

        if len(results['folds']) == 1 and 'tptn_bounds' in dataset:
            results['folds'][0]['tptn_bounds'] = dataset['tptn_bounds']
            results['tptn_bounds'] = dataset['tptn_bounds']
        if 'score_bounds' in dataset:
            for fold in results['folds']:
                fold['score_bounds'] = dataset['score_bounds']
            results['score_bounds'] = dataset['score_bounds']
    else:
        results['folds'] = dataset['folds']
        results['p'] = sum(fold['p'] for fold in dataset['folds'])
        results['n'] = sum(fold['n'] for fold in dataset['folds'])
    return results

def _expand_datasets(datasets):
    if isinstance(datasets, dict):
        return _create_folds(datasets)
    return [_create_folds(dataset) for dataset in datasets]

