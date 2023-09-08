"""
This module implements some functionalities related to fold structures
"""

import itertools
import copy

import numpy as np
from ..experiments import dataset_statistics

__all__ = ['stratified_configurations_sklearn',
            'determine_fold_configurations',
            '_create_folds',
            'fold_variations',
            'remainder_variations',
            'create_all_kfolds']

def stratified_configurations_sklearn(p,
                                        n,
                                        n_splits):
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

def determine_fold_configurations(p,
                                    n,
                                    n_folds,
                                    n_repeats,
                                    folding='stratified_sklearn'):
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

def _create_folds(p,
                    n,
                    *,
                    n_folds=None,
                    n_repeats=None,
                    folding=None,
                    score_bounds=None,
                    identifier=None):
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
        folds = [{'p': p, 'n': n} for _ in range(n_repeats)]

    elif folding is None:
        folds = [{'p': p * n_repeats, 'n': n * n_repeats}]
    else:
        folds = determine_fold_configurations(p,
                                                n,
                                                n_folds,
                                                n_repeats,
                                                folding)
        n_fold = 0
        n_repeat = 0
        for _, fold in enumerate(folds):
            fold['identifier'] = f'{identifier}_{n_repeat}_{n_fold}'
            n_fold += 1
            if n_fold % n_folds == 0:
                n_fold = 0
                n_repeat += 1

    for _, fold in enumerate(folds):
        if score_bounds is not None:
            fold['score_bounds'] = {**score_bounds}

    return folds

def fold_variations(n_items, n_folds, upper_bound_=None):
    """
    Create a list of fold variations for ``n_item`` items and ``n_folds`` folds

    Args:
        n_items (int): the number of items
        n_folds (int): the number of folds

    Returns:
        list(list): the list of potential distributions of ``n_items``
        items into ``n_folds`` folds
    """
    if n_folds == 1:
        return 1, [[n_items]]
    if n_items == 1:
        return 1, [[0]*(n_folds-1) + [1]]

    upper_bound = min(n_items, upper_bound_ if upper_bound_ is not None else n_items)
    lower_bound = n_items // n_folds

    total = 0
    all_configurations = []

    for value in reversed(list(range(lower_bound+1, upper_bound+1))):
        count, configurations = fold_variations(n_items - value, n_folds-1, value)
        total += count
        for conf in configurations:
            conf.append(value)
        all_configurations.extend(configurations)

    return total, all_configurations

def remainder_variations(n_remainders, n_folds):
    """
    Determines the potential distribution of ``n_remainders`` remainders into ``n_folds`` folds.

    Args:
        n_remainders (int): the number of remainders
        n_folds (int): the number of folds

    Returns:
        list(list): the potential distributions of ``n_remainders`` counts into
        ``n_folds`` folds.
    """
    indices = itertools.combinations(list(range(n_folds)), n_remainders)
    combinations = list(indices)

    if len(combinations) == 0:
        return [[0]*n_folds]

    lists = []
    for index in list(combinations):
        tmp = [0] * n_folds
        for idx in index:
            tmp[idx] = 1
        lists.append(tmp)
    return lists

def create_all_kfolds(p, n, n_folds):
    """
    Creates all potential foldings

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int): the number of folds

    Returns:
        list(list), list(list): the counts of positives and negatives in the potential
        foldings. The corresponding rows of the two list are the corresponding positive
        and negative counts.
    """
    total = p + n
    items_per_fold = total // n_folds
    remainder = total % n_folds

    totals = np.array(remainder_variations(remainder, n_folds)) + items_per_fold
    folds = np.array(fold_variations(p, n_folds)[1])

    positive_counts = []
    negative_counts = []

    for total_counts in totals:
        other = total_counts - folds
        fold_mask = np.sum(folds > 0, axis=1)
        other_mask = np.sum(other > 0, axis=1)
        positive_counts.append(folds[(fold_mask == n_folds) & (other_mask == n_folds)])
        negative_counts.append(other[(fold_mask == n_folds) & (other_mask == n_folds)])

    pos, neg = np.vstack(positive_counts).tolist(), np.vstack(negative_counts).tolist()

    return pos, neg

def generate_datasets_with_all_kfolds(dataset):
    """
    From a dataset specification generates all datasets with all possible
    fold specifications.

    Args:
        dataset (dict): a dataset specification

    Returns:
        list(dict): a list of dataset specifications
    """
    if 'folds' in dataset:
        raise ValueError('do not specify the "folds" key for the generation of folds')
    if ('p' in dataset and 'n' not in dataset)\
        or ('p' not in dataset and 'p' in dataset):
        raise ValueError('either specifiy both "p" and "n" or None of them')
    if 'name' in dataset and ('p' in dataset):
        raise ValueError('either specificy "name" or "p" and "n"')

    if 'name' in dataset:
        p = dataset_statistics[dataset["name"]]["p"]
        n = dataset_statistics[dataset["name"]]["n"]
    else:
        p = dataset["p"]
        n = dataset["n"]

    kfolds = create_all_kfolds(p=p, n=n, n_folds=dataset['n_folds'])
    results = []
    for positives, negatives in zip(*kfolds):
        results.append({'folds': [{'p': p_, 'n': n_} for p_, n_ in zip(positives, negatives)]})

    if dataset['n_repeats'] == 1:
        return results

    all_results = copy.deepcopy(results)

    n_repeats = dataset['n_repeats'] - 1
    while n_repeats > 0:
        tmp = []
        for result in all_results:
            for res in results:
                result = copy.deepcopy(result)
                result['folds'].extend(res['folds'])
                tmp.append(result)
        all_results = tmp
        n_repeats -= 1

    if 'fold_score_bounds' in dataset:
        for result in all_results:
            for fold in result['folds']:
                fold['score_bounds'] = {**dataset['fold_score_bounds']}

    return all_results
