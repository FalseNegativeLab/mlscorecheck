"""
This module implements some functionalities related to folding
"""

import copy
import itertools

import numpy as np

from ..experiments import dataset_statistics
from ._utils import random_identifier

__all__ = ['stratified_configurations_sklearn',
            'determine_fold_configurations',
            '_create_folds',
            'fold_variations',
            'remainder_variations',
            'create_all_kfolds',
            'generate_datasets_with_all_kfolds',
            '_check_specification_and_determine_p_n',
            'generate_experiments_with_all_kfolds']

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
        folds = [{'p': p,
                    'n': n,
                    'identifier': f'{identifier}_0_r{idx}'} for idx in range(n_repeats)]

    elif folding is None:
        folds = [{'p': p * n_repeats, 'n': n * n_repeats, 'identifier': f'{identifier}_0'}]
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

    for value in reversed(list(range(lower_bound, upper_bound+1))):
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

    return np.vstack(positive_counts).tolist(), np.vstack(negative_counts).tolist()

def _check_specification_and_determine_p_n(dataset, folding):
    """
    Checking if the dataset specification is correct and determining the p and n values

    Args:
        dataset (dict): a dataset specification

    Returns:
        int, int: the number of positives and negatives

    Raises:
        ValueError: if the specification is not suitable
    """
    if folding.get('folds') is not None:
        raise ValueError('do not specify the "folds" key for the generation of folds')
    if (dataset.get('p') is not None and dataset.get('n') is None)\
        or (dataset.get('p') is None and dataset.get('n') is not None):
        raise ValueError('either specifiy both "p" and "n" or None of them')
    if dataset.get('dataset_name') is not None and dataset.get('p') is not None:
        raise ValueError('either specificy "dataset_name" or "p" and "n"')

    p = (dataset_statistics[dataset['dataset_name']]["p"]
            if dataset.get('dataset_name') is not None else dataset["p"])
    n = (dataset_statistics[dataset['dataset_name']]["n"]
            if dataset.get('dataset_name') is not None else dataset["n"])

    return p, n

def create_folding_combinations(evaluations, n_repeats):
    """
    Generates all fold combinations according to n_repeats

    Args:
        evaluations (list(Evaluation)): the list of evaluations with different fold structures
        n_repeats (int): the number of repetitions to add

    Returns:
        list(Evaluations): the list of evaluations with all fold combinations
    """
    all_results = copy.deepcopy(evaluations)

    while n_repeats > 0:
        tmp = []
        for one_result in all_results:
            for res in evaluations:
                tmp_res = copy.deepcopy(one_result)
                to_add = copy.deepcopy(res['folding']['folds'])
                for fold in to_add:
                    fold['identifier'] = f'{fold["identifier"]}_r{n_repeats}'
                tmp_res['folding']['folds'].extend(to_add)

                tmp.append(tmp_res)
        all_results = tmp
        n_repeats -= 1

    return all_results

def generate_datasets_with_all_kfolds(evaluation):
    """
    From a dataset specification generates all datasets with all possible
    fold specifications.

    Args:
        dataset (dict): a dataset specification
        folding (dict): the folding specification

    Returns:
        list(dict): a list of dataset specifications

    Raises:
        ValueError: if the specification is not suitable
    """
    p, n = _check_specification_and_determine_p_n(evaluation.get('dataset'),
                                                    evaluation.get('folding'))

    kfolds = create_all_kfolds(p=p, n=n, n_folds=evaluation['folding'].get('n_folds', 1))
    results = []

    if evaluation['dataset'].get('dataset_name') is not None:
        evaluation['dataset']['identifier'] = \
            f'{evaluation["dataset"]["dataset_name"]}_{random_identifier(3)}'
    else:
        evaluation['dataset']['identifier'] = random_identifier(6)

    for jdx, (positives, negatives) in enumerate(zip(*kfolds)):
        results.append({'dataset': copy.deepcopy(evaluation['dataset']),
                        'folding': {'folds': [
                            {'p': p_,
                            'n': n_,
                            'identifier': f"{evaluation['dataset']['identifier']}_f{idx}_k{jdx}"}
                            for idx, (p_, n_) in enumerate(zip(positives,
                                                                negatives))]}})

    if evaluation['folding'].get('n_repeats', 1) == 1:
        for result in results:
            result['aggregation'] = evaluation.get('aggregation')
        return results

    # multiplicating the folds structures as many times as many repetitions there are
    n_repeats = evaluation['folding'].get('n_repeats', 1) - 1

    all_results = create_folding_combinations(results, n_repeats)

    # adding fold bounds
    if evaluation.get('fold_score_bounds') is not None:
        for result in all_results:
            result['fold_score_bounds'] = {**evaluation['fold_score_bounds']}
    for result in all_results:
        result['aggregation'] = evaluation.get('aggregation')

    return all_results

def generate_experiments_with_all_kfolds(experiment):
    """
    From a dataset specification generates all datasets with all possible
    fold specifications.

    Args:
        dataset (dict): a dataset specification
        folding (dict): the folding specification

    Returns:
        list(dict): a list of dataset specifications

    Raises:
        ValueError: if the specification is not suitable
    """
    evaluations = [generate_datasets_with_all_kfolds(evaluation)
                    for evaluation in experiment['evaluations']]

    results = []

    for evals in itertools.product(*evaluations):
        results.append({'evaluations': evals,
                        'dataset_score_bounds': experiment.get('dataset_score_bounds'),
                        'aggregation': experiment['aggregation']})

    return results
