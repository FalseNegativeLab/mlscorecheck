"""
This module implements some functionalities related to folding
"""

import copy
import itertools

import numpy as np

from sklearn.model_selection import StratifiedKFold

from ..core import logger
from ..experiments import dataset_statistics
from ._utils import random_identifier

__all__ = ['stratified_configurations_sklearn',
            'determine_fold_configurations',
            '_create_folds',
            'integer_partitioning_generator',
            'all_integer_partitioning_generator',
            'fold_partitioning_generator',
            '_check_specification_and_determine_p_n',
            'determine_min_max_p',
            'kfolds_generator',
            'repeated_kfolds_generator',
            'experiment_kfolds_generator',
            'multiclass_stratified_folds',
            'transform_multiclass_fold_to_binary',
            '_create_folds_multiclass',
            '_create_binary_folds_multiclass']

def stratified_configurations_sklearn(p: int,
                                        n: int,
                                        n_splits: int) -> list:
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

def determine_fold_configurations(p: int,
                                    n: int,
                                    n_folds: int,
                                    n_repeats: int,
                                    folding: str = 'stratified_sklearn') -> list:
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
    if folding != 'stratified_sklearn':
        raise ValueError(f'folding strategy {folding} is not supported yet')

    configurations = stratified_configurations_sklearn(p=p, n=n, n_splits=n_folds)
    configurations = [{'n': conf[0], 'p': conf[1]} for conf in configurations]
    return [{**item} for item in configurations for _ in range(n_repeats)]

def _create_folds(p: int,
                    n: int,
                    *,
                    n_folds: int = None,
                    n_repeats: int = None,
                    folding: str = None,
                    score_bounds: dict = None,
                    identifier: str = None) -> list:
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
        for fold in folds:
            fold['identifier'] = f'{identifier}_{n_repeat}_{n_fold}'
            n_fold += 1
            if n_fold % n_folds == 0:
                n_fold = 0
                n_repeat += 1

    for fold in folds:
        if score_bounds is not None:
            fold['score_bounds'] = {**score_bounds}

    return folds

def integer_partitioning_generator(n: int, m: int): # pylint: disable=invalid-name
    """
    Integer partitioning generator

    Integer partitioning algorithm implemented following the algorithm on page 343 in
    https://doi.org/10.1007/978-3-642-14764-7

    Args:
        n (int): the integer to partition
        m (int): the number of partitions

    Yields:
        list: the next configuration
    """
    x = [0] * (m+1) # pylint: disable=invalid-name
    s = [0] * (m+1) # pylint: disable=invalid-name

    for k in range(1, m): # pylint: disable=invalid-name
        x[k] = 1

    x[m] = n - m + 1

    for k in range(1, m + 1): # pylint: disable=invalid-name
        s[k] = x[k] + s[k-1]

    while True:
        yield x[1:]

        u = x[m] # pylint: disable=invalid-name
        k = m # pylint: disable=invalid-name
        while k > 0:
            k -= 1 # pylint: disable=invalid-name
            if x[k] + 2 <= u:
                break

        if k == 0:
            return

        f = x[k] + 1 # pylint: disable=invalid-name
        s_ = s[k-1] # pylint: disable=invalid-name
        while k < m:
            x[k] = f
            s_ += f # pylint: disable=invalid-name
            s[k] = s_
            k += 1 # pylint: disable=invalid-name

        x[m] = n - s[m-1]

def all_integer_partitioning_generator(n, k, non_zero, max_count): # pylint: disable=invalid-name
    """
    Generate all integer partitioning of n to k parts (including 0 parts)

    Args:
        n (int): the integer to partition
        k (int): the maximum number of parts
        non_zero (bool): True if all parts must be greater than zero, False otherwise
        max_count (int): the maximum value a part can take

    Yields:
        list: the list of parts
    """
    if n == 0:
        yield [0] * k
    else:
        lower_bound = min(k, n) if non_zero else 1
        upper_bound = min(k, n)
        for m in range(lower_bound, upper_bound + 1):# pylint: disable=invalid-name
            for positives in integer_partitioning_generator(n, m):
                if all(pos <= max_count for pos in positives):
                    yield [0] * (k - m) + positives

def not_enough_diverse_folds(p_values, n_values):
    """
    Checks if there are enough folds with positive and negative samples

    Args:
        p_values (list): the list of counts of positives
        n_values (list): the list of counts of negatives

    Returns:
        bool: True, if the configuration is incorrect, False otherwise
    """

    return (len(p_values) > 1 and (sum(p_tmp > 0 for p_tmp in p_values) < 2
                                    or sum(n_tmp > 0 for n_tmp in n_values) < 2))

def determine_min_max_p(*, p, n, k_a, k_b, c_a, p_non_zero, n_non_zero): # pylint: disable=too-many-locals
    """
    Determines the minimum and maximum number of positives that can appear in folds
    of type A

    Args:
        p (int): the total number of positives
        n (int): the total number of negatives
        k_a (int): the number of folds of type A
        k_b (int): the number of folds of type B
        c_a (int): the count of elements in folds of type A
        p_non_zero (bool): whether all p should be non-zero
        n_non_zero (bool): whether all n should be non-zero

    Returns:
        int, int: the minimum and maximum number of positives in all folds of
                    type A
    """

    total_a = k_a * c_a

    min_p_a = max(p_non_zero*k_a, total_a - (n - n_non_zero*k_b))
    max_p_a = min(p - p_non_zero*k_b, total_a - n_non_zero*k_a)

    return min_p_a, max_p_a

def fold_partitioning_generator(*, p, n, k, p_non_zero=True, n_non_zero=True, p_min=-1): #pylint: disable=invalid-name,too-many-locals
    """
    Generates the fold partitioning

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        k (int): the number of folds
        p_zero (bool): whether any p can be zero
        n_zero (bool): whether any n can be zero

    Yields:
        list, list: the list of positive and negative counts in folds
    """
    k_div = (p + n) // k
    k_mod = (p + n) % k

    k_b = k - k_mod
    k_a = k_mod
    c_a = k_div + 1
    c_b = k_div

    min_p_a, max_p_a = determine_min_max_p(p=p,
                                            n=n,
                                            k_a=k_a,
                                            k_b=k_b,
                                            c_a=c_a,
                                            p_non_zero=p_non_zero,
                                            n_non_zero=n_non_zero)

    for p_a in range(min_p_a, max_p_a + 1):
        p_b = p - p_a

        for ps_a in all_integer_partitioning_generator(p_a, k_a, p_non_zero, c_a - n_non_zero):

            if any(p_tmp < p_min for p_tmp in ps_a):
                continue

            ns_a = [c_a - tmp for tmp in ps_a]

            for ps_b in all_integer_partitioning_generator(p_b, k_b, p_non_zero, c_b - n_non_zero):

                if any(p_tmp < p_min for p_tmp in ps_b):
                    continue

                ns_b = [c_b - tmp for tmp in ps_b]

                ps_all = ps_a + ps_b
                ns_all = ns_a + ns_b

                if not_enough_diverse_folds(ps_all, ns_all):
                    continue

                yield ps_all, ns_all

def _check_specification_and_determine_p_n(dataset: dict, folding: dict) -> (int, int):
    """
    Checking if the dataset specification is correct and determining the p and n values

    Args:
        dataset (dict): the dataset specification
        folding (dict): the folding specification

    Returns:
        int, int: the number of positives and negatives

    Raises:
        ValueError: if the specification is not suitable
    """
    if folding.get('folds') is not None:
        raise ValueError('do not specify the "folds" key for the generation of folds')
    if (dataset.get('p') is not None and dataset.get('n') is None)\
        or (dataset.get('p') is None and dataset.get('n') is not None):
        raise ValueError('either specify both "p" and "n" or None of them')
    if dataset.get('dataset_name') is not None and dataset.get('p') is not None:
        raise ValueError('either specify "dataset_name" or "p" and "n"')

    p = (dataset_statistics[dataset['dataset_name']]["p"]
            if dataset.get('dataset_name') is not None else dataset["p"])
    n = (dataset_statistics[dataset['dataset_name']]["n"]
            if dataset.get('dataset_name') is not None else dataset["n"])

    return p, n

def kfolds_generator(evaluation: dict,
                        available_scores: list,
                        repeat_idx=0):
    """
    Generates the fold configurations

    Args:
        evaluation (dict): the evaluation to generate the configurations for
        available_scores (list): the list of available scores
        repeat_idx (int): the index of the repeat

    Returns:
        Generator: the generator

    Yields:
        list(dict): the list of fold specifications
    """
    p, n = _check_specification_and_determine_p_n(evaluation.get('dataset'),
                                                    evaluation.get('folding'))

    p_zero = False
    n_zero = False

    if 'sens' not in available_scores and 'bacc' not in available_scores:
        p_zero = True
        logger.info('sens and bacc not among the reported scores, p=0 folds are also considered')
    if 'spec' not in available_scores and 'bacc' not in available_scores:
        n_zero = True
        logger.info('spec and bacc not among the reported scores, n=0 folds are also considered')

    if evaluation['dataset'].get('dataset_name') is not None:
        evaluation['dataset']['identifier'] = \
            f'{evaluation["dataset"]["dataset_name"]}_{random_identifier(3)}'
    else:
        evaluation['dataset']['identifier'] = random_identifier(6)

    for jdx, (p_vals, n_vals) in enumerate(fold_partitioning_generator(p=p,
                                                n=n,
                                                k=evaluation['folding'].get('n_folds', 1),
                                                p_non_zero=not p_zero,
                                                n_non_zero=not n_zero)):
        yield [{'p': p_,
                'n': n_,
                'identifier': f"{evaluation['dataset']['identifier']}_f{idx}_k{jdx}_r{repeat_idx}"}
                for idx, (p_, n_) in enumerate(zip(p_vals, n_vals))]

def repeated_kfolds_generator(evaluation: dict,
                                available_scores: list):
    """
    Generates the evaluation variations

    Args:
        evaluation (dict): the evaluation to generate the configurations for
        available_scores (list): the list of available scores

    Returns:
        Generator: the generator

    Yields:
        dict: one evaluation
    """
    n_repeats = evaluation['folding'].get('n_repeats', 1)
    generators = [kfolds_generator(evaluation, available_scores, idx) for idx in range(n_repeats)]

    if n_repeats > 1:

        for folds in itertools.product(*generators):
            yield {'dataset': copy.deepcopy(evaluation['dataset']),
                    'folding': {
                        'folds': [fold for fold_list in folds for fold in fold_list]},
                    'fold_score_bounds': copy.deepcopy(evaluation.get('fold_score_bounds')),
                    'aggregation': evaluation.get('aggregation')}
    else:
        for fold_list in generators[0]:
            yield {'dataset': copy.deepcopy(evaluation['dataset']),
                    'folding': {
                        'folds': fold_list},
                    'fold_score_bounds': copy.deepcopy(evaluation.get('fold_score_bounds')),
                    'aggregation': evaluation.get('aggregation')}

def experiment_kfolds_generator(experiment: dict,
                                available_scores: list):
    """
    Generates the experiment variations

    Args:
        experiment (dict): the experiment to generate the configurations for
        available_scores (list): the list of available scores

    Returns:
        Generator: the generator

    Yields:
        dict: one experiment
    """
    generators = [repeated_kfolds_generator(evaluation, available_scores)
                    for evaluation in experiment['evaluations']]
    for evaluations in itertools.product(*generators):
        yield {'evaluations': list(evaluations),
                'dataset_score_bounds': experiment.get('dataset_score_bounds'),
                'aggregation': experiment['aggregation']}

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
    y = np.hstack([np.repeat(key, value) for key, value in dataset.items()])
    for _, test in StratifiedKFold(n_splits=n_folds).split(y.reshape(-1, 1), y, y):
        folds.append({idx: count for idx, count in enumerate(np.bincount(y[test]))})

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
    return [{'p': value, 'n': n_total - value} for value in fold.values()]

def _create_folds_multiclass(dataset, folding):
    if folding.get('folds') is not None and (folding.get('n_repeats') is not None \
                                                or folding.get('strategy') is not None
                                                or folding.get('n_folds') is not None):
        raise ValueError('either specify the folds or the folding strategy')

    if 'folds' in folding:
        return folding['folds']
    elif folding.get('strategy') == 'stratified_sklearn':
        folds = multiclass_stratified_folds(dataset, folding.get('n_folds', 1))
    else:
        folds = [dataset]

    n_repeats = folding.get('n_repeats', 1)

    folds = folds * n_repeats
    folds = [copy.deepcopy(fold) for fold in folds]

    return folds

def _create_binary_folds_multiclass(dataset, folding):
    if folding.get('folds') is not None and (folding.get('n_repeats') is not None \
                                                or folding.get('strategy') is not None
                                                or folding.get('n_folds') is not None):
        raise ValueError('either specify the folds or the folding strategy')

    folds = _create_folds_multiclass(dataset, folding)
    folds = [transform_multiclass_fold_to_binary(fold) for fold in folds]
    return [fold for fold_list in folds for fold in fold_list]
