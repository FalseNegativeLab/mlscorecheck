"""
This module implements some functionalities related to folding
"""

import copy
import itertools

from ..experiments import dataset_statistics
from ._utils import random_identifier

__all__ = ['stratified_configurations_sklearn',
            'determine_fold_configurations',
            '_create_folds',
            'integer_partitioning_generator',
            'fold_partitioning_generator',
            'create_all_kfolds',
            'generate_evaluations_with_all_kfolds',
            '_check_specification_and_determine_p_n',
            'generate_experiments_with_all_kfolds']

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
    x = [0]*(m+1) # pylint: disable=invalid-name
    s = [0]*(m+1) # pylint: disable=invalid-name

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
            k = k - 1 # pylint: disable=invalid-name
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

def fold_partitioning_generator(p, n, n_folds):
    """
    Generate the fold configurations

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int): the number of folds

    Yields:
        list, list: the next configurations' ``p`` and ``n`` counts
    """
    max_items = (p + n) // n_folds
    remainder = (p + n) % n_folds

    for ps in integer_partitioning_generator(p, n_folds): # pylint: disable=invalid-name
        if (sum(item > max_items for item in ps) != 0
                or sum(item == max_items for item in ps) > remainder):
            continue

        n_ordinary = len(ps) - sum(item == max_items for item in ps)
        ns = [max_items - p_val + (idx >= n_ordinary) for idx, p_val in enumerate(ps)] # pylint: disable=invalid-name

        # distributing the remainders between 0:idx
        combinations = {tuple(x)
                        for x in itertools.combinations(ps[:n_ordinary],
                                                        remainder - (len(ps) - n_ordinary))}

        n_variants = []
        for comb in combinations:
            tmp = copy.copy(ns)
            cdx = len(comb)-1
            pdx = n_ordinary-1
            while cdx >= 0 and pdx >= 0:
                if comb[cdx] == ps[pdx]:
                    tmp[pdx] += 1
                    pdx -= 1
                    cdx -= 1
                elif comb[cdx] <= ps[pdx]:
                    pdx -= 1
                # this cannot happen seemingly:
                #elif ps[pdx] <= comb[cdx]:
                #    cdx -= 1

            n_variants.append(tmp)

        for ns in n_variants: # pylint: disable=invalid-name
            yield ps, ns

def create_all_kfolds(p: int, n: int, n_folds: int) -> (list, list):
    """
    Create all kfold configurations

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int): the number of folds

    Returns:
        list, list: the lists of the counts of positives and negatives, the corresponding
        entries of the list describe one folding
    """
    positives = []
    negatives = []

    for pos, neg in fold_partitioning_generator(p, n, n_folds):
        positives.append(tuple(pos))
        negatives.append(tuple(neg))

    return positives, negatives

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
        raise ValueError('either specifiy both "p" and "n" or None of them')
    if dataset.get('dataset_name') is not None and dataset.get('p') is not None:
        raise ValueError('either specificy "dataset_name" or "p" and "n"')

    p = (dataset_statistics[dataset['dataset_name']]["p"]
            if dataset.get('dataset_name') is not None else dataset["p"])
    n = (dataset_statistics[dataset['dataset_name']]["n"]
            if dataset.get('dataset_name') is not None else dataset["n"])

    return p, n

def create_folding_combinations(evaluations: dict, n_repeats: dict) -> list:
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

def generate_evaluations_with_all_kfolds(evaluation: dict) -> list:
    """
    From a dataset specification generates all datasets with all possible
    fold specifications.

    Args:
        evaluation (dict): the specification of an evaluation

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

    print(kfolds)

    for jdx, (positives, negatives) in enumerate(zip(*kfolds)):
        print(positives, negatives)
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

def generate_experiments_with_all_kfolds(experiment: dict) -> list:
    """
    From a dataset specification generates all datasets with all possible
    fold specifications.

    Args:
        experiment (dict): the specification of an experiment

    Returns:
        list(dict): a list of dataset specifications

    Raises:
        ValueError: if the specification is not suitable
    """
    evaluations = [generate_evaluations_with_all_kfolds(evaluation)
                    for evaluation in experiment['evaluations']]

    results = []

    for evals in itertools.product(*evaluations):
        results.append({'evaluations': evals,
                        'dataset_score_bounds': experiment.get('dataset_score_bounds'),
                        'aggregation': experiment['aggregation']})

    return results
