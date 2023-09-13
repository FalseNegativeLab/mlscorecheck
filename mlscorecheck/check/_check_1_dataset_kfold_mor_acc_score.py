"""
This module tests the accuracy score reported from a mean-of-ratios
k-fold evaluation with unknown folds.
"""

import itertools

from ..core import NUMERICAL_TOLERANCE
from ..experiments import resolve_pn

__all__ = ['check_kfold_accuracy_score']

def check_kfold_accuracy_score(dataset,
                                acc,
                                eps,
                                numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Checking if the reported accuracy score can be the mean accuracy score for k folds

    Args:
        dataset (dict): a dataset specification (p,n,n_folds,n_repeats) or (name,n_folds,n_repeats)
        acc (float): the accuracy score to test
        eps (float): the numerical uncertainty in the reported acc score
        numerical_tolerance (float): the additional numerical tolerance

    Returns:
        dict: the result of the analysis, the flag under the key ``inconsistency`` indicates
        if inconsistency have been identified.
    """
    if 'folds' in dataset:
        raise ValueError('This check does not work with arbitrary folds, please '\
                        'specify the dataset by p,n,n_folds,n_repeats or the '\
                        'name of the dataset instead of p and n.')

    if 'name' in dataset:
        dataset = resolve_pn(dataset)

    total = dataset['p'] + dataset['n']

    n_folds = dataset.get('n_folds', 1)
    n_repeats = dataset.get('n_repeats', 1)

    n_items = total // n_folds
    n_high_fold = total % n_folds
    n_low_fold = n_folds - n_high_fold

    total_low = n_items * n_low_fold * n_repeats
    total_high = n_items * n_high_fold * n_repeats

    for true0, true1 in itertools.product(range(total_low + 1), range(total_high + 1)):
        if total_high > 0:
            acc_test = (true0 / total_low * n_low_fold + true1 / total_high * n_high_fold) \
                            / (n_low_fold + n_high_fold)
        elif total_high == 0:
            acc_test = true0 / total_low
        if abs(acc_test - acc) <= eps + numerical_tolerance:
            return {'inconsistency': False,
                    'evidence_n_tptn_low': true0,
                    'evidence_n_tptn_high': true1}

    return {'inconsistency': True}
