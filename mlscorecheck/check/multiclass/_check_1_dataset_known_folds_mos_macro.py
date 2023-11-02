"""
This module implements the multiclass tests in a k-fold MoS scenario with micro
averaging of scores.
"""

import copy

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary, _create_folds_multiclass, _create_binary_folds_multiclass

from ..binary import (check_n_datasets_mos_kfold_som,
                        check_n_datasets_mos_known_folds_mos)

__all__ = ['check_1_dataset_known_folds_mos_micro']

def check_1_dataset_known_folds_mos_micro(dataset: dict,
                                    folding: dict,
                                    scores: dict,
                                    eps,
                                    *,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores calculated by taking the micro average
    on one single multiclass dataset. The test follows the methodology of the
    1_dataset_som case.

    Args:
        testset (dict): the specification of the testset
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: a summary of the results. When the ``inconsistency`` flag is True, it indicates
        that the set of feasible ``tp``, ``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
        Under the key ``n_valid_tptn_pairs`` one finds the number of tp and tn pairs compatible with
        all scores. Under the key ``prefiltering_details`` one finds the results of the prefiltering
        by using the solutions for the score pairs.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        from mlscorecheck.check.multiclass import check_1_dataset_known_folds_mos_macro
        >>> dataset = {0: 149, 1: 118, 2: 83, 3: 154}
        >>> folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}
        >>> scores = {'acc': 0.626, 'sens': 0.2483, 'spec': 0.7509, 'f1p': 0.2469}
        >>> result = check_1_dataset_known_folds_mos_macro(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # False

        >>> scores['acc'] = 0.8745
        >>> result = check_1_dataset_known_folds_mos_macro(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    folds = _create_folds_multiclass(dataset, folding)
    binary_folds = [transform_multiclass_fold_to_binary(fold) for fold in folds]

    evaluations = []

    for binary_folding in binary_folds:
        folding = {'folds': binary_folding}
        dataset = {'p': sum(tmp['p'] for tmp in binary_folding),
                    'n': sum(tmp['n'] for tmp in binary_folding)}
        evaluations.append({'dataset': dataset,
                            'folding': folding})

    return check_n_datasets_mos_kfold_som(evaluations=evaluations,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance)
