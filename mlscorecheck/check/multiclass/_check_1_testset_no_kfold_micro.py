"""
This module implements the consistency testing for multiclass micro averages
supposing one evaluation set
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary

from ..binary import check_1_dataset_som

__all__ = ['check_1_testset_no_kfold_micro']

def check_1_testset_no_kfold_micro(testset: dict,
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
        >>> from mlscorecheck.check.multiclass import check_1_testset_no_kfold_micro
        >>> testset = {0: 10, 1: 100, 2: 80}
        >>> scores = {'acc': 0.5158, 'sens': 0.2737, 'spec': 0.6368,
            'bacc': 0.4553, 'ppv': 0.2737, 'npv': 0.6368}
        >>> results = check_1_testset_no_kfold_micro(testset=testset,
                                            scores=scores,
                                            eps=1e-4)
        >>> results['inconsistency']
        # False

        >>> scores['acc'] = 0.8184
        >>> results = check_1_testset_no_kfold_micro(testset=testset,
                                            scores=scores,
                                            eps=1e-4)
        >>> results['inconsistency']
        # True
    """
    folds = transform_multiclass_fold_to_binary(testset)
    dataset = {'p': sum(fold['p'] for fold in folds),
                'n': sum(fold['n'] for fold in folds)}

    return check_1_dataset_som(scores=scores,
                                eps=eps,
                                dataset=dataset,
                                folding={'folds': folds},
                                numerical_tolerance=numerical_tolerance)
