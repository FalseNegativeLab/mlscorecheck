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
    on one single multiclass dataset. Note that this
    test can only check the consistency of the 'acc', 'sens', 'spec'
    and 'bacc' scores. Note that without bounds, if there is a large
    number of classes, it is likely that there will be a configuration
    matching the scores provided. In order to increase the strength of
    the test, one can add class_scores_bounds if
    for example, besides the average score, the minimum and the maximum
    scores over the classes are also provided.

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
        dict: the dictionary of the results of the analysis, the
        ``inconsistency`` entry indicates if inconsistencies have
        been found. The aggregated_results entry is empty if
        the execution of the linear programming based check was
        unnecessary. The result has four more keys. Under ``lp_status``
        one finds the status of the lp solver, under ``lp_configuration_scores_match``
        one finds a flag indicating if the scores from the lp configuration
        match the scores provided, ``lp_configuration_bounds_match`` indicates
        if the specified bounds match the actual figures and finally
        ``lp_configuration`` contains the actual configuration of the
        linear programming solver.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> from mlscorecheck.check.multiclass import check_1_testset_no_kfold_macro
        >>> testset = {0: 10, 1: 100, 2: 80}
        >>> scores = {'acc': 0.6, 'sens': 0.3417, 'spec': 0.6928, 'f1p': 0.3308}
        >>> results = check_1_testset_no_kfold_macro(scores=scores, testset=testset, eps=1e-4)
        >>> results['inconsistency']
        # False

        >>> scores['acc'] = 0.8464
        >>> results = check_1_testset_no_kfold_macro(scores=scores, testset=testset, eps=1e-4)
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
