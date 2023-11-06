"""
This module implements the consistency testing for multiclass micro averages
supposing one evaluation set
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary

from ..binary import check_1_dataset_kfold_som

__all__ = ["check_1_testset_no_kfold_micro"]


def check_1_testset_no_kfold_micro(
    testset: dict,
    scores: dict,
    eps,
    *,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
    prefilter_by_pairs: bool = True
) -> dict:
    """
    Checking the consistency of scores calculated by taking the micro average of class level
    scores on one single multiclass dataset.

    The test operates by the exhaustive enumeration of all potential confusion matrices.

    Args:
        testset (dict): the specification of the testset
        scores (dict(str,float)): the scores to check ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'), when using
                                    f-beta positive or f-beta negative, also set
                                    'beta_positive' and 'beta_negative'.
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.
        prefilter_by_pairs (bool): whether to prefilter the solution space by pair
                                    solutions when possible to speed up the process

    Returns:
        dict: A dictionary containing the results of the consistency check. The dictionary
        includes the following keys:

            - ``'inconsistency'``:
                A boolean flag indicating whether the set of feasible true
                positive (tp) and true negative (tn) pairs is empty. If True,
                it indicates that the provided scores are not consistent with the experiment.
            - ``'details'``:
                A list providing further details from the analysis of the scores one
                after the other.
            - ``'n_valid_tptn_pairs'``:
                The number of tp and tn pairs that are compatible with all
                scores.
            - ``'prefiltering_details'``:
                The results of the prefiltering by using the solutions for
                the score pairs.
            - ``'evidence'``:
                The evidence for satisfying the consistency constraints.

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

        >>> scores['acc'] = 0.5258
        >>> results = check_1_testset_no_kfold_micro(testset=testset,
                                            scores=scores,
                                            eps=1e-4)
        >>> results['inconsistency']
        # True
    """
    folds = transform_multiclass_fold_to_binary(testset)
    dataset = {
        "p": sum(fold["p"] for fold in folds),
        "n": sum(fold["n"] for fold in folds),
    }

    return check_1_dataset_kfold_som(
        scores=scores,
        eps=eps,
        dataset=dataset,
        folding={"folds": folds},
        numerical_tolerance=numerical_tolerance,
        prefilter_by_pairs=prefilter_by_pairs,
    )
