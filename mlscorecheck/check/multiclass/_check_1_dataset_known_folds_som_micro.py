"""
This module implements the micro-averaged multiclass tests in a k-fold scenario
with SoM aggregation.
"""

import copy

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import create_folds_multiclass

from ._check_1_testset_no_kfold_micro import check_1_testset_no_kfold_micro

__all__ = ["check_1_dataset_known_folds_som_micro"]


def check_1_dataset_known_folds_som_micro(
    dataset: dict,
    folding: dict,
    scores: dict,
    eps,
    *,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
    prefilter_by_pairs: bool = True
) -> dict:
    """
    This function checks the consistency of scores calculated by taking the micro average of
    class level scores on a single multiclass dataset and averaging across the folds in the
    SoM manner.

    The test is performed by exhaustively testing all possible confusion matrices.

    Args:
        dataset (dict): The specification of the dataset.
        folding (dict): The specification of the folding strategy.
        scores (dict(str,float)): The scores to check.
        eps (float|dict(str,float)): The numerical uncertainty(ies) of the scores.
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
        ValueError: If the provided scores are not consistent with the dataset.

    Examples:
        >>> from mlscorecheck.check.multiclass import check_1_dataset_known_folds_som_micro
        >>> dataset = {0: 86, 1: 96, 2: 59, 3: 105}
        >>> folding = {'folds': [{0: 43, 1: 48, 2: 30, 3: 52}, {0: 43, 1: 48, 2: 29, 3: 53}]}
        >>> scores =  {'acc': 0.6272, 'sens': 0.2543, 'spec': 0.7514, 'f1p': 0.2543}
        >>> result = check_1_dataset_known_folds_som_micro(dataset=dataset,
                                                            folding=folding,
                                                            scores=scores,
                                                            eps=1e-4)
        >>> result['inconsistency']
        # False

        >>> scores['sens'] = 0.2553
        >>> result = check_1_dataset_known_folds_som_micro(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    folds = create_folds_multiclass(dataset, folding)

    testset = copy.deepcopy(folds[0])
    for fold in folds[1:]:
        for key in fold:
            testset[key] += fold[key]

    return check_1_testset_no_kfold_micro(
        testset=testset,
        scores=scores,
        eps=eps,
        numerical_tolerance=numerical_tolerance,
        prefilter_by_pairs=prefilter_by_pairs,
    )
