"""
This module implements the tests for the ISIC2016 dataset
"""

from ....experiments import get_experiment
from ...binary import check_1_testset_no_kfold
from ....core import NUMERICAL_TOLERANCE

__all__ = ['check_isic2016']

def check_isic2016(*,
                    scores: dict,
                    eps: float,
                    numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Tests if the scores are consistent with the test set of the ISIC2016
    melanoma classification dataset

    Args:
        scores (dict): the scores to check ('acc', 'sens', 'spec',
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
    Returns:
        dict: A dictionary containing the results of the consistency check. The dictionary
        includes the following keys:

            - ``'inconsistency'``:
                A boolean flag indicating whether the set of feasible true
                positive (tp) and true negative (tn) pairs is empty. If True,
                it indicates that the provided scores are not consistent with the dataset.
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

    Examples:
        >>> from mlscorecheck.check.bundles.skinlesion import check_isic2016
        >>> scores = {'acc': 0.7916, 'sens': 0.2933, 'spec': 0.9145}
        >>> results = check_isic2016(scores=scores, eps=1e-4)
        >>> results['inconsistency']
        # False
    """
    data = get_experiment('skinlesion.isic2016')
    return check_1_testset_no_kfold(scores=scores,
                                            testset=data,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance,
                                            prefilter_by_pairs=True)
