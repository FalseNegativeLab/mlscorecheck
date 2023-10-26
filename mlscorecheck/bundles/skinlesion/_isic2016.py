"""
This module implements the tests for the ISIC2016 dataset
"""

from ...experiments import get_experiment
from ...check import check_1_testset_no_kfold_scores
from ...core import NUMERICAL_TOLERANCE

__all__ = ['check_isic2016']

def check_isic2016(*,
                    scores: dict,
                    eps: float,
                    numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Tests if the scores are consistent with the test set of the ISIC2016
    melanoma classification dataset

    Args:
        scores (dict): the scores to check
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
        Under the key ``n_valid_tptn_pairs`` one finds the number of tp and tn pairs compatible
        with all scores. Under the key ``prefiltering_details`` one finds the results of the
        prefiltering by using the solutions for the score pairs.

    Examples:
        >>> from mlscorecheck.bundles.skinlesion import check_isic2016
        >>> scores = {'acc': 0.7916, 'sens': 0.2933, 'spec': 0.9145}
        >>> results = check_isic2016(scores=scores, eps=1e-4)
        >>> results['inconsistency']
        # False
    """
    data = get_experiment('skinlesion.isic2016')
    return check_1_testset_no_kfold_scores(scores=scores,
                                            testset=data,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance)
