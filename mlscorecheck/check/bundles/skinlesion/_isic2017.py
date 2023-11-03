"""
This module implements the tests for the ISIC2017 dataset
"""

from ....experiments import get_experiment
from ...binary import check_1_testset_no_kfold
from ....core import NUMERICAL_TOLERANCE

__all__ = ['check_isic2017',
            '_prepare_testset_isic2017']

def _prepare_testset_isic2017(target, against):
    """
    Preperation of the test set

    Args:
        target (str|list): the target (positive) class(es), with the encoding 'M' for melanoma,
                            'SK' for seborrheic keratosis and 'N' for nevus.
        against (str|list): specification of the negative classes, with the encoding 'M' for
                            melanoma, 'SK' for seborrheic keratosis and 'N' for nevus.

    Returns:
        dict: the testset
    """
    data = get_experiment('skinlesion.isic2017')

    target = [target] if isinstance(target, str) else target
    against = [against] if isinstance(against, str) else against

    mapping = {'M': 'melanoma',
                'SK': 'seborrheic keratosis',
                'N': 'nevus'}

    return {'p': sum(data[mapping[tmp]] for tmp in target),
            'n': sum(data[mapping[tmp]] for tmp in against)}

def check_isic2017(*,
                    target,
                    against,
                    scores: dict,
                    eps: float,
                    numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Tests if the scores are consistent with the test set of the ISIC2017
    skin lesion classification dataset. The dataset contains three classes,
    the test covers the binary classification aspect of the problem, when
    one (or two) of the classes are classified against the other two (or one)
    class.

    Args:
        target (str|list): the target (positive) class(es), with the encoding 'M' for melanoma,
                            'SK' for seborrheic keratosis and 'N' for nevus.
        against (str|list): specification of the negative classes, with the encoding 'M' for
                            melanoma, 'SK' for seborrheic keratosis and 'N' for nevus.
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
        >>> from mlscorecheck.check.bundles.skinlesion import check_isic2017
        >>> scores = {'acc': 0.6183, 'sens': 0.4957, 'ppv': 0.2544, 'f1p': 0.3362}
        >>> results = check_isic2017(target='M',
                            against=['SK', 'N'],
                            scores=scores,
                            eps=1e-4)
        >>> results['inconsistency']
        # False
    """

    testset = _prepare_testset_isic2017(target, against)

    return check_1_testset_no_kfold(scores=scores,
                                            testset=testset,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance,
                                            prefilter_by_pairs=True)
