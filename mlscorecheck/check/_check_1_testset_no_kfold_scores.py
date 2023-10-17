"""
This module implements the top level check function for
scores calculated from raw figures
"""

import warnings

from ..core import logger, NUMERICAL_TOLERANCE
from ..individual import check_scores_tptn_pairs
from ..experiments import dataset_statistics

__all__ = ['check_1_testset_no_kfold_scores']

def check_1_testset_no_kfold_scores(testset: dict,
                                    scores: dict,
                                    eps,
                                    *,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE,
                                    prefilter_by_pairs: bool = True) -> dict:
    """
    Use this check if the scores are calculated on one single test set
    with no kfolding or aggregation over multiple datasets.

    Args:
        testset (dict): the specification of a testset with p, n or its name
        scores (dict(str,float)): the scores to check ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'), when using
                                    f-beta positive or f-beta negative, also set
                                    'beta_positive' and 'beta_negative'.
        eps (float|dict(str,float)): the numerical uncertainty (potentially for each score)
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.
        prefilter_by_pairs (bool): whether to do a prefiltering based on the score-pair tp-tn
                                    solutions (faster)

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
        >>> from mlscorecheck.check import check_1_testset_no_kfold_scores
        >>> testset = {'p': 530, 'n': 902}
        >>> scores = {'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}
        >>> result = check_1_testset_no_kfold_scores(testset=testset,
                                                    scores=scores,
                                                    eps=1e-2)
        >>> result['inconsistency']
        # False

        >>> testset = {'p': 530, 'n': 902}
        >>> scores = {'acc': 0.92, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}
        >>> result = check_1_testset_no_kfold_scores(testset=testset,
                                                    scores=scores,
                                                    eps=1e-2)
        >>> result['inconsistency']
        # True

    """
    logger.info('Use this function if the scores originate from the '\
                'tp and tn statistics calculated on one test set with '\
                'no aggregation of any kind.')

    if ('p' not in testset or 'n' not in testset) and ('name' not in testset):
        raise ValueError('either "p" and "n" or "name" should be specified')

    if ('n_repeats' in testset) or ('n_folds' in testset) \
        or ('folds' in testset) or ('aggregation' in testset):
        warnings.warn('Additional fields beyond ("p", "n") or "name" present ' \
                        'in the specification, you might want to use another check '\
                        'function specialized to datasets (e.g. check_kfold_mos_scores)')

    p = testset.get('p')
    n = testset.get('n')
    if 'name' in testset:
        p = dataset_statistics[testset['name']]['p']
        n = dataset_statistics[testset['name']]['n']

    logger.info('calling the score check with scores %s, uncertainty %s, p %d and n %d',
                str(scores), str(eps), p, n)

    return check_scores_tptn_pairs(scores=scores,
                                    eps=eps,
                                    p=p,
                                    n=n,
                                    numerical_tolerance=numerical_tolerance,
                                    prefilter_by_pairs=prefilter_by_pairs)
