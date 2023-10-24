"""
This module implements the top level check function for
scores calculated by the score-of-means aggregation
over multiple testsets.
"""

from ..core import NUMERICAL_TOLERANCE
from ..individual import check_scores_tptn_pairs
from ..aggregated import Experiment, Dataset

__all__ = ['check_n_testsets_som_no_kfold_scores']

def check_n_testsets_som_no_kfold_scores(testsets: list,
                                        scores: dict,
                                        eps,
                                        *,
                                        numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated by aggregating the figures
    over testsets in the score of means fashion. If
    score bounds are specified and some of the 'acc', 'sens', 'spec' and
    'bacc' scores are supplied, the linear programming based check is
    executed to see if the bound conditions can be satisfied.

    Args:
        datasets (list(dict)): the specification of the evaluations
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
        >>> testsets = [{'p': 405, 'n': 223}, {'p': 3, 'n': 422}, {'p': 109, 'n': 404}]
        >>> scores = {'acc': 0.4719, 'npv': 0.6253, 'f1p': 0.3091}
        >>> result = check_n_datasets_som_no_kfold_scores(testsets=testsets,
                                                            scores=scores,
                                                            eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> scores['npv'] = 0.6263
        >>> result = check_n_datasets_som_no_kfold_scores(testsets=testsets,
                                                            scores=scores,
                                                            eps=1e-3)
        >>> result['inconsistency']
        # True
    """

    datasets = [Dataset(**dataset) for dataset in testsets]

    evaluations = [{'dataset': dataset.to_dict(),
                    'folding': {'folds': [{'p': dataset.p, 'n': dataset.n}]},
                    'aggregation': 'mos'}
                    for dataset in datasets]

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(evaluations=evaluations,
                            aggregation='som')

    # executing the individual tests
    return check_scores_tptn_pairs(scores=scores,
                                            eps=eps,
                                            p=experiment.figures['p'],
                                            n=experiment.figures['n'],
                                            numerical_tolerance=numerical_tolerance)
