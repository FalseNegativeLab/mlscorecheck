"""
This module implements the top level check function for
scores calculated by the score-of-means aggregation
in a kfold scenario on multiple datasets.
"""

import copy

from ...core import NUMERICAL_TOLERANCE
from ...individual import check_scores_tptn_pairs
from ...aggregated import Experiment

__all__ = ['check_n_datasets_som_kfold_som']

def check_n_datasets_som_kfold_som(evaluations: list,
                                            scores: dict,
                                            eps,
                                            *,
                                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to multiple datasets and aggregating the figures
    over the folds and datasets in the score of means fashion. If
    score bounds are specified and some of the 'acc', 'sens', 'spec' and
    'bacc' scores are supplied, the linear programming based check is
    executed to see if the bound conditions can be satisfied.

    Args:
        evaluations (list(dict)): the specification of the evaluations
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
        >>> from mlscorecheck.check.binary import check_n_datasets_som_kfold_som
        >>> evaluation0 = {'dataset': {'p': 389, 'n': 630},
                            'folding': {'n_folds': 5, 'n_repeats': 2,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.saheart'},
                            'folding': {'n_folds': 5, 'n_repeats': 2,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.631, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}
        >>> result = check_n_datasets_som_kfold_som(scores=scores,
                                                    evaluations=evaluations,
                                                    eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> evaluation0 = {'dataset': {'p': 389, 'n': 630},
                            'folding': {'n_folds': 5, 'n_repeats': 2,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.saheart'},
                            'folding': {'n_folds': 5, 'n_repeats': 2,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.731, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}
        >>> result = check_n_datasets_som_kfold_som(scores=scores,
                                                    evaluations=evaluations,
                                                    eps=1e-3)
        >>> result['inconsistency']
        # True
    """
    if any(evaluation.get('aggregation', 'som') != 'som' for evaluation in evaluations):
        raise ValueError('the aggregation specifications cannot be anything else '\
                            'but "rom"')

    evaluations = copy.deepcopy(evaluations)

    for evaluation in evaluations:
        evaluation['aggregation'] = 'som'

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
