"""
This module implements the top level check function for
scores calculated by the ratio-of-means aggregation
in a kfold scenario on one single dataset.
"""

import copy

from ..core import NUMERICAL_TOLERANCE
from ..individual import check_scores_tptn_pairs
from ..aggregated import Experiment

__all__ = ['check_n_datasets_rom_kfold_rom_scores']

def check_n_datasets_rom_kfold_rom_scores(scores: dict,
                                            eps,
                                            evaluations: list,
                                            *,
                                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to multiple datasets and aggregating the figures
    over the folds and datasets in the ratio of means fashion. If
    score bounds are specified and some of the 'acc', 'sens', 'spec' and
    'bacc' scores are supplied, the linear programming based check is
    executed to see if the bound conditions can be satisfied.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        evaluations (list[dict]): the specification of the evaluations
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: a summary of the results. When the ``inconsistency`` flag is True, it indicates
        that the set of feasible ``tp``,``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
        Under the key ``n_valid_tptn_pairs`` one finds the number of tp and tn pairs compatible with
        all scores. Under the key ``prefiltering_details`` one finds the results of the prefiltering
        by using the solutions for the score pairs.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> evaluation0 = {'dataset': {'p': 39, 'n': 822},
                            'folding': {'n_folds': 8, 'n_repeats': 4, 'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.winequality-white-3_vs_7'},
                            'folding': {'n_folds': 3, 'n_repeats': 3, 'strategy': 'stratified_sklearn'}}
        >>> experiment = {'evaluations': [evaluation0, evaluation1],
                            'dataset_score_bounds': {'acc': (0.5, 1.0)}}
        >>> scores = {'acc': 0.6586, 'sens': 0.7603, 'spec': 0.6508, 'bacc': 0.7055}
        >>> result = check_n_datasets_mor_kfold_rom_scores(experiment=experiment,
                                                            eps=1e-4,
                                                            scores=scores)
        >>> result['inconsistency']
        # False

        >>> evaluation0 = {'dataset': {'p': 39, 'n': 822},
                            'folding': {'n_folds': 8, 'n_repeats': 4, 'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.winequality-white-3_vs_7'},
                            'folding': {'n_folds': 3, 'n_repeats': 3, 'strategy': 'stratified_sklearn'}}
        >>> experiment = {'evaluations': [evaluation0, evaluation1],
                            'dataset_score_bounds': {'acc': (0.5, 1.0)}}
        >>> scores = {'acc': 0.7586, 'sens': 0.7603, 'spec': 0.6508, 'bacc': 0.7055}
        >>> result = check_n_datasets_mor_kfold_rom_scores(experiment=experiment,
                                                            eps=1e-4,
                                                            scores=scores)
        >>> result['inconsistency']
        # True
    """
    if any(evaluation.get('aggregation', 'rom') != 'rom' for evaluation in evaluations):
        raise ValueError('the aggregation specifications cannot be anything else '\
                            'but "rom"')

    evaluations = copy.deepcopy(evaluations)

    for evaluation in evaluations:
        evaluation['aggregation'] = 'rom'

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(evaluations=evaluations,
                            aggregation='rom')

    # executing the individual tests
    return check_scores_tptn_pairs(scores=scores,
                                            eps=eps,
                                            p=experiment.figures['p'],
                                            n=experiment.figures['n'],
                                            numerical_tolerance=numerical_tolerance)
