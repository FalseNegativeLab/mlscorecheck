"""
This module implements the top level check function for
scores calculated by the ratio-of-means aggregation
in a kfold scenario on one single dataset.
"""

from ..core import NUMERICAL_TOLERANCE
from ..individual import check_scores_tptn_pairs
from ..aggregated import Experiment

__all__ = ['check_1_dataset_rom_scores']

def check_1_dataset_rom_scores(dataset: dict,
                                folding: dict,
                                scores: dict,
                                eps,
                                *,
                                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to one single dataset and aggregating the figures
    over the folds in the ratio of means fashion. All pairs of
    the supported individual scores are checked against all other as in
    the 1_dataset_no_kfold case, however, additionally, if score_bounds
    are specified in the folds, the aggregated check is also executed
    on the supported acc, bacc, sens and spec scores.

    Args:
        dataset (dict): the dataset specification
        folding (dict): the folding specification
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
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
        >>> dataset = {'dataset_name': 'common_datasets.monk-2'}
        >>> folding = {'n_folds': 4, 'n_repeats': 3, 'strategy': 'stratified_sklearn'}
        >>> scores = {'spec': 0.668, 'npv': 0.744, 'ppv': 0.667,
                        'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}
        >>> result = check_1_dataset_rom_scores(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 10, 'n': 20}
        >>> folding = {'n_folds': 5, 'n_repeats': 1}
        >>> scores = {'acc': 0.428, 'npv': 0.392, 'bacc': 0.442, 'f1p': 0.391}
        >>> result = check_1_dataset_rom_scores(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-3)
        >>> result['inconsistency']
        # True

    """
    if folding.get('folds') is None and folding.get('strategy') is None:
        # any folding strategy results the same
        folding = {**folding} | {'strategy': 'stratified_sklearn'}

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(evaluations=[{'dataset': dataset,
                                            'folding': folding,
                                            'aggregation': 'rom'}],
                            aggregation='rom')

    # executing the individual tests
    return check_scores_tptn_pairs(scores=scores,
                                            eps=eps,
                                            p=experiment.figures['p'],
                                            n=experiment.figures['n'],
                                            numerical_tolerance=numerical_tolerance)
