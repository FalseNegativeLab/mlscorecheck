"""
This module implements the top level check function for
scores calculated by the ratio-of-means aggregation
in a kfold scenario on one single dataset.
"""

import warnings

from ..core import logger, NUMERICAL_TOLERANCE
from ..individual import check_scores_tptn_pairs
from ..aggregated import check_aggregated_scores, Experiment

__all__ = ['check_n_datasets_rom_kfold_rom_scores']

def check_n_datasets_rom_kfold_rom_scores(scores,
                                            eps,
                                            experiment,
                                            *,
                                            solver_name=None,
                                            timeout=None,
                                            verbosity=1,
                                            numerical_tolerance=NUMERICAL_TOLERANCE):
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
        datasets (list[dict]): the dataset specification
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): verbosity of the pulp linear programming solver
                            0: silent, non-zero: verbose
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the dictionary of the results of the analysis, the
        ``inconsistency`` entry indicates if inconsistencies have
        been found. The aggregated_results entry is empty if
        the execution of the linear programming based check was
        unnecessary. The result contains two more keys. Under the key
        ``individual_results`` one finds a structure similar to that of
        the 1_dataset_no_kfold output, summarizing the findings of
        checking the consistency of each pair of scores against a third
        score. Additionally, the key ``aggregated_results`` is available
        if score_bounds are specified and the aggregated checks are
        executed. In the aggregated_results one finds again the
        ``inconsistency`` flag, the status of the linear programming solver
        (``lp_status``) and under the ``lp_configuration`` key one finds
        the actual configuration of tp and tn variables with all
        scores calculated and the score_bounds checked where they
        were specified. Also, under the ``lp_status`` key one finds the
        status message of the linear solver, and in the as a double check, the
        flag under the key ``lp_configuration_scores_match`` indicates if the
        scores of the final configuration match the specified ones, similarly,
        the ``lp_configuration_bounds_match`` indicates if all bounds are
        satisfied.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> datasets = [{'p': 389, 'n': 630,
                        'n_folds': 6, 'n_repeats': 3,
                        'folding': 'stratified_sklearn',
                        'fold_score_bounds': {'acc': (0, 1)}},
                    {'name': 'common_datasets.saheart',
                        'n_folds': 2, 'n_repeats': 5,
                        'folding': 'stratified_sklearn'}]
        >>> scores = {'acc': 0.467, 'sens': 0.432, 'spec': 0.488, 'f1p': 0.373}
        >>> result = check_n_datasets_rom_kfold_rom_scores(scores=scores,
                                                            datasets=datasets,
                                                            eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> datasets = [{'folds': [{'p': 98, 'n': 8},
                                    {'p': 68, 'n': 25},
                                    {'p': 92, 'n': 19},
                                    {'p': 78, 'n': 61},
                                    {'p': 76, 'n': 67}]},
                        {'name': 'common_datasets.zoo-3',
                            'n_folds': 3,
                            'n_repeats': 4,
                            'folding': 'stratified_sklearn'},
                        {'name': 'common_datasets.winequality-red-3_vs_5',
                            'n_folds': 5,
                            'n_repeats': 5,
                            'folding': 'stratified_sklearn'}]
        >>> scores = {'acc': 0.4532, 'sens': 0.6639, 'npv': 0.9129, 'f1p': 0.2082}
        >>> result = check_n_datasets_rom_kfold_rom_scores(scores=scores,
                                                            datasets=datasets,
                                                            eps=1e-4)
        >>> result['inconsistency']
        # False

        >>> datasets = [{'folds': [{'p': 98, 'n': 8},
                                    {'p': 68, 'n': 25},
                                    {'p': 92, 'n': 19},
                                    {'p': 78, 'n': 61},
                                    {'p': 76, 'n': 67}]},
                        {'name': 'common_datasets.zoo-3',
                            'n_folds': 3,
                            'n_repeats': 4,
                            'folding': 'stratified_sklearn'}]
        >>> scores = {'acc': 0.9, 'spec': 0.85, 'ppv': 0.7}
        >>> result = check_n_datasets_rom_kfold_rom_scores(scores=scores,
                                                            datasets=datasets,
                                                            eps=1e-4)
        >>> result['inconsistency']
        # True

    """
    if any(evaluation.get('aggregation', 'rom') != 'rom'
            for evaluation in experiment['evaluations']) or experiment['aggregation'] != 'rom':
        raise ValueError('the aggregation specifications cannot be anything else '\
                            'but "rom"')

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(**experiment)

    # executing the individual tests
    return check_scores_tptn_pairs(scores=scores,
                                            eps=eps,
                                            p=experiment.p,
                                            n=experiment.n,
                                            numerical_tolerance=numerical_tolerance)
