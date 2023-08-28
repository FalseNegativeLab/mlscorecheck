"""
This module implements the top level check function for
scores calculated by the ratio-of-means aggregation
in a kfold scenario on one single dataset.
"""

import warnings

from ..core import logger
from ..individual import check_individual_scores
from ..aggregated import check_aggregated_scores, Experiment

__all__ = ['check_n_datasets_rom_kfold_rom_scores']

def check_n_datasets_rom_kfold_rom_scores(scores,
                                            eps,
                                            datasets,
                                            solver_name=None,
                                            timeout=None):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to one single dataset and aggregating the figures
    over the folds in the ratio of means fashion.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty(ies) of the scores
        datasets (list[dict]): the dataset specification
        solver_name (None/str): the solver to use
        timeout (None/int): the timeout for the linear programming solver in seconds

    Returns:
        dict: the dictionary of the results of the analysis, the
                'inconsistency' entry indicates if inconsistencies have
                been found. The aggregated_results entry is empty if
                the execution of the linear programming based check was
                unnecessary.

    Example:

    """
    if any(dataset.get('aggregation', 'rom') != 'rom' for dataset in datasets):
        raise ValueError('the aggregation specifications cannot be anything else '\
                            'but "rom"')

    # adjusting the dataset specification to ensure the aggregation is set
    for dataset in datasets:
        dataset['aggregation'] = 'rom'

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(datasets=datasets,
                            aggregation='rom')

    # calculating the raw tp, tn, p and n figures
    figures = experiment.calculate_figures()

    # executing the individual tests
    ind_results = check_individual_scores(scores=scores,
                                            eps=eps,
                                            p=figures['p'],
                                            n=figures['n'])

    result = {'inconsistency': ind_results['inconsistency'],
                'individual_results': ind_results}

    if not experiment.has_downstream_bounds():
        logger.info('In the lack of score bounds, the linear programming '\
                    'based aggregated test is not considered.')
        return result

    warnings.warn('It is not a common situation that score bounds '\
                        'can be specified with ratio-of-means aggregation '\
                        'on one single dataset. Please double check the '\
                        'configuration.')

    # for the aggregated figures the original dataset is constructed
    # the difference is that for this step the folding structure is not
    # arbitrary

    agg_results = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=eps,
                                        solver_name=solver_name,
                                        timeout=timeout)

    result['aggregated_results'] = agg_results
    result['inconsistency'] = result['inconsistency'] or agg_results['inconsistency']

    return result
