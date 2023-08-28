"""
This module implements the top level check function for
scores calculated by the ratio-of-means aggregation
in a kfold scenario on one single dataset.
"""

import warnings

from ..core import logger
from ..individual import check_individual_scores
from ..aggregated import check_aggregated_scores, Experiment

__all__ = ['check_1_dataset_kfold_rom_scores']

def check_1_dataset_kfold_rom_scores(scores,
                                        eps,
                                        dataset,
                                        solver_name=None,
                                        timeout=None,
                                        verbosity=1):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to one single dataset and aggregating the figures
    over the folds in the ratio of means fashion.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty(ies) of the scores
        dataset (dict): the dataset specification
        solver_name (None/str): the solver to use
        timeout (None/int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity level of the pulp solver (0: silent, non-0: verbose)

    Returns:
        dict: the dictionary of the results of the analysis, the
                'inconsistency' entry indicates if inconsistencies have
                been found. The aggregated_results entry is empty if
                the execution of the linear programming based check was
                unnecessary.

    Example:
        check_1_dataset_kfold_rom_scores(
            scores = {'acc': 0.954, 'sens': 0.934, 'spec': 0.98},
            eps = 1e-3,
            dataset = {'p': 5, 'n': 70, 'n_folds': 3, 'n_repeats': 10}
        )

        check_1_dataset_kfold_rom_scores(
            scores = {'acc': 0.954, 'sens': 0.934, 'spec': 0.98},
            eps = 1e-3,
            dataset = {'name': 'common_datasets.ADA',
                        'n_folds': 1}
        )

        check_1_dataset_kfold_rom_scores(
            scores = {'acc': 0.954, 'sens': 0.934, 'spec': 0.98},
            eps = 1e-3,
            dataset = {'name': 'common_datasets.ADA',
                        'n_folds': 3,
                        'n_repeats': 1,
                        'folding': 'stratified_sklearn'}
        )

        check_1_dataset_kfold_rom_scores(
            scores = {'acc': 0.954, 'sens': 0.934, 'spec': 0.98},
            eps = 1e-3,
            dataset = {'folds': [{'p': 2, 'n': 20},
                                    {'p': 2, 'n': 30},
                                    {'p': 1, 'n': 20}]}
        )
    """
    if dataset.get('aggregation', 'rom') != 'rom':
        raise ValueError(f'the aggregation {dataset.get("aggregation")} specified '\
                        'in the dataset specification is not suitable for this test, '\
                        'consider removing the mode of aggregation or specify "rom".')

    # adjusting the dataset specification to ensure the aggregation is set
    dataset = dataset | {'aggregation' : 'rom'}

    if dataset.get('score_bounds') is not None:
        raise ValueError('it is unnecessary to set score bounds at the dataset level, '\
                            'the scores are implicitly bounded by the numerical uncertainty')

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(datasets=[dataset],
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
                                        timeout=timeout,
                                        verbosity=verbosity)

    result['aggregated_results'] = agg_results
    result['inconsistency'] = result['inconsistency'] or agg_results['inconsistency']

    return result
