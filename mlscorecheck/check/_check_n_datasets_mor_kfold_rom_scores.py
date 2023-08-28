"""
This module implements the top level check function for
scores calculated by the ratio of means aggregation
in a kfold scenarios and mean of ratios aggregation on multiple datastes.
"""

from ..aggregated import check_aggregated_scores, Experiment

__all__ = ['check_n_datasets_mor_kfold_rom_scores']

def check_n_datasets_mor_kfold_rom_scores(scores,
                                            eps,
                                            datasets,
                                            solver_name=None,
                                            timeout=None):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to multiple datasets and aggregating the figures
    over the folds in the ratio of means fashion and over the datasets
    in the mean of ratios fashion.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty(ies) of the scores
        datasets (list): the dataset specification
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
        raise ValueError('the aggregation specified in each dataset must be "rom" or nothing.')

    # adjusting the dataset specification to ensure the aggregation is set
    for dataset in datasets:
        dataset['aggregation'] = 'rom'

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(datasets=datasets,
                            aggregation='mor')

    return check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=eps,
                                        solver_name=solver_name,
                                        timeout=timeout)
