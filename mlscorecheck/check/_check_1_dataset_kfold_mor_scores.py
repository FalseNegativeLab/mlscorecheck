"""
This module implements the top level check function for
scores calculated by the mean-of-ratios aggregation
in a kfold scenario on one single dataset.
"""

from ..aggregated import check_aggregated_scores, Experiment

__all__ = ['check_1_dataset_kfold_mor_scores']

def check_1_dataset_kfold_mor_scores(scores,
                                    eps,
                                    dataset,
                                    solver_name=None,
                                    timeout=None):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to one single dataset and aggregating the figures
    over the folds in the mean of ratios fashion.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty(ies) of the scores
        dataset (dict): the dataset specification
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
    if dataset.get('aggregation', 'mor') != 'mor':
        raise ValueError(f'the aggregation {dataset.get("aggregation")} specified '\
                        'in the dataset specification is not suitable for this test, '\
                        'consider removing the mode of aggregation or specify "rom".')

    # adjusting the dataset specification to ensure the aggregation is set
    dataset = dataset | {'aggregation' : 'mor'}

    if dataset.get('score_bounds') is not None:
        raise ValueError('it is unnecessary to set score bounds at the dataset level, '\
                            'the scores are implicitly bounded by the numerical uncertainty')

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(datasets=[dataset],
                            aggregation='mor')

    return check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=eps,
                                        solver_name=solver_name,
                                        timeout=timeout)
