"""
This module implements the top level check function for
scores calculated by the mean-of-ratios aggregation
in a kfold scenario on one single dataset.
"""

from ..core import NUMERICAL_TOLERANCE
from ..aggregated import check_aggregated_scores, Experiment

__all__ = ['check_1_dataset_kfold_mor_scores']

def check_1_dataset_kfold_mor_scores(scores,
                                    eps,
                                    dataset,
                                    *,
                                    solver_name=None,
                                    timeout=None,
                                    verbosity=1,
                                    numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to one single dataset and aggregating the figures
    over the folds in the mean of ratios fashion. Note that this
    test can only check the consistency of the 'acc', 'sens', 'spec'
    and 'bacc' scores. Note that without bounds, if there is a large
    number of folds, it is likely that there will be a configuration
    matching the scores provided. In order to increase the strength of
    the test, one can add score_bounds to the individual folds if
    for example, besides the average score, the minimum and the maximum
    scores over the folds are also provided.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        dataset (dict): the dataset specification
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity level of the pulp linear programming solver
                            0: silent, non-zero: verbose.
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
        unnecessary. The result has four more keys. Under ``lp_status``
        one finds the status of the lp solver, under ``lp_configuration_scores_match``
        one finds a flag indicating if the scores from the lp configuration
        match the scores provided, ``lp_configuration_bounds_match`` indicates
        if the specified bounds match the actual figures and finally
        ``lp_configuration`` contains the actual configuration of the
        linear programming solver.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> dataset = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}
        >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}
        >>> result = check_1_dataset_kfold_mor_scores(scores=scores,
                                                    eps=1e-3,
                                                    dataset=dataset)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 398,
                        'n': 569,
                        'n_folds': 4,
                        'n_repeats': 2,
                        'folding': 'stratified_sklearn'}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6}
        >>> result = check_1_dataset_kfold_mor_scores(scores=scores,
                                                    eps=1e-2,
                                                    dataset=dataset)
        >>> result['inconsistency']
        # True

        >>> dataset = {'name': 'common_datasets.glass_0_1_6_vs_2',
                        'n_folds': 4,
                        'n_repeats': 2,
                        'folding': 'stratified_sklearn',
                        'fold_score_bounds': {'acc': (0.8, 1.0)}}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1p': 0.95}
        >>> result = check_1_dataset_kfold_mor_scores(scores=scores,
                                                        eps=1e-2,
                                                        dataset=dataset)
        >>> result['inconsistency']
        # True

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
                                        timeout=timeout,
                                        verbosity=verbosity,
                                        numerical_tolerance=numerical_tolerance)
