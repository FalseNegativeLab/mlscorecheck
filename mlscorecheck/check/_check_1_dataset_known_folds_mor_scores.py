"""
This module implements the top level check function for
scores calculated by the mean-of-ratios aggregation
in a kfold scenario on one single dataset.
"""

from ..core import NUMERICAL_TOLERANCE
from ..aggregated import check_aggregated_scores, Experiment, Evaluation

__all__ = ['check_1_dataset_known_folds_mor_scores']

def check_1_dataset_known_folds_mor_scores(dataset: dict,
                                    folding: dict,
                                    scores: dict,
                                    eps,
                                    fold_score_bounds: dict = None,
                                    *,
                                    solver_name: str = None,
                                    timeout: int = None,
                                    verbosity: int = 1,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
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
        dataset (dict): the specification of the dataset
        folding (dict): the specification of the folding
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        fold_score_bounds (dict): bounds on the scores in the folds
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
        >>> dataset = {'p': 126, 'n': 131}
        >>> folding = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}
        >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}
        >>> result = check_1_dataset_known_folds_mor_scores(dataset=dataset,
                                                            folding=folding,
                                                            scores=scores,
                                                            eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 398, 'n': 569}
        >>> folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6}
        >>> result = check_1_dataset_known_folds_mor_scores(dataset=dataset,
                                                            folding=folding,
                                                            scores=scores,
                                                            eps=1e-2)
        >>> result['inconsistency']
        # True

        >>> dataset = {'dataset_name': 'common_datasets.glass_0_1_6_vs_2'}
        >>> folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1': 0.95}
        >>> result = check_1_dataset_known_folds_mor_scores(dataset=dataset,
                                                            folding=folding,
                                                            fold_score_bounds={'acc': (0.8, 1.0)},
                                                            scores=scores,
                                                            eps=1e-2,
                                                            numerical_tolerance=1e-6)
        >>> result['inconsistency']
        # True
    """

    evaluation = Evaluation(dataset=dataset,
                            folding=folding,
                            fold_score_bounds=fold_score_bounds,
                            aggregation='mor')

    experiment = Experiment(evaluations=[evaluation.to_dict()],
                            aggregation='mor')

    return check_aggregated_scores(experiment=experiment.to_dict(),
                                        scores=scores,
                                        eps=eps,
                                        solver_name=solver_name,
                                        timeout=timeout,
                                        verbosity=verbosity,
                                        numerical_tolerance=numerical_tolerance)
