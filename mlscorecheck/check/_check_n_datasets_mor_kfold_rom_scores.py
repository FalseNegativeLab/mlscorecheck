"""
This module implements the top level check function for
scores calculated by the ratio of means aggregation
in a kfold scenarios and mean of ratios aggregation on multiple datasets.
"""

import copy

from ..aggregated import check_aggregated_scores, Experiment
from ..core import NUMERICAL_TOLERANCE

__all__ = ['check_n_datasets_mor_kfold_rom_scores']

def check_n_datasets_mor_kfold_rom_scores(scores: dict,
                                        eps,
                                        evaluations: list,
                                        dataset_score_bounds: dict = None,
                                        *,
                                        solver_name: str = None,
                                        timeout: int = None,
                                        verbosity: int = 1,
                                        numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to multiple datasets and aggregating the figures
    over the folds in the ratio of means fashion and over the datasets
    in the mean of ratios fashion. This aggregated check can be applied
    only if some of the acc, sens, spec and bacc scores are provided.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        experiment (dict): the experiment specification
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the linear programming solver,
                            0: silent, 1: verbose.
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
        >>> evaluation0 = {'dataset': {'p': 13, 'n': 73},
                            'folding': {'n_folds': 4, 'n_repeats': 1,
                                    'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'p': 7, 'n': 26},
                            'folding': {'n_folds': 3, 'n_repeats': 1,
                                    'strategy': 'stratified_sklearn'}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.343}
        >>> result = check_n_datasets_mor_unknown_folds_mor_scores(evaluations=evaluations,
                                                                scores=scores,
                                                                eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> evaluation0 = {'dataset': {'p': 13, 'n': 73},
                            'folding': {'n_folds': 4, 'n_repeats': 1,
                                    'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'p': 7, 'n': 26},
                            'folding': {'n_folds': 3, 'n_repeats': 1,
                                    'strategy': 'stratified_sklearn'}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.9}
        >>> result = check_n_datasets_mor_unknown_folds_mor_scores(evaluations=evaluations,
                                                                scores=scores,
                                                                eps=1e-3)
        >>> result['inconsistency']
        # True
    """

    if any(evaluation.get('aggregation', 'rom') != 'rom' for evaluation in evaluations):
        raise ValueError('the aggregation specified in each dataset must be "rom" or nothing.')

    if any(evaluation.get('fold_score_bounds') is not None for evaluation in evaluations):
        raise ValueError('do not specify fold_score_bounds for a RoM evaluation')

    evaluations = copy.deepcopy(evaluations)

    for evaluation in evaluations:
        evaluation['aggregation'] = 'rom'

    experiment = Experiment(evaluations=evaluations,
                            dataset_score_bounds=dataset_score_bounds,
                            aggregation='mor')

    return check_aggregated_scores(experiment=experiment.to_dict(),
                                    scores=scores,
                                    eps=eps,
                                    solver_name=solver_name,
                                    timeout=timeout,
                                    verbosity=verbosity,
                                    numerical_tolerance=numerical_tolerance)
