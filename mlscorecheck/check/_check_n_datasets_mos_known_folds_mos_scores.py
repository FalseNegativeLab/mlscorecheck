"""
This module implements the top level check function for
scores calculated by the mean of scores aggregation
in a kfold scenarios and mean of scores aggregation on multiple datastes.
"""

import copy

from ..aggregated import check_aggregated_scores, Experiment
from ..core import NUMERICAL_TOLERANCE

__all__ = ['check_n_datasets_mos_known_folds_mos_scores']

def check_n_datasets_mos_known_folds_mos_scores(evaluations: list,
                                        scores: dict,
                                        eps,
                                        dataset_score_bounds: dict = None,
                                        *,
                                        solver_name: str = None,
                                        timeout: int = None,
                                        verbosity: int = 1,
                                        numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to multiple datasets and aggregating the figures
    over the folds in the mean of scores fashion and over the datasets
    in the mean of scores fashion.

    Args:
        evaluations (list(dict)): the list of evaluation specifications
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the pulp linear programming solver,
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
        >>> evaluation0 = {'dataset': {'p': 118, 'n': 95},
                        'folding': {'folds': [{'p': 22, 'n': 23}, {'p': 96, 'n': 72}]}}
        >>> evaluation1 = {'dataset': {'p': 781, 'n': 423},
                        'folding': {'folds': [{'p': 300, 'n': 200}, {'p': 481, 'n': 223}]}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.61, 'sens': 0.709, 'spec': 0.461, 'bacc': 0.585}
        >>> result = check_n_datasets_mos_known_folds_mos_scores(evaluations=evaluations,
                                                                scores=scores,
                                                                eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> evaluation0 = {'dataset': {'p': 118, 'n': 95},
                        'folding': {'folds': [{'p': 22, 'n': 23}, {'p': 96, 'n': 72}]}}
        >>> evaluation1 = {'dataset': {'p': 781, 'n': 423},
                        'folding': {'folds': [{'p': 300, 'n': 200}, {'p': 481, 'n': 223}]}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.71, 'sens': 0.709, 'spec': 0.461}
        >>> result = check_n_datasets_mos_known_folds_mos_scores(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
        >>> result['inconsistency']
        # True
    """
    if any(evaluation.get('aggregation', 'mos') != 'mos' for evaluation in evaluations):
        raise ValueError('the aggregation specified in each dataset must be "mor" or nothing.')
    if any(evaluation.get('fold_score_bounds') is not None for evaluation in evaluations):
        raise ValueError('do not specify fold_score_bounds through this interface')

    evaluations = copy.deepcopy(evaluations)

    for evaluation in evaluations:
        evaluation['aggregation'] = 'mos'

    experiment = Experiment(evaluations=evaluations,
                            dataset_score_bounds=dataset_score_bounds,
                            aggregation='mos')

    return check_aggregated_scores(experiment=experiment.to_dict(),
                                    scores=scores,
                                    eps=eps,
                                    solver_name=solver_name,
                                    timeout=timeout,
                                    verbosity=verbosity,
                                    numerical_tolerance=numerical_tolerance)
