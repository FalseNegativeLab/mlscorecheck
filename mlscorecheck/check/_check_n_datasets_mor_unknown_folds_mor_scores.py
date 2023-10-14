"""
This module implements the top level check function for
scores calculated by the mean of ratios aggregation
in a kfold scenarios and mean of ratios aggregation on multiple datasets.
"""

import copy

import numpy as np

from ._check_n_datasets_mor_known_folds_mor_scores \
            import check_n_datasets_mor_known_folds_mor_scores
from ._check_1_dataset_unknown_folds_mor_scores import estimate_n_evaluations
from ..core import NUMERICAL_TOLERANCE
from ..aggregated import generate_experiments_with_all_kfolds

__all__ = ['check_n_datasets_mor_unknown_folds_mor_scores',
            'estimate_n_experiments']

def estimate_n_experiments(evaluations: list) -> int:
    """
    Estimates the number of estimations with different fold combinations.

    Args:
        evaluations (list): a list of evaluation specifications

    Returns:
        int: the estimated number of different fold configurations.
    """
    counts = [estimate_n_evaluations(dataset=evaluation['dataset'],
                                        folding=evaluation['folding'])
                for evaluation in evaluations]
    return np.prod(counts)

def check_n_datasets_mor_unknown_folds_mor_scores(evaluations: list,
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
    over the folds in the mean of ratios fashion and over the datasets
    in the mean of ratios fashion.

    Note that depending on the number of the minority instances and on the
    folding structure, this test might lead to enormous execution times.
    Use the function ``estimate_n_experiments`` to get a rough upper bound estimate
    on the number of experiments with different fold combinations.

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
        been found. The ``details`` entry contains all possible folding
        combinations and the corresponding detailed results.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> from mlscorecheck.check import check_n_datasets_mor_unknown_folds_mor_scores
        >>> evaluation0 = {'dataset': {'p': 13, 'n': 73},
                        'folding': {'n_folds': 4, 'n_repeats': 1}}
        >>> evaluation1 = {'dataset': {'p': 7, 'n': 26},
                        'folding': {'n_folds': 3, 'n_repeats': 1}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.343}
        >>> result = check_n_datasets_mor_unknown_folds_mor_scores(evaluations=evaluations,
                                                                scores=scores,
                                                                eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> evaluation0 = {'dataset': {'p': 13, 'n': 73},
                        'folding': {'n_folds': 4, 'n_repeats': 1}}
        >>> evaluation1 = {'dataset': {'p': 7, 'n': 26},
                        'folding': {'n_folds': 3, 'n_repeats': 1}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.9}
        >>> result = check_n_datasets_mor_unknown_folds_mor_scores(evaluations=evaluations,
                                                                scores=scores,
                                                                eps=1e-3)
        >>> result['inconsistency']
        # True
    """
    if any(evaluation.get('aggregation', 'mor') != 'mor' for evaluation in evaluations):
        raise ValueError('the aggregation specified in each dataset must be "mor" or nothing.')
    if any(evaluation.get('fold_score_bounds') is not None for evaluation in evaluations):
        raise ValueError('do not specify fold score bounds through this interface')

    evaluations = copy.deepcopy(evaluations)

    for evaluation in evaluations:
        evaluation['aggregation'] = 'mor'

    experiment = {'evaluations': evaluations,
                    'dataset_score_bounds': dataset_score_bounds,
                    'aggregation': 'mor'}

    experiments = generate_experiments_with_all_kfolds(experiment=experiment,
                                                        available_scores=list(scores.keys()))

    results = {'details': [],
                'inconsistency': True}

    for experiment in experiments:
        result = check_n_datasets_mor_known_folds_mor_scores(
                                    evaluations=experiment['evaluations'],
                                    dataset_score_bounds=experiment.get('dataset_score_bounds'),
                                    scores=scores,
                                    eps=eps,
                                    timeout=timeout,
                                    solver_name=solver_name,
                                    verbosity=verbosity,
                                    numerical_tolerance=numerical_tolerance)

        results['details'].append(result)
        results['inconsistency'] = results['inconsistency'] and result['inconsistency']

        if not result['inconsistency']:
            break

    return results
