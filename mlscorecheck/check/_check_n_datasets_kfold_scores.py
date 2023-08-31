"""
This module implements the test for multiple datasets with k-fold cross-validation,
with no assumption on the mode of aggregation (mean-of-ratios/ratio-of-means).
"""

from ..core import NUMERICAL_TOLERANCE
from ._check_n_datasets_mor_kfold_mor_scores import check_n_datasets_mor_kfold_mor_scores
from ._check_n_datasets_rom_kfold_rom_scores import check_n_datasets_rom_kfold_rom_scores
from ._check_n_datasets_mor_kfold_rom_scores import check_n_datasets_mor_kfold_rom_scores

__all__ = ['check_n_datasets_kfold_scores']

def check_n_datasets_kfold_scores(scores,
                                eps,
                                datasets,
                                *,
                                solver_name=None,
                                timeout=None,
                                verbosity=1,
                                numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated in a k-fold cross validation on multiple
    datasets, without knowing if the aggregation was ratio-of-means or mean-of-ratios. The
    function tests the scores by both tests, and unifies the results by the rule that
    if both tests identify inconsistencies, then the configuration is inconsistent,
    however, if any of the tests passes without identifying inconsistencies, then
    the discovery of inconsistencies is not confirmed.

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
        been found. The details of the mean-of-ratios and ratio-of-means
        testing can be found under the ``mor_results`` and ``rom_results``
        keys.

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
        >>> result = check_n_datasets_kfold_scores(scores=scores,
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
        >>> result = check_n_datasets_kfold_scores(scores=scores,
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
        >>> result = check_n_datasets_kfold_scores(scores=scores,
                                                    datasets=datasets,
                                                    eps=1e-4)
        >>> result['inconsistency']
        # True

    """

    results_mor_mor = check_n_datasets_mor_kfold_mor_scores(scores=scores,
                                                        eps=eps,
                                                        datasets=datasets,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)
    results_mor_rom = check_n_datasets_mor_kfold_rom_scores(scores=scores,
                                                        eps=eps,
                                                        datasets=datasets,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)

    results_rom_rom = check_n_datasets_rom_kfold_rom_scores(scores=scores,
                                                        eps=eps,
                                                        datasets=datasets,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)

    return {'inconsistency': results_mor_mor['inconsistency'] \
                            and results_mor_rom['inconsistency']\
                            and results_rom_rom['inconsistency'],
            'mor_mor_results': results_mor_mor,
            'mor_rom_results': results_mor_rom,
            'rom_rom_results': results_rom_rom}
